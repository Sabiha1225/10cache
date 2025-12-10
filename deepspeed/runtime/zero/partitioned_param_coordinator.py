# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from dataclasses import dataclass
import collections
from collections import UserDict
from typing import Deque, Set

from deepspeed import comm as dist
from deepspeed.utils import z3_leaf_module
from deepspeed.utils.logging import logger
from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum
from deepspeed.runtime.zero.partition_parameters import *
from deepspeed.runtime.zero.partitioned_param_profiler import PartitionedParameterProfiler
from deepspeed.runtime.swap_tensor.partitioned_param_swapper import PartitionedParamStatus
from deepspeed.utils.debug import debug_module2name_id, debug_param2name_id
from deepspeed.accelerator import get_accelerator
import deepspeed.runtime.compiler as compiler

from deepspeed.runtime.utils import prefetchtable, memory_manager, active_tensor_window, params_id_list, gpu_threshold, GPU_MEMORY_CACHE, CPU_MEMORY_CACHE

import logging
import time
import csv
import pandas as pd

ENABLE_PROFILER = False


def debug_rank0(message: str) -> None:
    if dist.get_rank() == 0:
        logger.debug(message)


@instrument_w_nvtx
def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())


def iter_params(module: Module, recurse=False) -> Iterable[Parameter]:
    return map(lambda pair: pair[1], get_all_parameters(module, recurse))


class ZeRoTraceMode(Enum):
    # Record trace of the network during a single forward+backward (for training) or forward (for inference)
    RECORD = 1
    # Use recorded network trace to optimize current forward+backward or forward
    COMPLETE = 2
    # Recorded trace does not match current forward+backward or forward pass.
    INVALID = 3


class InflightParamRegistry(UserDict):
    """registry for parameters in flight"""

    def __setitem__(self, param: Parameter, handle: AllGatherCoalescedHandle) -> None:
        if param in self.data:
            raise RuntimeError(f"{param.ds_summary()} already in registry")
        if param.ds_status != ZeroParamStatus.INFLIGHT:
            raise RuntimeError(f"attempted to add non-inflight parameter to registry {param.ds_summary()}")
        self.data[param] = handle


class PartitionedParameterCoordinator:
    FORWARD_FETCH_SUBMIT = 'forward_fetch_submit'
    FORWARD_FETCH_WAIT = 'forward_fetch_wait'
    FORWARD_PREFETCH_SUBMIT = 'forward_prefetch_submit'
    BACKWARD_FETCH_SUBMIT = 'backward_fetch_submit'
    BACKWARD_FETCH_WAIT = 'backward_fetch_wait'
    BACKWARD_PREFETCH_SUBMIT = 'backward_prefetch_wait'
    FORWARD_ALL_GATHER = 'forward_all_gather'
    BACKWARD_ALL_GATHER = 'backward_all_gather'
    """Handles partitioning and gathering of parameters."""

    @dataclass
    class __ParamInTrace:
        param: Parameter
        step_id_last_used_at: int

    def __init__(
        self,
        prefetch_bucket_sz: int,
        max_reuse_distance_in_numel: int,
        max_available_parameters_in_numel: int,
        allgather_stream: get_accelerator().Stream,
        inflight_param_registry: InflightParamRegistry,
        prefetch_nvme: bool = False,
        timers=None,
        zero_quantized_weights=False,
        zero_quantized_nontrainable_weights=False,
    ) -> None:
        # mapping of param -> handle for each param that is currently in flight
        self.__inflight_param_registry = inflight_param_registry
        # keeps track of the number of submodules invoked so far.
        self.__step_id: int = 0
        # network tracing mode
        self.__trace_mode: ZeRoTraceMode = ZeRoTraceMode.RECORD
        # sequence of submodules/parameters in forward pass + backward pass
        self.__submodule_order: Iterable[Module] = []
        self.__param_order: Iterable[__class__.__ParamInTrace] = []
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(lambda: int(-1e10))
        self.__step_id_module_fetched_for = collections.defaultdict(lambda: collections.deque())
        # number of available params, and max number of available params
        self.__n_available_params: int = 0
        self.__max_n_available_params: int = max_available_parameters_in_numel
        # max distance between two use of the module beyond which module is released
        self.__max_reuse_dist_in_numel: int = max_reuse_distance_in_numel
        # queue for parameters to fetch. parameters will be popped off the left
        # side of the dequeue as they are fetched
        self.__param_queue: Deque[__class__.__ParamInTrace] = None
        self.__prefetch_bucket_sz: int = prefetch_bucket_sz
        self.__prefetch_nvme: bool = prefetch_nvme
        self.hierarchy: int = 0
        self.zero_quantized_weights = zero_quantized_weights
        self.zero_quantized_nontrainable_weights = zero_quantized_nontrainable_weights
        self.timers = timers

        # stream that will be used for allgather operations
        self.__allgather_stream: get_accelerator().Stream = allgather_stream

        # limit the number of fetch events that can be queued at once
        # otherwise, what happens is memory is allocated by the host thread at the
        # time of the call, but not used until later by the asynchronous cuda stream.
        # allowing an infinite number of these to queue up causes a lot of memory
        # pressure that then becomes detrimental to performance.
        # this is a much less elegant way of fixing this vs something like using
        # cudaMallocAsync/cudaFreeAsync. Choosing to not expose this to the user now
        # because ideally in the future its replaced by an async allocation
        # mechanism which doesn't require any configuration by the user.
        self.__ongoing_fetch_events: Deque[get_accelerator().Event] = collections.deque()
        # TODO. make this configurable via JSON
        self.__max_ongoing_fetch_events: int = 2
        self.__profiler = PartitionedParameterProfiler(timers if ENABLE_PROFILER else None)
        self.release_param = True
        self.tensor_size_dict = {}
        self.prefetch_from_nvme = set()
        self.new_prefetch_from_nvme = set()

    """Tracing and Tracking
    TODO. consider performing trace before initializing PartitionedParameterCoordinator
    and passing trace results into constructor. This way all the code in here can
    just assume that the trace is complete and the results can be entirely
    immutable.

    Bookkeeping operations used to track where we are in the forward/backward pass
    """

    def _clear_trace_structures(self) -> None:
        self.__submodule_order = []
        self.__param_order = []
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(lambda: int(-1e10))
        self.__param_queue = None

    def is_complete_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.COMPLETE

    def is_invalid_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.INVALID

    def is_record_trace(self) -> bool:
        return self.__trace_mode == ZeRoTraceMode.RECORD

    def _invalidate_trace(self) -> None:
        if self.is_invalid_trace():
            raise RuntimeError("attempted to invalidate already invalid trace")
        self.__trace_mode = ZeRoTraceMode.INVALID
        self._clear_trace_structures()

    def trace_prologue(self, sub_module: Module) -> None:
        if self.is_complete_trace():
            # sub_module must match expectation else invalidate trace cache
            if len(self.__submodule_order) <= self.__step_id:
                print_rank_0(
                    f"Invalidate trace cache @ step {self.__step_id} and module {sub_module.id}: "
                    f"cache has only {len(self.__submodule_order)} modules",
                    force=True)
                self._invalidate_trace()
                return

            if sub_module != self.__submodule_order[self.__step_id]:
                expected_module_id = self.__submodule_order[self.__step_id].id
                print_rank_0(
                    f"Invalidate trace cache @ step {self.__step_id}: "
                    f"expected module {expected_module_id}, but got module {sub_module.id}",
                    force=True)
                self._invalidate_trace()

    @compiler.disable
    def record_module(self, sub_module: Module) -> None:
        """adds sub module to trace"""
        if not self.is_record_trace():
            raise RuntimeError(f"attempted to record trace when status = {self.__trace_mode}")

        self.__submodule_order.append(sub_module)
        self.__step_id_module_fetched_for[sub_module.id].append(self.__step_id)

    def record_parameters(self, sub_module: Module) -> None:
        """adds sub module to trace"""
        if not self.is_record_trace():
            raise RuntimeError(f"attempted to record trace when status = {self.__trace_mode}")

        step_id = self.__step_id_module_fetched_for[sub_module.id].popleft()
        for param in sorted(set(iter_params(sub_module, recurse=z3_leaf_module(sub_module))), key=lambda p: p.ds_id):
            self.__param_order.append(__class__.__ParamInTrace(param=param, step_id_last_used_at=step_id))

    def construct_parameter_trace_from_module_trace(self):
        """use module trace to construct parameter trace"""
        self.__param_order = []
        for sub_module in self.__submodule_order:
            self.record_parameters(sub_module)

    def check_active_tensor_window(self):
        prefetch_information_list = prefetchtable.get_prefetch_information_list()
        param_to_remove_id = prefetch_information_list[prefetchtable.current_row].tensor_id
        param_to_add_id = prefetch_information_list[prefetchtable.current_row - 1].tensor_id
        index1 = prefetchtable.get_tensor_id_to_prefetch_list_id(param_to_add_id, "fwd")
        param_to_add_current_location = prefetchtable.get_prefetch_information_list_item(index1).current_location
        index1 = prefetchtable.get_tensor_id_to_prefetch_list_id(param_to_remove_id, "fwd")
        param_to_remove_current_location = prefetchtable.get_prefetch_information_list_item(index1).current_location

        if param_to_add_id not in active_tensor_window:
            if param_to_remove_id in active_tensor_window:
                self.__release_param(params_id_list[param_to_remove_id])
                active_tensor_window.remove(param_to_remove_id)
            
            active_tensor_window.append(param_to_add_id)
            param = params_id_list[param_to_add_id]
            tensor_element = prefetch_information_list[prefetchtable.current_row - 1].tensor_element
            gpu_buffer_id = GPU_MEMORY_CACHE.get_free_buffer_id(tensor_element)
            GPU_MEMORY_CACHE.add_param_id_to_buffer_id(param.ds_id, gpu_buffer_id)
            GPU_MEMORY_CACHE.add_occupied_buffer_id(param.ds_id, tensor_element)

            if param_to_add_current_location == OffloadDeviceEnum.nvme:
                param.nvme_swapper.swap_in([param], async_op=False)
                new_tensor = GPU_MEMORY_CACHE.add(param.ds_tensor, gpu_buffer_id)
                param.nvme_swapper.remove_partition_and_release_buffers([param])
                param.ds_tensor.data = new_tensor
                param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
                param.ds_tensor.final_location = OffloadDeviceEnum.gpu
            
            else:
                cpu_buffer_id = CPU_MEMORY_CACHE.get_param_id_to_buffer_id(param.ds_id)
                param.ds_tensor.data = GPU_MEMORY_CACHE.add(param.ds_tensor, gpu_buffer_id)
                CPU_MEMORY_CACHE.release_buffer_id(param.ds_id, cpu_buffer_id, tensor_element)
                CPU_MEMORY_CACHE.release_gpu_tensors_cpu_cache_id(param.ds_id, tensor_element)
    
            param.data = param.ds_tensor.data.to(device=get_accelerator().current_device_name(),
                                                    non_blocking=True).view(param.ds_shape)
            param.ds_status = ZeroParamStatus.AVAILABLE
            param.ds_tensor.final_location = OffloadDeviceEnum.gpu
            index = prefetchtable.get_tensor_id_to_prefetch_list_id(param.ds_id, "fwd")
            prefetchtable.get_prefetch_information_list_item(index).current_location = OffloadDeviceEnum.gpu
            index = prefetchtable.get_tensor_id_to_prefetch_list_id(param.ds_id, "bwd")
            prefetchtable.get_prefetch_information_list_item(index).current_location = OffloadDeviceEnum.gpu

    def reset_step(self) -> None:
        """indicate that we have completed one fwd+bwd for the model"""
        if self.__inflight_param_registry:
            for param, handle in self.__inflight_param_registry.items():
                handle.wait()
                #self.__release_param(param)
            self.__inflight_param_registry.clear()
            # raise RuntimeError(f"still have inflight params "
            #                    f"{[p.ds_summary() for p in self.__inflight_param_registry.keys()]}")
        
        if prefetchtable.eval:
            prefetch_information_list = prefetchtable.get_prefetch_information_list()
            new_activation_list = []
            length = len(active_tensor_window)
            for i in range(length):
                prefetch_info = prefetch_information_list[i]
                tensor_id = prefetch_info.tensor_id
                tensor_element = prefetch_info.tensor_element
                if tensor_id in active_tensor_window:
                    new_activation_list.append(tensor_id)
                    active_tensor_window.remove(tensor_id)
                else:
                    param_id = active_tensor_window.pop()
                    params_id_list[param_id].ds_active_sub_modules.clear()
                    self.__release_param(params_id_list[param_id])

                    index = prefetchtable.get_tensor_id_to_prefetch_list_id(tensor_id, "fwd")
                    current_location_prefetch_tensor = prefetchtable.get_prefetch_information_list_item(index).current_location

                    if current_location_prefetch_tensor == OffloadDeviceEnum.gpu:
                        new_activation_list.append(tensor_id)
                    else:
                        new_activation_list.append(tensor_id)
                        param = params_id_list[tensor_id]
                        gpu_buffer_id = GPU_MEMORY_CACHE.get_free_buffer_id(tensor_element)
                        GPU_MEMORY_CACHE.add_param_id_to_buffer_id(param.ds_id, gpu_buffer_id)
                        GPU_MEMORY_CACHE.add_occupied_buffer_id(param.ds_id, tensor_element)

                        if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
                            param.nvme_swapper.swap_in([param], async_op=False)
                            new_tensor = GPU_MEMORY_CACHE.add(param.ds_tensor, gpu_buffer_id)
                            param.nvme_swapper.remove_partition_and_release_buffers([param])
                            param.ds_tensor.data = new_tensor
                            param.ds_tensor.status = PartitionedParamStatus.AVAILABLE
                        else:
                            cpu_buffer_id = CPU_MEMORY_CACHE.get_param_id_to_buffer_id(param.ds_id)
                            param.ds_tensor.data = GPU_MEMORY_CACHE.add(param.ds_tensor, gpu_buffer_id)
                            CPU_MEMORY_CACHE.release_buffer_id(param.ds_id, cpu_buffer_id, tensor_element)
                            if prefetchtable.PARAMS_IN_NVME:
                                CPU_MEMORY_CACHE.release_gpu_tensors_cpu_cache_id(param.ds_id, tensor_element)

                        param.data = param.ds_tensor.data.to(device=get_accelerator().current_device_name(),
                                                            non_blocking=True).view(param.ds_shape)
                        param.ds_status = ZeroParamStatus.AVAILABLE

                        index = prefetchtable.get_tensor_id_to_prefetch_list_id(param.ds_id, "fwd")
                        prefetchtable.get_prefetch_information_list_item(index).current_location = OffloadDeviceEnum.gpu
                        index = prefetchtable.get_tensor_id_to_prefetch_list_id(param.ds_id, "bwd")
                        prefetchtable.get_prefetch_information_list_item(index).current_location = OffloadDeviceEnum.gpu

            prefetchtable.current_row = length
            active_tensor_window.clear()
            active_tensor_window.extend(new_activation_list)
            self.release_param = True

        if not self.is_complete_trace():  # not self.trace_complete:
            # Make sure that recorded submodule orders are identical across ranks
            assert_ints_same_as_other_ranks([m.id for m in self.__submodule_order])

            if self.is_record_trace():
                # Successfully recorded a trace
                self.construct_parameter_trace_from_module_trace()
                # Make sure that recorded parameter orders are identical across ranks
                assert_ints_same_as_other_ranks([p.param.ds_id for p in self.__param_order])
                assert_ints_same_as_other_ranks([p.step_id_last_used_at for p in self.__param_order])

                self.__submodule_order = tuple(self.__submodule_order)  # freeze
                self.__param_order = tuple(self.__param_order)  # freeze
                self.__trace_mode = ZeRoTraceMode.COMPLETE
                print_rank_0(
                    f"completed record trace of {len(self.__submodule_order)} sub modules: {[m.id for m in self.__submodule_order]}",
                    force=False)
            else:
                # Enable trace recording for next forward/backward pass
                self.__trace_mode = ZeRoTraceMode.RECORD

        else:
            if self.__profiler is not None:
                self.__profiler.log_events()

        self.__param_queue = collections.deque(self.__param_order)  # reset fetch queue
        self.__most_recent_step_id_param_fetched_for = collections.defaultdict(lambda: int(-1e10))
        self.__step_id_module_fetched_for = collections.defaultdict(lambda: collections.deque())
        self.__step_id = 0
        self.__n_available_params = 0
        self.__profiler.reset_events()

    def _dump_params(self, tag, sub_module, params, step_id=None):
        if step_id is None:
            step_id = self.__step_id
        param_names = [debug_param2name_id(p) for p in params]
        print_rank_0(f'{tag} step = {step_id} mod = {debug_module2name_id(sub_module)} p_names = {param_names}',
                     force=False)

    def _dump_param_ids(self, tag, mod_id, p_ids, step_id=None):
        if step_id is None:
            step_id = self.__step_id
        print_rank_0(f'{tag} mod = {mod_id}, step = {step_id}, p_ids = {p_ids}', force=False)

    """Fetch and Release
    Fetching, prefetching, and releasing parameters
    """

    @compiler.disable
    @instrument_w_nvtx
    @torch.no_grad()
    def fetch_sub_module(self, current_submodule: Module, forward: bool) -> None:
        """This method does the following (in order):
        1. kick off fetch for parameters in immediately required sub module
        2. kick off fetch for next few parameters we will need later (prefetch)
        3. block on parameters in immediately required sub module
        """
        
        if logger.isEnabledFor(logging.DEBUG):
            debug_rank0(
                f"{self.__step_id}: M{current_submodule.id}({type(current_submodule).__name__}) P{[p.ds_id for p in iter_params(current_submodule, recurse=z3_leaf_module(current_submodule))]} "
                + str({
                    "avail": f"{self.__n_available_params:.1e}",
                    "queue_sz": f"{len(self.__param_queue or [])}",
                    "inflight": [p.ds_id for p in self.__inflight_param_registry],
                }))

        if not forward and not self.release_param:
            self.release_param = True

        params_to_fetch = frozenset(iter_params(current_submodule, recurse=z3_leaf_module(current_submodule)))
        fp16_element_size = torch.tensor([], dtype=torch.float16).element_size()
        if forward and prefetchtable.get_warmup():
            for param in params_to_fetch:
                size = param.ds_tensor.ds_numel * fp16_element_size
                if size in self.tensor_size_dict:
                    self.tensor_size_dict[size] += 1
                else:
                    self.tensor_size_dict[size] = 1

        if prefetchtable.get_warmup():
            for param in params_to_fetch:
                current_time = time.time()
                fwd = "fwd" if forward else "bwd"
                prefetchtable.add_tensor_information(param.ds_id, "parameter_16", current_submodule.id, fwd, param.ds_tensor.ds_numel)
                index = prefetchtable.get_prefetch_information_list_len() - 1
                prefetchtable.get_prefetch_information_list_item(index).set_start_time(current_time)
                prefetchtable.add_tensor_id_to_prefetch_list_id(param.ds_id, fwd, index)
                if fwd == "fwd":
                    prefetchtable.add_param_id_to_module_id(param.ds_id, current_submodule.id)

                

        fetch_numel = sum(
            [p.partition_numel() for p in params_to_fetch if p.ds_status == ZeroParamStatus.NOT_AVAILABLE])

        if fetch_numel > 0:
            event_name = __class__.FORWARD_FETCH_SUBMIT if forward else __class__.BACKWARD_FETCH_SUBMIT
            self._dump_param_ids(event_name, current_submodule.id,
                                 [p.ds_id for p in params_to_fetch if p.ds_status == ZeroParamStatus.NOT_AVAILABLE])
            self.__profiler.start_event(event_name)
            # kick off all gather for params in the immediately required submodule
            #for param in params_to_fetch:
            if logger.isEnabledFor(logging.DEBUG):
                for param in params_to_fetch:
                    debug_rank0(f"-fetch: {param.ds_summary()}")
            self.__all_gather_params(params_to_fetch, forward)
            self.__profiler.stop_event(event_name, fetch_numel)

        wait_numel = 0
        wait_event_name = __class__.FORWARD_FETCH_WAIT if forward else __class__.BACKWARD_FETCH_WAIT
        self.__profiler.start_event(wait_event_name)
        # wait for parameters in the immediately needed submodule to become available
        for param in params_to_fetch:
            param.ds_active_sub_modules.add(current_submodule.id)
            if logger.isEnabledFor(logging.DEBUG):
                debug_rank0(f"-wait: {param.ds_summary()}")
            if param in self.__inflight_param_registry:
                wait_numel += param.partition_numel()
                with get_accelerator().stream(self.__allgather_stream):
                    while self.__ongoing_fetch_events and self.__ongoing_fetch_events[0].query():
                        self.__ongoing_fetch_events.popleft()
                    if len(self.__ongoing_fetch_events) > self.__max_ongoing_fetch_events:
                        self.__ongoing_fetch_events.popleft().synchronize()

                    self.__inflight_param_registry.pop(param).wait()

                    if not get_accelerator().handles_memory_backpressure():
                        event = get_accelerator().Event()
                        event.record()
                        self.__ongoing_fetch_events.append(event)

            assert param.ds_status == ZeroParamStatus.AVAILABLE, param.ds_summary()
        if not get_accelerator().resolves_data_dependency():
            get_accelerator().current_stream().wait_stream(self.__allgather_stream)
        self.__profiler.stop_event(wait_event_name, wait_numel)
        
        self.__step_id += 1

    @instrument_w_nvtx
    @torch.no_grad()
    def release_sub_module(self, submodule: Module, forward: bool) -> None:
        """release the parameters of a sub module, assuming they meet conditions to
        be released."""
        
        if not self.release_param:
            return

        params_to_release = (set(p.ds_id for p in iter_params(submodule, recurse=z3_leaf_module(submodule))))
        
        prefetch_information_list = prefetchtable.get_prefetch_information_list()
        
        params_to_prefetch = set()
        self.prefetch_from_nvme.clear()
        if len(self.new_prefetch_from_nvme) > 0:
            self.prefetch_from_nvme.update(self.new_prefetch_from_nvme)
            self.new_prefetch_from_nvme.clear()

        slide =  "fwd" if forward else "bwd"

        for param in iter_params(submodule, recurse=z3_leaf_module(submodule)):
            
            param.ds_active_sub_modules.discard(submodule.id)

            if prefetchtable.get_warmup():
                current_time = time.time()
                fwd = "fwd" if forward else "bwd"
                index = prefetchtable.get_tensor_id_to_prefetch_list_id(param.ds_id, fwd)
                prefetchtable.get_prefetch_information_list_item(index).set_end_time(current_time)
                prefetchtable.get_prefetch_information_list_item(index).set_activation_duration()
            
            if param.ds_id in params_to_release and not param.is_external_param:
                if prefetchtable.get_warmup():
                    self.__release_param(param)

                if not prefetchtable.get_warmup() and prefetchtable.current_row < len(prefetch_information_list):
                    active_tensor_window.remove(param.ds_id)
                    row_id = prefetchtable.get_tensor_id_to_prefetch_list_id(param.ds_id, "fwd" if forward else "bwd")
                    prefetch_information_list[row_id].loc = "CPU" 

                    while prefetch_information_list[prefetchtable.current_row].tensor_id in active_tensor_window:
                        prefetchtable.current_row += 1
                        self.release_param = False

                    current_tensor_id = prefetch_information_list[prefetchtable.current_row].tensor_id
                    active_tensor_window.append(current_tensor_id)
                    row_id = prefetchtable.get_tensor_id_to_prefetch_list_id(current_tensor_id, "fwd" if forward else "bwd")
                    prefetch_information_list[row_id].loc = "GPU"
                    current_param = params_id_list[current_tensor_id]
                    if current_param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
                        self.new_prefetch_from_nvme.add(current_param)
                    else:
                        params_to_prefetch.add(current_param)
                    prefetchtable.current_row += 1
                    self.__release_param(param)

        if len(self.prefetch_from_nvme) > 0:
            params_to_prefetch.update(self.prefetch_from_nvme)

        if len(params_to_prefetch) > 0:
            self.__all_gather_params(params_to_prefetch, forward)

        if len(self.new_prefetch_from_nvme) > 0:
            self.__prefetch_nvme_param_partitions()

    
    @instrument_w_nvtx
    @torch.no_grad()
    def get_grad(self, submodule: Module):
        params_grad = {}
        for param in iter_params(submodule, recurse=z3_leaf_module(submodule)):
            if param.grad is not None:
                params_grad[param.ds_id] = param.grad
        return params_grad

    @instrument_w_nvtx
    @torch.no_grad()
    def release_and_reset_all(self, module: Module) -> None:
        """release all module parameters"""
        for param in iter_params(module, recurse=True):
            if param in self.__inflight_param_registry:
                raise RuntimeError(f"param {param.ds_summary()} still in flight")

            # TODO. make this throw if if there are still active submodules. currently
            # there's a hook execution issue
            param.ds_active_sub_modules.clear()
            
            if prefetchtable.get_warmup():
                self.__release_param(param)

        if prefetchtable.get_warmup():
            for param in iter_params(module, recurse=True):
                if param.ds_status != ZeroParamStatus.NOT_AVAILABLE:
                    raise RuntimeError(f"{param.ds_summary()} expected to be released")

    @instrument_w_nvtx
    def __all_gather_params(self, params: Set[Parameter], forward: bool) -> None:
        quantized_params = []
        nonquantized_params = []
        for param in params:
            if hasattr(param.ds_tensor, 'ds_quant_scale'):
                quantized_params.append(param)
            else:
                nonquantized_params.append(param)
        if quantized_params:
            self.__all_gather_params_(quantized_params, forward, quantize=True)
        if nonquantized_params:
            self.__all_gather_params_(nonquantized_params, forward, quantize=self.zero_quantized_weights)

    def __all_gather_params_(self, params: Set[Parameter], forward: bool, quantize: bool = False) -> None:
        """for each partitioned parameter, kick off an async allgather and store
        the work handle for the in flight parameters."""
        partitioned_params = []
        all_gather_numel = 0  # numel = num of elements
        for param in params:
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                partitioned_params.append(param)
                all_gather_numel += param.ds_numel

        if partitioned_params:
            self.__n_available_params += all_gather_numel
            # here we need to handle a special case where some of the parameters have a valid hpz secondary tensor (e.g. they are not trainable so their secondary tensor never expire) but others do not.
            partitioned_params_with_secondary_tensors = [
                p for p in partitioned_params if p.ds_secondary_tensor is not None
            ]
            partitioned_params_without_secondary_tensors = [
                p for p in partitioned_params if p.ds_secondary_tensor is None
            ]
            for param_group in [
                    partitioned_params_with_secondary_tensors, partitioned_params_without_secondary_tensors
            ]:
                if not param_group:
                    continue
                with get_accelerator().stream(self.__allgather_stream):
                    event_name = __class__.FORWARD_ALL_GATHER if forward else __class__.BACKWARD_ALL_GATHER
                    self.__profiler.start_event(event_name)
                    handle = param_group[0].all_gather_coalesced(param_group, quantize=quantize)
                    self.__profiler.stop_event(event_name, all_gather_numel)
                for param in param_group:
                    assert param.ds_status == ZeroParamStatus.INFLIGHT, param.ds_summary()
                    self.__inflight_param_registry[param] = handle

            # Release swap buffers for persisted params on nvme since they will never be partitioned or evicted from GPU
            swap_persisted_params = [
                p for p in partitioned_params if p.ds_persist and p.ds_tensor.final_location == OffloadDeviceEnum.nvme
            ]
            if swap_persisted_params:
                swap_persisted_params[0].nvme_swapper.remove_partition_and_release_buffers(swap_persisted_params)

    @compiler.disable
    @instrument_w_nvtx
    def __release_param(self, param: Parameter) -> None:
        if param.ds_status == ZeroParamStatus.AVAILABLE and not param.ds_active_sub_modules:
            if logger.isEnabledFor(logging.DEBUG):
                debug_rank0(f"-release: {param.ds_summary()}")
            param.partition()
            self.__n_available_params -= param.ds_numel

    @instrument_w_nvtx
    @functools.lru_cache(maxsize=None)
    def __params_to_release(self, submodule_to_release: Module, step_id: int) -> Set[int]:
        if not self.is_complete_trace():
            raise RuntimeError("expected trace to be complete")

        params_to_release = set(
            p.ds_id for p in iter_params(submodule_to_release, recurse=z3_leaf_module(submodule_to_release))
            if not p.ds_persist)

        # Problem: When prefetcher scans the param trace, it skips AVAILABLE params.
        # This creates issues if those params are released before the skipped uses:
        # 1) It hurts performance as the skipped uses are never prefetched.
        # 2) For nvme params, we run out of swap buffers because the prefetch order
        # diverges from the trace.
        # Solution: Don't release params whose reuse was skipped by prefetch. This is
        # possible because we detect such skips during prefetch and mark those params.
        for param in iter_params(submodule_to_release, recurse=z3_leaf_module(submodule_to_release)):
            if self.__most_recent_step_id_param_fetched_for[param] > step_id:
                params_to_release.discard(param.ds_id)

        # examine all modules within `max_reuse_dist_in_numel` of the current step,
        # if we see any of the candidate parameters to be released reoccur while
        # doing this, remove them from the set of parameters to release.
        params_traversed = 0
        for module in self.__submodule_order[step_id:]:
            if params_traversed >= self.__max_reuse_dist_in_numel:
                break
            for param in iter_params(module, recurse=z3_leaf_module(submodule_to_release)):
                params_to_release.discard(param.ds_id)
                params_traversed += param.ds_numel
        return params_to_release

    @instrument_w_nvtx
    def __prefetch_nvme_param_partitions(self) -> None:
        """swap in parameter partitions from nvme for those parameters that will be used
        after the ones that are already being prefetched into full parameters
        """

        if prefetchtable.get_warmup():
            if not self.is_complete_trace():
                return

            numel_in_flight = sum(param.ds_numel for param in self.__inflight_param_registry)

            numel_considered = 0
            swap_in_params = []
            swap_params_id = []
            for param_in_trace in self.__param_queue:
                param = param_in_trace.param
                if param.nvme_swapper is None:
                    continue
                if (numel_considered > 2 * numel_in_flight
                        or len(swap_in_params) >= param.nvme_swapper.available_swap_in_buffers()):
                    break
                if param.ds_id in swap_params_id:
                    continue
                if param.ds_tensor.status == PartitionedParamStatus.NOT_AVAILABLE:
                    swap_in_params.append(param)
                numel_considered += param.ds_numel

            if swap_in_params:
                swap_in_params[0].nvme_swapper.swap_in(swap_in_params, async_op=True)

        elif not prefetchtable.get_warmup():
            params_list = list(self.new_prefetch_from_nvme)

            if len(params_list) >= params_list[0].nvme_swapper.available_swap_in_buffers():
                return

            if params_list:
                params_list[0].nvme_swapper.swap_in(params_list, async_op=True)