# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Functionality of swapping optimizer tensors to/from (NVMe) storage devices.
"""

import torch

from deepspeed.utils.logging import logger
from deepspeed.ops.op_builder import AsyncIOBuilder
from deepspeed import comm as dist

from deepspeed.runtime.swap_tensor.constants import *
from deepspeed.runtime.swap_tensor.utils import swap_in_tensors, swap_out_tensors, print_object, \
    get_sized_buffers
from deepspeed.runtime.swap_tensor.async_swapper import AsyncTensorSwapper
from deepspeed.runtime.swap_tensor.optimizer_utils import OptimizerSwapper
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.utils import prefetchtable, OPTIMIZER_STATES_CACHE
# import signal
# from contextlib import contextmanager

DEBUG_MODE = False

SWAP_IN_PARAM_TIMER = 'swap_in_param'
SWAP_OUT_PARAM_TIMER = 'swap_out_param'
SWAP_IN_GRADIENT_TIMER = 'swap_in_gradient'


class PartitionedOptimizerSwapper(OptimizerSwapper):

    def __init__(self, swap_config, aio_config, base_folder, optimizer, largest_numel, device, dtype, timers):
        super(PartitionedOptimizerSwapper, self).__init__(swap_config, aio_config, base_folder, optimizer,
                                                          largest_numel, device, dtype, timers)

        aio_op = AsyncIOBuilder().load(verbose=False)
        self.aio_handle = aio_op.aio_handle(aio_config[AIO_BLOCK_SIZE], aio_config[AIO_QUEUE_DEPTH],
                                            aio_config[AIO_SINGLE_SUBMIT], aio_config[AIO_OVERLAP_EVENTS],
                                            aio_config[AIO_THREAD_COUNT])

        # aio_config[AIO_QUEUE_DEPTH]
        self.aio_read_handle = aio_op.aio_handle(aio_config[AIO_BLOCK_SIZE], aio_config[AIO_QUEUE_DEPTH],
                                            aio_config[AIO_SINGLE_SUBMIT], aio_config[AIO_OVERLAP_EVENTS],
                                            aio_config[AIO_THREAD_COUNT])

        self.aio_read_handle_grad = aio_op.aio_handle(aio_config[AIO_BLOCK_SIZE], aio_config[AIO_QUEUE_DEPTH],
                                            aio_config[AIO_SINGLE_SUBMIT], aio_config[AIO_OVERLAP_EVENTS],
                                            aio_config[AIO_THREAD_COUNT])
        self.aio_write_handle = aio_op.aio_handle(aio_config[AIO_BLOCK_SIZE], aio_config[AIO_QUEUE_DEPTH],
                                            aio_config[AIO_SINGLE_SUBMIT], aio_config[AIO_OVERLAP_EVENTS],
                                            aio_config[AIO_THREAD_COUNT])

        # Overlap swapping out
        self.gradient_swapper = AsyncTensorSwapper(aio_handle=self.aio_handle,
                                                   numel_alignment=self.numel_alignment,
                                                   timers=self.timers)

        self.print_exclude_list += ['aio_handle', 'aio_read_handle', 'aio_read_handle_grad', 'aio_write_handle', 'gradient_swapper', 'print_exclude_list']
        
        self.pending_reads = 0
        self.pending_reads_grad = 0
        #keep track of async swap in params and buffers
        self.inflight_params = []

        if dist.get_rank() == 0:
            print_object(obj=self, name='PartitionedOptimizerSwapper', exclude_list=self.print_exclude_list)

    def initialize_parameters(self, parameters, src_tensors):
        # if not prefetchtable.get_warmup():
        #     self._flush_gradient_swapper(self.gradient_swapper)
        self._initialize_parameters(parameters=parameters, src_tensors=src_tensors, aio_handle=self.aio_handle)

    def initialize_from_swapped_fp16_params(self, fp16_partitions_info, fp16_num_elems, fp16_pinned_buffers,
                                            fp32_parameters):
        self._initialize_from_swapped_fp16_params(aio_handle=self.aio_handle,
                                                  fp16_partitions_info=fp16_partitions_info,
                                                  fp16_num_elems=fp16_num_elems,
                                                  fp16_pinned_buffers=fp16_pinned_buffers,
                                                  fp32_parameters=fp32_parameters)

    def flush_gradients(self):
        self._flush_gradient_swapper(self.gradient_swapper)

    def clear_swap_info(self):
        self.swap_params_info = {}

    def swap_in_optimizer_state(self, parameter, async_parameter=None):
        swap_info = self._get_param_swap_info(parameter)
        if swap_info is None:
            return

        self._flush_gradient_swapper(self.gradient_swapper)

        required_buffer_count = len(swap_info.tensors) + (1 if swap_info.has_gradients() else 0)
        aligned_numel = self._io_aligned_numel(swap_info.numel())
        pinned_buffers = self.swap_buffer_manager.allocate(num_elems=aligned_numel,
                                                           count=required_buffer_count,
                                                           dtype=parameter.dtype)
        assert pinned_buffers is not None
        self.allocated_swap_buffers = pinned_buffers.copy()

        self._start_timer(SWAP_IN_PARAM_TIMER)
        self._swap_in_parameter(aio_handle=self.aio_handle,
                                parameter=parameter,
                                dest_buffers=pinned_buffers[:required_buffer_count])
        self._stop_timer(SWAP_IN_PARAM_TIMER)
        self.timer_names.add(SWAP_IN_PARAM_TIMER)

        self._start_timer(SWAP_IN_GRADIENT_TIMER)
        self._swap_in_gradients(aio_handle=self.aio_handle, parameter=parameter, dest_buffer=pinned_buffers[-1])
        self._stop_timer(SWAP_IN_GRADIENT_TIMER)
        self.timer_names.add(SWAP_IN_GRADIENT_TIMER)

    def swap_in_optimizer_state_from_buffer(self, parameter, async_parameter=None):
        buffer_id = OPTIMIZER_STATES_CACHE.get_param_id_to_buffer_id(parameter.ds_id)
        swap_info = self._get_param_swap_info(parameter)
        if swap_info is None:
            return

        pinned_buffers = []
        pinned_buffers.append(OPTIMIZER_STATES_CACHE.get(buffer_id, parameter.ds_numel))
        pinned_buffers.append(OPTIMIZER_STATES_CACHE.get_momentum_buffer(buffer_id, parameter.ds_numel))
        pinned_buffers.append(OPTIMIZER_STATES_CACHE.get_variance_buffer(buffer_id, parameter.ds_numel))
        
        for t, buffer in zip(swap_info.tensors, pinned_buffers):
            t.data = buffer

        READ_TIMER = 'swap_submit_read_param'
        self._start_timer(SWAP_IN_PARAM_TIMER)
        self._start_timer(READ_TIMER)
        
        swap_in_tensors(self.aio_read_handle, pinned_buffers, swap_info.swap_paths)
        self._stop_timer(READ_TIMER)
        self._stop_timer(SWAP_IN_PARAM_TIMER)
        self.timer_names.add(SWAP_IN_PARAM_TIMER)

        self.pending_reads += len(pinned_buffers)
        self.inflight_params.append(parameter.ds_id)

        swap_bytes = sum([buffer.numel() * buffer.element_size() for buffer in pinned_buffers])

        self.aio_read_handle.wait()
        self.pending_reads = 0
        self.inflight_params = []
        
        if DEBUG_MODE and dist.get_rank() == 0:
            logger.info(f'optimizer_param_swap_in: {(swap_bytes/(1024**3)):5.2f} GB')

        gradient_tensor = OPTIMIZER_STATES_CACHE.get_gradient_buffer(buffer_id, parameter.ds_numel)
        parameter.grad = gradient_tensor

        self._start_timer(SWAP_IN_GRADIENT_TIMER)
        if swap_info.has_gradients():
            param_gradients = swap_info.swapped_gradients.values()
            swap_buffers = [gradient_tensor.narrow(0, grad.offset, grad.length) for grad in param_gradients]
            swap_paths = [grad.path for grad in param_gradients]
            SWAP_READ_GRADIENTS = 'swap_submit_read_gradient'
            SWAP_WAIT_GRADIENTS = 'swap_submit_wait_gradient'

            self._start_timer(SWAP_READ_GRADIENTS)
            swap_in_tensors(self.aio_read_handle_grad, swap_buffers, swap_paths)
            self._stop_timer(SWAP_READ_GRADIENTS)

            self.pending_reads_grad += len(swap_buffers)

            self._start_timer(SWAP_WAIT_GRADIENTS)
            self.aio_read_handle_grad.wait()
            self.pending_reads_grad = 0
            self._stop_timer(SWAP_WAIT_GRADIENTS)

        self._stop_timer(SWAP_IN_GRADIENT_TIMER)
        self.timer_names.add(SWAP_IN_GRADIENT_TIMER)


    def synchronize_reads(self):
        WAIT_TIMER = 'swap_wait_read_param'
        self._start_timer(WAIT_TIMER)
        
        if self.pending_reads > 0:
            self.aio_read_handle.wait()
            self.pending_reads = 0
            self.inflight_params = []
        
        if self.pending_reads_grad > 0:
            assert self.pending_reads_grad == self.aio_read_handle_grad.wait()
            self.pending_reads_grad = 0
        self._stop_timer(WAIT_TIMER)

    def check_inflight(self, parameter):
        if parameter.ds_id in self.inflight_params:
            return True
        return False

    def swap_out_optimizer_state(self, parameter, async_swap=False):
        swap_info = self._get_param_swap_info(parameter=parameter)

        if swap_info is None:
            return

        self._start_timer(SWAP_OUT_PARAM_TIMER)
        pinned_tensors, pinned_paths, unpinned_tensors, unpinned_paths = self._separate_pinned_tensors(swap_info)
        swap_bytes = sum([self._io_aligned_numel(t.numel()) * t.element_size() for t in swap_info.tensors])

        WRITE_TIMER = 'swap_submit_write'
        self._start_timer(WRITE_TIMER)

        swap_out_tensors(self.aio_handle, pinned_tensors, pinned_paths)
        assert self.aio_handle.wait() == len(pinned_tensors)
        for t in pinned_tensors:
            t.data = torch.Tensor()

        if len(unpinned_tensors) > 0:
            pinned_buffers = self.swap_buffer_manager.allocate_all(num_elems=self.largest_numel, dtype=self.dtype)
            self._swap_out_unpinned_tensors(aio_handle=self.aio_handle,
                                            unpinned_tensors=unpinned_tensors,
                                            dest_paths=unpinned_paths,
                                            pinned_buffers=pinned_buffers)
            self.allocated_swap_buffers += pinned_buffers

            for t in unpinned_tensors:
                t.data = torch.Tensor()
        self._stop_timer(WRITE_TIMER)

        self.swap_buffer_manager.free(self.allocated_swap_buffers)
        self.allocated_swap_buffers = []

        self._stop_timer(SWAP_OUT_PARAM_TIMER)
        self.timer_names.add(SWAP_OUT_PARAM_TIMER)

        self._log_timers([WRITE_TIMER])

        if DEBUG_MODE and dist.get_rank() == 0:
            logger.info(f'optimizer_param_swap_out: {(swap_bytes/(1024**3)):5.2f} GB')

    def swap_out_optimizer_state_from_buffer(self, parameter, async_swap=False):
        swap_info = self._get_param_swap_info(parameter=parameter)

        if swap_info is None:
            return

        self._start_timer(SWAP_OUT_PARAM_TIMER)
        pinned_tensors = []
        pinned_paths = []

        for tensor, path in zip(swap_info.tensors, swap_info.swap_paths):
            pinned_tensors.append(tensor.data)
            pinned_paths.append(path)
        swap_bytes = sum([self._io_aligned_numel(t.numel()) * t.element_size() for t in swap_info.tensors])

        WRITE_TIMER = 'swap_submit_write'
        self._start_timer(WRITE_TIMER)
        swap_out_tensors(self.aio_write_handle, pinned_tensors, pinned_paths)
        value = self.aio_write_handle.wait()
        assert value == len(pinned_tensors)
        for t in pinned_tensors:
            t.data = torch.Tensor()

        self._stop_timer(WRITE_TIMER)

        self._stop_timer(SWAP_OUT_PARAM_TIMER)
        self.timer_names.add(SWAP_OUT_PARAM_TIMER)

        self._log_timers([WRITE_TIMER])

        if DEBUG_MODE and dist.get_rank() == 0:
            logger.info(f'optimizer_param_swap_out: {(swap_bytes/(1024**3)):5.2f} GB')

    def swap_out_gradients(self, parameter, gradient_offsets, gradient_tensors):
        self._swap_out_gradients(parameter=parameter,
                                 gradient_offsets=gradient_offsets,
                                 gradient_tensors=gradient_tensors,
                                 gradient_swapper=self.gradient_swapper)

    def _swap_in_parameter(self, aio_handle, parameter, dest_buffers):
        swap_info = self._get_param_swap_info(parameter)
        if swap_info is None:
            return

        assert len(swap_info.tensors) <= len(dest_buffers)

        swap_lengths = [self._io_aligned_numel(swap_info.numel())] * len(swap_info.tensors)
        swap_buffers = get_sized_buffers(dest_buffers, swap_lengths)

        READ_TIMER = 'swap_submit_read_param'
        WAIT_TIMER = 'swap_wait_read_param'

        self._start_timer(READ_TIMER)
        swap_in_tensors(aio_handle, swap_buffers, swap_info.swap_paths)
        self._stop_timer(READ_TIMER)

        swap_bytes = sum([buffer.numel() * buffer.element_size() for buffer in swap_buffers])

        self._start_timer(WAIT_TIMER)
        aio_handle.wait()
        self._stop_timer(WAIT_TIMER)

        compute_lengths = [swap_info.numel()] * len(swap_info.tensors)
        compute_buffers = get_sized_buffers(dest_buffers, compute_lengths)
        for t, buffer in zip(swap_info.tensors, compute_buffers):
            t.data = buffer.data

        self._log_timers([READ_TIMER, WAIT_TIMER])
        if DEBUG_MODE and dist.get_rank() == 0:
            logger.info(f'optimizer_param_swap_in: {(swap_bytes/(1024**3)):5.2f} GB')

    def _separate_pinned_tensors(self, swap_info):
        pinned_tensors = []
        pinned_paths = []

        unpinned_tensors = []
        unpinned_paths = []

        for tensor, path in zip(swap_info.tensors, swap_info.swap_paths):
            if get_accelerator().is_pinned(tensor):
                pinned_tensors.append(tensor)
                pinned_paths.append(path)
            else:
                unpinned_tensors.append(tensor)
                unpinned_paths.append(path)

        return pinned_tensors, pinned_paths, unpinned_tensors, unpinned_paths

    def _swap_in_pinned_gradients(self, aio_handle, parameter, gradient_tensor):
        swap_info = self.swap_params_info[OptimizerSwapper.parameter_id(parameter)]
        param_gradients = swap_info.swapped_gradients.values()
        swap_buffers = [gradient_tensor.narrow(0, grad.offset, grad.length) for grad in param_gradients]
        swap_paths = [grad.path for grad in param_gradients]
        SWAP_READ_GRADIENTS = 'swap_submit_read_gradient'
        SWAP_WAIT_GRADIENTS = 'swap_submit_wait_gradient'

        self._start_timer(SWAP_READ_GRADIENTS)
        swap_in_tensors(aio_handle, swap_buffers, swap_paths)
        self._stop_timer(SWAP_READ_GRADIENTS)

        self._start_timer(SWAP_WAIT_GRADIENTS)
        assert len(swap_buffers) == aio_handle.wait()
        self._stop_timer(SWAP_WAIT_GRADIENTS)

        self._log_timers([SWAP_READ_GRADIENTS, SWAP_WAIT_GRADIENTS])

    def _swap_in_gradients(self, aio_handle, parameter, dest_buffer):
        swap_info = self.swap_params_info.get(OptimizerSwapper.parameter_id(parameter), None)
        if not (swap_info and swap_info.has_gradients()):
            return

        assert get_accelerator().is_pinned(dest_buffer)
        assert parameter.numel() <= dest_buffer.numel()

        parameter.grad = dest_buffer.narrow(0, 0, parameter.numel())

        if swap_info.swapped_gradients:
            self._swap_in_pinned_gradients(aio_handle, parameter, parameter.grad)

        if swap_info.unswapped_gradients:
            self._retrieve_unswapped_grad_partitions(swap_info=swap_info, dest_buffer=parameter.grad)