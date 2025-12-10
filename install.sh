#!/bin/bash

cd ..

BASE_PATH=$(pwd)

#git clone --recursive https://github.com/microsoft/DeepSpeed.git
git clone --recursive https://github.com/deepspeedai/DeepSpeed.git
# git clone --recursive https://github.com/Sabiha1225/10cache.git

cd DeepSpeed/
git checkout tags/v0.14.2
# SRC_FILE="${BASE_PATH}/DeepSpeed/environment.yml"
# DEST_FILE="${BASE_PATH}/10cache/environment.yml"
cp "${BASE_PATH}/10cache/environment.yml" "${BASE_PATH}/DeepSpeed/environment.yml"

cp "${BASE_PATH}/10cache/deepspeed/ops/adam/cpu_adam.py" "${BASE_PATH}/DeepSpeed/deepspeed/ops/adam/cpu_adam.py"
#cp "${BASE_PATH}/10cache/deepspeed/runtime/swap_tensor/async_swapper.py" "${BASE_PATH}/DeepSpeed/deepspeed/runtime/swap_tensor/async_swapper.py"
cp "${BASE_PATH}/10cache/deepspeed/runtime/swap_tensor/optimizer_utils.py" "${BASE_PATH}/DeepSpeed/deepspeed/runtime/swap_tensor/optimizer_utils.py"
cp "${BASE_PATH}/10cache/deepspeed/runtime/swap_tensor/partitioned_optimizer_swapper.py" "${BASE_PATH}/DeepSpeed/deepspeed/runtime/swap_tensor/partitioned_optimizer_swapper.py"
cp "${BASE_PATH}/10cache/deepspeed/runtime/swap_tensor/partitioned_param_swapper.py" "${BASE_PATH}/DeepSpeed/deepspeed/runtime/swap_tensor/partitioned_param_swapper.py"
cp "${BASE_PATH}/10cache/deepspeed/runtime/zero/offload_config.py" "${BASE_PATH}/DeepSpeed/deepspeed/runtime/zero/offload_config.py"
cp "${BASE_PATH}/10cache/deepspeed/runtime/zero/parameter_offload.py" "${BASE_PATH}/DeepSpeed/deepspeed/runtime/zero/parameter_offload.py"
cp "${BASE_PATH}/10cache/deepspeed/runtime/zero/partition_parameters.py" "${BASE_PATH}/DeepSpeed/deepspeed/runtime/zero/partition_parameters.py"
cp "${BASE_PATH}/10cache/deepspeed/runtime/zero/partitioned_param_coordinator.py" "${BASE_PATH}/DeepSpeed/deepspeed/runtime/zero/partitioned_param_coordinator.py"
cp "${BASE_PATH}/10cache/deepspeed/runtime/zero/stage3.py" "${BASE_PATH}/DeepSpeed/deepspeed/runtime/zero/stage3.py"
cp "${BASE_PATH}/10cache/deepspeed/runtime/engine.py" "${BASE_PATH}/DeepSpeed/deepspeed/runtime/engine.py"
cp "${BASE_PATH}/10cache/deepspeed/runtime/utils.py" "${BASE_PATH}/DeepSpeed/deepspeed/runtime/utils.py"

#cd ..

#cd 10cache/
conda env create -n 10cache -f environment.yml --yes
conda activate 10cache
export CUDA_HOME=$CONDA_PREFIX
./install.sh -l