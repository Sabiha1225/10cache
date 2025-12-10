import torch
import time

# Allocate a large tensor on GPU
size = 1_000_000_000 // 4  # 1 GB (float32 takes 4 bytes)
cpu_data = torch.ones(size, dtype=torch.float32, device="cpu")

# Measure CPU to GPU transfer bandwidth
torch.cuda.synchronize()
start = time.time()
gpu_data = cpu_data.to("cuda")
torch.cuda.synchronize()
end = time.time()
cpu_to_gpu_bw = (cpu_data.element_size() * cpu_data.numel() / (end - start)) / 1e9
print(f"CPU to GPU Transfer Bandwidth: {cpu_to_gpu_bw:.2f} GB/s")

# Measure GPU to CPU transfer bandwidth
torch.cuda.synchronize()
start = time.time()
cpu_data = gpu_data.to("cpu")
torch.cuda.synchronize()
end = time.time()
gpu_to_cpu_bw = (gpu_data.element_size() * gpu_data.numel() / (end - start)) / 1e9
print(f"GPU to CPU Transfer Bandwidth: {gpu_to_cpu_bw:.2f} GB/s")