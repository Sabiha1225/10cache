import numpy as np
import time
import os

file_size = 1_000_000_000  # 1 GB file
file_path = "/mnt/nvme/test_file.bin"  # Change this to your NVMe path

# Generate random data in memory
data = np.random.randint(0, 255, file_size, dtype=np.uint8)

# Measure read speed
with open(file_path, "rb") as f:
    start = time.time()
    data_read = f.read()
end = time.time()

read_bandwidth = file_size / (end - start) / 1e9  # GB/s
print(f"NVMe to CPU Read Bandwidth: {read_bandwidth:.2f} GB/s")