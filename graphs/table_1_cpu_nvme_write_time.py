import numpy as np
import time
import os

file_size = 1_000_000_000  # 1 GB file
file_path = "/mnt/nvme/test_file.bin"  # Change this to your NVMe path

# Generate random data in memory
data = np.random.randint(0, 255, file_size, dtype=np.uint8)

# Measure write speed
with open(file_path, "wb") as f:
    start = time.time()
    f.write(data.tobytes())
    f.flush()
    os.fsync(f.fileno())  # Ensure data is fully written to disk
end = time.time()

write_bandwidth = file_size / (end - start) / 1e9  # GB/s
print(f"CPU to NVMe Write Bandwidth: {write_bandwidth:.2f} GB/s")