import torch
import time

print("GPU available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

# Warm up CUDA
torch.cuda.synchronize()

start = time.perf_counter()

x = torch.randn(2000, 2000, device="cuda")
y = x @ x

torch.cuda.synchronize()

end = time.perf_counter()

print(f"GPU computation time: {end - start:.6f} seconds")