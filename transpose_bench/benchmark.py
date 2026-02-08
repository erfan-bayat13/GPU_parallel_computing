import torch
import time
import matplotlib.pyplot as plt
import os
import transpose_ext

device = "cuda"

sizes = [256, 512, 1024, 2048, 4096]
naive_times = []
tiled_times = []
torch_times = []

os.makedirs("plots", exist_ok=True)

def time_kernel(fn, A, iters=50):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = fn(A)
    torch.cuda.synchronize()
    return (time.time() - start) / iters

print("Benchmarking transpose kernels (time per operation in ms):")
print(f"{'N':>6} | {'Naive':>10} | {'Tiled':>10} | {'PyTorch':>10}")
print("-"*44)

for n in sizes:
    A = torch.randn((n, n), device=device)

    # Warmup
    for _ in range(5):
        transpose_ext.naive(A)
        transpose_ext.tiled(A)
        _ = A.T.contiguous()

    t_naive = time_kernel(transpose_ext.naive, A) * 1000
    t_tiled = time_kernel(transpose_ext.tiled, A) * 1000
    t_torch = time_kernel(lambda x: x.T.contiguous(), A) * 1000

    naive_times.append(t_naive)
    tiled_times.append(t_tiled)
    torch_times.append(t_torch)

    print(f"{n:6d} | {t_naive:10.3f} | {t_tiled:10.3f} | {t_torch:10.3f}")

# Save plot
plt.figure()
plt.plot(sizes, naive_times, label="Naive CUDA")
plt.plot(sizes, tiled_times, label="Tiled CUDA")
plt.plot(sizes, torch_times, label="PyTorch contiguous")
plt.xlabel("Matrix size (N x N)")
plt.ylabel("Time per transpose (ms)")
plt.title("Transpose Kernel Benchmark")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/benchmark.png")
print("\n✅ Benchmark plot saved to plots/benchmark.png")
