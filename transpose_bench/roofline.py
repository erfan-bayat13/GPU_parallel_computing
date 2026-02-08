import torch
import time
import os
import matplotlib.pyplot as plt
import transpose_ext

device = "cuda"
sizes = [512, 1024, 2048, 4096, 8192]

os.makedirs("plots", exist_ok=True)

def benchmark(fn, A, iters=50):
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters):
        _ = fn(A)
    torch.cuda.synchronize()
    return (time.time() - start) / iters

print("Roofline-style memory bandwidth (GB/s):")
print(f"{'N':>6} | {'Naive':>10} | {'Tiled':>10} | {'PyTorch':>10}")
print("-"*44)

naive_bw, tiled_bw, torch_bw = [], [], []

for n in sizes:
    A = torch.randn((n, n), device=device)

    # Warmup
    for _ in range(5):
        transpose_ext.naive(A)
        transpose_ext.tiled(A)
        _ = A.T.contiguous()

    t_naive = benchmark(transpose_ext.naive, A)
    t_tiled = benchmark(transpose_ext.tiled, A)
    t_torch = benchmark(lambda x: x.T.contiguous(), A)

    # Bytes moved = read + write (float32)
    bytes_moved = 2 * n * n * 4
    naive_bw_val = bytes_moved / t_naive / 1e9
    tiled_bw_val = bytes_moved / t_tiled / 1e9
    torch_bw_val = bytes_moved / t_torch / 1e9

    naive_bw.append(naive_bw_val)
    tiled_bw.append(tiled_bw_val)
    torch_bw.append(torch_bw_val)

    print(f"{n:6d} | {naive_bw_val:10.2f} | {tiled_bw_val:10.2f} | {torch_bw_val:10.2f}")

# Save roofline plot
plt.figure()
plt.plot(sizes, naive_bw, label="Naive CUDA")
plt.plot(sizes, tiled_bw, label="Tiled CUDA")
plt.plot(sizes, torch_bw, label="PyTorch contiguous")
plt.xlabel("Matrix size (N x N)")
plt.ylabel("Effective bandwidth (GB/s)")
plt.title("Transpose Roofline")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/roofline.png")
print("\n✅ Roofline plot saved to plots/roofline.png")
