import numpy as np
import time

from lu_numpy import invert_matrix as numpy_inv
from pathlib import Path
import sys

# Add Phase 1 path
sys.path.insert(0, str(Path(__file__).parent.parent / "01_naive_python"))
from lu_decomposition import invert_matrix as naive_inv


def benchmark(n=100, trials=3):
    A = np.random.rand(n, n)
    A += n * np.eye(n)  # improve conditioning

    # Naive Python
    A_list = A.tolist()
    t0 = time.time()
    for _ in range(trials):
        naive_inv(A_list)
    t_naive = (time.time() - t0) / trials

    # NumPy LU
    t0 = time.time()
    for _ in range(trials):
        numpy_inv(A)
    t_numpy_lu = (time.time() - t0) / trials

    # NumPy built-in
    t0 = time.time()
    for _ in range(trials):
        np.linalg.inv(A)
    t_np_inv = (time.time() - t0) / trials

    print(f"Matrix size: {n}x{n}")
    print(f"Naive Python LU: {t_naive:.4f} s")
    print(f"NumPy LU:        {t_numpy_lu:.6f} s")
    print(f"NumPy inv:       {t_np_inv:.6f} s")
    print(f"Speedup (NumPy LU vs naive): {t_naive/t_numpy_lu:.1f}x")


if __name__ == "__main__":
    for n in [50, 100, 200, 500]:
        benchmark(n)
        print("-" * 40)
