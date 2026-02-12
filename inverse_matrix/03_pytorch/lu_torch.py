import torch
import time
import numpy as np
from typing import Tuple, Optional

class LUPyTorch:
    """
    Matrix inversion using PyTorch's GPU-accelerated operations.
    
    This implementation shows how PyTorch abstracts CUDA operations.    
    """
    
    def __init__(self, device: str = 'cuda'):
        """
        Args:
            device: 'cuda' for GPU, 'cpu' for CPU comparison
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        if device == 'cuda' and not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
    
    def lu_decomposition(self, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute LU decomposition with partial pivoting: PA = LU
        
        Uses PyTorch's built-in torch.linalg.lu_factor which calls cuSOLVER under the hood.
        
        Args:
            A: Input matrix [n, n] on self.device
            
        Returns:
            L: Lower triangular with ones on diagonal [n, n]
            U: Upper triangular [n, n]
            P: Permutation matrix [n, n]
        """
        A = A.to(self.device)
        
        # torch.linalg.lu_factor returns compact LU representation + pivots
        # This internally calls cusolverDnSgetrf on GPU
        LU, pivots = torch.linalg.lu_factor(A)
        
        # Extract L and U from compact representation
        n = A.shape[0]
        L = torch.tril(LU, diagonal=-1) + torch.eye(n, device=self.device)
        U = torch.triu(LU)
        
        # Convert pivot indices to permutation matrix
        P = torch.eye(n, device=self.device)
        for i, pivot in enumerate(pivots -1): # Subtract 1 to convert to 0-based indexing
            if i != pivot:
                # Swap rows i and pivot
                P[[i, pivot]] = P[[pivot, i]]
        
        return L, U, P
    
    def invert_triangular(self, T: torch.Tensor, lower: bool = True) -> torch.Tensor:
        """
        Invert triangular matrix using PyTorch's optimized routine.
        
        Args:
            T: Triangular matrix [n, n]
            lower: True for lower triangular, False for upper
            
        Returns:
            T_inv: Inverse of T [n, n]
        """
        # torch.linalg.solve_triangular is faster than torch.inverse for triangular matrices
        # It calls cuBLAS trsm (triangular solve with multiple RHS) on GPU
        n = T.shape[0]
        I = torch.eye(n, device=self.device)
        
        T_inv = torch.linalg.solve_triangular(T, I, upper=not lower)
        return T_inv
    
    def invert_via_lu(self, A: torch.Tensor) -> torch.Tensor:
        """
        Invert matrix using LU decomposition: A^(-1) = U^(-1) @ L^(-1) @ P
        
        Args:
            A: Input matrix [n, n]
            
        Returns:
            A_inv: Inverse of A [n, n]
        """
        L, U, P = self.lu_decomposition(A)
        
        # Invert triangular matrices
        L_inv = self.invert_triangular(L, lower=True)
        U_inv = self.invert_triangular(U, lower=False)
        
        # A^(-1) = U^(-1) @ L^(-1) @ P
        # Note: P is its own inverse for permutation matrices (P @ P.T = I)
        A_inv = U_inv @ L_inv @ P
        
        return A_inv
    
    def invert_direct(self, A: torch.Tensor) -> torch.Tensor:
        """
        Direct inversion using PyTorch's built-in (for comparison).
        
        torch.linalg.inv internally uses:
        1. LU factorization (cusolverDnSgetrf)
        2. Triangular solve (cuBLAS trsm)
        
        This is the "production" way to invert matrices in PyTorch.
        """
        A = A.to(self.device)
        return torch.linalg.inv(A)
    
    def solve_system(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Solve Ax = b without explicitly computing A^(-1).
        
        This is almost always preferred over inversion in practice!
        
        Args:
            A: Coefficient matrix [n, n]
            b: Right-hand side [n, k] (k can be 1 for single vector)
            
        Returns:
            x: Solution [n, k]
        """
        A = A.to(self.device)
        b = b.to(self.device)
        
        # torch.linalg.solve uses LU factorization + forward/backward substitution
        # More numerically stable and faster than x = A_inv @ b
        return torch.linalg.solve(A, b)


def benchmark_pytorch(size: int = 2048, num_trials: int = 10):
    """
    Benchmark PyTorch GPU vs CPU performance.
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking PyTorch Matrix Inversion (size={size}x{size})")
    print(f"{'='*60}\n")
    
    # Generate random test matrix
    np.random.seed(42)
    A_np = np.random.randn(size, size).astype(np.float32)
    
    # Make sure it's invertible by adding to diagonal
    A_np += np.eye(size) * size * 0.1
    
    # Test on GPU
    if torch.cuda.is_available():
        print("GPU (CUDA) Benchmark:")
        print("-" * 60)
        
        lu_gpu = LUPyTorch(device='cuda')
        A_gpu = torch.from_numpy(A_np).to('cuda')
        
        # Warmup
        _ = lu_gpu.invert_via_lu(A_gpu)
        torch.cuda.synchronize()
        
        # Benchmark custom LU implementation
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_trials):
            A_inv_custom = lu_gpu.invert_via_lu(A_gpu)
            torch.cuda.synchronize()
        end = time.perf_counter()
        time_custom = (end - start) / num_trials * 1000
        
        # Benchmark PyTorch's built-in
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_trials):
            A_inv_direct = lu_gpu.invert_direct(A_gpu)
            torch.cuda.synchronize()
        end = time.perf_counter()
        time_direct = (end - start) / num_trials * 1000
        
        # Verify correctness
        I_custom = A_gpu @ A_inv_custom
        I_direct = A_gpu @ A_inv_direct
        error_custom = torch.norm(I_custom - torch.eye(size, device='cuda')).item()
        error_direct = torch.norm(I_direct - torch.eye(size, device='cuda')).item()
        
        print(f"Custom LU:        {time_custom:.3f} ms (error: {error_custom:.2e})")
        print(f"PyTorch built-in: {time_direct:.3f} ms (error: {error_direct:.2e})")
        print(f"Overhead:         {(time_custom/time_direct - 1)*100:.1f}%")
        
        # Memory info
        print(f"\nGPU Memory:")
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.3f} GB")
        print(f"Reserved:  {torch.cuda.memory_reserved()/1e9:.3f} GB")
    
    # Test on CPU for comparison
    print("\n" + "="*60)
    print("CPU Benchmark:")
    print("-" * 60)
    
    lu_cpu = LUPyTorch(device='cpu')
    A_cpu = torch.from_numpy(A_np)
    
    # Warmup
    _ = lu_cpu.invert_via_lu(A_cpu)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_trials):
        A_inv_cpu = lu_cpu.invert_via_lu(A_cpu)
    end = time.perf_counter()
    time_cpu = (end - start) / num_trials * 1000
    
    print(f"CPU LU: {time_cpu:.3f} ms")
    
    if torch.cuda.is_available():
        print(f"\nGPU Speedup: {time_cpu/time_direct:.1f}x over CPU")


def test_correctness():
    """
    Verify that our PyTorch implementation matches NumPy.
    """
    print("\nCorrectness Tests:")
    print("-" * 60)
    
    # Small test case
    A_np = np.array([
        [4.0, 3.0, 2.0],
        [3.0, 2.0, 1.0],
        [2.0, 1.0, 3.0]
    ], dtype=np.float32)
    
    A_torch = torch.from_numpy(A_np)
    
    # Test on available device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lu = LUPyTorch(device=device)
    
    # Test LU decomposition
    L, U, P = lu.lu_decomposition(A_torch)
    PA_reconstructed = L @ U
    PA_actual = P @ A_torch.to(device)
    
    error_lu = torch.norm(PA_reconstructed - PA_actual).item()
    print(f"LU Decomposition error: {error_lu:.2e}")
    
    # Test inversion
    A_inv = lu.invert_via_lu(A_torch)
    I = A_torch.to(device) @ A_inv
    error_inv = torch.norm(I - torch.eye(3, device=device)).item()
    print(f"Inversion error: {error_inv:.2e}")
    
    # Compare with PyTorch built-in
    A_inv_pytorch = torch.linalg.inv(A_torch.to(device))
    diff = torch.norm(A_inv - A_inv_pytorch).item()
    print(f"Difference from torch.linalg.inv: {diff:.2e}")
    
    if error_lu < 1e-5 and error_inv < 1e-5 and diff < 1e-5:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed!")


def demonstrate_solve_vs_invert():
    """
    Show why solving Ax=b is better than computing A^(-1)b.
    """
    print("\n" + "="*60)
    print("Solve vs Invert Comparison")
    print("="*60 + "\n")
    
    size = 2048
    A = torch.randn(size, size, device='cuda' if torch.cuda.is_available() else 'cpu')
    A = A @ A.T + torch.eye(size, device=A.device) * 0.1  # Make SPD
    b = torch.randn(size, 10, device=A.device)  # 10 RHS vectors
    
    lu = LUPyTorch(device=A.device.type)
    
    # Method 1: Invert then multiply
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    A_inv = lu.invert_direct(A)
    x1 = A_inv @ b
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_invert = (time.perf_counter() - start) * 1000
    
    # Method 2: Direct solve
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    x2 = lu.solve_system(A, b)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_solve = (time.perf_counter() - start) * 1000
    
    # Check accuracy
    residual1 = torch.norm(A @ x1 - b) / torch.norm(b)
    residual2 = torch.norm(A @ x2 - b) / torch.norm(b)
    
    print(f"Method 1 (Invert + Multiply): {time_invert:.3f} ms, residual: {residual1:.2e}")
    print(f"Method 2 (Direct Solve):      {time_solve:.3f} ms, residual: {residual2:.2e}")
    print(f"\nDirect solve is {time_invert/time_solve:.2f}x faster and more accurate!")


if __name__ == "__main__":
    # Run tests
    test_correctness()
    
    # Run benchmarks
    if torch.cuda.is_available():
        print(f"\nDetected GPU: {torch.cuda.get_device_name(0)}")
    
    benchmark_pytorch(size=1024, num_trials=20)
    benchmark_pytorch(size=2048, num_trials=10)
    
    # Show why you shouldn't invert
    if torch.cuda.is_available():
        demonstrate_solve_vs_invert()