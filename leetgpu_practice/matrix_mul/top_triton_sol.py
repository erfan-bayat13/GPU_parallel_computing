import torch
import triton
import triton.language as tl

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, 
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)

    num_pid_in_group = GROUP_SIZE * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE
    group_size = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_k = (pid % num_pid_in_group) // group_size

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bk = (pid_k * BLOCK_K + tl.arange(0, BLOCK_K)) % K
    offs_n = tl.arange(0, BLOCK_N)
    a_ptrs = a + (offs_am[:, None] * N + offs_n[None, :] * 1)
    b_ptrs = b + (offs_n[:, None] * K + offs_bk[None, :] * 1)

    accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_N)):
        # Load the next block of A and B, generate a mask by checking the N dimension.
        # If it is out of bounds, set it to 0.
        a_vals = tl.load(a_ptrs, mask=offs_n[None, :] < N - n * BLOCK_N, other=0.0)
        b_vals = tl.load(b_ptrs, mask=offs_n[:, None] < N - n * BLOCK_N, other=0.0)
        # We accumulate along the N dimension.
        accumulator = tl.dot(a_vals, b_vals, accumulator)
        # Advance the ptrs to the next N block.
        a_ptrs += BLOCK_N * 1
        b_ptrs += BLOCK_N * K
    
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_ck = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    c_ptrs = c + K * offs_cm[:, None] + 1 * offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    BLOCK_M = 64
    BLOCK_N = 32
    BLOCK_K = 128
    GROUP_SIZE=4
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(K, BLOCK_K), )

    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GROUP_SIZE=GROUP_SIZE
    )