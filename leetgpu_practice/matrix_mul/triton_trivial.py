import torch
import triton
import triton.language as tl


@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck):
    # program ids = output coordinates
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # bounds check
    if pid_m >= M or pid_k >= K:
        return

    acc = 0.0

    # loop over N dimension
    for n in range(0, N):
        a_val = tl.load(a + pid_m * stride_am + n * stride_an)
        b_val = tl.load(b + n * stride_bn + pid_k * stride_bk)
        acc += a_val * b_val

    # store result
    tl.store(c + pid_m * stride_cm + pid_k * stride_ck, acc)


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1

    grid = (M, K)
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
    )
