import triton
import triton.language as tl

@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr,
    rows, cols,
    TILE_X: tl.constexpr,
    TILE_Y: tl.constexpr
):
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)

    r0 = pid_x * TILE_X
    c0 = pid_y * TILE_Y

    offs_r = tl.arange(0, TILE_X)
    offs_c = tl.arange(0, TILE_Y)

    r = r0 + offs_r
    c = c0 + offs_c

    in_ptrs = input_ptr + r[:, None] * cols + c[None, :]
    in_mask = (r[:, None] < rows) & (c[None, :] < cols)
    x = tl.load(in_ptrs, mask=in_mask, other=0.0)

    out_r = c0 + offs_c
    out_c = r0 + offs_r

    out_ptrs = output_ptr + out_r[:, None] * rows + out_c[None, :]
    out_mask = (out_r[:, None] < cols) & (out_c[None, :] < rows)
    tl.store(out_ptrs, tl.trans(x), mask=out_mask)

def solve(input_ptr: int, output_ptr: int, rows: int, cols: int):
    TILE_X = 128
    TILE_Y = 64
    grid = (triton.cdiv(rows, TILE_X), triton.cdiv(cols, TILE_Y))
    matrix_transpose_kernel[grid](
        input_ptr, output_ptr,
        rows, cols,
        TILE_X=TILE_X,
        TILE_Y=TILE_Y
    )