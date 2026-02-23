import torch
import triton
import triton.language as tl


@triton.jit
def invert_kernel(image, width, height, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_pixels = width * height
    mask = offsets < n_pixels

    # Each pixel has 4 bytes
    base = offsets * 4

    # Load R, G, B
    r = tl.load(image + base + 0, mask=mask)
    g = tl.load(image + base + 1, mask=mask)
    b = tl.load(image + base + 2, mask=mask)

    # Invert
    r = 255 - r
    g = 255 - g
    b = 255 - b

    # Store back
    tl.store(image + base + 0, r, mask=mask)
    tl.store(image + base + 1, g, mask=mask)
    tl.store(image + base + 2, b, mask=mask)

# image is a tensor on the GPU
def solve(image: torch.Tensor, width: int, height: int):
    BLOCK_SIZE = 1024
    n_pixels = width * height
    grid = (triton.cdiv(n_pixels, BLOCK_SIZE),)

    invert_kernel[grid](image, width, height, BLOCK_SIZE)
