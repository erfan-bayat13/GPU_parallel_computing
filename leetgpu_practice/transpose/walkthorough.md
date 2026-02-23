
---

## 1️⃣ Kernel Definition

```python
@triton.jit
def matrix_transpose_kernel(
    input_ptr, output_ptr,
    rows, cols,
    TILE_X: tl.constexpr,
    TILE_Y: tl.constexpr
):
```

* `@triton.jit` → JIT-compiles this function into a GPU kernel.
* `input_ptr`, `output_ptr` → pointers to the GPU input/output matrices.
* `rows, cols` → dimensions of the input matrix.
* `TILE_X`, `TILE_Y` → **tile sizes in X and Y directions**, marked as `constexpr` so Triton can use them at compile time.

---

## 2️⃣ Program IDs (Block Coordinates)

```python
pid_x = tl.program_id(axis=0)
pid_y = tl.program_id(axis=1)
```

* Each kernel instance (program) is responsible for **one tile** of the matrix.
* `axis=0` → row blocks, `axis=1` → column blocks (or vice versa depending on your convention).

---

## 3️⃣ Compute Starting Indices for the Tile

```python
r0 = pid_x * TILE_X
c0 = pid_y * TILE_Y
```

* `(r0, c0)` is the **top-left corner** of the tile in the input matrix.
* Each program works on a tile starting at `(r0, c0)`.

---

## 4️⃣ Generate Offsets Within the Tile

```python
offs_r = tl.arange(0, TILE_X)
offs_c = tl.arange(0, TILE_Y)
```

* `offs_r` → row offsets inside the tile (0..TILE_X-1)
* `offs_c` → column offsets inside the tile (0..TILE_Y-1)

```python
r = r0 + offs_r
c = c0 + offs_c
```

* Absolute coordinates in the **input matrix** for this tile.

---

## 5️⃣ Compute Pointers and Mask for Loading

```python
in_ptrs = input_ptr + r[:, None] * cols + c[None, :]
in_mask = (r[:, None] < rows) & (c[None, :] < cols)
x = tl.load(in_ptrs, mask=in_mask, other=0.0)
```

* `r[:, None] * cols + c[None, :]` → **broadcast** to get all `(row, col)` indices in the tile.
* `in_mask` → ensures that if the tile goes past the matrix boundary (partial tiles), we **don’t read invalid memory**.
* `tl.load` → load values from memory into registers. Out-of-bounds elements are set to `0.0`.

---

## 6️⃣ Compute Output Coordinates (Transpose)

```python
out_r = c0 + offs_c
out_c = r0 + offs_r
```

* This **swaps row and column offsets**, which is the essence of a transpose.
* The tile in the input at `(r0..r0+TILE_X, c0..c0+TILE_Y)` will go to `(c0..c0+TILE_Y, r0..r0+TILE_X)` in the output.

---

## 7️⃣ Compute Output Pointers and Mask for Storing

```python
out_ptrs = output_ptr + out_r[:, None] * rows + out_c[None, :]
out_mask = (out_r[:, None] < cols) & (out_c[None, :] < rows)
tl.store(out_ptrs, tl.trans(x), mask=out_mask)
```

* `tl.trans(x)` → **transpose the tile** in registers.
* `out_mask` → ensures **partial tiles** are handled safely.
* `tl.store` → write the tile to the output. Only valid elements are written.

---

## 8️⃣ Host Launch Function

```python
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
```

* `triton.cdiv(rows, TILE_X)` → number of **tiles in the row direction**
* `triton.cdiv(cols, TILE_Y)` → number of **tiles in the column direction**
* Launches a **2D grid** of programs, one per tile.
* Tiles near the edges may be **partial tiles**, handled by `mask`.

---

## ✅ TL;DR

* Each program handles a **TILE_X × TILE_Y block** of the input matrix.
* Loads it into registers (`x`).
* Transposes it (`tl.trans`).
* Stores it in the correct place in the output.
* Masking handles **partial tiles** at the boundaries.
* The kernel works for **any matrix size**, not just multiples of the tile.

---
