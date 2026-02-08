#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void transpose_tiled_kernel(
    const float* A, float* B, int M, int N
) {
    __shared__ float tile[TILE][TILE + 1]; // +1 avoids bank conflicts

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < N && y < M)
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];

    __syncthreads();

    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    if (x < M && y < N)
        B[y * M + x] = tile[threadIdx.x][threadIdx.y];
}

void transpose_tiled(const float* A, float* B, int M, int N) {
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    transpose_tiled_kernel<<<grid, block>>>(A, B, M, N);
}
