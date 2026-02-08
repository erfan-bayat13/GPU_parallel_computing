#include <cuda.h>
#include <cuda_runtime.h>

__global__ void transpose_kernel(const float* __restrict__ A, float* __restrict__ B, int M, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N){
        B[col * M + row] = A[row * N + col];
    }
}

void transpose_cuda(const float* A, float* B, int M, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    transpose_kernel<<<grid, block>>>(A, B, M, N);
}