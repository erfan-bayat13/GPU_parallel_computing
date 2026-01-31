#include "matmul.h"
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16

// CUDA kernel
__global__ void naiveGEMMKernel(Matrix A, Matrix B, Matrix C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= C.height || col >= C.width) return;

    float Cvalue = 0;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];

    C.elements[row * C.width + col] = Cvalue;
}

// Host function that wraps kernel call
void naiveGEMM(const Matrix A, const Matrix B, Matrix C)
{
    Matrix d_A, d_B, d_C;
    d_A.width = A.width; d_A.height = A.height;
    d_B.width = B.width; d_B.height = B.height;
    d_C.width = C.width; d_C.height = C.height;

    size_t sizeA = A.width * A.height * sizeof(float);
    size_t sizeB = B.width * B.height * sizeof(float);
    size_t sizeC = C.width * C.height * sizeof(float);

    cudaMalloc(&d_A.elements, sizeA);
    cudaMalloc(&d_B.elements, sizeB);
    cudaMalloc(&d_C.elements, sizeC);

    cudaMemcpy(d_A.elements, A.elements, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, B.elements, sizeB, cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (A.height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    naiveGEMMKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // error checking (important!)
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
    }

    cudaMemcpy(C.elements, d_C.elements, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}
