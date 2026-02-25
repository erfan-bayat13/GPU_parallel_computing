// simple implementation of 1D convolution in CUDA
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output, int input_size, int kernel_size) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int M = input_size - kernel_size + 1;
    if (n < M){
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            sum += input[n + k] * kernel[k];
        }
        output[n] = sum;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
