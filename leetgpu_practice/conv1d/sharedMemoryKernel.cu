#include <cuda_runtime.h>
#define BLOCK_SIZE 256

__global__ void convolution_1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int input_size,
    int kernel_size)
{
    extern __shared__ float tile[];

    int tx = threadIdx.x;
    int global_out = blockIdx.x * BLOCK_SIZE + tx;
    int output_size = input_size - kernel_size + 1;

    int tile_start = blockIdx.x * BLOCK_SIZE;
    int tile_size = BLOCK_SIZE + kernel_size - 1;

    // Load input tile collaboratively
    for (int i = tx; i < tile_size; i += BLOCK_SIZE) {
        int global_in = tile_start + i;

        if (global_in < input_size)
            tile[i] = input[global_in];  
        else
            tile[i] = 0.0f;
    }

    __syncthreads();  // REQUIRED before using shared memory

    // Compute convolution
    if (global_out < output_size) {
        float sum = 0.0f;

        for (int k = 0; k < kernel_size; k++) {
            sum += tile[tx + k] * kernel[k];
        }

        output[global_out] = sum;
    }
}


// input, kernel, output are device pointers
extern "C" void solve(const float* input,
                      const float* kernel,
                      float* output,
                      int input_size,
                      int kernel_size)
{
    int output_size = input_size - kernel_size + 1;

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    // Shared memory size must be specified
    size_t shared_mem_size =
        (BLOCK_SIZE + kernel_size - 1) * sizeof(float);

    convolution_1d_kernel<<<blocksPerGrid,
                            threadsPerBlock,
                            shared_mem_size>>>(
        input, kernel, output,
        input_size, kernel_size);

    cudaDeviceSynchronize();
}