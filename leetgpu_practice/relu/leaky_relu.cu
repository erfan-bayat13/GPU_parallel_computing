#include <cuda_runtime.h>

__global__ void leaky_relu4_kernel(const float4* input, float4* output, int N4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N4) {
        float4 x = input[i];    
        x.x = fmaxf(x.x, 0.01f * x.x);
        x.y = fmaxf(x.y, 0.01f * x.y);
        x.z = fmaxf(x.z, 0.01f * x.z);
        x.w = fmaxf(x.w, 0.01f * x.w);
        output[i] = x;
    }
}

__global__ void leaky_relu_tail_kernel(const float* input, float* output, int start, int N) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        output[i] = fmaxf(input[i], 0.01f * input[i]);
    }
}
// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int N4 = N / 4;
    int blocks4 = (N4 + threadsPerBlock - 1) / threadsPerBlock;

    leaky_relu4_kernel<<<blocks4, threadsPerBlock>>>(reinterpret_cast<const float4*>(input), reinterpret_cast<float4*>(output), N4);

    int start = N4 * 4;
    int tail = N - start;

    if (tail > 0) {
        int blocksTail = (tail + threadsPerBlock - 1) / threadsPerBlock;
        leaky_relu_tail_kernel<<<blocksTail, threadsPerBlock>>>(input, output, start, N);
    }
    cudaDeviceSynchronize();
}
