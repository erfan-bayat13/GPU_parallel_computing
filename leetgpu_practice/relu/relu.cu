#include <cuda_runtime.h>

__global__ void relu4_kernel(const float4* input, float4* output, int N4) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N4) {
        float4 x = input[i];
        x.x = fmaxf(x.x, 0.0f);
        x.y = fmaxf(x.y, 0.0f);
        x.z = fmaxf(x.z, 0.0f);
        x.w = fmaxf(x.w, 0.0f);
        output[i] = x;
    }
}

__global__ void relu_tail_kernel(const float* input, float* output, int start, int N) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        output[i] = fmaxf(input[i], 0.0f);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;

    int N4 = N / 4;
    int blocks4 = (N4 + threads - 1) / threads;

    relu4_kernel<<<blocks4, threads>>>(
        reinterpret_cast<const float4*>(input),
        reinterpret_cast<float4*>(output),
        N4
    );

    int start = N4 * 4;
    int tail = N - start;

    if (tail > 0) {
        int blocksTail = (tail + threads - 1) / threads;
        relu_tail_kernel<<<blocksTail, threads>>>(input, output, start, N);
    }
}