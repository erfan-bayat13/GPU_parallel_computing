#include <cuda_runtime.h>

__device__ __forceinline__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME = 16777619u;
    const unsigned int OFFSET_BASIS = 2166136261u;

    unsigned int hash = OFFSET_BASIS;

    hash = (hash ^ ((input >> 0)  & 0xFFu)) * FNV_PRIME;
    hash = (hash ^ ((input >> 8)  & 0xFFu)) * FNV_PRIME;
    hash = (hash ^ ((input >> 16) & 0xFFu)) * FNV_PRIME;
    hash = (hash ^ ((input >> 24) & 0xFFu)) * FNV_PRIME;

    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output, int N, int R) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        unsigned int x = (unsigned int)input[i];

        for (int r = 0; r < R; r++) {
            x = fnv1a_hash(x);
        }

        output[i] = x;
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    cudaDeviceSynchronize();
}
