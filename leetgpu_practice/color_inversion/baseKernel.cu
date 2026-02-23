#include <cuda_runtime.h>


__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // global thread index
    int num_pixels = width * height;

    if (idx < num_pixels) {
        int base = idx * 4; // each pixel has 4 components: R, G, B, A
        image[base + 0] = 255 - image[base + 0]; // R
        image[base + 1] = 255 - image[base + 1]; // G
        image[base + 2] = 255 - image[base + 2]; // B
        // A remains unchanged: image[base + 3]
    }
}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
