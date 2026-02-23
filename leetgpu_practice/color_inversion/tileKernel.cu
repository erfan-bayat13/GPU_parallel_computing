#include <cuda_runtime.h>
#define TILE_WIDTH 16 


__global__ void invert_kernel(unsigned char* image, int width, int height) {
    __shared__ unsigned char tile[TILE_WIDTH][TILE_WIDTH * 4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    if(row < height && col < width) {
        int pixel_idx = (row * width + col) * 4;
        // Load RGBA into shared memory
        tile[ty][tx * 4 + 0] = image[pixel_idx + 0];
        tile[ty][tx * 4 + 1] = image[pixel_idx + 1];
        tile[ty][tx * 4 + 2] = image[pixel_idx + 2];
        tile[ty][tx * 4 + 3] = image[pixel_idx + 3];
    }

    __syncthreads();

    if(row < height && col < width) {
        // invert colors in shared memory
        tile[ty][tx * 4 + 0] = 255 - tile[ty][tx * 4 + 0];
        tile[ty][tx * 4 + 1] = 255 - tile[ty][tx * 4 + 1];
        tile[ty][tx * 4 + 2] = 255 - tile[ty][tx * 4 + 2];
        // tile[ty][tx * 4 + 3] stays the same

        // write back to global memory
        int pixel_idx = (row * width + col) * 4;
        image[pixel_idx + 0] = tile[ty][tx * 4 + 0];
        image[pixel_idx + 1] = tile[ty][tx * 4 + 1];
        image[pixel_idx + 2] = tile[ty][tx * 4 + 2];
        image[pixel_idx + 3] = tile[ty][tx * 4 + 3];
    }

}
// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char* image, int width, int height) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((width + TILE_WIDTH - 1)/TILE_WIDTH,
                       (height + TILE_WIDTH - 1)/TILE_WIDTH);

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}
