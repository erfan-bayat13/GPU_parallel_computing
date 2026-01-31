#include <iostream>
#include <cstdlib>
#include "matmul.h"
#include "CycleTimer.h"

int main() {
    const int WIDTH = 512;
    const int HEIGHT = 512;

    float* h_A = new float[WIDTH * HEIGHT];
    float* h_B = new float[WIDTH * HEIGHT];
    float* h_C = new float[WIDTH * HEIGHT];

    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        h_A[i] = rand() % 10;
        h_B[i] = rand() % 10;
    }

    Matrix A = {WIDTH, HEIGHT, h_A};
    Matrix B = {WIDTH, HEIGHT, h_B};
    Matrix C = {WIDTH, HEIGHT, h_C};

    double start = CycleTimer::currentSeconds();
    MatMul(A, B, C);
    cudaDeviceSynchronize();
    double end = CycleTimer::currentSeconds();

    std::cout << "GPU MatMul took " << (end - start) << " seconds\n";

    // Print first 4x4 block
    for (int i = 0; i < 4 && i < HEIGHT; ++i) {
        for (int j = 0; j < 4 && j < WIDTH; ++j)
            std::cout << C.elements[i * WIDTH + j] << " ";
        std::cout << "\n";
    }

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}
