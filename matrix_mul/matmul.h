#pragma once
#include <cuda_runtime.h>

// Matrix structure
typedef struct {
    int width;
    int height;
    int stride;
    float* elements;
} Matrix;

// Forward declaration of the GPU matrix multiplication
void naiveGEMM(const Matrix A, const Matrix B, Matrix C);
void MatMul(const Matrix, const Matrix, Matrix);