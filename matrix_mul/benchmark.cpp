#include "matmul.h"
#include "CycleTimer.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cstdlib>
#include <cmath>

// Helper: Initialize matrix with random values
void initMatrix(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

// Helper: Verify results match (within tolerance)
bool verifyResults(const float* C_test, const float* C_ref, int size, float tolerance = 1e-3) {
    int errors = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabs(C_test[i] - C_ref[i]);
        if (diff > tolerance) {
            if (errors < 10) {  // Only print first 10 errors
                printf("  Mismatch at index %d: test=%.6f, ref=%.6f, diff=%.6f\n",
                       i, C_test[i], C_ref[i], diff);
            }
            errors++;
        }
    }
    
    if (errors > 0) {
        printf("  Total errors: %d / %d (%.2f%%)\n", 
               errors, size, 100.0f * errors / size);
        return false;
    }
    return true;
}

// Helper: Calculate TFLOPS
float calculateTFLOPS(int M, int N, int K, float milliseconds) {
    float operations = 2.0f * M * N * K;
    float seconds = milliseconds / 1000.0f;
    float TFLOPS = (operations / seconds) / 1e12;
    return TFLOPS;
}

// Benchmark naive GEMM kernel
float benchmarkNaive(int M, int N, int K, 
                     const float* h_A, const float* h_B, float* h_C) {
    printf("\n--- Naive GEMM ---\n");
    
    // Create Matrix structs
    Matrix A = {K, M, K, const_cast<float*>(h_A)};
    Matrix B = {N, K, N, const_cast<float*>(h_B)};
    Matrix C = {N, M, N, h_C};
    
    // Warm-up run
    naiveGEMM(A, B, C);
    cudaDeviceSynchronize();
    
    // Timed run using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    naiveGEMM(A, B, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate performance
    float tflops = calculateTFLOPS(M, N, K, milliseconds);
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  Performance: %.3f TFLOPS\n", tflops);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return tflops;
}

// benchmark MatMul -- tiled version
float benchmarkMatMul(int M, int N, int K,
                      const float* h_A, const float* h_B, float* h_C) {
    printf("\n--- Tiled MatMul ---\n");
    
    // Create Matrix structs
    Matrix A = {K, M, K, const_cast<float*>(h_A)};
    Matrix B = {N, K, N, const_cast<float*>(h_B)};
    Matrix C = {N, M, N, h_C};
    
    // Warm-up run
    MatMul(A, B, C);
    cudaDeviceSynchronize();
    
    // Timed run using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    MatMul(A, B, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate performance
    float tflops = calculateTFLOPS(M, N, K, milliseconds);
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  Performance: %.3f TFLOPS\n", tflops);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return tflops;
}


// Benchmark cuBLAS
float benchmarkCublas(int M, int N, int K,
                      const float* h_A, const float* h_B, float* h_C) {
    printf("\n--- cuBLAS ---\n");
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);
    
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Warm-up
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
    cudaDeviceSynchronize();
    
    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,
                d_A, K,
                &beta,
                d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);
    
    // Calculate performance
    float tflops = calculateTFLOPS(M, N, K, milliseconds);
    printf("  Time: %.3f ms\n", milliseconds);
    printf("  Performance: %.3f TFLOPS\n", tflops);
    
    // T4 theoretical peak FP32: ~8.1 TFLOPS
    float efficiency = (tflops / 8.1f) * 100.0f;
    printf("  Efficiency vs T4 peak FP32 (8.1 TFLOPS): %.1f%%\n", efficiency);
    
    // Cleanup
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return tflops;
}

int main() {
    printf("========================================\n");
    printf("GEMM Benchmark: Naive vs Tiled vs cuBLAS\n");
    printf("========================================\n");
    
    // Print GPU info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Memory: %.2f GB\n\n", prop.totalGlobalMem / 1e9);
    
    // Test different matrix sizes
    int sizes[] = {128, 256, 512, 1024, 2048};
    int numSizes = 5;
    
    for (int i = 0; i < numSizes; i++) {
        int M = sizes[i];
        int N = sizes[i];
        int K = sizes[i];
        
        printf("\n========================================\n");
        printf("Matrix Size: %d × %d × %d\n", M, K, N);
        printf("========================================\n");
        
        // Allocate host memory
        float* h_A = new float[M * K];
        float* h_B = new float[K * N];
        float* h_C_naive = new float[M * N];
        float* h_C_tiled = new float[M * N];
        float* h_C_cublas = new float[M * N];
        
        // Initialize with random data
        initMatrix(h_A, M * K);
        initMatrix(h_B, K * N);
        
        // Benchmark naive
        float tflops_naive = benchmarkNaive(M, N, K, h_A, h_B, h_C_naive);
        
        // Benchmark tiled MatMul
        float tflops_tiled = benchmarkMatMul(M, N, K, h_A, h_B, h_C_tiled);
        
        // Benchmark cuBLAS
        float tflops_cublas = benchmarkCublas(M, N, K, h_A, h_B, h_C_cublas);
        
        // Verify correctness
        printf("\n--- Verification ---\n");
        bool correct_naive = verifyResults(h_C_naive, h_C_cublas, M * N);
        bool correct_tiled = verifyResults(h_C_tiled, h_C_cublas, M * N);
        if (correct_naive && correct_tiled) {
            printf("  ✓ Results match for both naive and tiled implementations!\n");
        } else {
            if (!correct_naive) {
                printf("  ✗ Naive results DO NOT match cuBLAS!\n");
            }
            if (!correct_tiled) {
                printf("  ✗ Tiled results DO NOT match cuBLAS!\n");
            }
        }
        
        // Performance comparison
        if (tflops_naive > 0 && tflops_tiled > 0 && tflops_cublas > 0) {
            printf("\n--- Performance Summary ---\n");
            printf("  Naive: %.3f TFLOPS\n", tflops_naive);
            printf("  Tiled: %.3f TFLOPS\n", tflops_tiled);
            printf("  cuBLAS: %.3f TFLOPS\n", tflops_cublas);
            printf("  Tiled is %.2fx faster than naive\n", tflops_tiled / tflops_naive);
            printf("  cuBLAS is %.2fx faster than tiled\n", tflops_cublas / tflops_tiled);
            printf("  Naive is %.1f%% of cuBLAS performance\n", 
                   (tflops_naive / tflops_cublas) * 100.0f);
            printf("  Tiled is %.1f%% of cuBLAS performance\n", 
                   (tflops_tiled / tflops_cublas) * 100.0f);
        }
        
        // Cleanup
        delete[] h_A;
        delete[] h_B;
        delete[] h_C_naive;
        delete[] h_C_tiled;
        delete[] h_C_cublas;
    }
    
    return 0;
}