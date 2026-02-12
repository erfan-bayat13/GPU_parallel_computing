#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// ============================================================================
// ERROR CHECKING MACROS
// ============================================================================

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUSOLVER_CHECK(call) \
do { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER error at %s:%d: code %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_CHECK(call) \
do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: code %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Convert row-major (C/NumPy) to column-major (Fortran/cuBLAS)
 */
void transpose_matrix(const float *input, float *output, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            output[j * n + i] = input[i * n + j];
        }
    }
}

/**
 * Print matrix in readable format (row-major)
 */
void print_matrix(const char *name, const float *A, int n, int max_print = 5) {
    printf("%s:\n", name);
    int print_size = (n < max_print) ? n : max_print;
    for (int i = 0; i < print_size; i++) {
        for (int j = 0; j < print_size; j++) {
            printf("%8.4f ", A[i * n + j]);
        }
        if (n > max_print) printf("...");
        printf("\n");
    }
    if (n > max_print) printf("...\n");
    printf("\n");
}

/**
 * Verify A * A_inv ≈ I
 */
float verify_inverse(const float *A, const float *A_inv, int n) {
    float *result = (float *)malloc(n * n * sizeof(float));
    
    // Compute result = A * A_inv
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * A_inv[k * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    
    // Compute ||result - I||_F
    float error = 0.0f;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float expected = (i == j) ? 1.0f : 0.0f;
            float diff = result[i * n + j] - expected;
            error += diff * diff;
        }
    }
    error = sqrtf(error);
    
    free(result);
    return error;
}

// ============================================================================
// MATRIX INVERSION USING cuSOLVER + cuBLAS
// ============================================================================

/**
 * Invert matrix using LU decomposition + triangular solve
 * 
 * Algorithm:
 * 1. Compute LU decomposition: PA = LU
 * 2. Solve PA X = I for X (which gives X = A^(-1))
 *    This is done by solving: LU X = P^T I
 *    Step 2a: Solve L Y = P^T I for Y (forward substitution)
 *    Step 2b: Solve U X = Y for X (backward substitution)
 * 
 * cuSOLVER provides: getrf (LU), getrs (solve)
 */
void invert_matrix_cusolver(float *h_A, float *h_A_inv, int n) {
    // ------------------------------------------------------------------------
    // STEP 1: Create handles
    // ------------------------------------------------------------------------
    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    // ------------------------------------------------------------------------
    // STEP 2: Allocate device memory
    // ------------------------------------------------------------------------
    float *d_A;           // Matrix (will be overwritten with LU)
    float *d_B;           // Right-hand side (identity matrix)
    int *d_pivot;         // Pivot indices
    int *d_info;          // Status code
    float *d_work;        // Workspace
    
    CUDA_CHECK(cudaMalloc((void**)&d_A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_pivot, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_info, sizeof(int)));
    
    // Convert input to column-major and copy to GPU
    float *h_A_colmajor = (float *)malloc(n * n * sizeof(float));
    transpose_matrix(h_A, h_A_colmajor, n);
    CUDA_CHECK(cudaMemcpy(d_A, h_A_colmajor, n * n * sizeof(float), 
                         cudaMemcpyHostToDevice));
    
    // Create identity matrix in column-major format
    float *h_I = (float *)calloc(n * n, sizeof(float));
    for (int i = 0; i < n; i++) {
        h_I[i * n + i] = 1.0f;  // Column-major: [col * n + row]
    }
    CUDA_CHECK(cudaMemcpy(d_B, h_I, n * n * sizeof(float), 
                         cudaMemcpyHostToDevice));
    free(h_I);
    
    // ------------------------------------------------------------------------
    // STEP 3: Query workspace size and allocate
    // ------------------------------------------------------------------------
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
        cusolver_handle,
        n, n,
        d_A, n,
        &lwork
    ));
    
    printf("Workspace needed: %d floats (%.2f MB)\n", 
           lwork, lwork * sizeof(float) / 1e6);
    
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(float)));
    
    // ------------------------------------------------------------------------
    // STEP 4: LU Decomposition
    // ------------------------------------------------------------------------
    printf("Computing LU decomposition...\n");
    
    CUSOLVER_CHECK(cusolverDnSgetrf(
        cusolver_handle,
        n, n,           // Matrix dimensions
        d_A, n,         // Matrix and leading dimension
        d_work,         // Workspace
        d_pivot,        // OUTPUT: pivot indices
        d_info          // OUTPUT: status
    ));
    
    // Check for errors
    int h_info;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (h_info != 0) {
        if (h_info > 0) {
            fprintf(stderr, "ERROR: Matrix is singular at element %d\n", h_info);
        } else {
            fprintf(stderr, "ERROR: Invalid parameter at position %d\n", -h_info);
        }
        // Cleanup
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_pivot);
        cudaFree(d_info); cudaFree(d_work);
        cusolverDnDestroy(cusolver_handle);
        free(h_A_colmajor);
        exit(EXIT_FAILURE);
    }
    
    printf("  [OK] LU factorization successful (info = %d)\n", h_info);
    
    // ------------------------------------------------------------------------
    // STEP 5: Solve A X = I using LU factors
    // ------------------------------------------------------------------------
    printf("Solving for inverse...\n");
    
    // cusolverDnSgetrs solves: A X = B given LU(A)
    // We set B = I, so X = A^(-1)
    CUSOLVER_CHECK(cusolverDnSgetrs(
        cusolver_handle,
        CUBLAS_OP_N,    // No transpose (solve A X = B, not A^T X = B)
        n,              // Matrix size
        n,              // Number of right-hand sides (n columns of identity)
        d_A, n,         // LU factorization
        d_pivot,        // Pivot indices from LU
        d_B, n,         // INPUT: I, OUTPUT: A^(-1)
        d_info          // Status
    ));
    
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    
    if (h_info != 0) {
        fprintf(stderr, "ERROR: Solve failed with code %d\n", h_info);
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_pivot);
        cudaFree(d_info); cudaFree(d_work);
        cusolverDnDestroy(cusolver_handle);
        free(h_A_colmajor);
        exit(EXIT_FAILURE);
    }
    
    printf("  [OK] Matrix inversion successful (info = %d)\n", h_info);
    
    // ------------------------------------------------------------------------
    // STEP 6: Copy result back (convert to row-major)
    // ------------------------------------------------------------------------
    float *h_inv_colmajor = (float *)malloc(n * n * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_inv_colmajor, d_B, n * n * sizeof(float),
                         cudaMemcpyDeviceToHost));
    transpose_matrix(h_inv_colmajor, h_A_inv, n);
    
    // ------------------------------------------------------------------------
    // CLEANUP
    // ------------------------------------------------------------------------
    free(h_A_colmajor);
    free(h_inv_colmajor);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_pivot));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
}

// ============================================================================
// BENCHMARKING WITH CUDA EVENTS
// ============================================================================

float benchmark_inversion(float *h_A, int n, int num_trials) {
    // Setup
    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    
    float *d_A, *d_B, *d_work;
    int *d_pivot, *d_info;
    
    CUDA_CHECK(cudaMalloc((void**)&d_A, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_pivot, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_info, sizeof(int)));
    
    // Prepare data
    float *h_A_colmajor = (float *)malloc(n * n * sizeof(float));
    transpose_matrix(h_A, h_A_colmajor, n);
    
    float *h_I = (float *)calloc(n * n, sizeof(float));
    for (int i = 0; i < n; i++) {
        h_I[i * n + i] = 1.0f;
    }
    
    // Query workspace
    int lwork;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(cusolver_handle, n, n, d_A, n, &lwork));
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(float)));
    
    // Warmup
    CUDA_CHECK(cudaMemcpy(d_A, h_A_colmajor, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_I, n * n * sizeof(float), cudaMemcpyHostToDevice));
    CUSOLVER_CHECK(cusolverDnSgetrf(cusolver_handle, n, n, d_A, n, d_work, d_pivot, d_info));
    CUSOLVER_CHECK(cusolverDnSgetrs(cusolver_handle, CUBLAS_OP_N, n, n, d_A, n, d_pivot, d_B, n, d_info));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Create timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Benchmark
    float total_time = 0.0f;
    for (int trial = 0; trial < num_trials; trial++) {
        // Reset data
        CUDA_CHECK(cudaMemcpy(d_A, h_A_colmajor, n * n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_I, n * n * sizeof(float), cudaMemcpyHostToDevice));
        
        // Start timing
        CUDA_CHECK(cudaEventRecord(start));
        
        // LU decomposition
        CUSOLVER_CHECK(cusolverDnSgetrf(cusolver_handle, n, n, d_A, n, d_work, d_pivot, d_info));
        
        // Solve for inverse
        CUSOLVER_CHECK(cusolverDnSgetrs(cusolver_handle, CUBLAS_OP_N, n, n, d_A, n, d_pivot, d_B, n, d_info));
        
        // Stop timing
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        total_time += milliseconds;
    }
    
    float avg_time = total_time / num_trials;
    
    // Cleanup
    free(h_A_colmajor);
    free(h_I);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_pivot));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    
    return avg_time;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char **argv) {
    printf("============================================================\n");
    printf("Phase 4: cuSOLVER Matrix Inversion\n");
    printf("============================================================\n\n");
    
    // ------------------------------------------------------------------------
    // Test 1: Small matrix for correctness
    // ------------------------------------------------------------------------
    printf("Test 1: Small Matrix (3x3)\n");
    printf("------------------------------------------------------------\n");
    
    int n_small = 3;
    float h_A_small[] = {
        4.0f, 3.0f, 2.0f,
        3.0f, 2.0f, 1.0f,
        2.0f, 1.0f, 3.0f
    };
    
    float *h_A_inv_small = (float *)malloc(n_small * n_small * sizeof(float));
    
    print_matrix("Input A", h_A_small, n_small, n_small);
    
    invert_matrix_cusolver(h_A_small, h_A_inv_small, n_small);
    
    print_matrix("Inverse A^(-1)", h_A_inv_small, n_small, n_small);
    
    float error = verify_inverse(h_A_small, h_A_inv_small, n_small);
    printf("Verification: ||A * A^(-1) - I||_F = %.2e\n", error);
    
    if (error < 1e-4) {
        printf("[PASS] Correctness test passed!\n\n");
    } else {
        printf("[FAIL] Correctness test failed!\n\n");
    }
    
    free(h_A_inv_small);
    
    // ------------------------------------------------------------------------
    // Test 2: Benchmark different sizes
    // ------------------------------------------------------------------------
    printf("============================================================\n");
    printf("Performance Benchmarks\n");
    printf("============================================================\n\n");
    
    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    printf("  Size    | Time (ms) | GFLOPS\n");
    printf("------------------------------------------\n");
    
    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        
        // Generate random matrix
        float *h_A = (float *)malloc(n * n * sizeof(float));
        for (int j = 0; j < n * n; j++) {
            h_A[j] = (float)rand() / RAND_MAX - 0.5f;
        }
        
        // Make it well-conditioned
        for (int j = 0; j < n; j++) {
            h_A[j * n + j] += n * 0.1f;
        }
        
        // Benchmark
        int num_trials = (n < 1024) ? 20 : 10;
        float avg_time = benchmark_inversion(h_A, n, num_trials);
        
        // Calculate GFLOPS
        // LU: (2/3)n^3, Solve: 2n^3 ≈ (8/3)n^3 total
        float flops = (8.0f / 3.0f) * n * n * n;
        float gflops = (flops / 1e9) / (avg_time / 1000.0f);
        
        printf("%4dx%-4d | %9.3f | %6.2f\n", n, n, avg_time, gflops);
        
        free(h_A);
    }
    
    printf("\n");
    printf("============================================================\n");
    printf("Phase 4 Complete!\n");
    printf("============================================================\n");
    
    return 0;
}