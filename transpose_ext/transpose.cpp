#include <torch/extension.h>

// CUDA function declaration
void transpose_cuda(const float* A, float* B, int M, int N);

torch::Tensor transpose_forward(torch::Tensor A) {
    TORCH_CHECK(A.is_cuda(), "Tensor must be on CUDA");
    TORCH_CHECK(A.dim() == 2, "Only 2D tensors supported");

    int M = A.size(0);
    int N = A.size(1);

    auto B = torch::empty({N, M}, A.options());

    transpose_cuda(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        M, N
    );

    return B;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_forward", &transpose_forward, "CUDA Transpose");
}
