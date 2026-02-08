#include <torch/extension.h>

void transpose_naive(const float*, float*, int, int);
void transpose_tiled(const float*, float*, int, int);

torch::Tensor run_naive(torch::Tensor A) {
    int M = A.size(0), N = A.size(1);
    auto B = torch::empty({N, M}, A.options());
    transpose_naive(A.data_ptr<float>(), B.data_ptr<float>(), M, N);
    return B;
}

torch::Tensor run_tiled(torch::Tensor A) {
    int M = A.size(0), N = A.size(1);
    auto B = torch::empty({N, M}, A.options());
    transpose_tiled(A.data_ptr<float>(), B.data_ptr<float>(), M, N);
    return B;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive", &run_naive);
    m.def("tiled", &run_tiled);
}
