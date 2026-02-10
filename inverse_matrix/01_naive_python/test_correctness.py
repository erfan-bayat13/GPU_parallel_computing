import numpy as np
from lu_decomposition import invert_matrix as lu_inv
from gauss_jordan import invert_matrix as gj_inv


def test():
    np.random.seed(0)
    A = np.random.rand(5,5) + 5*np.eye(5)
    A_list = A.tolist()

    A_inv_np = np.linalg.inv(A)

    A_inv_lu = np.array(lu_inv(A_list))
    A_inv_gj = np.array(gj_inv(A_list))

    print("LU error:", np.max(np.abs(A_inv_np - A_inv_lu)))
    print("Gauss-Jordan error:", np.max(np.abs(A_inv_np - A_inv_gj)))


if __name__ == "__main__":
    test()
