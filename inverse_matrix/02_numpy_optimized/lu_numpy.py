import numpy as np

def lu_decomposition(A):
    A = A.copy().astype(np.float64)
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    P = np.eye(n)
    for i in range(n):
        # Pivot selection
        max_row = np.argmax(np.abs(U[i:, i])) + i
        if U[max_row, i] == 0:
            raise ValueError("Singular matrix")

        # Swap rows in U
        U[[i, max_row]] = U[[max_row, i]]
        P[[i, max_row]] = P[[max_row, i]]

        # Swap rows in L (only left part)
        if i > 0:
            L[[i, max_row], :i] = L[[max_row, i], :i]

        # Elimination
        for j in range(i+1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] -= factor * U[i, i:]
    return P, L, U

def invert_matrix(A):
    n = A.shape[0]
    P, L, U = lu_decomposition(A)

    # Solve PA = LU → A⁻¹ = U⁻¹ L⁻¹ P
    I = np.eye(n)
    PI = P @ I

    # Forward substitution (vectorized solve)
    Y = np.linalg.solve(L, PI)
    X = np.linalg.solve(U, Y)

    return X