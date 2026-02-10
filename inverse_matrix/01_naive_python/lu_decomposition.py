def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def lu_decomposition(A):
    n = len(A)
    L = identity(n)
    U = [row[:] for row in A]
    P = identity(n)

    for i in range(n):
        # Pivot selection
        max_row = max(range(i, n), key=lambda r: abs(U[r][i]))
        if U[max_row][i] == 0:
            raise ValueError("Singular matrix")

        # Swap rows in U
        U[i], U[max_row] = U[max_row], U[i]
        P[i], P[max_row] = P[max_row], P[i]

        # Swap rows in L (only left part)
        if i > 0:
            L[i][:i], L[max_row][:i] = L[max_row][:i], L[i][:i]

        # Elimination
        for j in range(i+1, n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor
            for k in range(i, n):
                U[j][k] -= factor * U[i][k]

    return P, L, U


def forward_substitution(L, b):
    n = len(L)
    y = [0.0]*n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j]*y[j] for j in range(i))
    return y


def backward_substitution(U, y):
    n = len(U)
    x = [0.0]*n
    for i in reversed(range(n)):
        if abs(U[i][i]) < 1e-12:
            raise ZeroDivisionError("Zero pivot in U")
        x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]
    return x


def invert_matrix(A):
    n = len(A)
    P, L, U = lu_decomposition(A)
    I = identity(n)
    A_inv = [[0.0]*n for _ in range(n)]

    # Apply P to I
    PI = [[sum(P[i][k]*I[k][j] for k in range(n)) for j in range(n)] for i in range(n)]

    for col in range(n):
        e = [PI[row][col] for row in range(n)]
        y = forward_substitution(L, e)
        x = backward_substitution(U, y)
        for row in range(n):
            A_inv[row][col] = x[row]

    return A_inv
