def matmul(A,B):
    n, m, p = len(A), len(B), len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for k in range(m):
            for j in range(p):
                C[i][j] += A[i][k] * B[k][j]
    return C

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

def lu_decomposition(A):
    n = len(A)
    L = identity(n)
    U = [row[:] for row in A]

    for i in range(n):
        for j in range(i+1,n):
            if U[i][i] == 0:
                raise ZeroDivisionError("Zero pivot encountered")
        factor = U[j][i] / U[i][i]
        L[j][i] = factor
        for k in range(i,n):
            U[j][k] -= factor * U[i][k]
    return L,U

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
        x[i] = (y[i] - sum(U[i][j]*x[j] for j in range(i+1, n))) / U[i][i]
    return x


def invert_matrix(A):
    n = len(A)
    L, U = lu_decomposition(A)
    I = identity(n)
    A_inv = [[0.0]*n for _ in range(n)]

    for col in range(n):
        e = [I[row][col] for row in range(n)]
        y = forward_substitution(L, e)
        x = backward_substitution(U, y)
        for row in range(n):
            A_inv[row][col] = x[row]

    return A_inv

