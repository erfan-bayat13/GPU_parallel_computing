# Pure Python Gauss-Jordan inversion

def identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def invert_matrix(A):
    n = len(A)
    M = [A[i] + identity(n)[i] for i in range(n)]

    for i in range(n):
        pivot = M[i][i]
        if pivot == 0:
            raise ZeroDivisionError("Zero pivot encountered")

        # Normalize row
        for j in range(2*n):
            M[i][j] /= pivot

        # Eliminate column
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(2*n):
                    M[k][j] -= factor * M[i][j]

    # Extract inverse
    return [row[n:] for row in M]
