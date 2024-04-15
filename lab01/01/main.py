import numpy as np


def LU(A):
    n = len(A)
    L = [[0.0 if i != j else 1.0 for j in range(n)] for i in range(n)]
    U = A

    for k in range(n):
        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    return L, U


def solve_system(A, b):
    n = len(A)
    L, U = LU(A)
    # Lz = b
    z = [0 for _ in range(n)]
    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i][j] * z[j]
        z[i] = (b[i] - s) / L[i][i]

    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(n - 1, i - 1, -1):
            s += U[i][j] * x[j]
        x[i] = (z[i] - s) / U[i][i]

    return x


def det(A):
    n = len(A)
    L, U = LU(A)
    result = 1
    for i in range(n):
        result *= U[i][i]

    for i in range(n):
        result *= L[i][i]

    return result


def transpose(A):
    A_T = [[] for _ in range(len(A[0]))]
    for i in range(len(A_T)):
        A_T[i].extend([A[j][i] for j in range(len(A))])
    return A_T


def inverse(A):
    n = len(A)
    # E^-1
    E = [[float(i == j) for i in range(n)] for j in range(n)]

    inversed = []

    for column in E:
        x = solve_system(A, column)
        inversed.append(x)

    return transpose(inversed)


if __name__ == "__main__":
    A = [[-5, -1, -3, -1], [-2, 0, 8, -4], [-7, -2, 2, -2], [2, -4, -4, 4]]
    b = [18, -12, 6, -12]

    A = [
        [23994, 340923, 239492, -12],
        [1239, 2304, 20, -20],
        [234, 2342, 23, 0],
        [0, 1, 0, 4],
    ]
    L, U = LU(A)

    print(np.array(L) @ np.array(U))

    x = solve_system(A, b)

    print(f"Ax = b, x = {x} \n")
    print(f"detA = {det(A)} \n")
    # print(np.linalg.det(np.array(A)))

    inversed = inverse(A)

    print("A inversed = ")
    print(np.array(inversed))
