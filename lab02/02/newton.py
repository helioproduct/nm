from functions import *
from numpy.linalg import solve


def LU(A):
    n = len(A)
    L = [[0.0 if i != j else 1.0 for j in range(n)] for i in range(n)]
    U = [row[:] for row in A]
    P = list(range(n))

    for k in range(n):
        max_index = max(range(k, n), key=lambda i: abs(U[i][k]))
        if k != max_index:
            U[k], U[max_index] = U[max_index], U[k]
            P[k], P[max_index] = P[max_index], P[k]
            if k > 0:
                L[k][:k], L[max_index][:k] = L[max_index][:k], L[k][:k]

        for i in range(k + 1, n):
            L[i][k] = U[i][k] / U[k][k]
            for j in range(k, n):
                U[i][j] -= L[i][k] * U[k][j]

    return L, U, P


def solve_system(L, U, b):
    # n = len(A)
    # L, U, P = LU(A)
    # Lz = b

    n = len(b)
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


def Newton(a, b, eps):

    def JacobiMatrix(x):
        return [[df1_dx1(x), df1_dx2(x)], [df2_dx1(x), df2_dx2(x)]]

    x0 = [a[0], b[0]]
    x1 = [a[1], b[1]]

    x_prev = [(x0[1] + x0[0]) / 2, (x1[1] + x1[0]) / 2]

    iteration = 0
    while True <= 1000:
        iteration += 1
        jacobi = np.array(JacobiMatrix(x_prev))
        b = np.array([-f1(x_prev), -f2(x_prev)])
        # Jx = b
        # b = -F

        L, U, _ = LU(jacobi)
        delta_x = solve_system(L, U, b)

        x = [px + dx for px, dx in zip(x_prev, delta_x)]
        if norm(x, x_prev) <= eps:
            break
        x_prev = x

    return x, iteration
