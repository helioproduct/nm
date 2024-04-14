import numpy as np
from math import sqrt
import copy


def matrix_norm2(A):
    sm2 = 0
    for row in A:
        sm2 += sum([x * x for x in row])
    return sqrt(sm2)


def solve_interations(A, b, eps):
    A = np.array(A)
    b = np.array(b)
    n = len(A)

    beta = [0] * n
    alpha = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                alpha[i][j] = 0
            else:
                alpha[i][j] = -A[i][j] / A[i][i]
        beta[i] = b[i] / A[i][i]

    x_curr = beta
    iterations = 0

    while True:
        x_prev = x_curr
        x_curr = beta + alpha @ x_prev
        iterations += 1

        if matrix_norm2([x_prev - x_curr]) <= eps:
            break

    return x_curr, iterations


def solve_seidel2(A, b, eps):
    A = np.array(A)
    b = np.array(b)
    n = len(A)

    beta = [0] * n
    alpha = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                alpha[i][j] = 0
            else:
                alpha[i][j] = -A[i][j] / A[i][i]
        beta[i] = b[i] / A[i][i]

    iterations = 0
    x_curr = [0] * n

    while True:
        x_prev = copy.deepcopy(x_curr)

        for i in range(n):
            s1 = sum(alpha[i][j] * x_curr[j] for j in range(i))
            s2 = sum(alpha[i][j] * x_prev[j] for j in range(i + 1, n))
            x_curr[i] = beta[i] + s1 + s2

        iterations += 1

        if matrix_norm2([np.array(x_prev) - np.array(x_curr)]) <= eps:
            break

    return x_curr, iterations


def solve_seidel(A, b, eps):
    A = np.array(A)
    b = np.array(b)
    n = len(A)

    beta = np.zeros(n)
    alpha = np.zeros((n, n))

    for i in range(n):
        beta[i] = b[i] / A[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = -A[i][j] / A[i][i]

    x_curr = np.array(beta)
    iterations = 0

    while True:
        x_prev = x_curr.copy()

        for i in range(n):
            s1 = np.dot(alpha[i, :i], x_curr[:i])
            s2 = np.dot(alpha[i, i + 1 :], x_prev[i + 1 :])
            x_curr[i] = beta[i] + s1 + s2

        iterations += 1

        if matrix_norm2([np.array(x_prev) - np.array(x_curr)]) <= eps:
            break

    return x_curr.tolist(), iterations


A = [
    [21, -6, -9, -4],
    [-6, 20, -4, 2],
    [-2, -7, -20, 3],
    [4, 9, 6, 24],
]


b = [127, -144, 236, -5]

eps = 0.001

iteractions_solution, iterations = solve_interations(A, b, eps)


print(
    f"Решение методом простых итераций: {iteractions_solution}, кол-во итераций = {iterations}"
)

seidel_solution, iterations = solve_seidel(A, b, eps)

print(f"Решение методом Зейделя: {seidel_solution}, кол-во итераций = {iterations}")
