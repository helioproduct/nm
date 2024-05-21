import numpy as np


def solve(A, d):
    n = len(A)
    # прямой ход
    a, b, c = 0, A[0][0], A[0][1]
    P, Q = [0] * n, [0] * n
    P[0], Q[0] = -c / b, d[0] / b

    # xi = Pi x_i + 1 + Qi
    for i in range(1, n):
        a, c = 0, 0
        if i - 1 >= 0:
            a = A[i][i - 1]
        if i + 1 < n:
            c = A[i][i + 1]

        b = A[i][i]
        P[i] = -c / (b + a * P[i - 1])
        Q[i] = (d[i] - a * Q[i - 1]) / (b + a * P[i - 1])

    # обратный ход
    x = [0] * n
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


A = [
    [18, -9, 0, 0, 0],
    [2, -9, -4, 0, 0],
    [0, -9, 21, -8, 0],
    [0, 0, -4, -10, 5],
    [0, 0, 0, 7, 12],
]

b = [-81, 71, -39, 64, 3]


result = solve(A, b)
print("Решение методом прогонки: ")
print(result)
