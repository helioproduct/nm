import numpy as np


def TMA(a, b, c, d):
    n = len(d)
    P = np.zeros(n)
    Q = np.zeros(n)

    # Прямой ход
    denom = b[0]
    P[0] = -c[0] / denom
    Q[0] = d[0] / denom

    for i in range(1, n - 1):
        denom = b[i] + a[i]*P[i-1]
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i]*Q[i-1]) / denom

    # Последний элемент
    denom = b[-1] + a[-1]*P[-2]
    Q[-1] = (d[-1] - a[-1]*Q[-2]) / denom

    # Обратный ход
    x = np.zeros(n)
    x[-1] = Q[-1]
    for i in range(n-2, -1, -1):
        x[i] = P[i]*x[i+1] + Q[i]

    return x
