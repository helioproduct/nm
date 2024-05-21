import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def S(a, b, c, d, x, xi):
    return a + b * (x - xi) + c * (x - xi) ** 2 + d * (x - xi) ** 3


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


x = np.array([0.1, 0.5, 0.9, 1.3, 1.7])
f = np.array([-2.2026, -0.19315, 0.79464, 1.5624, 2.2306])


if __name__ == "__main__":

    h = np.diff(x)

    n = len(x)
    A = np.zeros((n - 2, n - 2))
    b = np.zeros(n - 2)

    for i in range(1, n - 1):
        if i == 1:
            A[i - 1, i - 1] = 2 * (h[i - 1] + h[i])
            A[i - 1, i] = h[i]
        elif i == n - 2:
            A[i - 1, i - 2] = h[i - 1]
            A[i - 1, i - 1] = 2 * (h[i - 1] + h[i])
        else:
            A[i - 1, i - 2] = h[i - 1]
            A[i - 1, i - 1] = 2 * (h[i - 1] + h[i])
            A[i - 1, i] = h[i]

        b[i - 1] = 3 * ((f[i + 1] - f[i]) / h[i] - (f[i] - f[i - 1]) / h[i - 1])

    c = np.zeros(n)
    c[1 : n - 1] = solve(A, b)

    a = f[:-1]
    b = [
        (f[i + 1] - f[i]) / h[i] - (h[i] * (2 * c[i] + c[i + 1])) / 3
        for i in range(len(h))
    ]
    d = [(c[i + 1] - c[i]) / (3 * h[i]) for i in range(len(h))]

    # Построение графика
    plt.figure(figsize=(10, 6))

    # График сплайна
    x_dense = np.linspace(x[0], x[-1], 400)
    y_dense = np.zeros_like(x_dense)

    x_test = 0.8

    for j in range(len(x) - 1):
        mask = (x_dense >= x[j]) & (x_dense <= x[j + 1])

        y_dense[mask] = S(a[j], b[j], c[j], d[j], x_dense[mask], x[j])

        if x_dense[mask][0] <= x_test <= x_dense[mask][-1]:
            s_value = S(a[j], b[j], c[j], d[j], x_test, x[j])
            print(f"S({x_test}) = {s_value}")

    plt.plot(x_dense, y_dense, label="Cubic Spline", color="blue")
    plt.scatter(x, f, color="red", label="Data Points")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Cubic Spline Interpolation")
    plt.legend()
    plt.grid(True)
    plt.show()
