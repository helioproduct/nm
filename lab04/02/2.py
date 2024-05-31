import numpy as np
import matplotlib.pyplot as plt


def reference(x):
    return np.sin(x) + 2 - np.sin(x) * np.log((1 + np.sin(x)) / (1 - np.sin(x)))


a = 0
b = np.pi / 6
n = 100
h = (b - a) / (n - 1)
x = np.linspace(a, b, n)

ya = 2
yb = 2.5 - 0.5 * np.log(3)

A = np.zeros((n, n))
d = np.zeros(n)

A[0, 0] = 1
d[0] = ya

for i in range(1, n - 1):
    A[i, i - 1] = 1 / h**2 - np.tan(x[i]) / (2 * h)
    A[i, i] = -2 / h**2 + 2
    A[i, i + 1] = 1 / h**2 + np.tan(x[i]) / (2 * h)

A[-1, -1] = 1
d[-1] = yb


def solve(A, d):
    n = len(A)
    a, b, c = 0, A[0][0], A[0][1]
    P, Q = [0] * n, [0] * n
    P[0], Q[0] = -c / b, d[0] / b

    for i in range(1, n):
        a, c = 0, 0
        if i - 1 >= 0:
            a = A[i][i - 1]
        if i + 1 < n:
            c = A[i][i + 1]

        b = A[i][i]
        P[i] = -c / (b + a * P[i - 1])
        Q[i] = (d[i] - a * Q[i - 1]) / (b + a * P[i - 1])

    x = [0] * n
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


y = solve(A, d)
y_exact = reference(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, "b", label="Finite Difference Solution")
plt.plot(x, y_exact, "r--", label="Exact Solution")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Finite Difference Solution and Exact Solution")
plt.legend()
plt.grid(True)
plt.show()
