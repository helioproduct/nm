import numpy as np
from math import sqrt, pi, atan


def matrix_norm2(A):
    return sqrt(np.sum(np.triu(A, k=1) ** 2))


def jacobi_method(A, eps):
    n = len(A)
    iterations = 0
    A = np.array(A, dtype=float)
    eigen_vectors = np.eye(n)

    while True:
        upper_triangle = np.triu(A, k=1)
        max_index_linear = np.argmax(np.abs(upper_triangle))
        i, j = np.unravel_index(max_index_linear, upper_triangle.shape)

        if A[i, i] == A[j, j]:
            phi = np.pi / 4
        else:
            phi = 0.5 * atan(2 * A[i, j] / (A[i, i] - A[j, j]))

        U = np.eye(n)
        U[i, i] = U[j, j] = np.cos(phi)
        U[i, j], U[j, i] = -np.sin(phi), np.sin(phi)

        A = U.T @ A @ U
        eigen_vectors = eigen_vectors @ U

        iterations += 1

        if matrix_norm2(A) <= eps:
            break

    eigen_values = np.diag(A)

    return eigen_values, [list(v) for v in eigen_vectors.T], iterations


A = np.array([[8, -3, 9], [-3, 8, -2], [9, -2, -8]])

eigen_values, eigen_vectors, iterations = jacobi_method(A, 0.001)
print("Приближенные собственные значения:", eigen_values)

print("Собственные векторы:")

for v in eigen_vectors:
    print(v)

print("Число итераций:", iterations)
