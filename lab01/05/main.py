import numpy as np
import copy

from math import sqrt


def sign(x):
    return 0 if x == 0 else x // abs(x)


def H_matrix(A, index):

    A = np.array(A)
    n = A.shape[0]
    v = np.zeros(n)

    b = A.T[index]
    v[index] = b[index] + sign(b[index]) * sqrt(sum([a * a for a in b]))

    for i in range(index + 1, n):
        v[i] = b[i]

    v = v[:, np.newaxis]

    E = np.eye(n)
    return E - 2 * (v @ v.T) / (v.T @ v)


def QR(A):
    A = np.array(A)
    n = A.shape[0]

    H = H_matrix(A, 0)
    Q = copy.deepcopy(H)

    A = H @ A
    for i in range(1, n - 1):
        H = H_matrix(A, i)
        A = H @ A
        Q = Q @ H
    return Q, A


def get_complex_values(A, i):
    n = A.shape[0]
    a11 = A[i][i]
    a12 = A[i][i + 1] if i + 1 < n else 0
    a21 = A[i + 1][i] if i + 1 < n else 0
    a22 = A[i + 1][i + 1] if i + 1 < n else 0
    return np.roots((1, -a11 - a22, a11 * a22 - a12 * a21))


def is_complex(A, i, eps):
    Q, R = QR(A)
    complex_curr = get_complex_values(A, i)
    A = R @ Q
    complex_next = get_complex_values(A, i)

    result = abs(complex_curr[0] - complex_next[0]) <= eps
    result = result and abs(complex_curr[1] - complex_next[1]) <= eps

    return result


def almost_zero(vec, eps):
    return all([abs(x) <= eps for x in vec])


def eigen_value(A, i, eps):
    A = np.copy(A)

    while True:
        Q, R = QR(A)
        A = R @ Q

        if almost_zero(A[i + 1 :, i], eps):
            return A[i][i], A
        elif is_complex(A, i, eps):
            return get_complex_values(A, i), A


def eigen_values(A, eps):
    n = A.shape[0]
    A_i = np.copy(A)
    eigen_values = []

    count = 0
    while count < n:
        cur_eigen_values, A = eigen_value(A_i, count, eps)
        if isinstance(cur_eigen_values, np.ndarray):
            eigen_values.extend(cur_eigen_values)
            count += 2
        else:
            eigen_values.append(cur_eigen_values)
            count += 1
    return eigen_values


A = [[1, 4, 2], [4, -1, 3], [2, 3, 1]]


A = [
    [1, 3, 1, 2, 5],
    [1, 1, 4, 34, 4],
    [4, 3, 1, 5, 4],
    [1, 1, 34, 4, 5],
    [1, 43, 43, 4, 5],
]


# A = [[134, 8234, 234],
#      [23, 0, 0],
#      [1, 1, 1]]


A = np.array(A)

eps = 0.01
result = eigen_values(A, eps)

print(result)
