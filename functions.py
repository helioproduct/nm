import numpy as np
from math import sqrt


def f1(X: np.array):
    x1, x2 = X[0], X[1]
    return 2 * x1**2 - x2 + x2**2 - 2


def df1_dx1(X: np.array):
    x1, x2 = X[0], X[1]
    return 4 * x2


def df1_dx2(X: np.array):
    _, x2 = X[0], X[1]
    return 2 * x2 - 1


def f2(X: np.array):
    x1, x2 = X[0], X[1]
    return x1 - sqrt(x2 + 2) + 1


def df2_dx1(X: np.array):
    _, _ = X[0], X[1]
    return 1


def df2_dx2(X: np.array):
    _, x2 = X[0], X[1]
    return -1 / (2 * sqrt(x2 + 2))
