import math
import numpy as np


def norm(x, x_prev):
    return math.sqrt(sum([(xn - xp) ** 2 for xn, xp in zip(x, x_prev)]))


def f1(x):
    return 2 * x[0] - np.cos(x[1])


def f2(x):
    return 2 * x[1] - np.exp(x[0])


def df1_dx1(x):
    return 2


def df1_dx2(x):
    return np.sin(x[1])


def df2_dx1(x):
    return -np.exp(x[0])


def df2_dx2(x):
    return 2


def phi1(x):
    return np.cos(x[1]) / 2


def phi2(x):
    return np.exp(x[0]) / 2


def dphi1_dx1(x):
    return 0


def dphi1_dx2(x):
    return -np.sin(x[1]) / 2


def dphi2_dx1(x):
    return np.exp(x[0]) / 2


def dphi2_dx2(x):
    return 0
