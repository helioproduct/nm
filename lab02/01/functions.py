from scipy.optimize import fsolve

from math import log10, log


def f(x):
    return log10(x + 1) - x + 0.5


# первая производная
def f1(x):
    return 1 / ((x + 1) * log(10)) - 1


# вторая производная
def f2(x):
    return -1 / (log(10) * x**2 + 2 * log(10) * x + log(10))


def phi(x):
    return 0.5 + log10(x + 1)


def phi1(x):
    return 1 / (log(10) * (x + 1))


def phi2(x):
    return -log(10) / ((log(10) * x + log(10)) ** 2)
