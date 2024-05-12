import numpy as np

from functions import *


def Newton(F, X0, eps):
    X = X0.copy()
    iterations = 0

    while np.linalg.norm(F(X)) > eps:
        J = JacobiMatrix(X)
        delta = np.linalg.solve(J, -F(X))
        X = X + delta
        iterations += 1

    return X, iterations
