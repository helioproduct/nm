from functions import F
from newton import Newton
import numpy as np


if __name__ == "__main__":

    X0 = np.array([1.0, 1.0])
    eps = 0.001
    solution, iterations = Newton(F, X0, eps)

    print("Решение методом Ньютона:")
    print(solution)
    print("Количество итераций:")
    print(iterations)
