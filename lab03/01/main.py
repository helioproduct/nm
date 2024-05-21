import numpy as np
import sympy as sp

import matplotlib.pyplot as plt

from math import log


def f(x):
    return np.log(x) + x


def lagrange_polynomial(xi, yi):
    x = sp.symbols("x")
    n = len(xi)
    L = 0

    for i in range(n):
        li = 1
        for j in range(n):
            if i != j:
                li *= (x - xi[j]) / (xi[i] - xi[j])
        L += yi[i] * li

    return sp.simplify(L)


if __name__ == "__main__":

    xi = np.array([0.1, 0.5, 0.9, 1.3])
    yi = np.array([f(x) for x in xi])

    L = lagrange_polynomial(xi, yi)

    print("Многочлен лагранжа")
    print(L)

    lagrange_func = sp.lambdify(sp.symbols("x"), L, "numpy")

    x_range = np.linspace(min(xi) - 0.1, max(xi) + 0.1, 1000)
    y_lagrange = lagrange_func(x_range)
    y_original = f(x_range)

    # Построение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_original, label="Исходная функция f(x)", color="blue")
    plt.plot(
        x_range, y_lagrange, label="Многочлен Лагранжа", color="red", linestyle="--"
    )
    plt.scatter(xi, yi, color="black", label="Узловые точки")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("График многочлена Лагранжа и исходной функции")
    plt.legend()
    plt.grid(True)
    plt.show()
