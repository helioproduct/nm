import numpy as np
import sympy as sp

import matplotlib.pyplot as plt


def f(x):
    return np.log(x) + x


def divided_diff(xi, yi):
    n = len(yi)
    factors = np.zeros([n, n])
    factors[:, 0] = yi

    for j in range(1, n):
        for i in range(n - j):
            factors[i][j] = (factors[i + 1][j - 1] - factors[i][j - 1]) / (
                xi[i + j] - xi[i]
            )

    return factors[0]


def newton_polynomial(xi, yi):
    x = sp.symbols("x")
    factors = divided_diff(xi, yi)

    n = len(factors)
    polynomial = factors[0]

    for i in range(1, n):
        term = factors[i]
        for j in range(i):
            term *= x - xi[j]
        polynomial += term

    return sp.simplify(polynomial)


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
    xi1 = np.array([0.1, 0.5, 0.9, 1.3])
    yi1 = np.array([f(x) for x in xi1])

    xi2 = np.array([0.1, 0.5, 1.1, 1.3])
    yi2 = np.array([f(x) for x in xi2])

    x_test, y_test = 0.8, f(0.8)

    # Многочлен Лагранжа
    L = lagrange_polynomial(xi2, yi2)
    print("Многочлен Лагранжа")
    print(L)
    lagrange_func = sp.lambdify(sp.symbols("x"), L, "numpy")

    print("Лагранж: погрешность", abs(y_test - lagrange_func(x_test)))
    # print(lagrange_func(x_test))

    # Многочлен Ньютона
    N = newton_polynomial(xi1, yi1)
    print("Многочлен Ньютона")
    print(N)
    newton_func = sp.lambdify(sp.symbols("x"), N, "numpy")
    print("Ньютон: погрешность", abs(y_test - newton_func(x_test)))

    x_range = np.linspace(0, 3, 1000)
    y_lagrange = lagrange_func(x_range)
    y_newton = newton_func(x_range)
    y_original = f(x_range)

    # Построение графиков
    plt.figure(figsize=(10, 6))
    plt.plot(x_range, y_original, label="Исходная функция f(x)", color="blue")
    plt.plot(
        x_range, y_lagrange, label="Многочлен Лагранжа", color="red", linestyle="--"
    )
    plt.plot(
        x_range, y_newton, label="Многочлен Ньютона", color="green", linestyle="-."
    )
    plt.scatter(xi1, yi1, color="black", label="Узловые точки (для Ньютона)")
    plt.scatter(xi2, yi2, color="gray", label="Узловые точки (для Лагранжа)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("График многочленов Лагранжа и Ньютона и исходной функции")
    plt.legend()
    plt.grid(True)
    plt.show()
