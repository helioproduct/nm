import numpy as np
import matplotlib.pyplot as plt

# Дифференциальное уравнение второго порядка:
# y'' - tg(x) y' + 2y = 0


def differential_equations(x, y, dy):
    ddy = np.tan(x) * dy - 2 * y
    return ddy


def rk4_step(f, x, y, dy, h):
    k1 = h * dy
    l1 = h * f(x, y, dy)

    k2 = h * (dy + 0.5 * l1)
    l2 = h * f(x + 0.5 * h, y + 0.5 * k1, dy + 0.5 * l1)

    k3 = h * (dy + 0.5 * l2)
    l3 = h * f(x + 0.5 * h, y + 0.5 * k2, dy + 0.5 * l2)

    k4 = h * (dy + l3)
    l4 = h * f(x + h, y + k3, dy + l3)

    y_next = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    dy_next = dy + (l1 + 2 * l2 + 2 * l3 + l4) / 6

    return y_next, dy_next


# Решение задачи методом стрельбы
def shooting_method(f, x0, y0, y_end, x_end, dy0_guess, tol=1e-6, max_iter=100):
    def integrate(f, x0, y0, dy0, x_end, h):
        x_values = [x0]
        y_values = [y0]
        dy = dy0
        y = y0
        x = x0

        while x < x_end:
            y, dy = rk4_step(f, x, y, dy, h)
            x += h
            x_values.append(x)
            y_values.append(y)

        return np.array(x_values), np.array(y_values)

    h = (x_end - x0) / 100
    dy0 = dy0_guess

    for i in range(max_iter):
        x_values, y_values = integrate(f, x0, y0, dy0, x_end, h)
        y_final = y_values[-1]

        if abs(y_final - y_end) < tol:
            break

        dy0 -= (y_final - y_end) / 10

    return x_values, y_values, dy0


if __name__ == "__main__":

    x0 = 0
    y0 = 2
    x_end = np.pi / 6
    y_end = 2.5 - 0.5 * np.log(3)
    dy0_guess = 0  # начальное предположение для y'(0)

    # Решение задачи
    x_values, y_values, dy0_final = shooting_method(
        differential_equations, x0, y0, y_end, x_end, dy0_guess
    )

    # Аналитическое решение
    def reference(x):
        return np.sin(x) + 2 - np.sin(x) * np.log((1 + np.sin(x)) / (1 - np.sin(x)))

    x_analytic = np.linspace(0, x_end, 100)
    y_analytic = reference(x_analytic)

    # Построение графика
    plt.plot(x_values, y_values, label="Численное решение")
    plt.plot(x_analytic, y_analytic, label="Аналитическое решение", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Решение краевой задачи методом стрельбы")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"Найденное начальное значение для y'(0): {dy0_final}")
