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


def shooting_method(f, x0, y0, y_end, x_end, dy0_guess, h, tol=1e-6, max_iter=100):
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

    dy0 = dy0_guess

    for i in range(max_iter):
        x_values, y_values = integrate(f, x0, y0, dy0, x_end, h)
        y_final = y_values[-1]

        if abs(y_final - y_end) < tol:
            break

        dy0 -= (y_final - y_end) / 10

    return x_values, y_values, dy0


def reference(x):
    return np.sin(x) + 2 - np.sin(x) * np.log((1 + np.sin(x)) / (1 - np.sin(x)))


x0 = 0
y0 = 2
x_end = np.pi / 6
y_end = 2.5 - 0.5 * np.log(3)
dy0_guess = 0

h = (x_end - x0) / 50
x_values_h, y_values_h, dy0_final_h = shooting_method(
    differential_equations, x0, y0, y_end, x_end, dy0_guess, h
)

h2 = h / 2
x_values_h2, y_values_h2, dy0_final_h2 = shooting_method(
    differential_equations, x0, y0, y_end, x_end, dy0_guess, h2
)

x_analytic = np.linspace(0, x_end, 100)
y_analytic = reference(x_analytic)


def runge_romberg(y_h, y_h2, p):
    return (y_h2 - y_h) / (2**p - 1)


p = 4
y_h_at_end = np.interp(x_end, x_values_h, y_values_h)
y_h2_at_end = np.interp(x_end, x_values_h2, y_values_h2)
error = runge_romberg(y_h_at_end, y_h2_at_end, p)

print(f"Погрешность методом Рунге-Ромберга: {error}")

plt.plot(x_values_h, y_values_h, label="Численное решение (шаг h)")
plt.plot(x_analytic, y_analytic, label="Аналитическое решение", linestyle="dashed")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Решение краевой задачи методом стрельбы")
plt.legend()
plt.grid()
plt.show()
