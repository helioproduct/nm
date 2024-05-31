import numpy as np
import matplotlib.pyplot as plt


def reference(x):
    return x**2 + x + 1


def f(x, y, z):
    return (2 * x * z - 2 * y) / (x**2 - 1)


def runge_kutta_4(x0, y0, z0, h, steps):
    x_values = [x0]
    y_values = [y0]
    z_values = [z0]

    x = x0
    y = y0
    z = z0

    for _ in range(steps):
        k1_y = h * z
        k1_z = h * f(x, y, z)

        k2_y = h * (z + 0.5 * k1_z)
        k2_z = h * f(x + 0.5 * h, y + 0.5 * k1_y, z + 0.5 * k1_z)

        k3_y = h * (z + 0.5 * k2_z)
        k3_z = h * f(x + 0.5 * h, y + 0.5 * k2_y, z + 0.5 * k2_z)

        k4_y = h * (z + k3_z)
        k4_z = h * f(x + h, y + k3_y, z + k3_z)

        y += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        z += (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6
        x += h

        x_values.append(x)
        y_values.append(y)
        z_values.append(z)

    return x_values, y_values, z_values


def runge_kutta_4_full(x0, y0, z0, h, x_end):
    n = int((x_end - x0) / h)
    x_values = np.linspace(x0, x_end, n + 1)
    y_values = np.zeros(n + 1)
    z_values = np.zeros(n + 1)

    y_values[0] = y0
    z_values[0] = z0

    x = x0
    y = y0
    z = z0

    for i in range(n):
        k1_y = h * z
        k1_z = h * f(x, y, z)

        k2_y = h * (z + 0.5 * k1_z)
        k2_z = h * f(x + 0.5 * h, y + 0.5 * k1_y, z + 0.5 * k1_z)

        k3_y = h * (z + 0.5 * k2_z)
        k3_z = h * f(x + 0.5 * h, y + 0.5 * k2_y, z + 0.5 * k2_z)

        k4_y = h * (z + k3_z)
        k4_z = h * f(x + h, y + k3_y, z + k3_z)

        y += (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        z += (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6
        x += h

        y_values[i + 1] = y
        z_values[i + 1] = z

    return x_values, y_values


def euler_method(x0, y0, z0, h, x_end):
    n = int((x_end - x0) / h)
    x_values = np.linspace(x0, x_end, n + 1)
    y_values = np.zeros(n + 1)
    z_values = np.zeros(n + 1)

    y_values[0] = y0
    z_values[0] = z0

    for i in range(n):
        y_values[i + 1] = y_values[i] + h * z_values[i]
        z_values[i + 1] = z_values[i] + h * f(x_values[i], y_values[i], z_values[i])

    return x_values, y_values


def adams_4(x0, y0, z0, h, x_end):
    _, y_rk, z_rk = runge_kutta_4(x0, y0, z0, h, 3)

    n = int((x_end - x0) / h)
    x_values = np.linspace(x0, x_end, n + 1)
    y_values = np.zeros(n + 1)
    z_values = np.zeros(n + 1)

    y_values[:4] = y_rk
    z_values[:4] = z_rk

    for i in range(3, n):
        f_k = f(x_values[i], y_values[i], z_values[i])
        f_k1 = f(x_values[i - 1], y_values[i - 1], z_values[i - 1])
        f_k2 = f(x_values[i - 2], y_values[i - 2], z_values[i - 2])
        f_k3 = f(x_values[i - 3], y_values[i - 3], z_values[i - 3])

        y_values[i + 1] = y_values[i] + h * z_values[i]
        z_values[i + 1] = z_values[i] + (h / 24) * (
            55 * f_k - 59 * f_k1 + 37 * f_k2 - 9 * f_k3
        )

    return x_values, y_values


x0 = 2
y0 = 7
z0 = 5
h = 0.1
x_end = 3

x_values_euler, y_values_euler = euler_method(x0, y0, z0, h, x_end)
x_values_rk, y_values_rk = runge_kutta_4_full(x0, y0, z0, h, x_end)
x_values_adams_h, y_values_adams_h = adams_4(x0, y0, z0, h, x_end)

exact_y_values = reference(x_values_euler)

errors_euler = np.abs(y_values_euler - exact_y_values)
errors_rk = np.abs(y_values_rk - exact_y_values)
errors_adams = np.abs(y_values_adams_h - exact_y_values)

print("Euler Error: ", np.max(errors_euler))
print("Runge-Kutta  Error: ", np.max(errors_rk))
print("Adams 4th  Error: ", np.max(errors_adams))

plt.figure(figsize=(10, 6))
plt.plot(x_values_euler, y_values_euler, "b", label="Euler Method Solution")
plt.plot(x_values_rk, y_values_rk, "c", label="Runge-Kutta 4th Order Solution")
plt.plot(
    x_values_adams_h, y_values_adams_h, "g", label="Adams-Bashforth 4th Order Solution"
)
plt.plot(x_values_euler, exact_y_values, "r", label="Exact Solution")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparison of Numerical Methods and Exact Solution")
plt.legend()
plt.grid(True)
plt.show()
