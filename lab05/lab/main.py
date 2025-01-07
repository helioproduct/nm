# Используя  явную  и  неявную  конечно-разностные  схемы,  а  также  схему  Кранка  - 
# Николсона,  решить  начально-краевую  задачу  для  дифференциального  уравнения 
# параболического  типа.  Осуществить  реализацию  трех  вариантов  аппроксимации 
# граничных условий, содержащих производные: двухточечная аппроксимация с первым 
# порядком,  трехточечная  аппроксимация  со  вторым  порядком,  двухточечная 
# аппроксимация  со  вторым  порядком.  В  различные  моменты  времени  вычислить 
# погрешность  численного  решения  путем  сравнения  результатов  с  приведенным  в 
# задании аналитическим решением Uxt( , ) . Исследовать зависимость погрешности от 
# сеточных параметров , h


import matplotlib.pyplot as plt
from math import sin, cos, pi

from triag import solve_triag
from utils import max_abs_error

import numpy as np


# time at k-index
def time(k, tau):
    return k * tau

def xi(L, i, h):
    return L + i * h

# ненулевая правая часть  
def f(t, x):
    return cos(x) * (cos(t) + sin(t))

# u(x=0, t)
def u_left(t):
    return sin(t)  

# u(x=pi/2, t)
def ux_right(t):
    return -sin(t) 

# начальное условие
def u_start(x):
    return 0  # u(x, 0) = 0

def u_real(t, x):
    return sin(t) * cos(x)

def explicit(n, L, R, K, T, approximation_type=1):
    u = np.zeros((K + 1, n + 1))
    tau = T / K
    h = (R - L) / n
    sigma = tau / (h * h)
    
    if sigma > 0.5:
        raise Exception(f"σ is greater than 1/2: tau / h**2 > 0.5 (tau={tau}, h={h})")
    
    for i in range(n + 1):
        u[0][i] = u_start(xi(L, i, h))
    
    for k in range(1, K + 1):
        for i in range(1, n):
           
            # u[k+1][j] = σu[k][j-1] + (1-2σ)u[k][j] + σu[k][j+1]           
            # k = k - 1
            # u[k][j] = σu[k-1][j-1] + (1-2σ)u[k-1][j] + σu[k-1][j+1]      
            
            u[k][i] = (
                sigma * u[k - 1][i - 1] +
                (1 - 2 * sigma) * u[k - 1][i] +
                sigma * u[k - 1][i + 1] + 
                tau * f(time(k - 1, tau), xi(L, i, h))
            )
       

        # x=0
        u[k][0] = u_left(time(k, tau))

        # Граничные условия    
        # ux_right[k][j] = (u[k+1][j] - u[k][j]) / h
        # u[k+1][j] = ux_right[k][j] * h + u[k][j]
        # u[k][j] = ux_right[k-1][j] * h + u[k - 1][j]
        # Двухточечная аппроксимация первого порядка    
        if approximation_type == 1:

            # u(R, t) = u(R - h, t) + h * u'x(t)
            u[k][-1] = u[k][-2] + ux_right(time(k, tau)) * h
       
        # Трехточечная аппроксимация второго порядка
        elif approximation_type == 2:
            u[k][-1] = (1 / 3) * (2 * h * ux_right(time(k, tau)) + 4 * u[k][-2] - u[k][-3])
        
        # Двухточечная аппроксимация второго порядка
        elif approximation_type == 3:
            g = (h * h) / (2 * tau)
            u[k][-1] = (
                (1 / (1 + g)) *
                (u[k][-2] + ux_right(time(k, tau)) * h + g * u[k - 1][-1] + g * tau * f(time(k, tau), xi(L, n, h)))
            )

    return u


def implicit(n, L, R, K, T, approx=1):
    u = np.zeros((K + 1, n + 1))
    tau = T / K
    h = (R - L) / n
    sigma = tau / (h * h)

    for i in range(n + 1):
        u[0][i] = u_start(xi(L, i, h))

    for k in range(K):
        a = np.zeros(n + 1)
        b = np.zeros(n + 1)
        c = np.zeros(n + 1)
        d = np.zeros(n + 1)

        for i in range(1, n):
            a[i] = -sigma
            b[i] = 1 + 2 * sigma
            c[i] = -sigma
            d[i] = u[k][i] + tau * f(time(k + 1, tau), xi(L, i, h))

        # Граничные условия
        if approx == 1:
            # x=0
            a[0] = 0
            b[0] = 1
            c[0] = 0
            d[0] = u_left(time(k + 1, tau))

            # x=R (дддвухточечная аппроксимация первого порядка)
            a[-1] = -1
            b[-1] = 1
            c[-1] = 0
            d[-1] = ux_right(time(k + 1, tau)) * h

        elif approx == 2:
            # x=0 
            a[0] = 0
            b[0] = 1
            c[0] = 0
            d[0] = u_left(time(k + 1, tau))

            # x=R трехточечная аппроксимация второго порядка)
            a[-1] = -1 / (2 * h)  # Коэффициент при u_{n-1}^{k+1}
            b[-1] = 0             # Коэффициент при u_n^{k+1} (нулевой, переносим в правую часть)
            c[-1] = 1 / (2 * h)   # Коэффициент при u_{n-2}^{k+1}
            d[-1] = ux_right(time(k + 1, tau))

        elif approx == 3:
            # x=0 (условие Дирихле)
            a[0] = 0
            b[0] = 1
            c[0] = 0
            d[0] = u_left(time(k + 1, tau))

            # x=R (условие Неймана, двухточечная аппроксимация второго порядка с учетом времени)
            g = (h * h) / (2 * tau)
            a[-1] = 1                 # Коэффициент при u_{n-1}^{k+1}
            b[-1] = -1 - g            # Коэффициент при u_n^{k+1}
            c[-1] = 0                 # Нет зависимости от u_{n+1}^{k+1}
            d[-1] = (
                -g * u[k][-1]
                - g * tau * f(time(k + 1, tau), xi(L, n, h))
                - h * ux_right(time(k + 1, tau))
            )

        else:
            raise ValueError("Invalid approximation type. Choose approx=1, 2, or 3.")

        solve = solve_triag(a, b, c, d)
        u[k + 1] = solve

    return u


def CN_method(n, L, R, K, T, approx=1, theta=0.5):
    u = np.zeros((K + 1, n + 1))
    tau = T / K
    h = (R - L) / n
    sigma = tau / (h * h)

    # time=0
    for i in range(n + 1):
        u[0][i] = u_start(xi(L, i, h))
    
    for k in range(K):
        a = np.zeros(n + 1)
        b = np.zeros(n + 1)
        c = np.zeros(n + 1)
        d = np.zeros(n + 1)
        
        f_k = np.array([f(time(k, tau), xi(L, i, h)) for i in range(n + 1)])
        f_k1 = np.array([f(time(k + 1, tau), xi(L, i, h)) for i in range(n + 1)])

        for i in range(1, n):
            a[i] = - theta * sigma
            b[i] = 1 + 2 * theta * sigma
            c[i] = - theta * sigma
            d[i] = (
                (1 - (1 - theta) * 2 * sigma) * u[k][i] +
                (1 - theta) * sigma * (u[k][i - 1] + u[k][i + 1]) +
                tau * (theta * f_k1[i] + (1 - theta) * f_k[i])
            )
        
        # Boundary conditions ??
        if approx == 1:
            # x=0
            a[0] = 0
            b[0] = 1
            c[0] = 0
            d[0] = u_left(time(k + 1, tau))
            
            # x=R, first-order approximation
            # (-u_{n-1}^{k+1} + u_n^{k+1}) = ux_right * h
            a[n] = -1
            b[n] = 1
            c[n] = 0
            d[n] = ux_right(time(k + 1, tau)) * h

        elif approx == 2:
            # x=0 (условие Дирихле)
            a[0] = 0
            b[0] = 1
            c[0] = 0
            d[0] = u_left(time(k + 1, tau))

            # x=R (условие Неймана, трехточечная аппроксимация второго порядка)
            a[-1] = -1 / (2 * h)  # Коэффициент при u_{n-1}^{k+1}
            b[-1] = 0             # Коэффициент при u_n^{k+1} (нулевой, переносим в правую часть)
            c[-1] = 1 / (2 * h)   # Коэффициент при u_{n-2}^{k+1}
            d[-1] = ux_right(time(k + 1, tau))
        
        elif approx == 3:
            # x=0
            a[0] = 0
            b[0] = 1
            c[0] = 0
            d[0] = u_left(time(k + 1, tau))
            
            # x=R, Neumann boundary condition, second-order approximation (two-point)
            g = (h * h) / (2 * tau)
            a[n] = 1
            b[n] = -1 - g
            c[n] = 0
            d[n] = (
                -g * u[k][n] -
                g * tau * (theta * f_k1[n] + (1 - theta) * f_k[n]) -
                h * ux_right(time(k + 1, tau))
            )
        
        solve = solve_triag(a, b, c, d)
        u[k + 1] = solve

    return u


import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, pi

# Функции и исходные данные остаются без изменений
# ...

if __name__ == "__main__":

    L = 0
    R = pi / 2
    T = 10
    n = 35
    K = 10000

    tau = T / K
    h = (R - L) / n

    print("h = ", h)
    print("sigma = ", tau / (h * h))

    # Вычисляем численные решения
    explicit_1 = explicit(n, L, R, K, T, approximation_type=1)
    explicit_2 = explicit(n, L, R, K, T, approximation_type=2)
    explicit_3 = explicit(n, L, R, K, T, approximation_type=3)

    implicit_1 = implicit(n, L, R, K, T, approx=1)
    implicit_2 = implicit(n, L, R, K, T, approx=2)
    implicit_3 = implicit(n, L, R, K, T, approx=3)

    theta = 0.5
    cn_1 = CN_method(n, L, R, K, T, approx=1, theta=theta)
    cn_2 = CN_method(n, L, R, K, T, approx=2, theta=theta)
    cn_3 = CN_method(n, L, R, K, T, approx=3, theta=theta)

    # Вычисление ошибок
    time_range = np.linspace(0, T, K + 1)
    
    def calculate_mae(numeric_solution):
        mae = np.zeros(K + 1)
        for k in range(K + 1):
            analytical = np.array([u_real(time(k, tau), xi(L, i, h)) for i in range(n + 1)])
            mae[k] = np.mean(np.abs(numeric_solution[k] - analytical))
        return mae

    mae_explicit_1 = calculate_mae(explicit_1)
    mae_explicit_2 = calculate_mae(explicit_2)
    mae_explicit_3 = calculate_mae(explicit_3)

    mae_implicit_1 = calculate_mae(implicit_1)
    mae_implicit_2 = calculate_mae(implicit_2)
    mae_implicit_3 = calculate_mae(implicit_3)

    mae_cn_1 = calculate_mae(cn_1)
    mae_cn_2 = calculate_mae(cn_2)
    mae_cn_3 = calculate_mae(cn_3)

    # Построение графиков погрешностей
    def plot_single_error(title, time_range, mae, label):
        plt.figure(figsize=(10, 5))
        plt.plot(time_range, mae, label=label)
        plt.xlabel('t')
        plt.ylabel('Средняя абсолютная ошибка')
        plt.title(title)
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.legend()
        plt.grid()
        plt.show()

    # Построение графиков для каждой погрешности отдельно
    plot_single_error("Явная конечно-разностная схема: 2т-ная 1-го порядка", time_range, mae_explicit_1, "2т-ная 1-го порядка")
    plot_single_error("Явная конечно-разностная схема: 3т-ная 2-го порядка", time_range, mae_explicit_2, "3т-ная 2-го порядка")
    plot_single_error("Явная конечно-разностная схема: 2т-ная 2-го порядка", time_range, mae_explicit_3, "2т-ная 2-го порядка")

    plot_single_error("Неявная конечно-разностная схема: 2т-ная 1-го порядка", time_range, mae_implicit_1, "2т-ная 1-го порядка")
    plot_single_error("Неявная конечно-разностная схема: 3т-ная 2-го порядка", time_range, mae_implicit_2, "3т-ная 2-го порядка")
    plot_single_error("Неявная конечно-разностная схема: 2т-ная 2-го порядка", time_range, mae_implicit_3, "2т-ная 2-го порядка")

    plot_single_error("Схема Кранка-Николсона: 2т-ная 1-го порядка", time_range, mae_cn_1, "2т-ная 1-го порядка")
    plot_single_error("Схема Кранка-Николсона: 3т-ная 2-го порядка", time_range, mae_cn_2, "3т-ная 2-го порядка")
    plot_single_error("Схема Кранка-Николсона: 2т-ная 2-го порядка", time_range, mae_cn_3, "2т-ная 2-го порядка")

    # Построение графиков численных решений отдельно
    time_index = 5000
    x = np.linspace(L, R, n + 1)
    current_time = time(time_index, tau)
    u_analytic_values = [u_real(current_time, xi(L, i, h)) for i in range(n + 1)]

    def plot_single_solution(title, x, analytical, numeric_solution, label):
        plt.figure(figsize=(10, 5))
        plt.plot(x, analytical, label='Аналитическое решение')
        plt.plot(x, numeric_solution, label=label)
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.title(title)
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.legend()
        plt.grid()
        plt.show()

    plot_single_solution("Явная конечно-разностная схема: 2т-ная 1-го порядка", x, u_analytic_values, explicit_1[time_index], "2т-ная 1-го порядка")
    plot_single_solution("Явная конечно-разностная схема: 3т-ная 2-го порядка", x, u_analytic_values, explicit_2[time_index], "3т-ная 2-го порядка")
    plot_single_solution("Явная конечно-разностная схема: 2т-ная 2-го порядка", x, u_analytic_values, explicit_3[time_index], "2т-ная 2-го порядка")

    plot_single_solution("Неявная конечно-разностная схема: 2т-ная 1-го порядка", x, u_analytic_values, implicit_1[time_index], "2т-ная 1-го порядка")
    plot_single_solution("Неявная конечно-разностная схема: 3т-ная 2-го порядка", x, u_analytic_values, implicit_2[time_index], "3т-ная 2-го порядка")
    plot_single_solution("Неявная конечно-разностная схема: 2т-ная 2-го порядка", x, u_analytic_values, implicit_3[time_index], "2т-ная 2-го порядка")

    plot_single_solution("Схема Кранка-Николсона: 2т-ная 1-го порядка", x, u_analytic_values, cn_1[time_index], "2т-ная 1-го порядка")
    plot_single_solution("Схема Кранка-Николсона: 3т-ная 2-го порядка", x, u_analytic_values, cn_2[time_index], "3т-ная 2-го порядка")
    plot_single_solution("Схема Кранка-Николсона: 2т-ная 2-го порядка", x, u_analytic_values, cn_3[time_index], "2т-ная 2-го порядка")
