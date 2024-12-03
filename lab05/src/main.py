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

            # x=R (условие Неймана, двухточечная аппроксимация первого порядка)
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
        
        # Boundary conditions
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


    explicit_1 = explicit(n, L, R, K, T, approximation_type=1)
    explicit_2 = explicit(n, L, R, K, T, approximation_type=2)
    explicit_3 = explicit(n, L, R, K, T, approximation_type=3)

    implicit_1 = implicit(n, L, R, K, T, approx=1)
    implicit_2 = implicit(n, L, R, K, T, approx=2)
    implicit_3 = implicit(n, L, R, K, T, approx=3)

    theta=0.5
    cn_1 = CN_method(n, L, R, K, T, approx=1, theta=theta)
    cn_2 = CN_method(n, L, R, K, T, approx=2, theta=theta)
    cn_3 = CN_method(n, L, R, K, T, approx=3, theta=theta)

    # вычисление ошибок
    tau = T / K
    h = (R - L) / n
    time_range = np.linspace(0, T, K + 1)
    mae_u_explicit_1 = np.zeros(K + 1)
    mae_u_explicit_2 = np.zeros(K + 1)
   
    mae_u_explicit_3 = np.zeros(K + 1)
    mae_u_implicit_1 = np.zeros(K + 1)
    mae_u_implicit_2 = np.zeros(K + 1)
    mae_u_implicit_3 = np.zeros(K + 1)
   
    mae_u_crank_nicolson_1 = np.zeros(K + 1)
    mae_u_crank_nicolson_2 = np.zeros(K + 1)
    mae_u_crank_nicolson_3 = np.zeros(K + 1)

    for k in range(K + 1):
        analytical = np.array([u_real(time(k, tau), xi(L, i, h)) for i in range(n + 1)])
        
        mae_u_explicit_1[k] = np.mean(np.abs(explicit_1[k] - analytical))
        mae_u_explicit_2[k] = np.mean(np.abs(explicit_2[k] - analytical))
        mae_u_explicit_3[k] = np.mean(np.abs(explicit_3[k] - analytical))

        mae_u_implicit_1[k] = np.mean(np.abs(implicit_1[k] - analytical))
        mae_u_implicit_2[k] = np.mean(np.abs(implicit_2[k] - analytical))
        mae_u_implicit_3[k] = np.mean(np.abs(implicit_3[k] - analytical))
    

        mae_u_crank_nicolson_1[k] = np.mean(np.abs(cn_1[k] - analytical))
        mae_u_crank_nicolson_2[k] = np.mean(np.abs(cn_2[k] - analytical))
        mae_u_crank_nicolson_3[k] = np.mean(np.abs(cn_3[k] - analytical))



    # Построение графика погрешности
    plt.figure(figsize=(10, 5))
    plt.plot(time_range, mae_u_explicit_1, label="Явная. 2т-ная. схема 1-го порядка")
    plt.plot(time_range, mae_u_explicit_2, label="Явная 3т-ная схема 2-го порядка")
    plt.plot(time_range, mae_u_explicit_3, label="Явная. 2т-ная схема 2-го порядка")
    
    plt.plot(time_range, mae_u_implicit_1, label="Неявная. 2т-ная. схема 1-го порядка")
    plt.plot(time_range, mae_u_implicit_2, label="Неявн. 3т-ная схема 2-го порядка")
    plt.plot(time_range, mae_u_implicit_3, label="Неявн. 2т-ная схема 2-го порядка")
    
    
    plt.plot(time_range, mae_u_crank_nicolson_1, label="KN. 2т-ная. схема 1-го порядка")
    plt.plot(time_range, mae_u_crank_nicolson_2, label="KN. 3т-ная схема 2-го порядка")
    plt.plot(time_range, mae_u_crank_nicolson_3, label="KN. 2т-ная схема 2-го порядка")

    plt.xlabel('t')
    plt.ylabel('mae(u(x, t), U(x, t))')
  
    plt.title('Прогрешности t')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.legend()
    plt.grid()
    plt.show()



    time_index = 5000
    x = np.linspace(L, R, n + 1)

    plt.figure(figsize=(10, 5))
    current_time = time(time_index, tau)
    u_analytic_values = [u_real(current_time, xi(L, i, h)) for i in range(n + 1)]
    
    
    u_explicit_1_values = [explicit_1[time_index][i] for i in range(n + 1)]
    u_explicit_2_values = [explicit_2[time_index][i] for i in range(n + 1)]
    u_explicit_3_values = [explicit_3[time_index][i] for i in range(n + 1)]
    
    u_implicit_1_values = [implicit_1[time_index][i] for i in range(n + 1)]
    u_implicit_2_values = [implicit_2[time_index][i] for i in range(n + 1)]
    u_implicit_3_values = [implicit_3[time_index][i] for i in range(n + 1)]
    
    u_crank_nicolson_1_values = [cn_1[time_index][i] for i in range(n + 1)]
    u_crank_nicolson_2_values = [cn_2[time_index][i] for i in range(n + 1)]
    u_crank_nicolson_3_values = [cn_3[time_index][i] for i in range(n + 1)]


    print("max abs error explicit 1", max_abs_error(u_explicit_1_values, u_analytic_values))
    print("max abs error explicit 2", max_abs_error(u_explicit_2_values, u_analytic_values))
    print("max abs error explicit 3", max_abs_error(u_explicit_3_values, u_analytic_values))


    print("max abs error implicit 1", max_abs_error(u_implicit_1_values, u_analytic_values))
    print("max abs error implicit 2", max_abs_error(u_implicit_2_values, u_analytic_values))
    print("max abs error implicit 3", max_abs_error(u_implicit_3_values, u_analytic_values))


    print("max abs error CN1", max_abs_error(u_crank_nicolson_1_values, u_analytic_values))
    print("max abs error CN2", max_abs_error(u_crank_nicolson_2_values, u_analytic_values))
    print("max abs error CN3", max_abs_error(u_crank_nicolson_3_values, u_analytic_values))




    plt.plot(x, u_analytic_values, label=f'Аналитическое, t = {current_time}')

    plt.plot(x, u_explicit_1_values, label=f'Явн. 2т. 1, t = {current_time}')
    plt.plot(x, u_explicit_2_values, label=f'Явн. 3т. 2, t = {current_time}')
    plt.plot(x, u_explicit_3_values, label=f'Явн. 2т. 2, t = {current_time}')

    plt.plot(x, u_implicit_1_values, label=f'неявн. 1т. 2, t = {current_time}')
    plt.plot(x, u_implicit_2_values, label=f'неявн. 2т. 2, t = {current_time}')
    plt.plot(x, u_implicit_3_values, label=f'неявн. 2т. 3, t = {current_time}')

    plt.plot(x, u_crank_nicolson_1_values, label="CN. 2т-ная. схема 1-го порядка")
    plt.plot(x, u_crank_nicolson_2_values, label="CN. 3т-ная схема 2-го порядка")
    plt.plot(x, u_crank_nicolson_3_values, label="CN. 2т-ная схема 2-го порядка")


    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Сравнение аналитического и численных решений')
    plt.xticks(ticks=[0, pi/4, pi/2], labels=['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.legend()
    plt.grid()
    plt.show()


