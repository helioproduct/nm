import matplotlib.pyplot as plt
from math import sin, cos, pi
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
    return sin(t)  # Дирихле на x=0

# u(x=pi/2, t)
def ux_right(t):
    return -sin(t)  # Нейман на x=pi/2

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
       

        # Дирихле на x=0
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


def solve_triag(a, b, c, d):
    n = len(d)
    P = np.zeros(n)
    Q = np.zeros(n)
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, n):
        denom = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom
    x = np.zeros(n)
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x


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
            a[i] = sigma
            b[i] = -1 - 2 * sigma
            c[i] = sigma
            d[i] = -u[k][i] - tau * f(time(k + 1, tau), xi(L, i, h))
        
        # Граничные условия
        if approx == 1:
            # x=0
            a[0] = 0
            b[0] = 1
            c[0] = 0
            d[0] = u_left(time(k + 1, tau))
            # x=R
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
            # Трехточечная аппроксимация второго порядка для Неймана на x=R
            a[-1] = -1 / (2 * h)
            b[-1] = 0
            c[-1] = 1 / (2 * h)
            d[-1] = ux_right(time(k + 1, tau))
       
        elif approx == 3:
            # Дирихле на x=0
            a[0] = 0
            b[0] = 1
            c[0] = 0
            d[0] = u_left(time(k + 1, tau))
            # Двухточечная аппроксимация второго порядка для Неймана на x=R
            g = (h * h) / (2 * tau)
            a[-1] = 1
            b[-1] = -1 - g
            c[-1] = 0
            d[-1] = (
                -g * u[k][-1] -
                g * tau * f(time(k + 1, tau), xi(L, n, h)) -
                h * ux_right(time(k + 1, tau))
            )
        solve = solve_triag(a, b, c, d)
        u[k + 1] = solve
    
    return u


if __name__ == "__main__":

    # Параметры задачи
    L = 0
    R = pi / 2
    T = 10
    n = 35
    K = 10000

    tau = T / K
    h = (R - L) / n

    print("h = ", h)
    print("sigma = ", tau / (h * h))


    # Явная конечно-разностная схема
    explicit_1 = explicit(n, L, R, K, T, approximation_type=1)
    explicit_2 = explicit(n, L, R, K, T, approximation_type=2)
    explicit_3 = explicit(n, L, R, K, T, approximation_type=3)

    # Неявная конечно-разностная схема
    implicit_1 = implicit(n, L, R, K, T, approx=1)
    implicit_2 = implicit(n, L, R, K, T, approx=2)
    implicit_3 = implicit(n, L, R, K, T, approx=3)

    # u_crank_nicolson_1 = crank_nicolson(n, L, R, K, T, approx=1)
    # u_crank_nicolson_2 = crank_nicolson(n, L, R, K, T, approx=2)
    # u_crank_nicolson_3 = crank_nicolson(n, L, R, K, T, approx=3)

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
    

        # mae_u_crank_nicolson_1[k] = np.mean(np.abs(u_crank_nicolson_1[k] - analytical))
        # mae_u_crank_nicolson_2[k] = np.mean(np.abs(u_crank_nicolson_2[k] - analytical))
        # mae_u_crank_nicolson_3[k] = np.mean(np.abs(u_crank_nicolson_3[k] - analytical))

    # Построение графика погрешности
    plt.figure(figsize=(12, 8))
    plt.plot(time_range, mae_u_explicit_1, label="Явн. 2т. 1")
    plt.plot(time_range, mae_u_explicit_2, label="Явн. 3т. 2")
    plt.plot(time_range, mae_u_explicit_3, label="Явн. 2т. 2")
    plt.plot(time_range, mae_u_implicit_1, label="Неявн. 2т. 1")
    plt.plot(time_range, mae_u_implicit_2, label="Неявн. 3т. 2")
    plt.plot(time_range, mae_u_implicit_3, label="Неявн. 2т. 2")
    plt.plot(time_range, mae_u_crank_nicolson_1, label="К-Н. 2т. 1")
    plt.plot(time_range, mae_u_crank_nicolson_2, label="К-Н. 3т. 2")
    plt.plot(time_range, mae_u_crank_nicolson_3, label="К-Н. 2т. 2")
    plt.xlabel('t')
    plt.ylabel('MAE(u(x, t), U(x, t))')
  
    plt.title('Зависимость погрешности от параметра t')
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.legend()
    plt.grid()
    plt.show()



    time_index = 5
    x = np.linspace(L, R, n + 1)

    plt.figure(figsize=(12, 8))
    current_time = time(time_index, tau)
    u_analytic_values = [u_real(current_time, xi(L, i, h)) for i in range(n + 1)]
    
    
    u_explicit_1_values = [explicit_1[time_index][i] for i in range(n + 1)]
    u_explicit_2_values = [explicit_2[time_index][i] for i in range(n + 1)]
    u_explicit_3_values = [explicit_3[time_index][i] for i in range(n + 1)]
    u_implicit_1_values = [implicit_1[time_index][i] for i in range(n + 1)]
    u_implicit_2_values = [implicit_2[time_index][i] for i in range(n + 1)]
    u_implicit_3_values = [implicit_3[time_index][i] for i in range(n + 1)]
    # u_crank_nicolson_1_values = [u_crank_nicolson_1[time_index][i] for i in range(n + 1)]
    # u_crank_nicolson_2_values = [u_crank_nicolson_2[time_index][i] for i in range(n + 1)]
    # u_crank_nicolson_3_values = [u_crank_nicolson_3[time_index][i] for i in range(n + 1)]


    plt.plot(x, u_analytic_values, label=f'Аналитическое, t = {current_time}')
    plt.plot(x, u_explicit_1_values, label=f'Явн. 2т. 1, t = {current_time}')
    plt.plot(x, u_explicit_2_values, label=f'Явн. 3т. 2, t = {current_time}')
    plt.plot(x, u_explicit_3_values, label=f'Явн. 2т. 2, t = {current_time}')

    plt.plot(x, u_implicit_1_values, label=f'неявн. 1т. 2, t = {current_time}')
    plt.plot(x, u_implicit_2_values, label=f'неявн. 2т. 2, t = {current_time}')
    plt.plot(x, u_implicit_3_values, label=f'неявн. 2т. 3, t = {current_time}')



    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title('Сравнение аналитического и численных решений')
    plt.xticks(ticks=[0, pi/4, pi/2], labels=['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$'])
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.legend()
    plt.grid()
    plt.show()
