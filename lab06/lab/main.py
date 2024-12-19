import matplotlib.pyplot as plt
from math import exp, cos, pi
from numpy import linspace
from math import pi
from triag import solve_triag

class StabilityException(Exception):
    def __init__(self, tau, h):
        self.tau = tau
        self.h = h
        super().__init__(self._error_message())

    def _error_message(self):
        return f"Stability condition violated: tau / (h * h) > 0.5 (tau={self.tau}, h={self.h})"


def print_grid(u, K, n):
    for i in range(K):
        string = ""
        for j in range(n):
            string += f"{u[i][j]:0.5f} "
        print(string)

def tk(k, tau):
    return k * tau

def xi(L, i, h):
    return L + i * h

# Граничные условия
def Phi_L(t):
    return cos(2 * t)  # u(0, t) = cos(2t)

def Phi_R(t):
    return 0.0         # u(pi/2, t) = 0

# Начальные условия
def Psi_1(x):
    return exp(-x) * cos(x)  # u(x, 0) = e^{-x}cos(x)

def Psi_2(x):
    return 0.0               # u_t(x, 0) = 0

# Аналитическое решение
def analytic(t, x):
    return exp(-x) * cos(x) * cos(2 * t)

def dx(f, x):
    eps = 1e-6
    return (f(x + eps) - f(x)) / eps

def ddx(f, x):
    eps = 1e-6
    return (dx(f, x + eps) - dx(f, x)) / eps

def explicit(n, L, R, K, T, approx=1):
    """
    Явная схема.
    Решаем уравнение:
    u_tt = u_xx + 2u_x - 2u

    Дискретизация по x:
    u_xx ≈ (u[i+1] - 2u[i] + u[i-1]) / h^2
    u_x  ≈ (u[i+1] - u[i-1]) / (2h)

    Подставляя:
    u_tt = (u[i+1]-2u[i]+u[i-1])/h^2 + (u[i+1]-u[i-1])/h - 2u[i]

    Итоговая формула на каждом шаге по времени (k):
    (u(k+1,i) - 2u(k,i) + u(k-1,i)) / tau^2 = u_xx + 2u_x - 2u

    Перегруппируем для явной схемы:
    u(k+1,i) = 2u(k,i) - u(k-1,i) + tau^2 * [ (u(k,i+1)-2u(k,i)+u(k,i-1))/h^2 
                                             + (u(k,i+1)-u(k,i-1))/h - 2u(k,i) ]
    """
    u = [[0.0 for _ in range(n + 1)] for _ in range(K + 1)]
    tau = T / K
    h = (R - L) / n
    sigma = (tau * tau) / (h * h)
    if sigma > 0.5:
        raise StabilityException(tau, h)

    # Инициализация по начальному условию
    for i in range(n + 1):
        u[0][i] = Psi_1(xi(L, i, h))
        if approx == 1:
            # u_t(x,0)=Psi_2(x)=0
            u[1][i] = u[0][i]  # так как Psi_2(x)=0, u(k+1)=u(k)+tau*0=u(k)
        elif approx == 2:
            # Второе приближение по времени
            # u_tt(x,0) из PDE можем вычислить численно:
            # u_tt(x,0) = u_xx(x,0) + 2u_x(x,0) - 2u(x,0)
            xx = xi(L, i, h)
            utt0 = ddx(Psi_1, xx) + 2*dx(Psi_1, xx) - 2*Psi_1(xx)
            u[1][i] = u[0][i] + tau * Psi_2(xx) + (tau*tau/2)*utt0
            # Psi_2=0 => u[1][i]=u[0][i]+(tau^2/2)*utt0

    # Граничные условия по времени
    for k in range(K + 1):
        u[k][0] = Phi_L(tk(k, tau))
        u[k][-1] = Phi_R(tk(k, tau))

    # Основной цикл по времени
    for k in range(1, K):
        for i in range(1, n):
            uxx = (u[k][i+1] - 2*u[k][i] + u[k][i-1]) / (h*h)
            ux = (u[k][i+1] - u[k][i-1]) / (2*h)
            # u_tt = u_xx + 2u_x - 2u
            utt = uxx + 2*ux - 2*u[k][i]
            u[k+1][i] = 2*u[k][i] - u[k-1][i] + tau*tau*utt

    return u


def implicit(n, L, R, K, T, approx=1):
    # Сетка по времени и пространству
    tau = T / K
    h = (R - L) / n
    u = [[0.0 for _ in range(n + 1)] for _ in range(K + 1)]

    # Начальное условие
    for i in range(n + 1):
        x_i = xi(L, i, h)
        u[0][i] = Psi_1(x_i)
    if approx == 1:
        for i in range(n + 1):
            x_i = xi(L, i, h)
            # u(1,i) = u(0,i) + tau*u_t(x,0) = u(0,i), т.к. Psi_2=0
            u[1][i] = u[0][i]
    elif approx == 2:
        for i in range(n + 1):
            x_i = xi(L, i, h)
            utt0 = ddx(Psi_1, x_i) + 2*dx(Psi_1, x_i) - 2*Psi_1(x_i)
            u[1][i] = u[0][i] + (tau*tau/2)*utt0

    # Применяем ГУ на всем временном слое
    for k in range(K + 1):
        u[k][0] = Phi_L(tk(k, tau))
        u[k][n] = Phi_R(tk(k, tau))

    a = [0.0 for _ in range(n + 1)]
    b = [0.0 for _ in range(n + 1)]
    c = [0.0 for _ in range(n + 1)]
    d = [0.0 for _ in range(n + 1)]

    # Формируем СЛАУ для каждого шага по времени
    for k in range(1, K):
        # Внутренние узлы: i = 1..n-1
        for i in range(1, n):
            # Коэффициенты для неявной схемы:
            # u_tt = u_xx + 2u_x - 2u
            # (u^{k+1}_i - 2u^k_i + u^{k-1}_i)/tau^2 = u_xx^{k+1,i} + 2u_x^{k+1,i} - 2u^{k+1}_i
            #
            # Приводим к виду:
            # a[i]*u^{k+1}_{i-1} + b[i]*u^{k+1}_i + c[i]*u^{k+1}_{i+1} = d[i]
            #
            # Вывод (см. предыдущие объяснения):
            a[i] = (-1/(h*h) + 1/h)
            b[i] = (1/(tau*tau) + 2/(h*h) + 2)
            c[i] = (-1/(h*h) - 1/h)
            d[i] = (2*u[k][i] - u[k-1][i])/(tau*tau)

        # Граничные условия
        # i=0
        a[0] = 0
        b[0] = 1
        c[0] = 0
        d[0] = Phi_L(tk(k, tau))

        # i=n
        a[n] = 0
        b[n] = 1
        c[n] = 0
        d[n] = Phi_R(tk(k, tau))

        # Решаем СЛАУ
        solve = solve_triag(a, b, c, d)
        for i in range(n + 1):
            u[k+1][i] = solve[i]

    return u


# Зададим параметры
L = 0
R = pi/2
T = 10
n = 100
K = 50000

u_explicit_1 = explicit(n, L, R, K, T, approx=1)
u_explicit_2 = explicit(n, L, R, K, T, approx=2)
u_implicit_1 = implicit(n, L, R, K, T, approx=1)
u_implicit_2 = implicit(n, L, R, K, T, approx=2)

tau = T / K
h = (R - L) / n
time = linspace(0, T, K + 1)

mae_u_explicit_1 = [0.0 for _ in range(K + 1)]
mae_u_explicit_2 = [0.0 for _ in range(K + 1)]
mae_u_implicit_1 = [0.0 for _ in range(K + 1)]
mae_u_implicit_2 = [0.0 for _ in range(K + 1)]
#
for k in range(K + 1):
    for i in range(n + 1):
        exact_val = analytic(tk(k, tau), xi(L, i, h))
        mae_u_explicit_1[k] += abs(u_explicit_1[k][i] - exact_val)
        mae_u_explicit_2[k] += abs(u_explicit_2[k][i] - exact_val)
        mae_u_implicit_1[k] += abs(u_implicit_1[k][i] - exact_val)
        mae_u_implicit_2[k] += abs(u_implicit_2[k][i] - exact_val)
    
    mae_u_explicit_1[k] /= n
    mae_u_explicit_2[k] /= n
    mae_u_implicit_1[k] /= n
    mae_u_implicit_2[k] /= n

plt.figure(figsize=(12, 8))
plt.plot(time, mae_u_explicit_1, label="Явная 1")
plt.plot(time, mae_u_explicit_2, label="Явная 2")
plt.plot(time, mae_u_implicit_1, label="Неявная 1")
plt.plot(time, mae_u_implicit_2, label="Неявная 2")
plt.xlabel('t')
plt.ylabel('MAE(u(x, t), U(x, t))')
plt.title('Зависимость погрешности от времени t')
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.legend()
plt.grid()
plt.show()



# Пример списка временных индексов, для которых хотим построить графики
time_indices = [1000, 2000, 5000]

# Расчетная сетка по x
x = linspace(L, R, n + 1)

for t_idx in time_indices:
    plt.figure(figsize=(12, 8))
    current_time = tk(t_idx, tau)
    # Вычисляем аналитическое решение
    u_analytic_values = [analytic(current_time, xi(L, i, h)) for i in range(n + 1)]
    # Численные решения
    u_explicit_1_values = [u_explicit_1[t_idx][i] for i in range(n + 1)]
    u_explicit_2_values = [u_explicit_2[t_idx][i] for i in range(n + 1)]
    u_implicit_1_values = [u_implicit_1[t_idx][i] for i in range(n + 1)]
    u_implicit_2_values = [u_implicit_2[t_idx][i] for i in range(n + 1)]

    # Построение графиков
    plt.plot(x, u_analytic_values, label=f'Аналитическое, t = {current_time}')
    plt.plot(x, u_explicit_1_values, label=f'Явная сх. 1, t = {current_time}', linestyle='--')
    plt.plot(x, u_explicit_2_values, label=f'Явная сх. 2, t = {current_time}', linestyle='-.')
    plt.plot(x, u_implicit_1_values, label=f'Неявная сх. 1, t = {current_time}', linestyle=':')
    plt.plot(x, u_implicit_2_values, label=f'Неявная сх. 2, t = {current_time}', linestyle='-')

    # Настройки осей
    plt.xlabel('x')
    plt.ylabel('u(x, t)')
    plt.title(f'Сравнение аналитического и численных решений при t={current_time}')
    plt.xticks(
        ticks=[0, pi/8, pi/4, 3*pi/8, pi/2], 
        labels=['0', r'$\frac{\pi}{8}$', r'$\frac{\pi}{4}$', r'$\frac{3\pi}{8}$', r'$\frac{\pi}{2}$']
    )
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.legend()
    plt.grid()
    plt.show()
