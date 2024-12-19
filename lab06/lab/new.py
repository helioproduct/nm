import matplotlib.pyplot as plt
from math import exp, cos, pi
from numpy import linspace

class StabilityException(Exception):
    def __init__(self, tau, h):
        self.tau = tau
        self.h = h
        super().__init__(self._error_message())

    def _error_message(self):
        return f"Stability condition violated: tau / (h * h) > 0.5 (tau={self.tau}, h={self.h})"


def tk(k, tau):
    return k * tau

def xi(L, i, h):
    return L + i * h

def Phi_L(t):
    return cos(2 * t)

def Phi_R(t):
    return 0.0

def Psi_1(x):
    return exp(-x) * cos(x)

def Psi_2(x):
    return 0.0  # u_t(x,0)=0

def analytic(t, x):
    return exp(-x) * cos(x) * cos(2 * t)

def dx(f, x):
    eps = 1e-6
    return (f(x + eps) - f(x)) / eps

def ddx(f, x):
    eps = 1e-6
    return (dx(f, x + eps) - dx(f, x)) / eps

def TMA(a, b, c, d):
    n = len(d)
    P = [0.0 for _ in range(n)]
    Q = [0.0 for _ in range(n)]
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / denom
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denom
    denom = b[-1] + a[-1] * P[-2]
    Q[-1] = (d[-1] - a[-1] * Q[-2]) / denom
    x = [0.0 for _ in range(n)]
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x

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
        solve = TMA(a, b, c, d)
        for i in range(n + 1):
            u[k+1][i] = solve[i]

    return u

# Пример использования
L = 0
R = pi / 2
T = 10
n = 100
K = 50000

u_implicit_1 = implicit(n, L, R, K, T, approx=2)

tau = T / K
h = (R - L) / n
time = linspace(0, T, K + 1)

# Проверка точности для неявной схемы
mae_u_implicit_1 = [0.0 for _ in range(K + 1)]
for k in range(K + 1):
    for i in range(n + 1):
        mae_u_implicit_1[k] += abs(u_implicit_1[k][i] - analytic(tk(k, tau), xi(L, i, h)))
    mae_u_implicit_1[k] /= n

plt.figure(figsize=(12, 8))
plt.plot(time, mae_u_implicit_1, label="Неявная сх. 1")
plt.xlabel('t')
plt.ylabel('MAE')
plt.title('Ошибка неявной схемы')
plt.grid()
plt.legend()
plt.show()
