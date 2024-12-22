import numpy as np
import matplotlib.pyplot as plt
from math import sinh, cos, exp, pi
from triag import TMA


def create_grid(a, Nx, Ny, K, T):
    x_left, x_right = 0.0, pi/4
    y_left, y_right = 0.0, np.log(2)
    
    x = np.linspace(x_left, x_right, Nx+1)
    y = np.linspace(y_left, y_right, Ny+1)
    t = np.linspace(0, T, K+1)
    
    hx = (x_right - x_left)/Nx
    hy = (y_right - y_left)/Ny
    tau = T/K

    return x, y, t, hx, hy, tau

def bc_x0(t, y, a):
    return sinh(y)*exp(-3*a*t)

def bc_xL(t, y, a):
    return 0.0

def bc_y0(t, x, a):
    return 0.0

def bc_yM(t, x, a):
    return 0.75*cos(2*x)*exp(-3*a*t)

def initial_condition(x, y):
    return cos(2*x)*sinh(y)

def exact_solution(t, x, y, a):
    return cos(2*x)*sinh(y)*exp(-3*a*t)

def solve_analytic(a, Nx, Ny, K, T):
    x, y, t, hx, hy, tau = create_grid(a, Nx, Ny, K, T)
    u_analytic = np.zeros((K+1, Nx+1, Ny+1))
    for n in range(K+1):
        for i in range(Nx+1):
            for j in range(Ny+1):
                u_analytic[n, i, j] = exact_solution(t[n], x[i], y[j], a)
    return u_analytic



def solve_mpn(a_coeff, Nx, Ny, K, T):
    """
    Метод переменных направлений (ADI).
    Возвращает 3D-массив (K+1, Nx+1, Ny+1).
    
    a_coeff -- коэффициент a из уравнения du/dt = a*(uxx + uyy).
    """
    x_vals, y_vals, t_vals, hx, hy, tau = create_grid(a_coeff, Nx, Ny, K, T)
    
    # Инициализация: слой t=0
    u_current = np.zeros((Nx+1, Ny+1))
    for i in range(Nx+1):
        for j in range(Ny+1):
            u_current[i,j] = initial_condition(x_vals[i], y_vals[j])

    solution = [u_current.copy()]

    # За один шаг (k -> k+1) делаем 2 полушага:
    for n in range(K):
        t_n = t_vals[n]

        # 1) Полушаг по x (неявно по x, явно по y)
        u_half = step_x_adi(u_current, x_vals, y_vals, a_coeff, hx, hy, tau, t_n)
        # 2) Полушаг по y (неявно по y, явно по x)
        u_next = step_y_adi(u_half, x_vals, y_vals, a_coeff, hx, hy, tau, t_n)

        solution.append(u_next)
        u_current = u_next
    
    return np.array(solution)


def step_x_adi(u_in, x_vals, y_vals, a_coeff, hx, hy, tau, t_n):
    """
    Первый полушаг МПН: неявная схема по x, оператор по y явно.
    alpha = 2/tau.
    """
    Nx, Ny = u_in.shape[0]-1, u_in.shape[1]-1
    alpha = 2.0 / tau  # (1+1)/tau для ADI
    # r_x = a/hx^2
    r_x = a_coeff/(hx*hx)

    # u_out (k+1/2)
    u_out = u_in.copy()
    t_half = t_n + 0.5*tau

    # Граничные условия на полушаге:
    for j in range(Ny+1):
        u_out[0, j]  = bc_x0(t_half, y_vals[j], a_coeff)
        u_out[Nx, j] = bc_xL(t_half, y_vals[j], a_coeff)
    for i in range(Nx+1):
        u_out[i, 0]  = bc_y0(t_half, x_vals[i], a_coeff)
        u_out[i, Ny] = bc_yM(t_half, x_vals[i], a_coeff)

    # Теперь собираем трёхдиагональную систему построчно (j=1..Ny-1).
    # Матрица в ADI (по x): 
    #   a = -r_x,   b = alpha + 2*r_x,  c = -r_x
    # Правая часть: b_ = alpha*u_in[i,j] + a_coeff*(u_in[i,j+1]-2u_in[i,j]+u_in[i,j-1])/hy^2
    # (т.к. оператор по y берётся явно из слоя u_in)

    a_val = -r_x
    b_val = alpha + 2*r_x
    c_val = -r_x

    for j in range(1, Ny):
        a_list = []
        b_list = []
        c_list = []
        d_list = []
        for i in range(1, Nx):
            a_list.append(a_val)
            b_list.append(b_val)
            c_list.append(c_val)

            # Явная часть по y:
            # lapl_y = (u_in[i,j+1] - 2u_in[i,j] + u_in[i,j-1]) / hy^2
            lapl_y = (u_in[i,j+1] - 2*u_in[i,j] + u_in[i,j-1])/(hy*hy)
            # => d = alpha*u_in[i,j] + a_coeff*lapl_y
            d_val = alpha*u_in[i,j] + a_coeff*lapl_y
            d_list.append(d_val)

        # Учет граничных узлов
        # d1 -= a_val*u_out[0,j], d_{N-1} -= c_val*u_out[Nx,j]
        d_list[0]  -= a_val*u_out[0,j]
        d_list[-1] -= c_val*u_out[Nx,j]

        sol = TMA(a_list, b_list, c_list, d_list)
        for i_int in range(1, Nx):
            u_out[i_int, j] = sol[i_int-1]

    return u_out


def step_y_adi(u_in, x_vals, y_vals, a_coeff, hx, hy, tau, t_n):
    """
    Второй полушаг МПН: неявная схема по y, оператор по x явно.
    alpha = 2/tau
    """
    Nx, Ny = u_in.shape[0]-1, u_in.shape[1]-1
    alpha = 2.0 / tau
    r_y = a_coeff/(hy*hy)

    u_out = u_in.copy()
    t_half = t_n + 0.5*tau

    # ГУ (t_{n+1}, но ставим t_half для единообразия)
    for j in range(Ny+1):
        u_out[0, j]  = bc_x0(t_half, y_vals[j], a_coeff)
        u_out[Nx, j] = bc_xL(t_half, y_vals[j], a_coeff)
    for i in range(Nx+1):
        u_out[i, 0]  = bc_y0(t_half, x_vals[i], a_coeff)
        u_out[i, Ny] = bc_yM(t_half, x_vals[i], a_coeff)

    # Матрица по y:
    #   a = -r_y, b = alpha+2*r_y, c = -r_y
    # Правая часть: alpha*u_in[i,j] + a_coeff*(u_in[i+1,j] - 2*u_in[i,j] + u_in[i-1,j])/hx^2
    #  (оператор по x явно)

    a_val = -r_y
    b_val = alpha + 2*r_y
    c_val = -r_y

    for i in range(1, Nx):
        a_list = []
        b_list = []
        c_list = []
        d_list = []
        for j in range(1, Ny):
            a_list.append(a_val)
            b_list.append(b_val)
            c_list.append(c_val)

            lapl_x = (u_in[i+1,j] - 2*u_in[i,j] + u_in[i-1,j])/(hx*hx)
            d_val = alpha*u_in[i,j] + a_coeff*lapl_x
            d_list.append(d_val)

        d_list[0]  -= a_val*u_out[i,0]
        d_list[-1] -= c_val*u_out[i,Ny]

        sol = TMA(a_list, b_list, c_list, d_list)
        for j_int in range(1, Ny):
            u_out[i, j_int] = sol[j_int-1]

    return u_out




def solve_mdsh(a_coeff, Nx, Ny, K, T):
    """
    Тот же метод дробных шагов (MДШ), 
    но с переменными, названными в стиле a, b, c, d (как в формульном описании).
    
    a_coeff -- это 'a' из уравнения du/dt = a*(uxx + uyy).
    """
    x_vals, y_vals, t_vals, hx, hy, tau = create_grid(a_coeff, Nx, Ny, K, T)

    # Инициализация: слой t=0
    u_current = np.zeros((Nx+1, Ny+1))
    for i in range(Nx+1):
        for j in range(Ny+1):
            u_current[i,j] = initial_condition(x_vals[i], y_vals[j])

    solution = [u_current.copy()]

    for n in range(K):
        t_n = t_vals[n]
        # Первый "дробный" шаг: неявно по x
        u_half = step_x_mdsh(u_current, x_vals, y_vals, a_coeff, hx, hy, tau, t_n)
        # Второй "дробный" шаг: неявно по y
        u_next = step_y_mdsh(u_half, x_vals, y_vals, a_coeff, hx, hy, tau, t_n)

        solution.append(u_next)
        u_current = u_next

    return np.array(solution)


def step_x_mdsh(u_in, x_vals, y_vals, a_coeff, hx, hy, tau, t_n):
    """
    Шаг по x (неявный). 
    Переменные a, b, c, d - как в теории.
    gamma=0 => alpha=1/tau => (b ~ alpha + 2*r_x, a=c= -r_x).
    """
    Nx, Ny = u_in.shape[0]-1, u_in.shape[1]-1

    alpha = 1.0 / tau
    r_x = a_coeff/(hx*hx)

    u_out = u_in.copy()
    t_half = t_n + 0.5*tau

    # ГУ (Dirichlet)
    for j in range(Ny+1):
        u_out[0, j]  = bc_x0(t_half, y_vals[j], a_coeff)
        u_out[Nx, j] = bc_xL(t_half, y_vals[j], a_coeff)
    for i in range(Nx+1):
        u_out[i, 0]  = bc_y0(t_half, x_vals[i], a_coeff)
        u_out[i, Ny] = bc_yM(t_half, x_vals[i], a_coeff)

    # Теперь собираем трёхдиагональную систему по каждой строке j=1..Ny-1
    # В теории пишут:
    #  a*u_{i-1} + b*u_i + c*u_{i+1} = d
    # где b = alpha + 2*r_x, a=c= -r_x, d = alpha * u_in[i,j].
    
    a_val = -r_x  # в теории "a"
    b_val = alpha + 2*r_x  # "b"
    c_val = -r_x  # "c"

    for j in range(1, Ny):
        a_list = []
        b_list = []
        c_list = []
        d_list = []

        # формируем систему размером (Nx-1)
        for i in range(1, Nx):
            a_list.append(a_val)
            b_list.append(b_val)
            c_list.append(c_val)

            # d = alpha * (старое u)
            d_list.append(alpha*u_in[i,j])

        # Учет граничных значений
        # d1 = d - a*u0 => d_list[0] -= a_val * u_out[0,j]
        d_list[0]  -= a_val*u_out[0,j]
        # dN-1 = d - c*uN => d_list[-1] -= c_val * u_out[Nx,j]
        d_list[-1] -= c_val*u_out[Nx,j]

        # Решаем прогонкой
        sol = TMA(a_list, b_list, c_list, d_list)
        for i_int in range(1, Nx):
            u_out[i_int, j] = sol[i_int-1]

    return u_out


def step_y_mdsh(u_in, x_vals, y_vals, a_coeff, hx, hy, tau, t_n):
    """
    Шаг по y (неявный).
    Аналогично, но теперь a, b, c соответствуют неявности по y.
    """
    Nx, Ny = u_in.shape[0]-1, u_in.shape[1]-1

    alpha = 1.0 / tau
    r_y = a_coeff/(hy*hy)

    u_out = u_in.copy()
    t_half = t_n + 0.5*tau

    # ГУ
    for j in range(Ny+1):
        u_out[0, j]  = bc_x0(t_half, y_vals[j], a_coeff)
        u_out[Nx, j] = bc_xL(t_half, y_vals[j], a_coeff)
    for i in range(Nx+1):
        u_out[i, 0]  = bc_y0(t_half, x_vals[i], a_coeff)
        u_out[i, Ny] = bc_yM(t_half, x_vals[i], a_coeff)

    # Матрица: a = -r_y, b = alpha + 2*r_y, c = -r_y.
    a_val = -r_y
    b_val = alpha + 2*r_y
    c_val = -r_y

    # Строим систему постолбцово (фиксируем i, решаем по j):
    for i in range(1, Nx):
        a_list = []
        b_list = []
        c_list = []
        d_list = []
        for j in range(1, Ny):
            a_list.append(a_val)
            b_list.append(b_val)
            c_list.append(c_val)

            # d = alpha * u_in[i,j]
            d_list.append(alpha*u_in[i,j])

        d_list[0]  -= a_val*u_out[i,0]
        d_list[-1] -= c_val*u_out[i,Ny]

        sol = TMA(a_list, b_list, c_list, d_list)
        for j_int in range(1, Ny):
            u_out[i, j_int] = sol[j_int-1]

    return u_out

if __name__ == "__main__":
    a = 1.0
    T = 1.0
    Nx, Ny, K = 30, 30, 30

    U_analytic = solve_analytic(a, Nx, Ny, K, T)
    U_mpn = solve_mpn(a, Nx, Ny, K, T)
    U_mdsh = solve_mdsh(a, Nx, Ny, K, T)

    x, y, t_arr, hx, hy, tau = create_grid(a, Nx, Ny, K, T)
    Xgrid, Ygrid = np.meshgrid(y, x)

    times_to_plot = [0, T/2, T]
    for tval in times_to_plot:
        idx = np.argmin(np.abs(t_arr - tval))

        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xgrid, Ygrid, U_analytic[idx], cmap='viridis')
        ax.set_title(f"Exact, t={tval:.2f}")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        plt.show()

        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xgrid, Ygrid, U_mpn[idx], cmap='plasma')
        ax.set_title(f"МПН, t={tval:.2f}")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        plt.show()

        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xgrid, Ygrid, U_mdsh[idx], cmap='inferno')
        ax.set_title(f"МДШ, t={tval:.2f}")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        plt.show()

    mae_mpn  = []
    mae_mdsh = []
    for n in range(K+1):
        diff_mpn  = np.abs(U_mpn[n]  - U_analytic[n])
        diff_mdsh = np.abs(U_mdsh[n] - U_analytic[n])
        mae_mpn.append( np.mean(diff_mpn)  )
        mae_mdsh.append(np.mean(diff_mdsh))

    plt.figure(figsize=(7,5))
    plt.plot(t_arr, mae_mpn,  label="МПН (MAE)")
    plt.plot(t_arr, mae_mdsh, label="МДШ (MAE)")
    plt.xlabel("t")
    plt.ylabel("Средняя абсолютная ошибка")
    plt.title("Сравнение ошибок МПН и МДШ")
    plt.legend()
    plt.grid(True)
    plt.show()
