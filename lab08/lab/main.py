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


def solve_mpn(a, Nx, Ny, K, T):
    x_vals, y_vals, t_vals, hx, hy, tau = create_grid(a, Nx, Ny, K, T)
    u = np.zeros((Nx+1, Ny+1))
    for i in range(Nx+1):
        for j in range(Ny+1):
            u[i, j] = initial_condition(x_vals[i], y_vals[j])

    solution = [u.copy()]

    for k in range(K):
        t_n = t_vals[k]

        u_half = step_x_adi(u, x_vals, y_vals, a, hx, hy, tau, t_n)
        u_next = step_y_adi(u_half, x_vals, y_vals, a, hx, hy, tau, t_n)

        solution.append(u_next)
        u = u_next
    
    return np.array(solution)

def step_x_adi(u_in, x_vals, y_vals, a, hx, hy, tau, t_n):
    Nx, Ny = u_in.shape[0] - 1, u_in.shape[1] - 1
    alpha = 2.0 / tau

    u_out = u_in.copy()
    t_half = t_n + 0.5 * tau

    for j in range(Ny+1):
        u_out[0, j] = bc_x0(t_half, y_vals[j], a)
        u_out[Nx, j] = bc_xL(t_half, y_vals[j], a)
    for i in range(Nx+1):
        u_out[i, 0] = bc_y0(t_half, x_vals[i], a)
        u_out[i, Ny] = bc_yM(t_half, x_vals[i], a)

    for j in range(1, Ny):
        a_coeff = -a / (hx * hx)
        b_coeff = alpha + 2 * a / (hx * hx)
        c_coeff = -a / (hx * hx)
        d = [
            alpha * u_in[i, j] + a * (u_in[i, j+1] - 2 * u_in[i, j] + u_in[i, j-1]) / (hy * hy)
            for i in range(1, Nx)
        ]
        d[0] -= a_coeff * u_out[0, j]
        d[-1] -= c_coeff * u_out[Nx, j]

        sol = TMA([a_coeff] * (Nx - 1), [b_coeff] * (Nx - 1), [c_coeff] * (Nx - 1), d)
        for i in range(1, Nx):
            u_out[i, j] = sol[i - 1]

    return u_out

def step_y_adi(u_in, x_vals, y_vals, a, hx, hy, tau, t_n):
    Nx, Ny = u_in.shape[0] - 1, u_in.shape[1] - 1
    alpha = 2.0 / tau

    u_out = u_in.copy()
    t_half = t_n + 0.5 * tau

    for j in range(Ny+1):
        u_out[0, j] = bc_x0(t_half, y_vals[j], a)
        u_out[Nx, j] = bc_xL(t_half, y_vals[j], a)
    for i in range(Nx+1):
        u_out[i, 0] = bc_y0(t_half, x_vals[i], a)
        u_out[i, Ny] = bc_yM(t_half, x_vals[i], a)

    for i in range(1, Nx):
        a_coeff = -a / (hy * hy)
        b_coeff = alpha + 2 * a / (hy * hy)
        c_coeff = -a / (hy * hy)
        d = [
            alpha * u_in[i, j] + a * (u_in[i+1, j] - 2 * u_in[i, j] + u_in[i-1, j]) / (hx * hx)
            for j in range(1, Ny)
        ]
        d[0] -= a_coeff * u_out[i, 0]
        d[-1] -= c_coeff * u_out[i, Ny]

        sol = TMA([a_coeff] * (Ny - 1), [b_coeff] * (Ny - 1), [c_coeff] * (Ny - 1), d)
        for j in range(1, Ny):
            u_out[i, j] = sol[j - 1]

    return u_out


def solve_fractional_steps(a, Nx, Ny, K, T):
    x_vals, y_vals, t_vals, hx, hy, tau = create_grid(a, Nx, Ny, K, T)

    u_current = np.zeros((Nx+1, Ny+1))
    for i in range(Nx+1):
        for j in range(Ny+1):
            u_current[i,j] = initial_condition(x_vals[i], y_vals[j])

    solution = [u_current.copy()]

    for n in range(K):
        t_n = t_vals[n]
        u_half = step_y_fractional(u_current, x_vals, y_vals, a, hx, hy, tau, t_n)
        u_next = step_x_fractional(u_half, x_vals, y_vals, a, hx, hy, tau, t_n)

        solution.append(u_next)
        u_current = u_next

    return np.array(solution)

def step_x_fractional(u_in, x_vals, y_vals, a, hx, hy, tau, t_n):
    Nx, Ny = u_in.shape[0]-1, u_in.shape[1]-1

    alpha = 1.0 / tau
    a_val = -a / (hx * hx)
    b_val = alpha + 2 * a / (hx * hx)
    c_val = -a / (hx * hx)

    u_out = u_in.copy()
    t_half = t_n + 0.5 * tau

    for j in range(Ny+1):
        u_out[0, j]  = bc_x0(t_half, y_vals[j], a)
        u_out[Nx, j] = bc_xL(t_half, y_vals[j], a)
    for i in range(Nx+1):
        u_out[i, 0]  = bc_y0(t_half, x_vals[i], a)
        u_out[i, Ny] = bc_yM(t_half, x_vals[i], a)

    for j in range(1, Ny):
        a_list = []
        b_list = []
        c_list = []
        d_list = []

        for i in range(1, Nx):
            a_list.append(a_val)
            b_list.append(b_val)
            c_list.append(c_val)
            d_list.append(alpha * u_in[i, j])

        d_list[0]  -= a_val * u_out[0, j]
        d_list[-1] -= c_val * u_out[Nx, j]

        sol = TMA(a_list, b_list, c_list, d_list)
        for i_int in range(1, Nx):
            u_out[i_int, j] = sol[i_int - 1]

    return u_out

def step_y_fractional(u_in, x_vals, y_vals, a, hx, hy, tau, t_n):
    Nx, Ny = u_in.shape[0]-1, u_in.shape[1]-1

    alpha = 1.0 / tau
    a_val = -a / (hy * hy)
    b_val = alpha + 2 * a / (hy * hy)
    c_val = -a / (hy * hy)

    u_out = u_in.copy()
    t_half = t_n + 0.5 * tau

    for j in range(Ny+1):
        u_out[0, j]  = bc_x0(t_half, y_vals[j], a)
        u_out[Nx, j] = bc_xL(t_half, y_vals[j], a)
    for i in range(Nx+1):
        u_out[i, 0]  = bc_y0(t_half, x_vals[i], a)
        u_out[i, Ny] = bc_yM(t_half, x_vals[i], a)

    for i in range(1, Nx):
        a_list = []
        b_list = []
        c_list = []
        d_list = []
        for j in range(1, Ny):
            a_list.append(a_val)
            b_list.append(b_val)
            c_list.append(c_val)
            d_list.append(alpha * u_in[i, j])

        d_list[0]  -= a_val * u_out[i, 0]
        d_list[-1] -= c_val * u_out[i, Ny]

        sol = TMA(a_list, b_list, c_list, d_list)
        for j_int in range(1, Ny):
            u_out[i, j_int] = sol[j_int - 1]

    return u_out



if __name__ == "__main__":
    a = 1.0
    T = 1.0
    Nx, Ny, K = 100, 100, 100

    U_analytic = solve_analytic(a, Nx, Ny, K, T)
    U_mpn = solve_mpn(a, Nx, Ny, K, T)
    U_mdsh = solve_fractional_steps(a, Nx, Ny, K, T)

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
