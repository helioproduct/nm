import math
import numpy as np
import matplotlib.pyplot as plt

# Граничные условия (см. условие задачи)
def phi_0(y):  # u(0,y)=0
    return 0.0

def phi_1(y):  # u(pi/2,y)=y
    return y

def psi_0(x):  # u(x,0)=0
    return 0.0

def psi_1(x):  # u(x,1)=sin(x)
    return math.sin(x)

def solution(x, y):
    return y * math.sin(x)

def get_analytical_solution(x_range, y_range, h_x, h_y):
    x_vals = np.arange(x_range[0], x_range[1] + 1e-12, h_x)
    y_vals = np.arange(y_range[0], y_range[1] + 1e-12, h_y)
    res = np.zeros((len(x_vals), len(y_vals)))
    for i,xv in enumerate(x_vals):
        for j,yv in enumerate(y_vals):
            res[i,j] = solution(xv,yv)
    return x_vals, y_vals, res

def max_abs_error(A, B):
    assert A.shape == B.shape
    return np.max(np.abs(A - B))

def mean_abs_error(A, B):
    assert A.shape == B.shape
    return np.mean(np.abs(A - B))

def L2_norm(vec: np.ndarray):
    return np.sqrt(np.sum(vec*vec))

def iterative(A, b, eps):
    """
    Якоби (простых итераций).
    """
    n = A.shape[0]
    alpha = np.zeros_like(A, dtype=float)
    beta  = np.zeros(n, dtype=float)
    for i in range(n):
        diag = A[i,i]
        for j in range(n):
            if i==j:
                alpha[i,j] = 0
            else:
                alpha[i,j] = -A[i,j]/diag
        beta[i] = b[i]/diag

    x_cur = np.copy(beta)
    iterations = 0
    while True:
        x_prev = x_cur.copy()
        x_cur  = alpha @ x_prev + beta
        iterations += 1
        if L2_norm(x_cur - x_prev) < eps:
            break
    return x_cur, iterations

def seidel_multiplication(alpha, x, beta):
    n = len(x)
    x_new = np.copy(x)
    for i in range(n):
        s = beta[i]
        for j in range(n):
            s += alpha[i,j]*x_new[j]
        x_new[i] = s
    return x_new

def seidel(A, b, eps):
    """
    Гаусс–Зейдель
    """
    n = A.shape[0]
    alpha = np.zeros_like(A, dtype=float)
    beta  = np.zeros(n, dtype=float)
    for i in range(n):
        diag = A[i,i]
        for j in range(n):
            if i==j:
                alpha[i,j] = 0
            else:
                alpha[i,j] = -A[i,j]/diag
        beta[i] = b[i]/diag

    x_cur = np.copy(beta)
    iterations = 0
    while True:
        x_prev = x_cur.copy()
        x_cur  = seidel_multiplication(alpha, x_cur, beta)
        iterations += 1
        if L2_norm(x_cur - x_prev) < eps:
            break
    return x_cur, iterations

def relaxation(A, b, eps, w=1.5):
    """
    Метод верхней релаксации (SOR).
    """
    n = A.shape[0]
    alpha = np.zeros_like(A, dtype=float)
    beta  = np.zeros(n, dtype=float)
    for i in range(n):
        diag = A[i,i]
        for j in range(n):
            if i==j:
                alpha[i,j] = 0
            else:
                alpha[i,j] = -A[i,j]/diag
        beta[i] = b[i]/diag

    x_cur = np.copy(beta)
    iterations = 0
    while True:
        x_prev = x_cur.copy()
        # seidel step
        x_inter = seidel_multiplication(alpha, x_cur, beta)
        # relaxation
        x_cur = w*x_inter + (1-w)*x_prev
        iterations += 1
        if L2_norm(x_cur - x_prev) < eps:
            break
    return x_cur, iterations

def finite_difference_schema(
    x_range, y_range,
    h_x, h_y,
    method,  # callable(A, b, eps) -> (x_sol, num_iters)
    phi_0, phi_1, psi_0, psi_1,
    eps=1e-7
):
    """
    Собирает СЛАУ для уравнения: ∆u + u=0
    с Дирихле ГУ:
        u(0,y)=phi_0(y),
        u(x_end,y)=phi_1(y),
        u(x,0)=psi_0(x),
        u(x,y_end)=psi_1(x).
    Решает методом 'method'.
    Возвращает (res, num_iters, x_vals, y_vals),
    где res.shape=(Nx,Ny) — это 2D-массив сеточного решения.
    """
    x_vals = np.arange(x_range[0], x_range[1]+1e-12, h_x)
    y_vals = np.arange(y_range[0], y_range[1]+1e-12, h_y)
    Nx = len(x_vals)
    Ny = len(y_vals)

    # Инициализируем массив решения, проставим граничные условия
    res = np.zeros((Nx, Ny))
    # ГУ по x=0, x=x_end
    for j in range(Ny):
        res[0, j]   = phi_0(y_vals[j])   # x=0
        res[Nx-1,j] = phi_1(y_vals[j])   # x= x_end
    # ГУ по y=0, y=y_end
    for i in range(Nx):
        res[i, 0]   = psi_0(x_vals[i])  # y=0
        res[i, Ny-1]= psi_1(x_vals[i])  # y=1

    # СЛАУ для внутренних узлов (i=1..Nx-2, j=1..Ny-2)
    # Шаблон:   (1/hx^2)(u[i+1,j]+u[i-1,j]) + (1/hy^2)(u[i,j+1]+u[i,j-1])
    #           + ( -2/hx^2 - 2/hy^2 + 1 )*u[i,j] = 0
    mapping = -1 * np.ones((Nx,Ny), dtype=int)
    eq_id = 0
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            mapping[i,j] = eq_id
            eq_id += 1
    N_equations = eq_id

    A = np.zeros((N_equations, N_equations), dtype=float)
    b = np.zeros(N_equations, dtype=float)

    def is_boundary(i,j):
        return (i==0 or i==Nx-1 or j==0 or j==Ny-1)

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            eq_index = mapping[i,j]
            # центр:
            center_coef = (-2.0/h_x**2 - 2.0/h_y**2 + 1.0)
            A[eq_index, eq_index] = center_coef

            # сосед (i+1,j)
            if is_boundary(i+1,j):
                val = res[i+1,j]
                b[eq_index] -= (val / h_x**2)
            else:
                A[eq_index, mapping[i+1,j]] += (1.0/h_x**2)

            # сосед (i-1,j)
            if is_boundary(i-1,j):
                val = res[i-1,j]
                b[eq_index] -= (val / h_x**2)
            else:
                A[eq_index, mapping[i-1,j]] += (1.0/h_x**2)

            # сосед (i,j+1)
            if is_boundary(i,j+1):
                val = res[i,j+1]
                b[eq_index] -= (val / h_y**2)
            else:
                A[eq_index, mapping[i,j+1]] += (1.0/h_y**2)

            # сосед (i,j-1)
            if is_boundary(i,j-1):
                val = res[i,j-1]
                b[eq_index] -= (val / h_y**2)
            else:
                A[eq_index, mapping[i,j-1]] += (1.0/h_y**2)

    # Решаем A x = b
    x_sol, num_iters = method(A, b, eps)

    # Заполним res
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            eq_index = mapping[i,j]
            res[i,j] = x_sol[eq_index]

    return res, num_iters, x_vals, y_vals

if __name__=="__main__":

    x_begin = 0.0
    x_end   = math.pi/2
    y_begin = 0.0
    y_end   = 1.0
    h_x     = 0.05
    h_y     = 0.05
    eps = 10e-4

    x_vals, y_vals, u_exact = get_analytical_solution(
        (x_begin, x_end),
        (y_begin, y_end),
        h_x, h_y
    )

    # 2) Методы: Якоби
    sol_jacobi, it_jacobi, Xn, Yn = finite_difference_schema(
        (x_begin, x_end),
        (y_begin, y_end),
        h_x, h_y,
        method=iterative,
        phi_0=phi_0, phi_1=phi_1,
        psi_0=psi_0, psi_1=psi_1,
        eps=eps
    )
    print("[Jacobi] iters =", it_jacobi)
    print("[Jacobi] max_error =", max_abs_error(sol_jacobi, u_exact))

    # метод Зейделя
    sol_seidel, it_seid, _, _ = finite_difference_schema(
        (x_begin, x_end),
        (y_begin, y_end),
        h_x, h_y,
        method=seidel,
        phi_0=phi_0, phi_1=phi_1,
        psi_0=psi_0, psi_1=psi_1,
        eps=eps
    )
    print("[Seidel] iters =", it_seid)
    print("[Seidel] max_error =", max_abs_error(sol_seidel, u_exact))

    # методы релаксации
    sol_relax, it_relax, _, _ = finite_difference_schema(
        (x_begin, x_end),
        (y_begin, y_end),
        h_x, h_y,
        method=lambda A,b,eps: relaxation(A,b,eps,w=1.5),
        phi_0=phi_0, phi_1=phi_1,
        psi_0=psi_0, psi_1=psi_1,
        eps=eps
    )
    print("[Relax w=1.5] iters =", it_relax)
    print("[Relax w=1.5] max_error =", max_abs_error(sol_relax, u_exact))

    def plot_3d_surface(x_arr, y_arr, z_2d, title="3D Surface"):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Xgrid, Ygrid = np.meshgrid(x_arr, y_arr)  # (ny,nx)
        Zplot = z_2d.T
        surf = ax.plot_surface(Xgrid, Ygrid, Zplot, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u(x,y)")
        plt.colorbar(surf, shrink=0.5, aspect=10)
        plt.show()

    plot_3d_surface(x_vals, y_vals, sol_jacobi, "Jacobi method: u(x,y)")
    plot_3d_surface(x_vals, y_vals, sol_seidel, "Seidel method: u(x,y)")
    plot_3d_surface(x_vals, y_vals, sol_relax, "Relaxation (w=1.5): u(x,y)")
    plot_3d_surface(x_vals, y_vals, u_exact, "Exact solution: u(x,y)")

    error_relax = np.abs(sol_relax - u_exact)
    plot_3d_surface(x_vals, y_vals, error_relax, "Error (Relaxation w=1.5)")
