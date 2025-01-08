import math
import numpy as np
import matplotlib.pyplot as plt

def L2_norm(vec: np.ndarray):
    return np.sqrt(np.sum(vec*vec))

def iterative(A, b, eps):
    '''
    метод простых итераций
    '''
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
    Метод Гаусса–Зейделя.
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
        # шаг Зейделя
        x_inter = seidel_multiplication(alpha, x_cur, beta)
        # релаксация
        x_cur = w*x_inter + (1-w)*x_prev
        iterations += 1
        if L2_norm(x_cur - x_prev) < eps:
            break
    return x_cur, iterations



def finite_difference_schema_mixed(
    x_range, y_range,
    h_x, h_y,
    method,
    eps=1e-7
):

    x_vals = np.arange(x_range[0], x_range[1] + 1e-14, h_x)
    y_vals = np.arange(y_range[0], y_range[1] + 1e-14, h_y)
    Nx = len(x_vals)
    Ny = len(y_vals)

    N = Nx * Ny
    A = np.zeros((N, N), dtype=float)
    b = np.zeros(N, dtype=float)

    def idx(i,j):
        return i*Ny + j

    res = np.zeros((Nx, Ny), dtype=float)

    for i in range(Nx):
        for j in range(Ny):
            I = idx(i,j) 

            x_ij = x_vals[i]
            y_ij = y_vals[j]

            if i == 0:
                A[I,I] = 1.0
                b[I]   = 0.0
                continue

            if i == Nx-1:
                A[I,I] = 1.0
                b[I]   = y_ij
                continue

            if j == 0:
                A[I,I] = -1.0/h_y
                A[I,idx(i, j+1)] = 1.0/h_y
                b[I] = math.sin(x_ij)
                continue

            if j == Ny-1:
                A[I,I] =  1.0/h_y - 1.0
                A[I, idx(i, j-1)] = -1.0/h_y
                b[I] = 0.0
                continue


            A[I,I] = -2.0/h_x**2 - 2.0/h_y**2 + 1.0
            A[I, idx(i+1,j)] += 1.0/h_x**2
            A[I, idx(i-1,j)] += 1.0/h_x**2
            A[I, idx(i,j+1)] += 1.0/h_y**2
            A[I, idx(i,j-1)] += 1.0/h_y**2

    x_sol, num_iters = method(A, b, eps)

    for i in range(Nx):
        for j in range(Ny):
            I = idx(i,j)
            res[i,j] = x_sol[I]

    return res, num_iters, x_vals, y_vals


if __name__=="__main__":

    x_begin = 0.0
    x_end   = math.pi/2
    y_begin = 0.0
    y_end   = 1.0
    h_x     = 0.1
    h_y     = 0.1
    eps = 1e-4

    def solution(x, y):
        return y * math.sin(x)

    x_vals = np.arange(x_begin, x_end+1e-14, h_x)
    y_vals = np.arange(y_begin, y_end+1e-14, h_y)
    u_exact = np.zeros((len(x_vals), len(y_vals)))
    for i, xv in enumerate(x_vals):
        for j, yv in enumerate(y_vals):
            u_exact[i,j] = solution(xv, yv)

    sol_jacobi, it_jacobi, Xn, Yn = finite_difference_schema_mixed(
        (x_begin, x_end),
        (y_begin, y_end),
        h_x, h_y,
        method=iterative,
        eps=eps
    )
    print("[Метод релаксации] iters =", it_jacobi)
    def max_abs_error(A, B):
        return np.max(np.abs(A - B))
    print("[Метод релаксации] max_error =", max_abs_error(sol_jacobi, u_exact))

    #--- (2) Зейдель
    sol_seidel, it_seid, _, _ = finite_difference_schema_mixed(
        (x_begin, x_end),
        (y_begin, y_end),
        h_x, h_y,
        method=seidel,
        eps=eps
    )
    print("[Seidel] iters =", it_seid)
    print("[Seidel] max_error =", max_abs_error(sol_seidel, u_exact))

    #--- (3) Релаксация
    sol_relax, it_relax, _, _ = finite_difference_schema_mixed(
        (x_begin, x_end),
        (y_begin, y_end),
        h_x, h_y,
        method=lambda A,b,eps: relaxation(A,b,eps,w=1.5),
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

    plot_3d_surface(x_vals, y_vals, u_exact, "Analytical solution")

    plot_3d_surface(x_vals, y_vals, sol_jacobi, "метод простых итераций")
    plot_3d_surface(x_vals, y_vals, np.abs(sol_jacobi - u_exact), "Error (метод простых итераций)")

    plot_3d_surface(x_vals, y_vals, sol_seidel, "Seidel method")
    plot_3d_surface(x_vals, y_vals, np.abs(sol_seidel - u_exact), "Error (Seidel)")

    plot_3d_surface(x_vals, y_vals, sol_relax, "Метод релаксации w=1.5")
    plot_3d_surface(x_vals, y_vals, np.abs(sol_relax - u_exact), "Error (Метод релаксации w=1.5)")
