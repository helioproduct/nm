from functions import *
from numpy.linalg import solve


def Newton(a, b, eps):

    def JacobiMatrix(x):
        return [[df1_dx1(x), df1_dx2(x)], [df2_dx1(x), df2_dx2(x)]]

    x0_interv = [a[0], b[0]]
    x1_interv = [a[1], b[1]]

    x_prev = [(x0_interv[1] + x0_interv[0]) / 2, (x1_interv[1] + x1_interv[0]) / 2]

    iteration = 0
    while True <= 1000:
        iteration += 1
        jacobi = np.array(JacobiMatrix(x_prev))
        b = np.array([-f1(x_prev), -f2(x_prev)])
        delta_x = solve(jacobi, b).tolist()
        x = [px + dx for px, dx in zip(x_prev, delta_x)]
        if norm(x, x_prev) <= eps:
            break
        x_prev = x

    return x, iteration
