from functions import *


def Iterations(a, b, eps):

    def JacobiMatrix(x):
        return [[dphi1_dx1(x), dphi1_dx2(x)], [dphi2_dx1(x), dphi2_dx2(x)]]

    def GetNorm(J):
        return max([max([abs(x) for x in row]) for row in J])

    x0_interv = [a[0], b[0]]
    x1_interv = [a[1], b[1]]

    x_prev = [(x0_interv[1] + x0_interv[0]) / 2, (x1_interv[1] + x1_interv[0]) / 2]

    q = GetNorm(JacobiMatrix(x_prev))
    if q >= 1:
        return None

    iterations = 0
    while True <= 1000:
        iterations += 1
        x = [phi1(x_prev), phi2(x_prev)]
        if q / (1 - q) * norm(x, x_prev) <= eps:
            break
        x_prev = x
    return x, iterations
