import math
from functions import *


def Iterations(a, b, eps):
    def get_phi_norm(x):
        return max(
            abs(dphi1_dx1(x)) + abs(dphi1_dx2(x)), abs(dphi2_dx1(x)) + abs(dphi2_dx2(x))
        )

    x0_interv = [a[0], b[0]]
    x1_interv = [a[1], b[1]]

    x_prev = [(x0_interv[1] + x0_interv[0]) / 2, (x1_interv[1] + x1_interv[0]) / 2]

    q = get_phi_norm(x_prev)
    if q >= 1:
        return None

    iterations = 0
    while iterations <= 1000:
        iterations += 1
        x = [phi1(x_prev), phi2(x_prev)]
        error = q / (1 - q) * norm(x, x_prev)
        if error <= eps:
            break
        x_prev = x
    return x, iterations
