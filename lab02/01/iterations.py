from scipy.optimize import fsolve

from functions import phi, phi1, phi2


def iterations(f, a, b, eps):

    root = fsolve(phi2, 0)[0]
    if a <= root <= b:
        mx = abs(phi1(root))
    else:
        mx = max(abs(phi1(a)), abs(phi1(b)))

    if mx > 1:
        print("Contraction condition not met.")
        return None

    q = mx
    x_prev = (b + a) / 2
    x = phi(x_prev)
    iterations = 0

    while q / (1 - q) * abs(x - x_prev) > eps:
        x_prev = x
        x = phi(x_prev)
        iterations += 1

    return (x, iterations)
