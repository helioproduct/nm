
def solve_triag(a, b, c, d):
    n = len(d)
    P = [0.0 for _ in range(n)]
    Q = [0.0 for _ in range(n)]
    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] + a[i]*P[i-1]
        P[i] = -c[i]/denom
        Q[i] = (d[i] - a[i]*Q[i-1])/denom
    P[-1] = 0
    Q[-1] = (d[-1] - a[-1]*Q[-2])/(b[-1] + a[-1]*P[-2])
    x = [0.0 for _ in range(n)]
    x[-1] = Q[-1]
    for i in range(n - 2, -1, -1):
        x[i] = P[i]*x[i+1] + Q[i]
    return x
