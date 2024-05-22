import copy
import matplotlib.pyplot as plt


def LU(A):
    n = len(A)
    L = [[0 for _ in range(n)] for _ in range(n)]
    U = copy.deepcopy(A)

    for k in range(1, n):
        for i in range(k - 1, n):
            for j in range(i, n):
                L[j][i] = U[j][i] / U[i][i]

        for i in range(k, n):
            for j in range(k - 1, n):
                U[i][j] = U[i][j] - L[i][k - 1] * U[k - 1][j]

    return L, U


def solve_system(A, b):
    L, U = LU(A)
    n = len(L)
    y = [0 for _ in range(n)]
    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i][j] * y[j]
        y[i] = (b[i] - s) / L[i][i]
    x = [0 for _ in range(n)]
    for i in range(n - 1, -1, -1):
        s = 0
        for j in range(n - 1, i - 1, -1):
            s += U[i][j] * x[j]
        x[i] = (y[i] - s) / U[i][i]
    return x


def MNK(x, y, n):
    assert len(x) == len(y)
    A = []
    b = []
    for k in range(n + 1):
        row = []
        for i in range(n + 1):
            sum_ai = 0
            for xi in x:
                sum_ai += xi ** (i + k)
            row.append(sum_ai)
        A.append(row)

        sum_bk = 0
        for i in range(len(x)):
            sum_bk += y[i] * (x[i] ** k)
        b.append(sum_bk)

    return solve_system(A, b)


def P(coefs, x):
    result = 0
    for i in range(len(coefs)):
        result += coefs[i] * (x**i)
    return result


def square_error(x, y, ls_coefs):
    y_not_real = []
    for x_i in x:
        y_not_real.append(P(ls_coefs, x_i))
    sse = 0
    for i in range(len(y)):
        sse += (y[i] - y_not_real[i]) ** 2
    return sse


if __name__ == "__main__":
    x = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
    y = [-2.2026, -0.19315, 0.79464, 1.5624, 2.2306, 2.8419]

    plt.scatter(x, y, color="r")

    print("Least squares method, degree = 1")
    ls1 = MNK(x, y, 1)
    print(f"P(x) = {ls1[0]} + {ls1[1]}x")
    plt.plot(x, [P(ls1, x_i) for x_i in x], color="b", label="degree = 1")
    print(f"Sum of squared errors = {square_error(x, y, ls1)}")

    print("Least squares method, degree = 2")
    ls2 = MNK(x, y, 2)
    print(f"P(x) = {ls2[0]} + {ls2[1]}x + {ls2[2]}x^2")
    plt.plot(x, [P(ls2, x_i) for x_i in x], color="g", label="degree = 2")
    print(f"Sum of squared errors = {square_error(x, y, ls2)}")

    plt.legend()
    plt.show()
