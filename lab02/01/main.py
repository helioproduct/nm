from newton import Newton
from iterations import Iterations

from functions import f, phi


if __name__ == "__main__":
    # f(x) = 0

    eps = 0.01
    solution, it = Newton(f, 0, 1, 0.01)
    print("Решение методом Ньютона:")
    print(solution)
    print("Количество итераций:", it)
    print()

    result = Iterations(phi, 0, 1, 0.01)
    if result:
        print("Решение методом простых итераций:")
        print(result[0])
        print("Количество итераций:", result[1])
