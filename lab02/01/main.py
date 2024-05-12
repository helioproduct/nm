from newton import Newton
from iterations import iterations

from functions import f, phi


if __name__ == "__main__":
    # f(x) = 0
    solution, iterations = Newton(f, 0, 1, 0.0001)
    print("Решение методом Ньютона:")
    print(solution)
    print("Количество итераций:")
    print(iterations)

    result = iterations(phi, 0, 1, 0.01)
    if result:
        print("Решение методом простых итераций:")
        print(result[0])
        print("Количество итераций:")
        print(result[1])
