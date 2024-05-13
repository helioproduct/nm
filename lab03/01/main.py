import numpy as np


from math import log


def f(x):
    return log(x) + x


def get_coefficients(_pl: int, _xi: np.ndarray):
    """
    Определение коэффициентов для множителей базисных полиномов l_i
    :param _pl: индекс базисного полинома
    :param _xi: массив значений x
    :return:
    """
    n = int(_xi.shape[0])
    coefficients = np.empty((n, 2), dtype=float)
    for i in range(n):
        if i == _pl:
            coefficients[i][0] = float("inf")
            coefficients[i][1] = float("inf")
        else:
            coefficients[i][0] = 1 / (_xi[_pl] - _xi[i])
            coefficients[i][1] = -_xi[i] / (_xi[_pl] - _xi[i])
    filtered_coefficients = np.empty((n - 1, 2), dtype=float)
    j = 0
    for i in range(n):
        if coefficients[i][0] != float("inf"):
            # изменение последовательности, степень увеличивается
            filtered_coefficients[j][0] = coefficients[i][1]
            filtered_coefficients[j][1] = coefficients[i][0]
            j += 1
    return filtered_coefficients


def get_polynomial_l(_xi: np.ndarray):
    """
    Определение базисных полиномов
    :param _xi: массив значений x
    :return:
    """
    n = int(_xi.shape[0])
    pli = np.zeros((n, n), dtype=float)
    for pl in range(n):
        coefficients = get_coefficients(pl, _xi)
        for i in range(1, n - 1):  # проходим по массиву coefficients
            if i == 1:  # на второй итерации занимаются 0-2 степени
                pli[pl][0] = coefficients[i - 1][0] * coefficients[i][0]
                pli[pl][1] = (
                    coefficients[i - 1][1] * coefficients[i][0]
                    + coefficients[i][1] * coefficients[i - 1][0]
                )
                pli[pl][2] = coefficients[i - 1][1] * coefficients[i][1]
            else:
                clone_pli = np.zeros(n, dtype=float)
                for val in range(n):
                    clone_pli[val] = pli[pl][val]
                zeros_pli = np.zeros(n, dtype=float)
                for j in range(n - 1):  # проходим по строке pl массива pli
                    product_1 = clone_pli[j] * coefficients[i][0]
                    product_2 = clone_pli[j] * coefficients[i][1]
                    zeros_pli[j] += product_1
                    zeros_pli[j + 1] += product_2
                for val in range(n):
                    pli[pl][val] = zeros_pli[val]
    return pli


def get_polynomial(_xi: np.ndarray, _yi: np.ndarray):
    """
    Определение интерполяционного многочлена Лагранжа
    :param _xi: массив значений x
    :param _yi: массив значений y
    :return:
    """
    n = int(_xi.shape[0])
    polynomial_l = get_polynomial_l(_xi)
    for i in range(n):
        for j in range(n):
            polynomial_l[i][j] *= _yi[i]
    L = np.zeros(n, dtype=float)
    for i in range(n):
        for j in range(n):
            L[i] += polynomial_l[j][i]
    return L


# результат в виде массива коэффициентов многочлена при x в порядке увеличения степени
# [ 0.         -1.47747378  0.          4.8348476   0.        ]
# т.е. результирующий многочлен имеет вид: y(x) = -1.47747378*x + 4.8348476*x^3


if __name__ == "__main__":

    # данные для примера
    # xi = np.array([-1.5, -0.75, 0, 0.75, 1.5])
    # yi = np.array([-14.1014, -0.931596, 0, 0.931596, 14.1014])

    x1 = np.array([0.1, 0.5, 0.9, 1.3])
    y1 = np.array([f(x) for x in x1])

    # L(x) = l0(x) + ... + ln(x)
    # l0 = [(x - x1)(x - x2)] / [(x0 - x1)(x0 -x2)] * y0

    # print(xi)
    # print(yi)
