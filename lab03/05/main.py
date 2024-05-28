from math import sqrt


def f(x):
    return x**2 / (256 - x**4)
    # return x**2 * sqrt(36 - x**2)


def RectangleMethod(f, l, r, h):
    sm = 0
    cur_x = l
    while cur_x < r:
        sm += h * f((cur_x + cur_x + h) * 0.5)
        cur_x += h
    return sm


def TrapezoidMethod(f, l, r, h):
    sm = 0
    cur_x = l
    while cur_x < r:
        sm += h * 0.5 * (f(cur_x + h) + f(cur_x))
        cur_x += h
    return sm


def SimpsonMethod(f, l, r, h):
    return (r - l) / 6 * (f(l) + 4 * f((l + r) / 2) + f(r))


if __name__ == "__main__":
    l, r = 0, 2
    h1, h2 = 0.5, 0.25

    # l, r = 1, 5
    # h1, h2 = 1.0, 0.05

    print("Метод прямоугольников:")
    integral1 = RectangleMethod(f, l, r, h1)
    integral2 = RectangleMethod(f, l, r, h2)
    error_rect = (integral2 - integral1) / 3
    print(f"Шаг = {h1}: I = {integral1}")
    print(f"Шаг = {h2}: I = {integral2}")
    print(f"Погрешность: {error_rect}")
    print()

    print("Метод трапеций")
    integral1 = TrapezoidMethod(f, l, r, h1)
    integral2 = TrapezoidMethod(f, l, r, h2)
    error_trap = (integral2 - integral1) / 3
    print(f"Step = {h1}: I = {integral1}")
    print(f"Step = {h2}: I = {integral2}")
    print(f"Ошибка: {error_trap}")
    print()

    print("Метод Симпсона")
    integral1 = SimpsonMethod(f, l, r, h1)
    integral2 = SimpsonMethod(f, l, r, h2)
    error_simp = (integral2 - integral1) / 15
    print(f"Step = {h1}: integral = {integral1}")
    print(f"Step = {h2}: integral = {integral2}")
    print(f"Ошибка: {error_simp}")
    print()
