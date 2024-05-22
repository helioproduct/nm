def f1(x_values, y_values, x_star):
    for i in range(len(x_values) - 1):
        if x_values[i] <= x_star <= x_values[i + 1]:
            break
    else:
        raise ValueError("x_star is out of the interpolation range")

    x_i = x_values[i]
    x_i1 = x_values[i + 1]
    x_i2 = x_values[i + 2]

    y_i = y_values[i]
    y_i1 = y_values[i + 1]
    y_i2 = y_values[i + 2]

    first_term = (y_i1 - y_i) / (x_i1 - x_i)
    second_term = ((y_i2 - y_i1) / (x_i2 - x_i1) - (y_i1 - y_i) / (x_i1 - x_i)) / (
        x_i2 - x_i
    )
    derivative = first_term + second_term * (2 * x_star - x_i - x_i1)

    return derivative


def f2(x_values, y_values, x_star):
    # Locate the interval x_star belongs to
    for i in range(len(x_values) - 2):
        if x_values[i] <= x_star <= x_values[i + 1]:
            break
    else:
        raise ValueError("x_star is out of the interpolation range")

    x_i = x_values[i]
    x_i1 = x_values[i + 1]
    x_i2 = x_values[i + 2]

    y_i = y_values[i]
    y_i1 = y_values[i + 1]
    y_i2 = y_values[i + 2]

    # Calculate the second derivative
    second_derivative = 2 * (
        ((y_i2 - y_i1) / (x_i2 - x_i1) - (y_i1 - y_i) / (x_i1 - x_i)) / (x_i2 - x_i)
    )

    return second_derivative


if __name__ == "__main__":
    x_values = [0.0, 1.0, 2.0, 3.0, 4.0]
    y_values = [0.0, 2.0, 3.4142, 4.7321, 6.0]
    x_star = 2.0

    derivative_at_x_star = f1(x_values, y_values, x_star)
    print("The derivative at x* = 2.0 is:", derivative_at_x_star)

    f2 = f2(x_values, y_values, x_star)
    print("The second derivative at x* = 2.0 is:", f2)
