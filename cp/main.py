import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from scipy.interpolate import UnivariateSpline

def generate_data_with_outliers(n_points=100, n_outliers=10):
    x = np.linspace(0, 2 * np.pi, n_points)
    y = np.sin(x)

    # Добавление нормального шума
    y += np.random.normal(0, 0.1, size=n_points)

    # Генерация выбросов
    outlier_indices = np.random.choice(n_points, n_outliers, replace=False)
    y[outlier_indices] += np.random.uniform(-2, 2, size=n_outliers)

    return x, y

def robust_spline_smoothing(data, smoothing_param):
    # Применяем дискретное косинусное преобразование
    dct_data = dct(data, type=2, norm='ortho')

    # Формируем диагональную матрицу Γ
    n = len(dct_data)
    gamma = np.array([1 / (1 + smoothing_param * (2 - 2 * np.cos(np.pi * i / n))) for i in range(n)])

    smoothed_dct = dct_data * gamma
    smoothed_data = idct(smoothed_dct, type=2, norm='ortho')
    return smoothed_data

def get_robust_spline_function(x, y, smoothing_param):
    smoothed_y = robust_spline_smoothing(y, smoothing_param)
    spline = UnivariateSpline(x, smoothed_y, s=0)
    return spline

if __name__ == "__main__":
    x, y = generate_data_with_outliers()

    smoothing_param = 100

    spline = get_robust_spline_function(x, y, smoothing_param)

    new_x = np.linspace(0, 2 * np.pi, 500)
    interpolated_y = spline(new_x)


    plt.scatter(x, y, label="Данные с выбросами", color="blue", alpha=0.7)
    plt.plot(new_x, interpolated_y, label="Интерполяция сплайна", color="purple", linestyle="--")
    plt.plot(x, np.sin(x), label="Оригинальная функция", color="green", linestyle="--")
    plt.title("Робастное сглаживание и сплайн-интерполяция")
    plt.legend()
    plt.show()
