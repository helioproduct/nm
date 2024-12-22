import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # для построения 3D графиков
from math import sin, pi, ceil
from abc import ABC, abstractmethod

MAX_ITERS = 100000

def norm(A: np.ndarray, B: np.ndarray) -> float:
    """
    Евклидова норма (L2) разности двух матриц.
    """
    assert A.shape == B.shape, "Матрицы должны быть одинакового размера"
    diff = A - B
    return np.sqrt(np.sum(diff**2))

# -------------------------------------------------------------------
# 1. Класс задачи: PDE = d^2u/dx^2 + d^2u/dy^2 + u = 0
#    Область: x in [0, pi/2], y in [0, 1]
#    ГУ: u(0,y)=0, u(pi/2,y)=y, u(x,0)=0, u(x,1)=sin(x)
#    Аналитика: U(x,y) = y sin x
# -------------------------------------------------------------------

class ElipticEquation(ABC):
    """
    Абстрактный класс-описание эллиптического уравнения и граничных условий.
    """
    @abstractmethod
    def Phi_1(self, y: float):
        """ ГУ при x=0 """
        pass

    @abstractmethod
    def Phi_2(self, y: float):
        """ ГУ при x = l1 (здесь pi/2) """
        pass
    
    @abstractmethod
    def Phi_3(self, x: float):
        """ ГУ при y=0 """
        pass
    
    @abstractmethod
    def Phi_4(self, x: float):
        """ ГУ при y = l2 (здесь 1) """
        pass

    @abstractmethod
    def U(self, x: float, y: float):
        """ Аналитическое решение (если известно). """
        pass

    @abstractmethod
    def f(self, x: float, y: float):
        """
        Правая часть в уравнении: ∆u - c*u = f(x,y).
        Здесь c=-1 => ∆u + u=0 => f=0
        """
        pass

    @property
    @abstractmethod
    def l1(self):
        """ Левая/правая граница по x: [0, l1]. """
        pass

    @property
    @abstractmethod
    def l2(self):
        """ Нижняя/верхняя граница по y: [0, l2]. """
        pass

    @property
    @abstractmethod
    def c(self):
        """
        Уравнение в виде: ∆u - c*u = 0.
        Если у нас ∆u + u=0 => -c=+1 => c=-1.
        """
        pass

class ElipticEquationUser(ElipticEquation):
    """
    d^2u/dx^2 + d^2u/dy^2 = -u  =>  ∆u + u=0
    x in [0, pi/2], y in [0, 1]

    ГУ:
      u(0,y)=0
      u(pi/2,y)=y
      u(x,0)=0
      u(x,1)=sin(x)

    Аналитика: U(x,y) = y*sin(x).
    """

    def Phi_1(self, y: float):
        # u(0,y)=0
        return 0

    def Phi_2(self, y: float):
        # u(pi/2,y)= y
        return y

    def Phi_3(self, x: float):
        # u(x,0)=0
        return 0

    def Phi_4(self, x: float):
        # u(x,1)= sin(x)
        return np.sin(x)

    def U(self, x: float, y: float):
        # Аналитическое решение
        return y * np.sin(x)

    def f(self, x: float, y: float):
        # ∆u + u=0 => f=0
        return 0

    @property
    def l1(self):
        # x in [0, pi/2]
        return pi/2

    @property
    def l2(self):
        # y in [0, 1]
        return 1

    @property
    def c(self):
        # ∆u - c*u=0 => c=-1 => ∆u + u=0
        return -1

# -------------------------------------------------------------------
# 2. Генерация сеток + Аналитический решатель (табулируем U).
# -------------------------------------------------------------------

class Solver(ABC):
    @property
    @abstractmethod
    def equation(self) -> ElipticEquation:
        pass

    @abstractmethod
    def solve(self):
        pass

def get_ranges(equation: ElipticEquation, delta: float):
    """
    Создаём 1D-сетки x и y с шагом delta, 
    покрываем [0..l1], [0..l2].
    Возвращаем x, y, hx, hy.
    """
    Nx = ceil(equation.l1 / delta)
    Ny = ceil(equation.l2 / delta)
    hx = equation.l1 / (Nx - 1)
    hy = equation.l2 / (Ny - 1)
    x = np.arange(0, equation.l1 + hx/2, hx)
    y = np.arange(0, equation.l2 + hy/2, hy)
    return x, y, hx, hy

class AnalyticSolver(Solver):
    """
    "Решатель", который просто вычисляет y*sin(x) 
    на сетке [0..pi/2]x[0..1].
    """
    def __init__(self, equation: ElipticEquation, delta=0.05):
        self._equation = equation
        self._delta = delta

    @property
    def equation(self) -> ElipticEquation:
        return self._equation

    def solve(self):
        x, y, hx, hy = get_ranges(self.equation, self._delta)
        u = np.zeros((len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                u[i,j] = self.equation.U(x[i], y[j])
        return x, y, u

# -------------------------------------------------------------------
# 3. Итерационные методы: Якоби, Зейдель, Релаксация (SOR)
#    для PDE: ∆u + u=0 => (2/hx^2 + 2/hy^2 + 1)*u_ij = ...
# -------------------------------------------------------------------

class ApproximationMethod(ABC):
    """
    Базовый класс для итерационных схем (шаблон).
    """
    @property
    @abstractmethod
    def hx(self):
        pass
    
    @property
    @abstractmethod
    def hy(self):
        pass

    @abstractmethod
    def update(self, u: np.ndarray, u_prev: np.ndarray, i: int, j: int):
        pass


class JacobiMethod(ApproximationMethod):
    """
    Метод простых итераций (Якоби) для d^2u/dx^2 + d^2u/dy^2 + u=0.
    Формула:
      (2/hx^2 + 2/hy^2 + 1)*u[i,j] = (u[i+1,j] + u[i-1,j])/hx^2 + (u[i,j+1] + u[i,j-1])/hy^2
    Используем все значения из "u_prev".
    """
    def __init__(self, hx, hy):
        self._hx = hx
        self._hy = hy

    @property
    def hx(self):
        return self._hx
    
    @property
    def hy(self):
        return self._hy

    def update(self, u: np.ndarray, u_prev: np.ndarray, i: int, j: int):
        numerator = (
            (u_prev[i+1,j] + u_prev[i-1,j]) / (self.hx**2)
          + (u_prev[i,j+1] + u_prev[i,j-1]) / (self.hy**2)
        )
        denominator = 2.0/(self.hx**2) + 2.0/(self.hy**2) + 1.0
        return numerator / denominator

    def __str__(self):
        return "JacobiMethod"


class SeidelMethod(ApproximationMethod):
    """
    Метод Зейделя (Gauss–Seidel) для ∆u + u=0.
    Отличие: u[i-1,j], u[i,j-1] берём из "u" (уже обновлено), 
    а i+1,j, i,j+1 — из "u_prev".
    """
    def __init__(self, hx, hy):
        self._hx = hx
        self._hy = hy

    @property
    def hx(self):
        return self._hx
    
    @property
    def hy(self):
        return self._hy

    def update(self, u: np.ndarray, u_prev: np.ndarray, i: int, j: int):
        numerator = (
            (u_prev[i+1,j] + u[i-1,j]) / (self.hx**2)
          + (u_prev[i,j+1] + u[i,j-1]) / (self.hy**2)
        )
        denominator = 2.0/(self.hx**2) + 2.0/(self.hy**2) + 1.0
        return numerator / denominator

    def __str__(self):
        return "SeidelMethod"


class RelaxationMethod(ApproximationMethod):
    """
    Метод верхней релаксации (SOR) для ∆u + u=0:
      new_val = Jacobi_formula
      u[i,j] = theta*new_val + (1 - theta)*u_prev[i,j]
    """
    def __init__(self, hx, hy, theta: float = 1.0):
        self._hx = hx
        self._hy = hy
        self._theta = theta

    @property
    def hx(self):
        return self._hx
    
    @property
    def hy(self):
        return self._hy

    def update(self, u: np.ndarray, u_prev: np.ndarray, i: int, j: int):
        numerator = (
            (u_prev[i+1,j] + u[i-1,j]) / (self.hx**2)
          + (u_prev[i,j+1] + u[i,j-1]) / (self.hy**2)
        )
        denominator = 2.0/(self.hx**2) + 2.0/(self.hy**2) + 1.0
        jacobi_val = numerator / denominator
        
        return self._theta * jacobi_val + (1 - self._theta) * u_prev[i,j]

    def __str__(self):
        return f"RelaxationMethod(theta={self._theta:.2f})"

# -------------------------------------------------------------------
# 4. Численный решатель: строим сетку, итерируем, проверяем сходимость
# -------------------------------------------------------------------

class NumericSolver(Solver):
    """
    - Строит сетку (x,y),
    - Инициализирует u граничными условиями,
    - На каждой итерации вызывает method.update(...) для внутренних узлов,
    - Проверяет \|u - u_prev\| < epsilon.
    - (Опционально) собирает "макс. ошибку" на каждой итерации.
    """
    def __init__(self, equation: ElipticEquation, method: ApproximationMethod,
                 delta=0.05, epsilon=1e-3):
        self._equation = equation
        self._method   = method
        self._delta    = delta
        self._epsilon  = epsilon

        self.x, self.y, self.hx, self.hy = get_ranges(equation, self._delta)
        self.nx, self.ny = len(self.x), len(self.y)

    @property
    def equation(self) -> ElipticEquation:
        return self._equation

    def _init_u(self):
        """
        Заполняем массив u граничными условиями (Дирихле).
        """
        u = np.zeros((self.nx, self.ny))

        # x=0 -> i=0
        for j in range(self.ny):
            u[0,j] = self.equation.Phi_1(self.y[j])

        # x=l1 -> i=nx-1
        for j in range(self.ny):
            u[self.nx-1,j] = self.equation.Phi_2(self.y[j])

        # y=0 -> j=0
        for i in range(self.nx):
            u[i,0] = self.equation.Phi_3(self.x[i])

        # y=l2 -> j=ny-1
        for i in range(self.nx):
            u[i,self.ny-1] = self.equation.Phi_4(self.x[i])

        return u

    def solve(self, return_history=False):
        """
        Если return_history=True, возвращает (u, history_max_error).
        Иначе возвращает просто u.
        """
        u = self._init_u()

        # Аналитика (для оценки ошибки)
        uA = np.zeros_like(u)
        for i in range(self.nx):
            for j in range(self.ny):
                uA[i,j] = self.equation.U(self.x[i], self.y[j])

        history_max_err = []
        count_iters = 0

        while True:
            count_iters += 1
            u_prev = u.copy()

            # Обновляем внутренние узлы
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    u[i,j] = self._method.update(u, u_prev, i, j)

            # Проверяем \|u - u_prev\|
            diff = norm(u, u_prev)
            if diff < self._epsilon:
                # добавим финальную ошибку в историю, если нужно
                curr_err = np.max(np.abs(u - uA))
                history_max_err.append(curr_err)
                break

            if count_iters > MAX_ITERS:
                raise StopIteration("Превышено максимальное число итераций")

            # Сохраняем макс. ошибку |u - u_analytic|
            if return_history:
                curr_err = np.max(np.abs(u - uA))
                history_max_err.append(curr_err)

        # Выводим итог:
        final_err = np.max(np.abs(u - uA))
        print(f"{self._method} завершился за {count_iters} итераций, финальная max ошибка={final_err:.3e}")

        if return_history:
            return u, history_max_err
        else:
            return u


# -------------------------------------------------------------------
# 5. "main": Решаем PDE тремя методами, строим графики
# -------------------------------------------------------------------

if __name__ == "__main__":
    # --- PDE: d^2u/dx^2 + d^2u/dy^2 + u=0, x in [0, pi/2], y in [0,1] ---
    eq = ElipticEquationUser()

    # Параметры сетки и точность
    delta   = 0.01
    epsilon = 1e-4
    theta   = 1.5  # для SOR

    # === Аналитическое решение ===
    anal_solver = AnalyticSolver(eq, delta=delta)
    xA, yA, uA = anal_solver.solve()  # uA.shape=(nx, ny)

    # === Три итерационных метода ===
    method_jacobi = JacobiMethod(hx=delta, hy=delta)
    method_seidel = SeidelMethod(hx=delta, hy=delta)
    method_relax  = RelaxationMethod(hx=delta, hy=delta, theta=theta)

    # === "Решатели" ===
    solver_jacobi = NumericSolver(eq, method_jacobi, delta=delta, epsilon=epsilon)
    solver_seidel = NumericSolver(eq, method_seidel, delta=delta, epsilon=epsilon)
    solver_relax  = NumericSolver(eq, method_relax,  delta=delta, epsilon=epsilon)

    # Получаем численные решения + историю ошибки
    u_jacobi, hist_jacobi  = solver_jacobi.solve(return_history=True)
    u_seidel, hist_seidel  = solver_seidel.solve(return_history=True)
    u_relax,  hist_relax   = solver_relax.solve(return_history=True)

    # ------------------ 5.1 График сходимости ------------------
    plt.figure(figsize=(7,4))
    plt.plot(hist_jacobi, label="Jacobi")
    plt.plot(hist_seidel, label="Seidel")
    plt.plot(hist_relax,  label=f"Relax (theta={theta})")
    plt.title("Сходимость: max|u - u_analytic| от итерации")
    plt.yscale("log")
    plt.xlabel("итерация")
    plt.ylabel("макс. ошибка")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ------------------ 5.2 3D-графики решений ------------------
    # Подготовим 2D-сетки X,Y через meshgrid
    X, Y = np.meshgrid(xA, yA)  
    #  у нас: xA.shape=(nx,), yA.shape=(ny,) => X.shape=(ny,nx), Y.shape=(ny,nx).

    # Функция для "транспонирования" решения (nx,ny)->(ny,nx)
    def as_surface(u_2d):
        return u_2d.T  # чтобы shape совпало с (ny,nx)

    # 1) Аналитика
    figA = plt.figure()
    axA = figA.add_subplot(111, projection='3d')
    surfA = axA.plot_surface(X, Y, as_surface(uA), cmap='viridis')
    axA.set_title("Аналитическое решение: u(x,y)=y*sin(x)")
    axA.set_xlabel("x")
    axA.set_ylabel("y")
    axA.set_zlabel("u")
    figA.colorbar(surfA, shrink=0.5, aspect=10)
    plt.show()

    # 2) Якоби
    figJ = plt.figure()
    axJ = figJ.add_subplot(111, projection='3d')
    surfJ = axJ.plot_surface(X, Y, as_surface(u_jacobi), cmap='plasma')
    axJ.set_title("Численное решение (Якоби)")
    axJ.set_xlabel("x")
    axJ.set_ylabel("y")
    axJ.set_zlabel("u")
    figJ.colorbar(surfJ, shrink=0.5, aspect=10)
    plt.show()

    # 3) Зейдель
    figS = plt.figure()
    axS = figS.add_subplot(111, projection='3d')
    surfS = axS.plot_surface(X, Y, as_surface(u_seidel), cmap='coolwarm')
    axS.set_title("Численное решение (Зейдель)")
    axS.set_xlabel("x")
    axS.set_ylabel("y")
    axS.set_zlabel("u")
    figS.colorbar(surfS, shrink=0.5, aspect=10)
    plt.show()

    # 4) Релаксация
    figR = plt.figure()
    axR = figR.add_subplot(111, projection='3d')
    surfR = axR.plot_surface(X, Y, as_surface(u_relax), cmap='inferno')
    axR.set_title(f"Численное решение (SOR, theta={theta})")
    axR.set_xlabel("x")
    axR.set_ylabel("y")
    axR.set_zlabel("u")
    figR.colorbar(surfR, shrink=0.5, aspect=10)
    plt.show()

    # ------------------ 5.3 3D-график ошибки (например, метод Релаксации) ------------------
    error_relax = np.abs(u_relax - uA)
    figE = plt.figure()
    axE = figE.add_subplot(111, projection='3d')
    surfE = axE.plot_surface(X, Y, as_surface(error_relax), cmap='hot')
    axE.set_title(f"Ошибка (метод Релаксации, theta={theta})")
    axE.set_xlabel("x")
    axE.set_ylabel("y")
    axE.set_zlabel("|u - u_analytic|")
    figE.colorbar(surfE, shrink=0.5, aspect=10)
    plt.show()
