import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # для 3D-графиков
from math import cos, sin, pi, ceil
from abc import ABC, abstractmethod

MAX_ITERS = 10000

def norm(A: np.ndarray, B: np.ndarray) -> float:
    """
    Евклидова норма разности двух матриц (A-B).
    """
    assert A.shape == B.shape, "Матрицы должны быть одинакового размера"
    diff = A - B
    return np.sqrt(np.sum(diff ** 2))


class ElipticEquation(ABC):
    """
    Интерфейс (абстрактный класс) для эллиптического уравнения с граничными условиями.
    """
    @abstractmethod
    def Phi_1(self, y: float):
        """ ГУ при x=0. """
        pass

    @abstractmethod
    def Phi_2(self, y: float):
        """ ГУ при x=l1. """
        pass
    
    @abstractmethod
    def Phi_3(self, x: float):
        """ ГУ при y=0. """
        pass
    
    @abstractmethod
    def Phi_4(self, x: float):
        """ ГУ при y=l2. """
        pass

    @abstractmethod
    def U(self, x: float, y: float):
        """ Аналитическое решение (если известно). """
        pass

    @abstractmethod
    def f(self, x, y):
        """
        Правая часть для уравнения ∆u - c*u = f(x,y).
        Если уравнение однородное (∆u - c*u=0), то f=0.
        """
        pass

    @property
    @abstractmethod
    def l1(self):
        """ Размер области по оси x (0..l1). """
        pass

    @property
    @abstractmethod
    def l2(self):
        """ Размер области по оси y (0..l2). """
        pass

    @property
    @abstractmethod
    def c(self):
        """ Уравнение: ∆u - c*u=0. """
        pass

    # Параметры a,b часто не нужны в конкретной задаче, поэтому опустим или оставим пустыми.
    # @property
    # @abstractmethod
    # def a(self):
    #     pass
    #
    # @property
    # @abstractmethod
    # def b(self):
    #     pass


class ElipticEquationVariant7(ElipticEquation):
    """
    Пример задачи: 
        d^2u/dx^2 + d^2u/dy^2 - 2u = 0
    на области x in [0, pi/2], y in [0, pi/2].
    
    Граничные условия:
        u(0, y) = cos(y)
        u(pi/2, y) = 0
        u(x, 0) = cos(x)
        u(x, pi/2) = 0

    Аналитическое решение:
        U(x,y) = cos(x)*cos(y).
    """

    def Phi_1(self, y: float):
        # u(0,y) = cos(0)*cos(y)=1*cos(y)=cos(y)
        return cos(y)

    def Phi_2(self, y: float):
        # u(pi/2, y)=0
        return 0

    def Phi_3(self, x: float):
        # u(x,0)=cos(x)*cos(0)=cos(x)
        return cos(x)

    def Phi_4(self, x: float):
        # u(x, pi/2)=0
        return 0

    def U(self, x: float, y: float):
        return cos(x)*cos(y)

    def f(self, x, y):
        # ∆u - 2u=0 => f=0
        return 0

    @property
    def l1(self):
        return pi/2  # x in [0, pi/2]

    @property
    def l2(self):
        return pi/2  # y in [0, pi/2]

    @property
    def c(self):
        return 2     # ∆u - 2u=0 => c=2


#--------------------------------------------------------------------
# 2. АНАЛИТИЧЕСКИЙ "РЕШАТЕЛЬ" (просто табулирует U(x,y) на сетке)
#--------------------------------------------------------------------

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
    Создаёт 1D-сетки (x и y) исходя из шага delta и длины [0, l1], [0, l2].
    Возвращает x, y, hx, hy.
    """
    Nx = ceil(equation.l1 / delta)
    Ny = ceil(equation.l2 / delta)
    hx = equation.l1 / (Nx - 1)
    hy = equation.l2 / (Ny - 1)
    x = np.arange(0, equation.l1 + hx/2, hx)
    y = np.arange(0, equation.l2 + hy/2, hy)
    return x, y, hx, hy


class Real(Solver):
    """
    "Решатель", который просто строит аналитическое решение на сетке.
    """
    def __init__(self, equation: ElipticEquation, delta: float = 0.05) -> None:
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


#--------------------------------------------------------------------
# 3. Итерационные методы (Якоби, Зейдель, Релаксация)
#--------------------------------------------------------------------

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

    def __str__(self) -> str:
        return "Approximation method"


class Iterations(ApproximationMethod):
    """
    Метод простых итераций (Якоби) для уравнения ∆u - 2u=0
    => -1/hx^2*(u[i+1,j]+u[i-1,j]) + -1/hy^2*(u[i,j+1]+u[i,j-1]) / divisor
    где divisor = 2 - 2/hx^2 - 2/hy^2
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
            (-1/self.hx**2) * (u_prev[i+1,j] + u_prev[i-1,j])
            + (-1/self.hy**2) * (u_prev[i,j+1] + u_prev[i,j-1])
        )
        denominator = 2 - 2/self.hx**2 - 2/self.hy**2
        return numerator / denominator
    
    def __str__(self):
        return "IterationsMethod (Якоби)"


class SeidelMethod(ApproximationMethod):
    """
    Метод Зейделя (Gauss–Seidel).
    Отличается от Якоби тем, что u[i-1,j] и u[i,j-1]
    берём уже обновлёнными (из u), а "вперёд" по i+1,j
    всё ещё из u_prev.
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
            (-1/self.hx**2)*(u_prev[i+1,j] + u[i-1,j]) +
            (-1/self.hy**2)*(u_prev[i,j+1] + u[i,j-1])
        )
        denominator = 2 - 2/self.hx**2 - 2/self.hy**2
        return numerator / denominator
    
    def __str__(self):
        return "SeidelMethod (Гаусс–Зейдель)"


class RelaxationMethod(ApproximationMethod):
    """
    Метод Верхней Релаксации (SOR).
    new_val = theta * (Jacobi_formula) + (1 - theta)*u_prev[i,j]
    """
    def __init__(self, hx, hy, theta: float = 1.0) -> None:
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
        # Сама "Якоби"-формула без релаксации:
        numerator = (
            (-1/self.hx**2)*(u_prev[i+1,j] + u[i-1,j])
            + (-1/self.hy**2)*(u_prev[i,j+1] + u[i,j-1])
        )
        denominator = 2 - 2/self.hx**2 - 2/self.hy**2
        jacobi_val = numerator / denominator
        
        # Верхняя релаксация:
        return self._theta*jacobi_val + (1 - self._theta)*u_prev[i,j]
    
    def __str__(self):
        return f"RelaxationMethod (theta={self._theta:.2f})"


#--------------------------------------------------------------------
# 4. Численный решатель (NumericSolverVariant7)
#--------------------------------------------------------------------

class NumericSolver(Solver):
    """
    - Строит сетку,
    - Инициализирует u граничными условиями,
    - На каждой итерации вызывает method.update(...),
    - Проверяет сходимость,
    - По желанию (return_history=True) возвращает историю max ошибки |u - u_analytic|.
    """
    def __init__(self, equation: ElipticEquation, method: ApproximationMethod,
                 delta: float = 0.05, epsilon: float = 1e-3) -> None:
        self._equation = equation
        self._method = method
        self._delta = delta
        self._epsilon = epsilon

        self.x, self.y, self.hx, self.hy = get_ranges(self.equation, self._delta)
        self.nx, self.ny = len(self.x), len(self.y)

    @property
    def equation(self) -> ElipticEquation:
        return self._equation

    def _init(self):
        """
        Инициализирует массив u, задаёт граничные значения.
        """
        u = np.zeros((self.nx, self.ny))

        # x=0 => i=0
        for j in range(self.ny):
            u[0, j] = self.equation.Phi_1(self.y[j])

        # x=l1 => i=nx-1
        for j in range(self.ny):
            u[self.nx-1, j] = self.equation.Phi_2(self.y[j])

        # y=0 => j=0
        for i in range(self.nx):
            u[i, 0] = self.equation.Phi_3(self.x[i])

        # y=l2 => j=ny-1
        for i in range(self.nx):
            u[i, self.ny-1] = self.equation.Phi_4(self.x[i])

        return u

    def solve(self, return_history=False):
        """
        Если return_history=True, возвращаем (u, history_max_err),
        иначе просто u.
        """
        u = self._init()

        # Аналитика (для подсчёта ошибки)
        uA = np.zeros((self.nx, self.ny))
        for i in range(self.nx):
            for j in range(self.ny):
                uA[i,j] = self.equation.U(self.x[i], self.y[j])

        history_max_err = []
        count_iterations = 0

        while True:
            count_iterations += 1
            u_prev = u.copy()

            # Обновляем внутренние точки
            for i in range(1, self.nx - 1):
                for j in range(1, self.ny - 1):
                    u[i,j] = self._method.update(u, u_prev, i, j)

            # Критерий сходимости
            if norm(u, u_prev) < self._epsilon:
                # добавим финальную ошибку в history (если надо)
                curr_err = np.abs(u - uA)
                history_max_err.append(np.max(curr_err))
                break

            if count_iterations > MAX_ITERS:
                raise StopIteration("Превышен лимит итераций")

            # собираем max|u-uA| на текущей итерации
            if return_history:
                curr_err = np.abs(u - uA)
                history_max_err.append(np.max(curr_err))

        # Вывод итогов
        final_error = np.max(np.abs(u - uA))
        print(f"{self._method}: итераций={count_iterations}, финальная max ошибка={final_error:.6g}")

        if return_history:
            return u, history_max_err
        return u


if __name__ == "__main__":
    eq = ElipticEquationVariant7()
    delta = 0.05     # шаг сетки
    epsilon = 1e-5   # точность
    theta = 1.5      # для метода SOR

    # ============ АНАЛИТИКА ============
    anal_solver = Real(eq, delta=delta)
    xA, yA, uA = anal_solver.solve()  # uA.shape=(len(xA), len(yA))

    # ============ Методы ==============
    iteration_method = Iterations(hx=delta, hy=delta)
    seidel_method    = SeidelMethod(hx=delta,   hy=delta)
    relax_method     = RelaxationMethod(hx=delta, hy=delta, theta=theta)

    solver_iter  = NumericSolver(eq, iteration_method, delta=delta, epsilon=epsilon)
    solver_seid  = NumericSolver(eq, seidel_method,    delta=delta, epsilon=epsilon)
    solver_relax = NumericSolver(eq, relax_method,     delta=delta, epsilon=epsilon)

    u_iter,  hist_iter  = solver_iter.solve(return_history=True)
    u_seid,  hist_seid  = solver_seid.solve(return_history=True)
    u_relax, hist_relax = solver_relax.solve(return_history=True)

    X, Y = np.meshgrid(xA, yA)  # (ny,nx)

    # (nx,ny) => (ny,nx)
    def to_2d(u_2d):
        """ Транспонируем, чтобы Z.shape=(ny,nx). """
        return u_2d.T

    # 1) Аналитика
    figA = plt.figure()
    axA = figA.add_subplot(111, projection='3d')
    surfA = axA.plot_surface(X, Y, to_2d(uA), cmap='viridis')
    axA.set_title("Аналитическое решение")
    axA.set_xlabel("x")
    axA.set_ylabel("y")
    axA.set_zlabel("u(x,y)")
    figA.colorbar(surfA, shrink=0.6)
    plt.show()

    # 2) Метод Якоби
    figI = plt.figure()
    axI = figI.add_subplot(111, projection='3d')
    surfI = axI.plot_surface(X, Y, to_2d(u_iter), cmap='plasma')
    axI.set_title("Численное решение (Метод Итераций)")
    axI.set_xlabel("x")
    axI.set_ylabel("y")
    axI.set_zlabel("u(x,y)")
    figI.colorbar(surfI, shrink=0.6)
    plt.show()


    error_relax = np.abs(u_iter - uA)
    figE = plt.figure()
    axE = figE.add_subplot(111, projection='3d')
    surfE = axE.plot_surface(X, Y, to_2d(error_relax), cmap='hot')
    axE.set_title(f"Ошибка (Метод Протых итераций)")
    axE.set_xlabel("x")
    axE.set_ylabel("y")
    axE.set_zlabel("error")
    figE.colorbar(surfE, shrink=0.6)
    plt.show()



    # 3) Метод Зейделя
    figS = plt.figure()
    axS = figS.add_subplot(111, projection='3d')
    surfS = axS.plot_surface(X, Y, to_2d(u_seid), cmap='coolwarm')
    axS.set_title("Численное решение (Метод Зейделя)")
    axS.set_xlabel("x")
    axS.set_ylabel("y")
    axS.set_zlabel("u(x,y)")
    figS.colorbar(surfS, shrink=0.6)
    plt.show()

    # график ошибки
    error_relax = np.abs(u_seid - uA)
    figE = plt.figure()
    axE = figE.add_subplot(111, projection='3d')
    surfE = axE.plot_surface(X, Y, to_2d(error_relax), cmap='hot')
    axE.set_title(f"Ошибка (Метод Зейделя)")
    axE.set_xlabel("x")
    axE.set_ylabel("y")
    axE.set_zlabel("error")
    figE.colorbar(surfE, shrink=0.6)
    plt.show()



    # 4) Метод Релаксации
    figR = plt.figure()
    axR = figR.add_subplot(111, projection='3d')
    surfR = axR.plot_surface(X, Y, to_2d(u_relax), cmap='inferno')
    axR.set_title(f"Численное решение (Метод Релаксации, theta={theta})")
    axR.set_xlabel("x")
    axR.set_ylabel("y")
    axR.set_zlabel("u(x,y)")
    figR.colorbar(surfR, shrink=0.6)
    plt.show()

    # 5) (Опционально) 3D-график ошибки для метода релаксации
    error_relax = np.abs(u_relax - uA)
    figE = plt.figure()
    axE = figE.add_subplot(111, projection='3d')
    surfE = axE.plot_surface(X, Y, to_2d(error_relax), cmap='hot')
    axE.set_title(f"Ошибка (Метод Релаксации, theta={theta})")
    axE.set_xlabel("x")
    axE.set_ylabel("y")
    axE.set_zlabel("error")
    figE.colorbar(surfE, shrink=0.6)
    plt.show()



