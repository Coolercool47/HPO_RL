import numpy as np
from typing import Dict, Any, Literal
from hpo_rl.backends.base import EvaluationBackend

# TODO: изменить набор функций так, чтобы сдвиг задавался параметром, а не отдельными функциями, в остальном все +- адекватно


class OptimizationBenchmarkBackend(EvaluationBackend):
    """
    Backend с классическими оптимизационными задачами.

    Особенности:
    - Известные глобальные оптимумы
    - Разная сложность ландшафта (от выпуклых до многоэкстремальных)
    - Разная размерность (2D-20D)
    - Четкие boundary условия (гиперкуб)
    """

    def __init__(
        self,
        function_name: Literal[
            "sphere",           # Простая выпуклая, min at (0, 0)
            "rosenbrock",       # Долина, min at (1, 1)
            "rastrigin",        # Много локальных минимумов, min at (0, 0)
            "ackley",           # Много локальных минимумов, min at (0, 0)
            "griewank",         # Многоэкстремальная, min at (0, 0)
            "schwefel",         # Очень сложная, min at (420.97, 420.97)
            "levy",             # Сложная многомодальная, min at (1, 1)
            "michalewicz",      # Крутые пики
            "booth",            # 2D only, min at (1, 3)
            "beale",            # 2D only, min at (3, 0.5)
            "goldstein_price",  # 2D only, min at (0, -1)
            "shifted_sphere",   # Сдвинутая сфера, min at (2, 2, ...)
            "shifted_rastrigin" # Сдвинутый rastrigin, min at (2.5, 2.5, ...)
        ] = "rastrigin",
        dimensions: int = 6,
        noise_std: float = 0.0,
        maximize: bool = True  # True = максимизируем reward (инвертируем функцию)
    ):
        """
        Args:
            function_name: Название функции для оптимизации
            dimensions: Размерность задачи (количество параметров)
            noise_std: Стандартное отклонение шума (0 = детерминистично)
            maximize: Если True, инвертируем функцию для максимизации reward
        """
        super().__init__()

        self.function_name = function_name
        self.dimensions = dimensions
        self.noise_std = noise_std
        self.maximize = maximize
        print(self.function_name, self.dimensions, self.noise_std, self.maximize)

        self.func_map = {
            "sphere": self._sphere,
            "rosenbrock": self._rosenbrock,
            "rastrigin": self._rastrigin,
            "ackley": self._ackley,
            "griewank": self._griewank,
            "schwefel": self._schwefel,
            "levy": self._levy,
            "michalewicz": self._michalewicz,
            "nondiff": self._nondiff,
            "booth": self._booth,
            "beale": self._beale,
            "goldstein_price": self._goldstein_price,
            "shifted_sphere": self._shifted_sphere,
            "shifted_rastrigin": self._shifted_rastrigin,
        }
        # Определяем границы и оптимум для каждой функции
        self._setup_function()

        print(f"OptimizationBenchmarkBackend: {function_name}")
        print(f"  Dimensions: {dimensions}")
        print(f"  Bounds: {self.bounds}")
        print(f"  Global optimum: {self.global_optimum}")
        print(f"  Global optimum value: {self.global_optimum_value:.6f}")

    def _setup_function(self):
        """Настройка параметров функции"""

        if self.function_name == "sphere":
            # f(x) = sum(x_i^2), min at (0, ..., 0)
            self.bounds = (-5.0, 5.0)  # Исправлено: правильные границы
            self.global_optimum = np.zeros(self.dimensions)
            self.global_optimum_value = 0.0  # Исправлено: правильное значение оптимума

        elif self.function_name == "rosenbrock":
            # f(x) = sum(100*(x_{i+1} - x_i^2)^2 + (1-x_i)^2), min at (1, ..., 1)
            self.bounds = (-1.0, 1.0)
            self.global_optimum = np.ones(self.dimensions)
            self.global_optimum_value = 0.0

        elif self.function_name == "rastrigin":
            # f(x) = 10n + sum(x_i^2 - 10*cos(2*pi*x_i)), min at (0, ..., 0)
            self.bounds = (-5.12, 5.12)
            self.global_optimum = np.zeros(self.dimensions)
            self.global_optimum_value = 0.0

        elif self.function_name == "ackley":
            # Ackley function, min at (0, ..., 0)
            self.bounds = (-32.768, 32.768)
            self.global_optimum = np.zeros(self.dimensions)
            self.global_optimum_value = 0.0

        elif self.function_name == "griewank":
            # Griewank function, min at (0, ..., 0)
            self.bounds = (-600.0, 600.0)
            self.global_optimum = np.zeros(self.dimensions)
            self.global_optimum_value = 0.0

        elif self.function_name == "schwefel":
            # Schwefel function, min at (420.9687, ..., 420.9687)
            self.bounds = (-500.0, 500.0)
            self.global_optimum = np.full(self.dimensions, 420.9687)
            self.global_optimum_value = 0.0

        elif self.function_name == "levy":
            # Levy function, min at (1, ..., 1)
            self.bounds = (-10.0, 10.0)
            self.global_optimum = np.zeros(self.dimensions)
            self.global_optimum_value = 0.0

        elif self.function_name == "michalewicz":
            # Michalewicz function, min varies with dimension
            self.bounds = (0.0, np.pi)
            # Приблизительный оптимум (зависит от размерности)
            self.global_optimum = None  # Нет аналитического решения
            self.global_optimum_value = None  # Неизвестно заранее

        # === Функции с минимумом НЕ в центре (для тестирования) ===
        
        elif self.function_name == "booth":
            # Booth function (2D only): f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
            # min at (1, 3) = 0
            if self.dimensions != 2:
                print(f"booth function is 2D only, setting dimensions=2")
                self.dimensions = 2
            self.bounds = (-10.0, 10.0)
            self.global_optimum = np.array([1.0, 3.0])
            self.global_optimum_value = 0.0

        elif self.function_name == "beale":
            # Beale function (2D only): min at (3, 0.5) = 0
            if self.dimensions != 2:
                print(f"beale function is 2D only, setting dimensions=2")
                self.dimensions = 2
            self.bounds = (-4.5, 4.5)
            self.global_optimum = np.array([3.0, 0.5])
            self.global_optimum_value = 0.0

        elif self.function_name == "goldstein_price":
            # Goldstein-Price function (2D only): min at (0, -1) = 3
            if self.dimensions != 2:
                print(f"goldstein_price function is 2D only, setting dimensions=2")
                self.dimensions = 2
            self.bounds = (-2.0, 2.0)
            self.global_optimum = np.array([0.0, -1.0])
            self.global_optimum_value = 3.0

        elif self.function_name == "shifted_sphere":
            # Shifted Sphere: f(x) = sum((x_i - 2)^2), min at (2, 2, ...)
            self.bounds = (-5.0, 5.0)
            self.global_optimum = np.full(self.dimensions, 2.0)
            self.global_optimum_value = 0.0

        elif self.function_name == "shifted_rastrigin":
            # Shifted Rastrigin: min at (2.5, 2.5, ...)
            self.bounds = (-5.12, 5.12)
            self.global_optimum = np.full(self.dimensions, 2.5)
            self.global_optimum_value = 0.0

    def evaluate(self, config: Dict[str, Any]) -> float:
        """
        Вычисляет значение функции для заданной конфигурации.
        Expected config: {"x0": value, "x1": value, ..., "x{n-1}": value}
        """

        # Собираем вектор из конфигурации
        x = np.array([config[f"x{i}"] for i in range(self.dimensions)])
        # Проверяем границы (на всякий случай)
        if isinstance(self.bounds, tuple):
            x = np.clip(x, self.bounds[0], self.bounds[1])
        # Вычисляем значение функции
        func = self.func_map.get(self.function_name)
        if func is None:
            raise ValueError(f"Unknown function: {self.function_name}")
        value = func(x)
        # Добавляем шум если нужно
        if self.noise_std > 0:
            value += np.random.normal(0, self.noise_std)
        # Конвертируем в reward
        # При maximize=False (минимизация): reward = -value
        # Агент максимизирует reward, поэтому максимизация reward = минимизация функции
        # При maximize=True (максимизация): reward = value
        # Агент максимизирует reward, поэтому максимизация reward = максимизация функции
        if self.maximize:
            reward = value
        else:
            reward = -value

        return reward

    def _sphere(self, x: np.ndarray) -> float:
        """Sphere function: f(x) = sum(x_i^2), minimum at (0, ..., 0)"""
        return np.sum(x ** 2)

    def _rosenbrock(self, x: np.ndarray) -> float:
        """Rosenbrock function (banana valley)"""
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def _rastrigin(self, x: np.ndarray) -> float:
        """Rastrigin function (lots of local minima)"""
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def _ackley(self, x: np.ndarray) -> float:
        """Ackley function"""
        n = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(2 * np.pi * x))

        term1 = -20 * np.exp(-0.2 * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)

        return term1 + term2 + 20 + np.e

    def _griewank(self, x: np.ndarray) -> float:
        """Griewank function"""
        sum_term = np.sum(x ** 2) / 4000
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return sum_term - prod_term + 1

    def _schwefel(self, x: np.ndarray) -> float:
        """Schwefel function (very deceptive)"""
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def _levy(self, x: np.ndarray) -> float:
        """Levy function"""
        w = 1 + (x - 0) / 4

        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)

        return term1 + term2 + term3

    def _michalewicz(self, x: np.ndarray) -> float:
        """Michalewicz function (steep ridges)"""
        m = 10
        i = np.arange(1, len(x) + 1)
        return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * m))

    def _nondiff(self, x: np.ndarray) -> float:
        return (x > 0).sum()

    # === Функции с минимумом НЕ в центре ===

    def _booth(self, x: np.ndarray) -> float:
        """Booth function (2D): f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2, min at (1, 3) = 0"""
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    def _beale(self, x: np.ndarray) -> float:
        """Beale function (2D): min at (3, 0.5) = 0"""
        term1 = (1.5 - x[0] + x[0]*x[1])**2
        term2 = (2.25 - x[0] + x[0]*x[1]**2)**2
        term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
        return term1 + term2 + term3

    def _goldstein_price(self, x: np.ndarray) -> float:
        """Goldstein-Price function (2D): min at (0, -1) = 3"""
        a = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
        b = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
        return a * b

    def _shifted_sphere(self, x: np.ndarray) -> float:
        """Shifted Sphere: f(x) = sum((x_i - 2)^2), min at (2, 2, ...) = 0"""
        shift = 2.0
        return np.sum((x - shift) ** 2)

    def _shifted_rastrigin(self, x: np.ndarray) -> float:
        """Shifted Rastrigin: min at (2.5, 2.5, ...) = 0"""
        shift = 2.5
        x_shifted = x - shift
        n = len(x)
        return 10 * n + np.sum(x_shifted**2 - 10 * np.cos(2 * np.pi * x_shifted))
