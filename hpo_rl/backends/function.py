import numpy as np
from typing import Dict, Any, Literal
from hpo_rl.backends.base import EvaluationBackend


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
            "sphere",           # Простая выпуклая (легко)
            "rosenbrock",       # Долина (средне)
            "rastrigin",        # Много локальных минимумов (сложно)
            "ackley",           # Много локальных минимумов + крутые склоны
            "griewank",         # Многоэкстремальная
            "schwefel",         # Очень сложная (обманчивые локальные минимумы)
            "levy",             # Сложная многомодальная
            "michalewicz"       # Крутые пики
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
            self.bounds = (5.0, 7.0)
            self.global_optimum = np.zeros(self.dimensions)
            self.global_optimum_value = 6.0

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
        # Конвертируем в reward (чем меньше функция, тем больше reward)
        if self.maximize:
            reward = value
        else:
            reward = -value

        return reward

    def _sphere(self, x: np.ndarray) -> float:
        """Sphere function: f(x) = sum(x_i^2)"""
        return np.sum((x-6) ** 2)

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
