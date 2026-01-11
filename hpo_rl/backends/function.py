import numpy as np
from typing import Dict, Any, Literal
from hpo_rl.backends.base import EvaluationBackend


class OptimizationBenchmarkBackend(EvaluationBackend):
    """
    Backend с классическими оптимизационными функциями.
    Известные оптимумы, разная сложность ландшафта (2D-20D).
    """

    FUNCTIONS = Literal[
        "sphere", "rosenbrock", "rastrigin", "ackley", "griewank",
        "schwefel", "levy", "michalewicz", "booth", "beale",
        "goldstein_price", "shifted_sphere", "shifted_rastrigin"
    ]

    def __init__(
        self,
        function_name: FUNCTIONS = "rastrigin",
        dimensions: int = 6,
        noise_std: float = 0.0,
        maximize: bool = True,
        use_cache: bool = True
    ):
        # Кэш имеет смысл только без шума
        super().__init__(use_cache=(use_cache and noise_std == 0))
        self.function_name = function_name
        self.dimensions = dimensions
        self.noise_std = noise_std
        self.maximize = maximize

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

        self._setup_function()
        print(f"{function_name}: dims={dimensions}, bounds={self.bounds}, opt={self.global_optimum_value:.6f}")

    def _setup_function(self):
        """Настройка границ и оптимума для функции."""
        d = self.dimensions

        configs = {
            "sphere":           ((-5.0, 5.0),    np.zeros(d),           0.0),
            "rosenbrock":       ((-1.0, 1.0),    np.ones(d),            0.0),
            "rastrigin":        ((-5.12, 5.12),  np.zeros(d),           0.0),
            "ackley":           ((-32.768, 32.768), np.zeros(d),        0.0),
            "griewank":         ((-600.0, 600.0), np.zeros(d),          0.0),
            "schwefel":         ((-500.0, 500.0), np.full(d, 420.9687), 0.0),
            "levy":             ((-10.0, 10.0),  np.zeros(d),           0.0),
            "michalewicz":      ((0.0, np.pi),   None,                  None),
            "shifted_sphere":   ((-5.0, 5.0),    np.full(d, 2.0),       0.0),
            "shifted_rastrigin":((-5.12, 5.12),  np.full(d, 2.5),       0.0),
        }

        # 2D-only функции
        if self.function_name in ("booth", "beale", "goldstein_price"):
            if d != 2:
                print(f"{self.function_name} только для 2D, меняем dimensions=2")
                self.dimensions = 2

        configs_2d = {
            "booth":           ((-10.0, 10.0), np.array([1.0, 3.0]),  0.0),
            "beale":           ((-4.5, 4.5),   np.array([3.0, 0.5]),  0.0),
            "goldstein_price": ((-2.0, 2.0),   np.array([0.0, -1.0]), 3.0),
        }

        if self.function_name in configs_2d:
            self.bounds, self.global_optimum, self.global_optimum_value = configs_2d[self.function_name]
        elif self.function_name in configs:
            self.bounds, self.global_optimum, self.global_optimum_value = configs[self.function_name]
        else:
            raise ValueError(f"Неизвестная функция: {self.function_name}")

    def _evaluate(self, config: Dict[str, Any]) -> float:
        """Вычисляет значение функции. config: {"x0": val, "x1": val, ...}"""
        x = np.array([config[f"x{i}"] for i in range(self.dimensions)])

        if isinstance(self.bounds, tuple):
            x = np.clip(x, self.bounds[0], self.bounds[1])

        func = self.func_map.get(self.function_name)
        if func is None:
            raise ValueError(f"Неизвестная функция: {self.function_name}")

        value = func(x)

        if self.noise_std > 0:
            value += np.random.normal(0, self.noise_std)

        return value if self.maximize else -value # плохое решение в общем случае, для RL корректно, где надо максимизировать reward, а для бейзлайнов где идет минимизация нет

    def _sphere(self, x):
        """min at (0, ..., 0) = 0"""
        return np.sum(x ** 2)

    def _rosenbrock(self, x):
        """min at (1, ..., 1) = 0"""
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def _rastrigin(self, x):
        """min at (0, ..., 0) = 0, много локальных минимумов"""
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def _ackley(self, x):
        """min at (0, ..., 0) = 0"""
        n = len(x)
        s1 = np.sum(x ** 2)
        s2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20 + np.e

    def _griewank(self, x):
        """min at (0, ..., 0) = 0"""
        s = np.sum(x ** 2) / 4000
        p = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return s - p + 1

    def _schwefel(self, x):
        """min at (420.97, ...) = 0, очень обманчивая"""
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def _levy(self, x):
        """min at (1, ..., 1) = 0"""
        w = 1 + (x - 0) / 4
        t1 = np.sin(np.pi * w[0]) ** 2
        t2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        t3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        return t1 + t2 + t3

    def _michalewicz(self, x):
        """крутые пики, оптимум зависит от размерности"""
        m = 10
        i = np.arange(1, len(x) + 1)
        return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * m))

    def _nondiff(self, x):
        return (x > 0).sum()

    def _booth(self, x):
        """min at (1, 3) = 0"""
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    def _beale(self, x):
        """min at (3, 0.5) = 0"""
        t1 = (1.5 - x[0] + x[0]*x[1])**2
        t2 = (2.25 - x[0] + x[0]*x[1]**2)**2
        t3 = (2.625 - x[0] + x[0]*x[1]**3)**2
        return t1 + t2 + t3

    def _goldstein_price(self, x):
        """min at (0, -1) = 3"""
        a = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
        b = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
        return a * b

    def _shifted_sphere(self, x):
        """min at (2, 2, ...) = 0"""
        return np.sum((x - 2.0) ** 2)

    def _shifted_rastrigin(self, x):
        """min at (2.5, 2.5, ...) = 0"""
        xs = x - 2.5
        n = len(x)
        return 10 * n + np.sum(xs**2 - 10 * np.cos(2 * np.pi * xs))
