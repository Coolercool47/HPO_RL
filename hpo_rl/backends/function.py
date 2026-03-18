"""Бэкенд с классическими тестовыми функциями оптимизации.

Модуль содержит :class:`OptimizationBenchmarkBackend` — бэкенд для оценки
конфигураций на основе известных математических функций с известными
глобальными оптимумами. Подходит для бенчмаркинга и тестирования
RL-алгоритмов оптимизации гиперпараметров.
"""

import numpy as np
from typing import Dict, Any, Literal
import hashlib
from hpo_rl.backends.base import EvaluationBackend


class OptimizationBenchmarkBackend(EvaluationBackend):
    """Бэкенд с классическими оптимизационными функциями.

    Предоставляет набор тестовых функций для бенчмаркинга алгоритмов
    оптимизации. Все функции имеют известные глобальные оптимумы и
    различную сложность ландшафта.

    Поддерживаемые многомерные функции:
        - ``sphere`` — простая унимодальная функция
        - ``rosenbrock`` — сложная унимодальная функция с узкой долиной
        - ``rastrigin`` — многоэкстремальная функция с множеством локальных минимумов
        - ``ackley`` — многоэкстремальная функция с экспоненциальным затуханием
        - ``griewank`` — многоэкстремальная функция
        - ``schwefel`` — многоэкстремальная функция с большим числом локальных минимумов
        - ``levy`` — сложная унимодальная функция с плоскими участками
        - ``michalewicz`` — функция с крутыми пиками (оптимум зависит от размерности)

    Поддерживаемые двухмерные функции:
        - ``booth`` — простая унимодальная функция
        - ``beale`` — унимодальная функция с узкой долиной
        - ``goldstein_price`` — многоэкстремальная функция

    Args:
        function_name: Название функции из :attr:`FUNCTIONS`.
        dimensions: Размерность пространства поиска.
        noise_std: Стандартное отклонение гауссова шума (0 = детерминированная).
        maximize: True для максимизации, False для минимизации.
        use_cache: Включить кэширование результатов (автоматически отключается при шуме).

    Attributes:
        function_name: Название используемой функции.
        dimensions: Размерность пространства поиска.
        bounds: Границы пространства поиска ``(min, max)``.
        global_optimum: Координаты глобального оптимума.
        global_optimum_value: Значение функции в глобальном оптимуме.
        noise_std: Стандартное отклонение шума.
        maximize: Направление оптимизации.

    Пример::

        # Минимизация функции Rastrigin в 2D
        backend = OptimizationBenchmarkBackend("rastrigin", dimensions=2, maximize=False)
        reward = backend.evaluate({"x0": 0.0, "x1": 0.0})  # Близко к оптимуму

        # С шумом (кэш автоматически отключён)
        backend = OptimizationBenchmarkBackend("sphere", dimensions=5, noise_std=0.1)

    Note:
        Функции ``booth``, ``beale``, ``goldstein_price`` работают только в 2D.
        При указании других размерностей автоматически устанавливается ``dimensions=2``.
    """

    FUNCTIONS = Literal[
        "sphere", "rosenbrock", "rastrigin", "ackley", "griewank",
        "schwefel", "levy", "michalewicz", "booth", "beale",
        "goldstein_price", "bukin_n6", "cross_in_tray", "drop_wave",
        "eggholder", "holder_table", "schaffer_n2", "schaffer_n4",
        "shubert", "dejong_n5", "easom", "levy_n13", "langermann"
    ]

    def __init__(
        self,
        function_name: FUNCTIONS = "rastrigin",
        dimensions: int = 2,
        noise_std: float = 0.0,
        maximize: bool = False,
        use_cache: bool = True
    ):
        """Инициализирует OptimizationBenchmarkBackend.

        Args:
            function_name: Название функции из :attr:`FUNCTIONS`.
            dimensions: Размерность пространства поиска.
            noise_std: Стандартное отклонение гауссова шума (0 = детерминированная).
            maximize: True для максимизации, False для минимизации.
            use_cache: Включить кэширование результатов (автоматически отключается при шуме).
        """
        # Кэш имеет смысл только без шума
        super().__init__(use_cache=(use_cache and noise_std == 0))
        self.function_name = function_name
        self.dimensions = dimensions
        self.noise_std = noise_std
        self.maximize = maximize

        # Уникальный сид для этого экземпляра бэкенда.
        # Позволяет генерировать разные поверхности шума в разных запусках алгоритма.
        self.instance_seed = np.random.randint(0, 2**31 - 1)

        self.func_map = {
            "sphere": self._sphere,
            "rosenbrock": self._rosenbrock,
            "rastrigin": self._rastrigin,
            "ackley": self._ackley,
            "griewank": self._griewank,
            "schwefel": self._schwefel,
            "levy": self._levy,
            "michalewicz": self._michalewicz,
            "booth": self._booth,
            "beale": self._beale,
            "goldstein_price": self._goldstein_price,
            "bukin_n6": self._bukin_n6,
            "cross_in_tray": self._cross_in_tray,
            "drop_wave": self._drop_wave,
            "eggholder": self._eggholder,
            "holder_table": self._holder_table,
            "schaffer_n2": self._schaffer_n2,
            "schaffer_n4": self._schaffer_n4,
            "shubert": self._shubert,
            "dejong_n5": self._dejong_n5,
            "easom": self._easom,
            "levy_n13": self._levy_n13,
            "langermann": self._langermann,
        }

        self._setup_function()
        print(f"{function_name}: dims={dimensions}, bounds={self.bounds}, opt={self.global_optimum_value:.6f}")

    def _setup_function(self) -> None:
        """Настраивает границы, оптимум и значение оптимума для выбранной функции.

        Автоматически определяет конфигурацию функции на основе её названия.
        Для 2D-функций автоматически устанавливает ``dimensions=2``.
        """
        d = self.dimensions

        configs = {
            "sphere":           ((-5.0, 5.0),    np.zeros(d), 0.0),
            "rosenbrock":       ((-1.0, 1.0),    np.ones(d), 0.0),
            "rastrigin":        ((-5.12, 5.12),  np.zeros(d),           0.0),
            "ackley":           ((-32.768, 32.768), np.zeros(d),        0.0),
            "griewank":         ((-600.0, 600.0), np.zeros(d),          0.0),
            "schwefel":         ((-500.0, 500.0), np.full(d, 420.9687), 0.0),
            "levy":             ((-10.0, 10.0),  np.zeros(d),           0.0),
            "michalewicz":      ((0.0, np.pi), None, 0.0),
        }

        # 2D-only функции
        if self.function_name in ("booth", "beale", "goldstein_price", "bukin_n6", "cross_in_tray", "drop_wave", "eggholder", "holder_table", "schaffer_n2", "schaffer_n4", "shubert", "dejong_n5", "easom", "levy_n13", "langermann"):
            if d != 2:
                print(f"{self.function_name} только для 2D, принимается dimensions=2")
                self.dimensions = 2

        configs_2d = {
            "booth":           ((-10.0, 10.0), np.array([1.0, 3.0]),  0.0),
            "beale":           ((-4.5, 4.5),   np.array([3.0, 0.5]),  0.0),
            "goldstein_price": ((-2.0, 2.0),   np.array([0.0, -1.0]), 3.0),
            "bukin_n6":        ((-15.0, 3.0),  np.array([-10.0, 1.0]), 0.0),
            "cross_in_tray":   ((-10.0, 10.0), np.array([1.34941, 1.34941]), -2.06261),
            "drop_wave":       ((-5.12, 5.12), np.array([0.0, 0.0]), -1.0),
            "eggholder":       ((-512.0, 512.0), np.array([512.0, 404.2319]), -959.6407),
            "holder_table":    ((-10.0, 10.0), np.array([8.05502, 9.66459]), -19.2085),
            "schaffer_n2":     ((-100.0, 100.0), np.array([0.0, 0.0]), 0.0),
            "schaffer_n4":     ((-100.0, 100.0), np.array([0.0, 1.25313]), 0.292579),
            "shubert":         ((-10.0, 10.0), None, -186.7309),
            "dejong_n5":       ((-65.536, 65.536), np.array([-32.0, -32.0]), 0.998004),
            "easom":           ((-100.0, 100.0), np.array([np.pi, np.pi]), -1.0),
            "levy_n13":        ((-10.0, 10.0), np.array([1.0, 1.0]), 0.0),
            "langermann":      ((0.0, 10.0), np.array([9.681, 4.774]), -1.493),
        }

        if self.function_name in configs_2d:
            self.bounds, self.global_optimum, self.global_optimum_value = configs_2d[self.function_name]
        elif self.function_name in configs:
            self.bounds, self.global_optimum, self.global_optimum_value = configs[self.function_name]
        else:
            raise ValueError(f"Неизвестная функция: {self.function_name}")

        self.hp_space = {
            f"x{i}": {"values": [float(self.bounds[0]), float(self.bounds[1])], "type": "float", "log": False}
            for i in range(self.dimensions)
        }

    def _evaluate(self, config: Dict[str, Any]) -> float:
        """Вычисляет значение тестовой функции для заданной конфигурации.

        Извлекает координаты из конфигурации (``x0``, ``x1``, ...), применяет
        функцию и добавляет шум (если задан). Всегда возвращает сырое значение
        функции; ответственность за интерпретацию направления оптимизации
        (``maximize``) лежит на потребителе (RL-среда, baseline и т.д.).

        Args:
            config: Словарь с координатами ``{"x0": val, "x1": val, ...}``.

        Returns:
            Сырое значение тестовой функции (с шумом, если задан).

        """
        x = np.array([config[f"x{i}"] for i in range(self.dimensions)])

        func = self.func_map.get(self.function_name)
        if func is None:
            raise ValueError(f"Неизвестная функция: {self.function_name}")

        value = func(x)

        if self.noise_std > 0:
            x_rounded = np.round(x, 8)
            # Смешиваем уникальный сид инстанса (чтобы поверхность отличалась в разных запусках) 
            # и координаты точки (чтобы зафиксировать шум в конкретной точке для графика,
            # и мы рисовали ровно то же, что "видел" алгоритм во время работы).
            h = hashlib.md5(x_rounded.tobytes() + str(self.instance_seed).encode('utf-8'))
            local_seed = int(h.hexdigest(), 16) % (2**32)
            rng = np.random.RandomState(local_seed)
            value += rng.normal(0, self.noise_std)

        return max(value, 0)

    def _sphere(self, x: np.ndarray) -> float:
        """Сфера (Sphere function).

        Простая унимодальная функция. Минимум в ``(0, ..., 0)`` со значением ``0``.

        Args:
            x: Вектор координат.

        Returns:
            Значение функции:

            .. math::

                f(x) = \\sum_{i=1}^{d} x_i^2

            где :math:`d` — размерность.
        """
        return np.sum(x ** 2)

    def _rosenbrock(self, x: np.ndarray) -> float:
        """Функция Розенброка (Rosenbrock function).

        Сложная унимодальная функция с узкой долиной. Минимум в ``(1, ..., 1)``
        со значением ``0``.

        Args:
            x: Вектор координат.

        Returns:
            Значение функции:

            .. math::

                f(x) = \\sum_{i=1}^{d-1} [100(x_{i+1} - x_i^2)^2 + (1 - x_i)^2]

            где :math:`d` — размерность.
        """
        return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

    def _rastrigin(self, x: np.ndarray) -> float:
        """Функция Растригина (Rastrigin function).

        Многоэкстремальная функция с множеством локальных минимумов.
        Минимум в ``(0, ..., 0)`` со значением ``0``. Сложна для глобальной
        оптимизации из-за большого количества локальных оптимумов.

        Args:
            x: Вектор координат.

        Returns:
            Значение функции:

            .. math::

                f(x) = 10d + \\sum_{i=1}^{d} [x_i^2 - 10\\cos(2\\pi x_i)]

            где :math:`d` — размерность.
        """
        n = len(x)
        return 10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))

    def _ackley(self, x: np.ndarray) -> float:
        """Функция Акли (Ackley function).

        Многоэкстремальная функция с множеством локальных минимумов.
        Минимум в ``(0, ..., 0)`` со значением ``0``. Характеризуется
        экспоненциальным затуханием и периодическими компонентами.

        Args:
            x: Вектор координат.

        Returns:
            Значение функции:

            .. math::

                f(x) = -20\\exp\\left(-0.2\\sqrt{\\frac{1}{d}\\sum_{i=1}^{d} x_i^2}\\right) - \\exp\\left(\\frac{1}{d}\\sum_{i=1}^{d} \\cos(2\\pi x_i)\\right) + 20 + e

            где :math:`d` — размерность, :math:`e` — число Эйлера.
        """
        n = len(x)
        s1 = np.sum(x ** 2)
        s2 = np.sum(np.cos(2 * np.pi * x))
        return -20 * np.exp(-0.2 * np.sqrt(s1 / n)) - np.exp(s2 / n) + 20 + np.e

    def _griewank(self, x: np.ndarray) -> float:
        """Функция Гриванка (Griewank function).

        Многоэкстремальная функция с произведением косинусов, создающим
        множество локальных минимумов. Минимум в ``(0, ..., 0)`` со значением ``0``.

        Args:
            x: Вектор координат.

        Returns:
            Значение функции:

            .. math::

                f(x) = 1 + \\frac{1}{4000}\\sum_{i=1}^{d} x_i^2 - \\prod_{i=1}^{d} \\cos\\left(\\frac{x_i}{\\sqrt{i}}\\right)

            где :math:`d` — размерность.
        """
        s = np.sum(x ** 2) / 4000
        p = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return s - p + 1

    def _schwefel(self, x: np.ndarray) -> float:
        """Функция Швефеля (Schwefel function).

        Очень обманчивая многоэкстремальная функция. Глобальный минимум
        в ``(420.9687, ..., 420.9687)`` со значением ``0``. Локальные минимумы
        находятся далеко от глобального, что затрудняет оптимизацию.

        Args:
            x: Вектор координат.

        Returns:
            Значение функции:

            .. math::

                f(x) = 418.9829d - \\sum_{i=1}^{d} x_i \\sin(\\sqrt{|x_i|})

            где :math:`d` — размерность.
        """
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def _levy(self, x: np.ndarray) -> float:
        """Функция Леви (Levy function).

        Сложная унимодальная функция с множеством плоских участков.
        Минимум в ``(1, ..., 1)`` со значением ``0``. Требует адаптивных
        алгоритмов оптимизации.

        Args:
            x: Вектор координат.

        Returns:
            Значение функции Леви:

            .. math::

                f(x) = \\sin^2(\\pi w_1) + \\sum_{i=1}^{d-1} (w_i - 1)^2 [1 + 10\\sin^2(\\pi w_i + 1)] + (w_d - 1)^2 [1 + \\sin^2(2\\pi w_d)]

            где :math:`w_i = 1 + (x_i - 1)/4`.
        """
        w = 1 + (x - 1) / 4
        t1 = np.sin(np.pi * w[0]) ** 2
        t2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
        t3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        return t1 + t2 + t3

    def _michalewicz(self, x: np.ndarray) -> float:
        """Функция Михалевича (Michalewicz function).

        Функция с крутыми пиками и множеством локальных минимумов.
        Глобальный оптимум зависит от размерности. Используется для
        тестирования алгоритмов на сложных ландшафтах.

        Args:
            x: Вектор координат.

        Returns:
            Значение функции:

            .. math::

                f(x) = -\\sum_{i=1}^{d} \\sin(x_i) \\left[\\sin\\left(\\frac{i x_i^2}{\\pi}\\right)\\right]^{2m}

            где :math:`d` — размерность, :math:`m = 10`.
        """
        m = 10
        i = np.arange(1, len(x) + 1)
        return -np.sum(np.sin(x) * np.sin(i * x**2 / np.pi) ** (2 * m))

    def _booth(self, x: np.ndarray) -> float:
        """Функция Бута (Booth function) — только 2D.

        Простая унимодальная функция. Минимум в ``(1, 3)`` со значением ``0``.

        Args:
            x: Вектор из 2 координат ``[x0, x1]``.

        Returns:
            Значение функции:

            .. math::

                f(x_0, x_1) = (x_0 + 2x_1 - 7)^2 + (2x_0 + x_1 - 5)^2
        """
        return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

    def _beale(self, x: np.ndarray) -> float:
        """Функция Била (Beale function) — только 2D.

        Унимодальная функция с узкой долиной. Минимум в ``(3, 0.5)``
        со значением ``0``.

        Args:
            x: Вектор из 2 координат ``[x0, x1]``.

        Returns:
            Значение функции Била:

            .. math::

                f(x_0, x_1) = (1.5 - x_0 + x_0 x_1)^2 + (2.25 - x_0 + x_0 x_1^2)^2 + (2.625 - x_0 + x_0 x_1^3)^2
        """
        t1 = (1.5 - x[0] + x[0]*x[1])**2
        t2 = (2.25 - x[0] + x[0]*x[1]**2)**2
        t3 = (2.625 - x[0] + x[0]*x[1]**3)**2
        return t1 + t2 + t3

    def _goldstein_price(self, x: np.ndarray) -> float:
        """Функция Гольдштейна-Прайса (Goldstein-Price function) — только 2D.

        Многоэкстремальная функция с несколькими локальными минимумами.
        Глобальный минимум в ``(0, -1)`` со значением ``3``.

        Args:
            x: Вектор из 2 координат ``[x0, x1]``.

        Returns:
            Значение функции Гольдштейна-Прайса:

            .. math::

                f(x_0, x_1) = a(x_0, x_1) \\cdot b(x_0, x_1)

            где:

            .. math::

                a(x_0, x_1) = 1 + (x_0 + x_1 + 1)^2 (19 - 14x_0 + 3x_0^2 - 14x_1 + 6x_0 x_1 + 3x_1^2)

            .. math::

                b(x_0, x_1) = 30 + (2x_0 - 3x_1)^2 (18 - 32x_0 + 12x_0^2 + 48x_1 - 36x_0 x_1 + 27x_1^2)
        """
        a = 1 + (x[0] + x[1] + 1)**2 * (19 - 14*x[0] + 3*x[0]**2 - 14*x[1] + 6*x[0]*x[1] + 3*x[1]**2)
        b = 30 + (2*x[0] - 3*x[1])**2 * (18 - 32*x[0] + 12*x[0]**2 + 48*x[1] - 36*x[0]*x[1] + 27*x[1]**2)
        return a * b

    def _bukin_n6(self, x: np.ndarray) -> float:
        """Bukin Function N. 6 — только 2D."""
        term1 = 100 * np.sqrt(np.abs(x[1] - 0.01 * x[0]**2))
        term2 = 0.01 * np.abs(x[0] + 10)
        return term1 + term2

    def _cross_in_tray(self, x: np.ndarray) -> float:
        """Cross-in-Tray Function — только 2D."""
        fact1 = np.sin(x[0]) * np.sin(x[1])
        fact2 = np.exp(np.abs(100 - np.sqrt(x[0]**2 + x[1]**2) / np.pi))
        return -0.0001 * (np.abs(fact1 * fact2) + 1)**0.1

    def _drop_wave(self, x: np.ndarray) -> float:
        """Drop-Wave Function — только 2D."""
        r = np.sqrt(x[0]**2 + x[1]**2)
        num = 1 + np.cos(12 * r)
        den = 0.5 * (r**2) + 2
        return -num / den

    def _eggholder(self, x: np.ndarray) -> float:
        """Eggholder Function — только 2D."""
        term1 = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0] / 2 + 47)))
        term2 = -x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
        return term1 + term2

    def _holder_table(self, x: np.ndarray) -> float:
        """Holder Table Function — только 2D."""
        fact1 = np.sin(x[0]) * np.cos(x[1])
        fact2 = np.exp(np.abs(1 - np.sqrt(x[0]**2 + x[1]**2) / np.pi))
        return -np.abs(fact1 * fact2)

    def _schaffer_n2(self, x: np.ndarray) -> float:
        """Schaffer Function N. 2 — только 2D."""
        num = (np.sin(x[0]**2 - x[1]**2))**2 - 0.5
        den = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
        return 0.5 + num / den

    def _schaffer_n4(self, x: np.ndarray) -> float:
        """Schaffer Function N. 4 — только 2D."""
        num = (np.cos(np.sin(np.abs(x[0]**2 - x[1]**2))))**2 - 0.5
        den = (1 + 0.001 * (x[0]**2 + x[1]**2))**2
        return 0.5 + num / den

    def _shubert(self, x: np.ndarray) -> float:
        """Shubert Function — только 2D."""
        sum1 = sum(i * np.cos((i + 1) * x[0] + i) for i in range(1, 6))
        sum2 = sum(i * np.cos((i + 1) * x[1] + i) for i in range(1, 6))
        return sum1 * sum2

    def _dejong_n5(self, x: np.ndarray) -> float:
        """De Jong Function N. 5 — только 2D."""
        A1 = np.array([-32, -16, 0, 16, 32] * 5)
        A2 = np.array([[-32]*5, [-16]*5, [0]*5, [16]*5, [32]*5]).flatten()
        i = np.arange(1, 26)
        term = 1.0 / (i + (x[0] - A1)**6 + (x[1] - A2)**6)
        return 1.0 / (0.002 + np.sum(term))

    def _easom(self, x: np.ndarray) -> float:
        """Easom Function — только 2D."""
        fact1 = -np.cos(x[0]) * np.cos(x[1])
        fact2 = np.exp(-((x[0] - np.pi)**2 + (x[1] - np.pi)**2))
        return fact1 * fact2

    def _levy_n13(self, x: np.ndarray) -> float:
        """Levy Function N. 13 — только 2D."""
        term1 = np.sin(3 * np.pi * x[0])**2
        term2 = (x[0] - 1)**2 * (1 + np.sin(3 * np.pi * x[1])**2)
        term3 = (x[1] - 1)**2 * (1 + np.sin(2 * np.pi * x[1])**2)
        return term1 + term2 + term3

    def _langermann(self, x: np.ndarray) -> float:
        """Langermann Function — только 2D."""
        c = np.array([1, 2, 5, 2, 3])
        A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
        dist_sq = (x[0] - A[:, 0])**2 + (x[1] - A[:, 1])**2
        return np.sum(c * np.exp(-dist_sq / np.pi) * np.cos(np.pi * dist_sq))
