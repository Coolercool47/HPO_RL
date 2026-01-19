"""Базовые классы для бэкендов оценки.

Этот модуль содержит абстрактный базовый класс :class:`EvaluationBackend`,
от которого наследуются все конкретные бэкенды.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

CATASTROPHIC_FAILURE_REWARD: float = -1e9
"""Большой штраф при сбое (ошибка обучения, некорректные параметры)."""

# Сделать выгрузку на диск или lru_cache

class EvaluationBackend(ABC):
    """Абстрактный базовый класс для бэкендов оценки конфигураций.

    Бэкенд принимает конфигурацию гиперпараметров и возвращает числовую
    оценку (награду). Поддерживает кэширование результатов.

    Args:
        use_cache: Включить кэширование результатов.

    Attributes:
        maximize: Направление оптимизации, True — максимизация (для метрик, таких как accuracy), False — минимизация (для функций потерь).
        use_cache: Флаг использования кэша.

    Пример::

        class MyBackend(EvaluationBackend):
            def _evaluate(self, config):
                return -sum([v**2 for v in config.values()])

        backend = MyBackend(use_cache=True)
        result = backend.evaluate({"x": 1.0, "y": 2.0})  # -5.0

    Note:
        Подклассы должны реализовать метод :meth:`_evaluate`.
    """

    def __init__(self, use_cache: bool = False):
        """Инициализирует бэкенд.

        Args:
            use_cache: Включить кэширование результатов оценки.
        """
        self.maximize = True
        self.use_cache = use_cache
        self._cache: Dict[Tuple, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self.n = 0 # для 10: 1901 для 1: 3822 для 5: 2138 для 100: 1763

    def evaluate(self, config: Dict[str, Any]) -> float:
        """Оценивает конфигурацию гиперпараметров.

        Если кэширование включено, сначала проверяет кэш, если результат найден, возвращает его.
        Иначе вызывает непосредственно метод для оценки: meth:`_evaluate`.

        Args:
            config: Словарь гиперпараметров ``{"имя": значение, ...}``.

        Returns:
            Числовая оценка конфигурации (награда).
        """
        self.n += 1
        # print(self.n)
        if not self.use_cache:
            return self._evaluate(config)

        key = self._config_to_key(config)
        if key in self._cache:
            self._cache_hits += 1
            # print(self._cache[key])
            return self._cache[key]

        self._cache_misses += 1
        result = self._evaluate(config)
        self._cache[key] = result
        # print(result)
        return result

    @abstractmethod
    def _evaluate(self, config: Dict[str, Any]) -> float:
        """Внутренний метод оценки (реализуется в подклассах).

        Args:
            config: Словарь гиперпараметров.

        Returns:
            Числовая оценка конфигурации.
        """
        pass

    def _config_to_key(self, config: Dict[str, Any]) -> Tuple:
        """Преобразует конфигурацию в хэшируемый ключ для кэша.

        Args:
            config: Словарь гиперпараметров.

        Returns:
            Кортеж пар (ключ, значение), отсортированный по ключам.
        """
        items = []
        for k in sorted(config.keys()):
            v = config[k]
            # Округляем float для устойчивости к погрешностям
            if isinstance(v, float):
                v = round(v, 10)
            items.append((k, v))
        return tuple(items)

    def clear_cache(self) -> None:
        """Очищает кэш и сбрасывает статистику."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Статистика использования кэша.

        Returns:
            Словарь со статистикой:
                - ``hits``: количество попаданий в кэш
                - ``misses``: количество промахов
                - ``size``: текущий размер кэша
                - ``hit_rate``: доля попаданий (0.0 - 1.0)
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "hit_rate": hit_rate
        }
