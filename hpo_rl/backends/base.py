from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

CATASTROPHIC_FAILURE_REWARD: float = -1e9


class EvaluationBackend(ABC):
    """Базовый класс для бэкендов оценки конфигураций."""

    def __init__(self, use_cache: bool = False):
        self.maximize = True  # True = чем выше, тем лучше
        self.use_cache = use_cache
        self._cache: Dict[Tuple, float] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def evaluate(self, config: Dict[str, Any]) -> float:
        """Оценивает конфигурацию (с кэшированием если включено)."""
        if not self.use_cache:
            return self._evaluate(config)

        key = self._config_to_key(config)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        self._cache_misses += 1
        result = self._evaluate(config)
        self._cache[key] = result
        return result

    @abstractmethod
    def _evaluate(self, config: Dict[str, Any]) -> float:
        """Реализация оценки конфигурации (переопределяется в наследниках)."""
        pass

    def _config_to_key(self, config: Dict[str, Any]) -> Tuple:
        """Конвертирует конфиг в hashable ключ для кэша."""
        items = []
        for k in sorted(config.keys()):
            v = config[k]
            # Округляем float для устойчивости к погрешностям
            if isinstance(v, float):
                v = round(v, 10)
            items.append((k, v))
        return tuple(items)

    def clear_cache(self):
        """Очищает кэш."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Статистика кэша."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
            "hit_rate": hit_rate
        }
