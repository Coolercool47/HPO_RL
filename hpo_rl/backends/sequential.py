"""Обёртка для переключения между несколькими бэкендами (рандомизация домена).

:class:`SequentialBackend` хранит список дочерних бэкендов и переключает
активный при каждом reset среды, улучшая обобщение RL-агента.
"""

import random
from typing import Any, Dict, List

from hpo_rl.backends.base import EvaluationBackend

_PROXIED_ATTRS = ("bounds", "dimensions", "global_optimum",
                  "global_optimum_value", "function_name")


def _backend_name(backend: EvaluationBackend) -> str:
    """Возвращает читаемое имя бэкенда."""
    if hasattr(backend, "function_name"):
        return backend.function_name
    if hasattr(backend, "objective_function"):
        fn = backend.objective_function
        return getattr(fn, "__name__", None) or type(backend).__name__
    return type(backend).__name__


class SequentialBackend(EvaluationBackend):
    """Обёртка для переключения между бэкендами при рандомизации домена.

        При каждом ``env.reset()`` активируется другой бэкенд.

        Режимы переключения:
            - ``"random"`` — случайный выбор (по умолчанию)
            - ``"sequential"`` — циклический обход по очереди
            - ``"shuffle"`` — случайная перестановка без повторов в раунде

        Args:
            backends: список дочерних :class:`EvaluationBackend`
            mode: ``"random"`` | ``"sequential"`` | ``"shuffle"``

        Attributes:
            backends: список дочерних бэкендов
            mode: режим переключения
    """

    VALID_MODES = ("random", "sequential", "shuffle")

    def __init__(
        self,
        backends: List[EvaluationBackend],
        mode: str = "random",
    ) -> None:
        """Инициализирует SequentialBackend.

        Args:
            backends: список дочерних бэкендов (не пустой)
            mode: режим переключения
        """
        if not backends:
            raise ValueError("backends list must not be empty")

        maximize_values = {b.maximize for b in backends}
        if len(maximize_values) > 1:
            raise ValueError(
                f"All backends must share the same optimization direction. "
                f"Got maximize={[b.maximize for b in backends]}"
            )
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")

        super().__init__(use_cache=False)

        self.backends = backends
        self.mode = mode
        self._switch_count = 0

        self._shuffle_order: List[int] = []
        self._shuffle_pos: int = 0

        self._locked: bool = False

        self._current_idx = 0
        if mode == "shuffle":
            self._reshuffle()

        self._sync_attributes()

        names = [_backend_name(b) for b in backends]
        print(f"SequentialBackend: {len(backends)} backends ({', '.join(names)}), mode={mode}")

    @property
    def current_backend(self) -> EvaluationBackend:
        """Возвращает активный дочерний бэкенд."""
        return self.backends[self._current_idx]

    def next_backend(self) -> None:
        """Переключает на следующий бэкенд (вызывается из env.reset())."""
        if self._locked:
            return

        self._switch_count += 1

        if self.mode == "sequential":
            self._current_idx = (self._switch_count - 1) % len(self.backends)
        elif self.mode == "shuffle":
            if self._shuffle_pos >= len(self._shuffle_order):
                self._reshuffle()
            self._current_idx = self._shuffle_order[self._shuffle_pos]
            self._shuffle_pos += 1
        else:
            self._current_idx = random.randint(0, len(self.backends) - 1)

        self._sync_attributes()

    def set_active_backend(self, idx: int, lock: bool = True) -> None:
        """Вручную задаёт активный бэкенд по индексу.

        Args:
            idx: индекс бэкенда
            lock: если True, блокирует автоматическое переключение
        """
        if not 0 <= idx < len(self.backends):
            raise IndexError(
                f"Backend index {idx} out of range [0, {len(self.backends)})"
            )
        self._current_idx = idx
        self._locked = lock
        self._sync_attributes()

    def unlock(self) -> None:
        """Снимает блокировку переключения после set_active_backend()."""
        self._locked = False

    def _evaluate(self, config: Dict[str, Any]) -> float:
        """Делегирует оценку текущему бэкенду."""
        return self.current_backend.evaluate(config)

    def clear_cache(self) -> None:
        """Очищает кэш всех дочерних бэкендов."""
        for b in self.backends:
            b.clear_cache()

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Агрегирует статистику кэша всех дочерних бэкендов.

        Returns:
            dict с ключами ``hits``, ``misses``, ``size``, ``hit_rate``.
        """
        total_hits = sum(b.cache_stats["hits"] for b in self.backends)
        total_misses = sum(b.cache_stats["misses"] for b in self.backends)
        total = total_hits + total_misses
        return {
            "hits": total_hits,
            "misses": total_misses,
            "size": sum(b.cache_stats["size"] for b in self.backends),
            "hit_rate": total_hits / total if total > 0 else 0.0,
        }

    @property
    def name(self) -> str:
        """Читаемое имя текущего активного бэкенда."""
        return _backend_name(self.current_backend)

    def __repr__(self) -> str:
        """Строковое представление с именами бэкендов и текущим активным."""
        names = [_backend_name(b) for b in self.backends]
        return (
            f"SequentialBackend(backends=[{', '.join(names)}], "
            f"mode={self.mode!r}, current={names[self._current_idx]})"
        )

    def _sync_attributes(self) -> None:
        """Проксирует атрибуты текущего дочернего бэкенда."""
        cb = self.current_backend
        self.maximize = cb.maximize
        for attr in _PROXIED_ATTRS:
            if hasattr(cb, attr):
                setattr(self, attr, getattr(cb, attr))
            elif hasattr(self, attr):
                delattr(self, attr)

    def _reshuffle(self) -> None:
        """Генерирует новую случайную перестановку индексов бэкендов."""
        order = list(range(len(self.backends)))
        random.shuffle(order)
        if len(order) > 1 and order[0] == self._current_idx:
            swap_idx = random.randint(1, len(order) - 1)
            order[0], order[swap_idx] = order[swap_idx], order[0]
        self._shuffle_order = order
        self._shuffle_pos = 0
