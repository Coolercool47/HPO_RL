"""Мульти-бэкенд для обучения на нескольких задачах одновременно.

Модуль содержит :class:`SequentialBackend` — обёртку, которая хранит
список дочерних бэкендов любого типа (функции, objective, real) и
переключает активный бэкенд между эпизодами. Это позволяет RL-агенту
обучаться на разных ландшафтах оптимизации, улучшая обобщающую способность.
"""

import random as _random
from typing import Dict, Any, List, Optional, Sequence
from hpo_rl.backends.base import EvaluationBackend


def _backend_name(b: EvaluationBackend) -> str:
    """Возвращает человекочитаемое имя бэкенда для логирования."""
    if hasattr(b, 'function_name'):
        return b.function_name
    if hasattr(b, 'objective_function'):
        fn = b.objective_function
        return getattr(fn, '__name__', None) or getattr(fn, '__qualname__', type(b).__name__)
    return type(b).__name__


class SequentialBackend(EvaluationBackend):
    """Бэкенд-обёртка, переключающий дочерние бэкенды между эпизодами.

    Содержит список бэкендов любого типа (:class:`OptimizationBenchmarkBackend`,
    :class:`ObjectiveBackend`, :class:`RealTrainingBackend` и т.д.) и при каждом
    вызове :meth:`next_backend` переключается на следующий. Среда вызывает
    ``backend.next_backend()`` автоматически при ``reset()``.

    Все атрибуты (``maximize``, ``bounds``, ``dimensions`` и т.д.)
    проксируются к текущему активному бэкенду.

    Поддерживает три режима переключения:
        - ``"random"`` — случайный выбор из списка (по умолчанию)
        - ``"sequential"`` — строго по порядку
        - ``"shuffle"`` — случайная перестановка всех бэкендов, затем
          проход по ней; когда перестановка исчерпана — новая перестановка.
          Гарантирует, что каждый бэкенд встретится ровно 1 раз за раунд.

    Переключение вызывается контроллером раз в эпоху через
    ``periodic_train_hook``.

    Args:
        backends: Список экземпляров :class:`EvaluationBackend`.
        mode: Режим переключения: ``"random"`` или ``"sequential"``.

    Raises:
        ValueError: Если список бэкендов пуст.
        ValueError: Если бэкенды имеют разные значения ``maximize``.

    Пример::

        from hpo_rl.backends import OptimizationBenchmarkBackend, ObjectiveBackend, SequentialBackend

        backends = [
            OptimizationBenchmarkBackend("sphere", dimensions=2),
            OptimizationBenchmarkBackend("rastrigin", dimensions=2),
            ObjectiveBackend(objective_function=my_func, hp_space=hp),
        ]
        # Случайный бэкенд каждую эпоху:
        backend = SequentialBackend(backends)
    """

    def __init__(
        self,
        backends: List[EvaluationBackend],
        mode: str = "random",
        merged_bounds: Optional[Dict[str, tuple]] = None,
    ):
        if not backends:
            raise ValueError("Список бэкендов не может быть пустым")

        # Проверяем, что все бэкенды оптимизируют в одну сторону
        maximize_values = {b.maximize for b in backends}
        if len(maximize_values) > 1:
            raise ValueError(
                "Все бэкенды должны иметь одинаковое направление оптимизации (maximize). "
                f"Получено: {[b.maximize for b in backends]}"
            )

        if mode not in ("sequential", "random", "shuffle"):
            raise ValueError(f"mode должен быть 'sequential', 'random' или 'shuffle', получено '{mode}'")

        # Не вызываем super().__init__() с use_cache, т.к. кэширование
        # делегируется дочерним бэкендам
        super().__init__(use_cache=False)

        self.backends = backends
        self.mode = mode
        self._current_idx = 0
        self._switch_count = 0

        # Shuffle mode: случайная перестановка индексов, обновляется каждый раунд
        self._shuffle_order: List[int] = []
        self._shuffle_pos: int = 0
        if mode == "shuffle":
            self._reshuffle()

        # Merged bounds для ремаппинга значений из env к дочерним бэкендам.
        # Env работает в merged (максимальном) диапазоне, но каждый дочерний
        # бэкенд может иметь свой диапазон. При evaluate() значения линейно
        # пересчитываются из merged bounds в bounds текущего бэкенда.
        self._merged_bounds = merged_bounds  # {"x0": (lo, hi), "x1": (lo, hi), ...}

        # Для каждого бэкенда сохраняем его собственные bounds
        self._child_bounds: List[Optional[Dict[str, tuple]]] = []
        for b in backends:
            if hasattr(b, 'bounds') and hasattr(b, 'dimensions'):
                self._child_bounds.append(
                    {f"x{i}": b.bounds for i in range(b.dimensions)}
                )
            else:
                self._child_bounds.append(None)

        # Проксируем атрибуты от первого (текущего) бэкенда
        self._sync_attributes()

        names = [_backend_name(b) for b in backends]
        print(f"SequentialBackend: {len(backends)} backends ({', '.join(names)}), mode={mode}, switch every epoch")

    @property
    def current_backend(self) -> EvaluationBackend:
        """Текущий активный бэкенд."""
        return self.backends[self._current_idx]

    def _sync_attributes(self) -> None:
        """Синхронизирует атрибуты обёртки с текущим активным бэкендом.

        Проксирует ``maximize``, ``bounds``, ``dimensions``,
        ``global_optimum``, ``global_optimum_value``, ``function_name``
        от текущего дочернего бэкенда.
        """
        cb = self.current_backend
        self.maximize = cb.maximize

        # Атрибуты OptimizationBenchmarkBackend (если есть)
        for attr in ('bounds', 'dimensions', 'global_optimum',
                     'global_optimum_value', 'function_name'):
            if hasattr(cb, attr):
                setattr(self, attr, getattr(cb, attr))

    def _reshuffle(self) -> None:
        """Создаёт новую случайную перестановку индексов бэкендов.

        Используется в режиме ``"shuffle"``: после исчерпания текущей
        перестановки генерируется новая, чтобы каждый бэкенд встретился
        ровно 1 раз за раунд.
        """
        self._shuffle_order = list(range(len(self.backends)))
        _random.shuffle(self._shuffle_order)
        self._shuffle_pos = 0
        names = [_backend_name(self.backends[i]) for i in self._shuffle_order]
        print(f"[SequentialBackend] New shuffle order: {', '.join(names)}")

    def next_backend(self) -> None:
        """Переключает на следующий бэкенд.

        Вызывается контроллером раз в эпоху через ``periodic_train_hook``.
        Каждый вызов — гарантированное переключение.
        """
        self._switch_count += 1

        if self.mode == "sequential":
            self._current_idx = self._switch_count % len(self.backends)
        elif self.mode == "shuffle":
            self._current_idx = self._shuffle_order[self._shuffle_pos]
            self._shuffle_pos += 1
            if self._shuffle_pos >= len(self._shuffle_order):
                self._reshuffle()
        else:  # random
            self._current_idx = _random.randint(0, len(self.backends) - 1)

        self._sync_attributes()
        print(f"[SequentialBackend] Epoch {self._switch_count}: switched to '{self.name}'")

    def set_active_backend(self, idx: int) -> None:
        """Явно переключает активный бэкенд по индексу.

        Используется для inference — позволяет прогнать агента
        на конкретном дочернем бэкенде.

        Args:
            idx: Индекс дочернего бэкенда в ``self.backends``.
        """
        if not 0 <= idx < len(self.backends):
            raise IndexError(f"Backend index {idx} out of range [0, {len(self.backends)})")
        self._current_idx = idx
        self._sync_attributes()
        print(f"[SequentialBackend] Manually switched to '{self.name}' (idx={idx})")

    def _remap_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Ремапит значения из merged bounds в bounds текущего дочернего бэкенда.

        Env создаёт bins по merged (максимальному) диапазону. Если дочерний
        бэкенд имеет более узкий диапазон, значения линейно пересчитываются:
        ``merged [lo_m, hi_m] → child [lo_c, hi_c]``.

        Если merged_bounds не заданы или дочерний бэкенд не имеет bounds —
        значения передаются как есть.
        """
        child_bounds = self._child_bounds[self._current_idx]
        if self._merged_bounds is None or child_bounds is None:
            return config

        remapped = {}
        for key, val in config.items():
            if key in self._merged_bounds and key in child_bounds:
                m_lo, m_hi = self._merged_bounds[key]
                c_lo, c_hi = child_bounds[key]
                if m_hi - m_lo > 1e-12 and (m_lo != c_lo or m_hi != c_hi):
                    # Линейный ремап: normalized → child bounds
                    t = (val - m_lo) / (m_hi - m_lo)  # [0, 1]
                    remapped[key] = c_lo + t * (c_hi - c_lo)
                else:
                    remapped[key] = val
            else:
                remapped[key] = val
        return remapped

    def _evaluate(self, config: Dict[str, Any]) -> float:
        """Делегирует оценку текущему активному бэкенду.

        Перед оценкой ремапит значения из merged bounds env
        в bounds текущего дочернего бэкенда.

        Args:
            config: Словарь гиперпараметров ``{"x0": val, "x1": val, ...}``.

        Returns:
            Значение от текущего активного бэкенда.
        """
        remapped = self._remap_config(config)
        return self.current_backend.evaluate(remapped)

    def clear_cache(self) -> None:
        """Очищает кэш всех дочерних бэкендов."""
        for b in self.backends:
            b.clear_cache()

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Агрегированная статистика кэша всех дочерних бэкендов.

        Returns:
            Словарь с суммарной статистикой по всем бэкендам.
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
        """Имя текущего активного бэкенда."""
        return _backend_name(self.current_backend)

    def __repr__(self) -> str:
        names = [_backend_name(b) for b in self.backends]
        return (
            f"SequentialBackend(backends=[{', '.join(names)}], "
            f"mode={self.mode!r}, current={names[self._current_idx]})"
        )