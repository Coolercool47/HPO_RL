"""Простой бэкенд для тестирования и отладки.

Модуль содержит :class:`DummyBackend` — бэкенд, который вычисляет
награду на основе расстояния до заданного оптимума. Не требует
реального обучения моделей, поэтому подходит для быстрого
тестирования.
"""

from hpo_rl.backends.base import EvaluationBackend
from typing import Dict, Any

import numpy as np
import logging

logger = logging.getLogger(__name__)


class DummyBackend(EvaluationBackend):
    """Бэкенд для тестирования на основе расстояния до заданного оптимума.

    Вычисляет награду как ``exp(-MSE)``, где MSE — среднеквадратичное
    отклонение от заданного оптимума. Чем ближе конфигурация к оптимуму,
    тем выше награда (максимум = 1.0 при точном совпадении).

    Args:
        optimum: Словарь с оптимальными значениями параметров.
        use_cache: Включить кэширование оценок.

    Attributes:
        optimum: Целевая конфигурация (оптимум).
        maximize: Всегда True (максимизируем близость к оптимуму).

    Пример::

        optimum = {"learning_rate": 0.001, "batch_size": 32}
        backend = DummyBackend(optimum)

        # Точное совпадение → награда ≈ 1.0
        reward = backend.evaluate({"learning_rate": 0.001, "batch_size": 32})

        # Отклонение → награда < 1.0
        reward = backend.evaluate({"learning_rate": 0.01, "batch_size": 64})
    """

    def __init__(self, optimum: Dict[str, Any], use_cache: bool = True) -> None:
        """Инициализирует DummyBackend.

        Args:
            optimum: Словарь оптимальных значений ``{"параметр": значение}``.
            use_cache: Включить кэширование (по умолчанию True).
        """
        super().__init__(use_cache=use_cache)

        if optimum is None:
            raise ValueError("optimum не может быть None")
        if not isinstance(optimum, dict):
            raise TypeError(f"optimum должен быть dict, получен {type(optimum).__name__}")
        if not optimum:
            raise ValueError("optimum не может быть пустым")

        self.optimum = optimum
        logger.info(f"DummyBackend: {len(optimum)} параметров: {list(optimum.keys())}")

    def _evaluate(self, config: Dict[str, Any]) -> float:
        """Вычисляет награду как exp(-MSE) от оптимума.

        Для числовых параметров используется относительное отклонение,
        для категориальных — бинарное (0 или 1).

        Args:
            config: Оцениваемая конфигурация.

        Returns:
            Награда в диапазоне [0, 1]:
            - ``1.0`` при точном совпадении всех параметров с оптимумом
            - ``0.0`` только в краевых случаях (невалидный config или нет общих параметров)
            - ``(0, 1)`` при отклонениях (чем больше отклонение, тем ближе к 0)
            
            Формула: ``reward = exp(-MSE)``, где MSE — среднее квадратичное
            относительное отклонение. Гарантированно в [0, 1].
        """
        if not isinstance(config, dict):
            logger.warning(f"config должен быть dict, получен {type(config).__name__}")
            return 0.0

        total_dist = 0.0
        count = 0

        for key, opt_val in self.optimum.items():
            if key not in config:
                continue

            count += 1
            cfg_val = config[key]

            if isinstance(opt_val, (int, float)):
                if not isinstance(cfg_val, (int, float)):
                    dist = 1.0
                else:
                    abs_opt = abs(opt_val)
                    abs_diff = abs(cfg_val - opt_val)
                    
                    if abs_opt < 1e-9:
                        dist = abs_diff
                    else:
                        dist = abs_diff / abs_opt
                    dist = min(dist, 10.0)
            else:
                dist = 0.0 if cfg_val == opt_val else 1.0

            total_dist += dist ** 2

        if count == 0:
            logger.warning(f"Нет общих параметров: config={list(config.keys())}, optimum={list(self.optimum.keys())}")
            return 0.0

        mse = total_dist / count
        return np.exp(-mse)
