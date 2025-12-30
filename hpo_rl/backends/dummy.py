"""Тестовый backend для отладки RL-агентов без реальных вычислений."""

from hpo_rl.backends.base import EvaluationBackend
from typing import Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class DummyBackend(EvaluationBackend):
    """
    Вычисляет награду на основе близости к заданному оптимуму.
    Награда = exp(-MSE), где MSE — среднеквадратичное расстояние до оптимума.
    """

    def __init__(self, optimum: Dict[str, Any]) -> None:
        super().__init__()

        if optimum is None:
            raise ValueError("optimum не может быть None")
        if not isinstance(optimum, dict):
            raise TypeError(f"optimum должен быть dict, получен {type(optimum).__name__}")
        if not optimum:
            raise ValueError("optimum не может быть пустым")

        self.optimum = optimum
        logger.info(f"DummyBackend: {len(optimum)} параметров: {list(optimum.keys())}")

    def evaluate(self, config: Dict[str, Any]) -> float:
        """
        Награда в диапазоне [0, 1]:
        - 1.0 = точное совпадение
        - 0.0 = полное несовпадение
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
                    if abs_opt < 1e-9:
                        dist = abs(cfg_val - opt_val)
                    else:
                        dist = abs(cfg_val - opt_val) / abs_opt
                    dist = min(dist, 10.0)
            else:
                dist = 0.0 if cfg_val == opt_val else 1.0

            total_dist += dist ** 2

        if count == 0:
            logger.warning(f"Нет общих параметров: config={list(config.keys())}, optimum={list(self.optimum.keys())}")
            return 0.0

        mse = total_dist / count
        return math.exp(-mse)
