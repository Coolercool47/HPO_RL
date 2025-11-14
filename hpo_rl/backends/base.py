from abc import ABC, abstractmethod
from typing import Dict, Any

CATASTROPHIC_FAILURE_REWARD: float = -1e9


class EvaluationBackend(ABC):
    @abstractmethod
    def evaluate(self, config: Dict[str, Any]) -> float:
        """
        Основной метод, выполняющий оценку одной конфигурации.

        Args:
            config (Dict[str, Any]): Словарь с гиперпараметрами,
                                     предложенный RL-агентом.

        Returns:
            float: Одно число, представляющее метрику качества (награду).
                   Чем выше, тем лучше.
        """
        pass
