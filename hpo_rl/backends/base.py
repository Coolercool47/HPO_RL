from abc import ABC, abstractmethod
from typing import Dict, Any

CATASTROPHIC_FAILURE_REWARD: float = -1e9


class EvaluationBackend(ABC):
    """Базовый класс для бэкендов оценки конфигураций."""

    def __init__(self):
        self.maximize = True  # True = чем выше, тем лучше

    @abstractmethod
    def evaluate(self, config: Dict[str, Any]) -> float:
        """Оценивает конфигурацию и возвращает метрику качества."""
        pass
