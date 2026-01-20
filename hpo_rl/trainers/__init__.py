"""Классы для обучения моделей

- :class:`BaseTrainer` - абстрактный базовый класс для обучения моделей
- :class:`TorchTrainer` - обучатель для моделей из pytorch
"""

from hpo_rl.trainers.base import BaseTrainer
from hpo_rl.trainers.torch_trainer import TorchTrainer

__all__ = [
    "BaseTrainer",
    "TorchTrainer",
]