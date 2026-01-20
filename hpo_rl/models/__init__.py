"""Модели для подбора гиперпараметров

- :class:`BaseModel` - абстрактный базовый класс для создания моделей
- :class:`SimpleCNN` - небольшая сверточная нейронная сеть
"""

from hpo_rl.models.base import BaseModel
from hpo_rl.models.simple_cnn import SimpleCNN

__all__ = [
    "BaseModel",
    "SimpleCNN",
]