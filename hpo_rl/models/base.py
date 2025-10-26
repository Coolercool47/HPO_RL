from abc import ABC, abstractmethod
from typing import Dict, Any
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """
    Абстрактный базовый класс для ML моделей в проекте.
    """

    HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {}

    def __init__(self, **kwargs):
        super().__init__()
        self._hyperparams = kwargs

    @abstractmethod
    def forward():
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """
        Конструктор класса для модели, принимающий словарь гиперпараметров
        """
        init_kwargs = {}
        for param_name, meta in cls.HYPERPARAMETERS.items():
            if param_name in config:
                init_kwargs[param_name] = config[param_name]
            elif "default" in meta:
                init_kwargs[param_name] = meta["default"]
            else:
                raise ValueError(
                    f"Обязательный параметр '{param_name}' отсутствует в конфиге для класса {cls.__name__}"
                )
