from abc import ABC, abstractmethod
from typing import Dict, Any

import torch.optim as optim


class OptimizerBuilder(ABC):
    """
    Абстрактный интерфейс для всех строителей оптимизаторов.
    Определяет публичный контракт - метод build().
    """
    @abstractmethod
    def build(self, model_params, hparams: Dict[str, Any]) -> optim.Optimizer:
        pass


class BaseOptimizerBuilder(OptimizerBuilder):

    HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {}

    def build(self, model_params, hparams: Dict[str, Any]) -> optim.Optimizer:

        params_to_pass = {}
        for name, meta in self.HYPERPARAMETERS.items():
            if name in hparams:
                params_to_pass[name] = hparams[name]
            elif "default" in meta:
                params_to_pass[name] = meta["default"]

        return self._build_optimizer(model_params, params_to_pass)

    @abstractmethod
    def _build_optimizer(self, model_params, final_params: Dict[str, Any]) -> optim.Optimizer:
        """
        Метод который вызывает конструктор оптимизатора и возвращает его инстанс
        """
        pass


class AdamBuilder(BaseOptimizerBuilder):
    """Строитель для torch.optim.Adam."""

    HYPERPARAMETERS = {
        "lr": {"type": float, "default": 0.001},
    }

    def _build_optimizer(self, model_params, final_params: Dict[str, Any]) -> optim.Optimizer:
        return optim.Adam(model_params, **final_params)


class SGDBuilder(BaseOptimizerBuilder):
    """Строитель для torch.optim.SGD."""

    HYPERPARAMETERS = {
        "lr": {"type": float, "default": 0.01},
    }

    def _build_optimizer(self, model_params, final_params: Dict[str, Any]) -> optim.Optimizer:
        return optim.SGD(model_params, **final_params)
