from abc import ABC, abstractmethod
from typing import Dict, Any

import torch.optim as optim


class OptimizerBuilder(ABC):
    """Интерфейс для строителей оптимизаторов."""

    @abstractmethod
    def build(self, model_params, hparams: Dict[str, Any]) -> optim.Optimizer:
        pass


class BaseOptimizerBuilder(OptimizerBuilder):
    """Базовый строитель с поддержкой HYPERPARAMETERS."""

    HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {}

    def build(self, model_params, hparams: Dict[str, Any]) -> optim.Optimizer:
        params_to_pass = {}
        for name, meta in self.HYPERPARAMETERS.items():
            if name in hparams:
                params_to_pass[name] = hparams[name]
            elif name == "lr" and "learning_rate" in hparams:
                params_to_pass[name] = hparams["learning_rate"]
            elif "default" in meta:
                params_to_pass[name] = meta["default"]

        return self._build_optimizer(model_params, params_to_pass)

    @abstractmethod
    def _build_optimizer(self, model_params, final_params: Dict[str, Any]) -> optim.Optimizer:
        pass


class AdamBuilder(BaseOptimizerBuilder):
    HYPERPARAMETERS = {"lr": {"type": float, "default": 0.001}}

    def _build_optimizer(self, model_params, final_params: Dict[str, Any]) -> optim.Optimizer:
        return optim.Adam(model_params, **final_params)


class SGDBuilder(BaseOptimizerBuilder):
    HYPERPARAMETERS = {"lr": {"type": float, "default": 0.01}}

    def _build_optimizer(self, model_params, final_params: Dict[str, Any]) -> optim.Optimizer:
        return optim.SGD(model_params, **final_params)
