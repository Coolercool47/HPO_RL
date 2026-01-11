"""Инструменты для инициализации оптимизаторов PyTorch.

Модуль содержит :class:`OptimizerBuilder` и его реализации для создания
экземпляров оптимизаторов на основе словарей конфигурации. Позволяет
отделить описание гиперпараметров от логики инициализации модели.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import torch.optim as optim


class OptimizerBuilder(ABC):
    """Абстрактный интерфейс для построения оптимизаторов.

    Определяет контракт для создания экземпляров :class:`torch.optim.Optimizer`
    на основе параметров модели и внешних гиперпараметров.

    Пример::

        class MyOptimizerBuilder(OptimizerBuilder):
            def build(self, model_params, hparams):
                return optim.Adam(model_params, lr=hparams.get("lr", 0.001))
    """

    @abstractmethod
    def build(self, model_params, hparams: Dict[str, Any]) -> optim.Optimizer:
        """Создает экземпляр оптимизатора.

        Args:
            model_params: Итерируемый объект с параметрами модели (обычно ``model.parameters()``).
            hparams: Словарь гиперпараметров из конфигурации.

        Returns:
            Инициализированный объект оптимизатора PyTorch.
        """
        pass


class BaseOptimizerBuilder(OptimizerBuilder):
    """Базовая реализация билдера с механизмом извлечения гиперпараметров.

    Обеспечивает автоматическое сопоставление входных настроек со схемой
    :attr:`HYPERPARAMETERS`, поддерживает значения по умолчанию и алиасы
    (например, замену ``learning_rate`` на ``lr``).

    Attributes:
        HYPERPARAMETERS: Метаданные ожидаемых параметров в формате:
            ``{"name": {"type": type, "default": value}}``.
    """

    HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {}

    def build(self, model_params, hparams: Dict[str, Any]) -> optim.Optimizer:
        """Подготавливает параметры и вызывает создание оптимизатора.

        Извлекает значения из ``hparams`` согласно схеме :attr:`HYPERPARAMETERS`.
        Если ключ отсутствует, использует значение ``default``. Поддерживает
        алиас ``learning_rate`` для параметра ``lr``.

        Args:
            model_params: Параметры модели.
            hparams: Словарь гиперпараметров.

        Returns:
            Экземпляр оптимизатора PyTorch с примененными настройками.
        """
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
        """Внутренний метод для инициализации конкретного класса оптимизатора.

        Args:
            model_params: Параметры модели.
            final_params: Обработанный словарь гиперпараметров.
        """
        pass


class AdamBuilder(BaseOptimizerBuilder):
    """Билдер для оптимизатора :class:`torch.optim.Adam`.

    По умолчанию использует ``lr=0.001``.

    Пример::

        builder = AdamBuilder()
        optimizer = builder.build(model.parameters(), {"lr": 1e-4})
    """

    HYPERPARAMETERS = {"lr": {"type": float, "default": 0.001}}

    def _build_optimizer(self, model_params, final_params: Dict[str, Any]) -> optim.Optimizer:
        """Создает экземпляр Adam."""
        return optim.Adam(model_params, **final_params)


class SGDBuilder(BaseOptimizerBuilder):
    """Билдер для оптимизатора :class:`torch.optim.SGD`.

    По умолчанию использует ``lr=0.01``.

    Пример::

        builder = SGDBuilder()
        optimizer = builder.build(model.parameters(), {"learning_rate": 0.1})
    """

    HYPERPARAMETERS = {"lr": {"type": float, "default": 0.01}}

    def _build_optimizer(self, model_params, final_params: Dict[str, Any]) -> optim.Optimizer:
        """Создает экземпляр SGD."""
        return optim.SGD(model_params, **final_params)