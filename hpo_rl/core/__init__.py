"""Ядро фреймворка HPO_RL.

Модуль содержит основные компоненты для построения и регистрации
компонентов системы оптимизации гиперпараметров:

- :class:`OptimizerBuilder` - абстрактный базовый класс для построения оптимизаторов
- :class:`BaseOptimizerBuilder` - базовый класс с поддержкой гиперпараметров
- Функции фабрики для создания бэкендов, окружений и моделей
- Регистрация компонентов системы
"""

from hpo_rl.core.builder import (
    OptimizerBuilder,
    BaseOptimizerBuilder,
    AdamBuilder,
    SGDBuilder,
)

__all__ = [
    "OptimizerBuilder",
    "BaseOptimizerBuilder",
    "AdamBuilder",
    "SGDBuilder",
]

