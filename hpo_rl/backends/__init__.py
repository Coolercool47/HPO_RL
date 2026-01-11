"""Бэкенды для оценки конфигураций гиперпараметров.

Модуль предоставляет различные бэкенды для вычисления награды
при оценке конфигураций гиперпараметров:

- :class:`EvaluationBackend` — абстрактный базовый класс
- :class:`OptimizationBenchmarkBackend` — тестовые функции оптимизации
- :class:`DummyBackend` — простой бэкенд для отладки
- :class:`RealTrainingBackend` — реальное обучение моделей

Пример::

    from hpo_rl.backends import OptimizationBenchmarkBackend

    backend = OptimizationBenchmarkBackend("rastrigin", dimensions=2)
    reward = backend.evaluate({"x0": 0.0, "x1": 0.0})
"""

from hpo_rl.backends.base import EvaluationBackend, CATASTROPHIC_FAILURE_REWARD
from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.backends.dummy import DummyBackend
from hpo_rl.backends.real import RealTrainingBackend

__all__ = [
    "EvaluationBackend",
    "CATASTROPHIC_FAILURE_REWARD",
    "OptimizationBenchmarkBackend",
    "DummyBackend",
    "RealTrainingBackend",
]

