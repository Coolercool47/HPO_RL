from typing import Callable, Dict, Any
import numpy as np

from hpo_rl.backends.base import EvaluationBackend, CATASTROPHIC_FAILURE_REWARD


class ObjectiveBackend(EvaluationBackend):
    """Бэкенд для обучения моделей с помощью `objective_function`.

    Для работы данного `backend` пользователю нужно предоставить только целевую функцию, возвращающую оценку, и набор гиперпараметров с ограничениями.
    Использует кэширование для экономии времени при повторных оценках одинаковых конфигураций.

    Args:
        objective_function: целевая функция
        hp_space: набор гиперпараметров для оптимизации

    Attributes:
        objective_function: целевая функция
        hp_space: набор гиперпараметров для оптимизации
        maximize: Всегда False (минимизируем loss).

    Пример::

        def objective_function(params): 
            score = ...
            return score

        dict_config = {
            "x0": {type: float, min: 0.0, max:1.0} , 
            "x1": {type: categorical, values: ["a", "b"]}
        }

        backend = ObjectiveBackend(objective_function = objective_function, hp_space = dict_config)

        reward = backend.evaluate({"x0": 0.1, "x1": "a"})
    """
    def __init__(
        self,
        objective_function: Callable,
        hp_space: Dict[str, Any],
    ):
        """
        Инициализирует ObjectiveBackend

        Args:
        objective_function: целевая функция
        hp_space: набор гиперпараметров для оптимизации
        
        """
        super().__init__(use_cache=False)  # кэш экономит много на повторных конфигах
        self.maximize = False  # минимизируем loss

        self.objective_function = objective_function
        self.hp_space = hp_space
       
    def _evaluate(self, config: Dict[str, Any]):
        """Выполняет обучение модели и вычисляет награду.

        Args:
            config: Конфигурация гиперпараметров.

        Returns:
            Оценка из `objective_function`
        """
        return self.objective_function(config, self.hp_space)
