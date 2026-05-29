"""Бэкенд для оценки конфигураций через пользовательскую целевую функцию."""

from typing import Callable, Dict, Any

from hpo_rl.backends.base import EvaluationBackend


class ObjectiveBackend(EvaluationBackend):
    """Бэкенд для обучения моделей через пользовательскую целевую функцию.

        Args:
            objective_function: целевая функция, возвращающая оценку
            hp_space: пространство гиперпараметров

        Attributes:
            objective_function: целевая функция
            hp_space: пространство гиперпараметров
            maximize: всегда False (минимизация loss)

        Пример::

            def objective_function(params, hp_space):
                return train_and_eval(params)

            hp_space = {"lr": {"type": "float", "values": [1e-4, 1e-1]}}

            backend = ObjectiveBackend(objective_function, hp_space)
            reward = backend.evaluate({"lr": 0.01})
    """

    def __init__(
        self,
        objective_function: Callable,
        hp_space: Dict[str, Any],
    ):
        """Инициализирует ObjectiveBackend.

        Args:
            objective_function: целевая функция
            hp_space: пространство гиперпараметров
        """
        super().__init__(use_cache=False)
        self.maximize = False

        self.objective_function = objective_function
        self.hp_space = hp_space

    def _evaluate(self, config: Dict[str, Any]):
        """Вычисляет оценку конфигурации.

        Args:
            config: конфигурация гиперпараметров

        Returns:
            значение целевой функции
        """
        return self.objective_function(config, self.hp_space)
