"""Бэкенд для оценки конфигураций через реальное обучение моделей.

Модуль содержит :class:`RealTrainingBackend` — бэкенд, который использует
реальные данные и обучение моделей для оценки гиперпараметров.
Поддерживает кэширование результатов для оптимизации производительности.
"""

from typing import Callable, Dict, Any

from hpo_rl.backends.base import EvaluationBackend, CATASTROPHIC_FAILURE_REWARD
from hpo_rl.models.base import BaseModel
from hpo_rl.trainers.base import BaseTrainer



class RealTrainingBackend(EvaluationBackend):
    """Бэкенд для оценки конфигураций через реальное обучение моделей.

    Использует реальные данные и обучение для оценки гиперпараметров.
    Поддерживает кэширование результатов для экономии времени на повторных конфигурациях.

    Args:
        model: Класс модели или строка с именем модели.
        trainer: Класс тренера или строка с именем тренера.
        hp_space: Словарь пространства гиперпараметров.
        data_processor: Функция, возвращающая кортеж (train_data, val_data).
            Включает загрузку и обработку данных.
    """

    def __init__(
        self,
        model: BaseModel | str,
        trainer: BaseTrainer | str,
        hp_space: Dict[str, Any],
        data_processor: Callable, # включает в себя загрузку данных и их обработку
    ):
        """Инициализирует бэкенд для реального обучения.

        Args:
            model: Класс модели или строка с именем модели.
            trainer: Класс тренера или строка с именем тренера.
            hp_space: Словарь пространства гиперпараметров.
            data_processor: Функция, возвращающая (train_data, val_data).
        """

        super().__init__(use_cache=True)  # кэш экономит много на повторных конфигах
        self.maximize = False  # минимизируем loss

        self.model_class = model if isinstance(model, type) and issubclass(model, BaseModel) else self.get_model_class(model)
        self.trainer = trainer(hp_space) if isinstance(trainer, type) and issubclass(trainer, BaseTrainer) else self.get_trainer(trainer)
        self.train_data, self.val_data = data_processor()
        self.hp_space = hp_space.copy()
       
    def _evaluate(self, config: Dict[str, Any]) -> float:
        """Оценивает конфигурацию через обучение модели.

        Args:
            config: Словарь гиперпараметров {"parameter_name": value}.

        Returns:
            Значение валидационной функции потерь после обучения.
        """
        model, losses = self.trainer.train(config, self.model_class, self.train_data, self.val_data)
        return losses["val_loss_history"][-1]

    def get_model_class(self, model_name: str):
        """Получает класс модели по имени.

        Args:
            model_name: Имя модели.

        Returns:
            Класс модели.

        Raises:
            NotImplementedError: Метод еще не реализован.
        """
        raise NotImplementedError("Method is yet to implement")
    
    def get_trainer(self, trainer_name: str):
        """Получает класс тренера по имени.

        Args:
            trainer_name: Имя тренера.

        Returns:
            Класс тренера.

        Raises:
            NotImplementedError: Метод еще не реализован.
        """
        raise NotImplementedError("Method is yet to implement")
