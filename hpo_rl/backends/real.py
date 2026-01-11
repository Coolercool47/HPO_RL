"""Бэкенд для реального обучения моделей.

Модуль содержит :class:`RealTrainingBackend` — бэкенд, который выполняет
реальное обучение моделей машинного обучения для оценки конфигураций
гиперпараметров.
"""

from typing import Dict, Any, Optional, List, Union
import warnings
from random import choice

from torch.utils.data import DataLoader

from hpo_rl.backends.base import EvaluationBackend, CATASTROPHIC_FAILURE_REWARD
from hpo_rl.core.factory import get_model_class, build_trainer
from hpo_rl.models.base import BaseModel


class RealTrainingBackend(EvaluationBackend):
    """Бэкенд для реального обучения моделей.

    Выполняет полный цикл: создание модели → обучение → вычисление награды
    на основе метрик обучения. Использует кэширование для экономии времени
    при повторных оценках одинаковых конфигураций.

    Args:
        model_names: Название модели или список названий. Если список,
            используется стратегия выбора модели.
        train_data: DataLoader с обучающими данными.
        model_choice_strategy: Стратегия выбора модели из списка:
            ``"random"`` — случайный выбор, ``"sequential"`` — последовательный.
        val_data: Опциональный DataLoader с валидационными данными.
        reward_strategy: Стратегия вычисления награды. Поддерживается:
            ``"neg_final_val_loss"`` — отрицательное значение финального
            валидационного loss.

    Attributes:
        model_names: Список названий моделей.
        model_choice_strategy: Стратегия выбора модели.
        train_data: Обучающие данные.
        val_data: Валидационные данные (если заданы).
        reward_strategy: Стратегия вычисления награды.
        maximize: Всегда False (минимизируем loss).

    Пример::

        from torch.utils.data import DataLoader

        train_loader = DataLoader(train_dataset, batch_size=32)
        val_loader = DataLoader(val_dataset, batch_size=32)

        backend = RealTrainingBackend(
            model_names="simple_cnn",
            train_data=train_loader,
            val_data=val_loader,
            reward_strategy="neg_final_val_loss"
        )

        reward = backend.evaluate({"trainer": {"epochs": 10, "lr": 0.001}})
    """

    def __init__(
        self,
        model_names: Union[str, List[str]],
        train_data: DataLoader,
        model_choice_strategy: str = "random",
        val_data: Optional[DataLoader] = None,
        reward_strategy: str = "neg_final_val_loss"
    ):
        """Инициализирует RealTrainingBackend.

        Args:
            model_names: Название модели или список названий.
            train_data: DataLoader с обучающими данными.
            model_choice_strategy: Стратегия выбора модели (случайный или последовательный).
            val_data: Опциональный DataLoader с валидационными данными.
            reward_strategy: Стратегия вычисления награды.
        """
        super().__init__(use_cache=True)  # кэш экономит много на повторных конфигах
        self.maximize = False  # минимизируем loss

        # Валидация model_names
        if isinstance(model_names, str):
            self.model_names = [model_names]
        elif isinstance(model_names, list):
            if not model_names:
                raise ValueError("model_names не может быть пустым")
            if not all(isinstance(n, str) for n in model_names):
                raise ValueError("Все элементы model_names должны быть строками")
            self.model_names = model_names
        else:
            raise TypeError(f"model_names: ожидался str или list, получен {type(model_names)}")

        if model_choice_strategy not in ("random", "sequential"):
            raise ValueError(f"model_choice_strategy: ожидался 'random' или 'sequential', получен '{model_choice_strategy}'")

        self.model_choice_strategy = model_choice_strategy
        self.train_data = train_data
        self.val_data = val_data
        self.reward_strategy = reward_strategy
        self.prev_model_idx = 0

    def _evaluate(self, config: Dict[str, Any]) -> float:
        """Выполняет обучение модели и вычисляет награду.

        Создаёт модель, обучает её на заданных данных и возвращает награду
        на основе выбранной стратегии. При ошибках возвращает
        :const:`CATASTROPHIC_FAILURE_REWARD`.

        Args:
            config: Конфигурация гиперпараметров. Должна содержать ключ
                ``"trainer"`` с параметрами обучения.

        Returns:
            Награда (отрицательное значение loss для минимизации).
            При ошибках возвращает :const:`CATASTROPHIC_FAILURE_REWARD`.
        """
        try:
            trainer_cfg = config.get("trainer", {})
            model_class = get_model_class(self._get_model_name())
            model = model_class()
            trainer = build_trainer(trainer_cfg)
            trained_model, history = trainer.train(model, self.train_data, self.val_data)
            return self._calculate_reward(trained_model, history, config)
        except Exception as e:
            print(f"Ошибка при оценке конфигурации: {e}")
            return CATASTROPHIC_FAILURE_REWARD

    def _calculate_reward(self, model: BaseModel, history: Dict[str, Any], config: Dict[str, Any]) -> float:
        """Вычисляет награду на основе истории обучения.

        Args:
            model: Обученная модель.
            history: Словарь с историей обучения (метрики, loss и т.д.).
            config: Конфигурация гиперпараметров.

        Returns:
            Награда на основе выбранной стратегии.
        """
        if self.reward_strategy == "neg_final_val_loss":
            val_losses = history.get("val_loss_history", [])
            if not val_losses:
                warnings.warn("Нет истории val_loss для стратегии 'neg_final_val_loss'", UserWarning)
                return CATASTROPHIC_FAILURE_REWARD
            return -val_losses[-1]
        else:
            raise NotImplementedError(f"Неизвестная стратегия награды: {self.reward_strategy}")

    def _get_model_name(self) -> str:
        """Выбирает название модели согласно стратегии.

        Returns:
            Название модели для использования.
        """
        if len(self.model_names) == 1:
            return self.model_names[0]

        if self.model_choice_strategy == "random":
            return choice(self.model_names)
        else:  # sequential
            name = self.model_names[self.prev_model_idx]
            self.prev_model_idx = (self.prev_model_idx + 1) % len(self.model_names)
            return name
