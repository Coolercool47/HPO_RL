from typing import Dict, Any, Optional, List, Union
import warnings
from random import choice

from torch.utils.data import DataLoader

from hpo_rl.backends.base import EvaluationBackend, CATASTROPHIC_FAILURE_REWARD
from hpo_rl.core.factory import get_model_class, build_trainer
from hpo_rl.models.base import BaseModel


class RealTrainingBackend(EvaluationBackend):
    """Бэкенд для реального обучения: создание модели -> обучение -> награда."""

    def __init__(
        self,
        model_names: Union[str, List[str]],
        train_data: DataLoader,
        model_choice_strategy: str = "random",
        val_data: Optional[DataLoader] = None,
        reward_strategy: str = "neg_final_val_loss"
    ):
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

        # TODO: разные данные для разных моделей через dict имя_модели -> данные
        warnings.warn("Разные данные для разных моделей пока не поддерживаются", UserWarning)

    def _evaluate(self, config: Dict[str, Any]) -> float:
        try:
            trainer_cfg = config.get("trainer", {})
            model_class = get_model_class(self._get_model_name())
            model = model_class()
            trainer = build_trainer(trainer_cfg)
            trained_model, history = trainer.train(model, self.train_data, self.val_data)
            return self._calculate_reward(trained_model, history, config)
        except Exception as e:
            print(f"КАТАСТРОФИЧЕСКАЯ ОШИБКА в цикле evaluate: {e}")
            return CATASTROPHIC_FAILURE_REWARD

    def _calculate_reward(self, model: BaseModel, history: Dict[str, Any], config: Dict[str, Any]) -> float:
        if self.reward_strategy == "neg_final_val_loss":
            val_losses = history.get("val_loss_history", [])
            if not val_losses:
                warnings.warn("Нет истории val_loss для стратегии 'neg_final_val_loss'", UserWarning)
                return CATASTROPHIC_FAILURE_REWARD
            return -val_losses[-1]
        else:
            raise NotImplementedError(f"Неизвестная стратегия награды: {self.reward_strategy}")

    def _get_model_name(self) -> str:
        if len(self.model_names) == 1:
            return self.model_names[0]

        if self.model_choice_strategy == "random":
            return choice(self.model_names)
        else:  # sequential
            name = self.model_names[self.prev_model_idx]
            self.prev_model_idx = (self.prev_model_idx + 1) % len(self.model_names)
            return name
