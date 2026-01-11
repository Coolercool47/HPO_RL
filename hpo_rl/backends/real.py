from typing import Dict, Any, Optional, List, Union
import warnings
import logging
from random import choice

from torch.utils.data import DataLoader

from hpo_rl.backends.base import EvaluationBackend, CATASTROPHIC_FAILURE_REWARD
from hpo_rl.core.factory import get_model_class, build_trainer
from hpo_rl.models.base import BaseModel

logger = logging.getLogger(__name__)


# TODO: отревьюить код, проверить что все работает (пока что не самая важная задача, но все же)

class RealTrainingBackend(EvaluationBackend):
    """
    Основной бэкенд, выполняющий полный цикл:
    создание -> обучение -> мониторинг -> вычисление награды.
    """
    def __init__(
        self,
        model_names: Union[str, List[str]],
        train_data: DataLoader,
        model_choice_strategy: str = "random",
        val_data: Optional[DataLoader] = None,
        reward_strategy: str = "neg_final_val_loss"
    ):
        # Валидация model_names
        if isinstance(model_names, str):
            self.model_names = [model_names]
        elif isinstance(model_names, list):
            if not model_names:
                raise ValueError("model_names не может быть пустым списком")
            if not all(isinstance(name, str) for name in model_names):
                raise ValueError("Все элементы model_names должны быть строками")
            self.model_names = model_names
        else:
            raise TypeError(f"model_names должен быть строкой или списком строк, получен {type(model_names)}")
        
        if model_choice_strategy in ["random", "sequential"]:
            self.model_choice_strategy = model_choice_strategy
        else:
            raise ValueError(
                f"model_choice_strategy должен быть 'random' или 'sequential', "
                f"получен '{model_choice_strategy}'"
            )
        warnings.warn("Different data for different models is not yet implemented, use unified task models", UserWarning)
        self.train_data = train_data  
        # TODO сделать подгрузку датасетов через dict-ы формата имя_модели: данные, в идеале что-то типа multi-key делать
        self.val_data = val_data
        self.reward_strategy = reward_strategy
        self.prev_model_idx = 0

    def evaluate(self, config: Dict[str, Any]) -> float:
        """
        Оркестрирует весь процесс оценки одной конфигурации.
        
        Создает модель, обучает её с использованием указанного тренера
        и вычисляет награду на основе результатов обучения.
        
        Args:
            config: Словарь конфигурации, должен содержать ключ "trainer"
                   с параметрами тренера.
        
        Returns:
            float: Награда за данную конфигурацию. В случае ошибки возвращает
                   CATASTROPHIC_FAILURE_REWARD.
        """
        try:
            trainer_config = config.get("trainer", {})

            model_class = get_model_class(self._get_model_name())
            model = model_class()

            trainer = build_trainer(trainer_config)

            trained_model, history = trainer.train(model, self.train_data, self.val_data)

            reward = self._calculate_reward(trained_model, history, config)

            return reward

        except Exception as e:
            logger.error(
                f"Catastrophic error in evaluate cycle: {e}",
                exc_info=True,
                extra={"config": config}
            )
            return CATASTROPHIC_FAILURE_REWARD

    def _calculate_reward(
        self,
        model: BaseModel,
        history: Dict[str, Any],
        config: Dict[str, Any]
    ) -> float:
        """Реализует различные стратегии вычисления награды."""

        if self.reward_strategy == "neg_final_val_loss":
            val_losses = history.get("val_loss_history", [])
            if not val_losses:
                warnings.warn("Для стратегии 'neg_final_val_loss' нет истории val_loss.", UserWarning)
                return CATASTROPHIC_FAILURE_REWARD

            return -val_losses[-1]
        else:
            raise NotImplementedError(f"Неизвестная стратегия награды: {self.reward_strategy}")

    def _get_model_name(self):
        if len(self.model_names) == 1 or type(self.model_names) is str:
            return self.model_names[0]
        if self.model_choice_strategy == "random":
            return choice(self.model_names)
        elif self.model_choice_strategy == "sequential":
            self.prev_model_idx += 1
            self.prev_model_idx %= len(self.model_names)
            return self.model_names[self.prev_model_idx]
