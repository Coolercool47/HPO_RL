from typing import Dict, Any, Optional
import warnings

from torch.utils.data import DataLoader

from hpo_rl.backends.base import EvaluationBackend, CATASTROPHIC_FAILURE_REWARD
from hpo_rl.core.factory import get_model_class, build_trainer
from hpo_rl.models.base import BaseModel


class RealTrainingBackend(EvaluationBackend):
    """
    Основной бэкенд, выполняющий полный цикл:
    создание -> обучение -> мониторинг -> вычисление награды.
    """

    def __init__(
        self,
        train_data: DataLoader,
        val_data: Optional[DataLoader] = None,
        reward_strategy: str = "neg_final_val_loss"
    ):
        self.train_data = train_data
        self.val_data = val_data
        self.reward_strategy = reward_strategy

    def evaluate(self, config: Dict[str, Any]) -> float:
        """Оркестрирует весь процесс оценки одной конфигурации."""
        try:
            model_config = config.get("model", {})
            trainer_config = config.get("trainer", {})  # <-- Получаем весь блок "trainer"

            model_class = get_model_class(model_config.get("name"))
            model = model_class.from_config(model_config.get("params", {}))

            trainer = build_trainer(trainer_config)

            trained_model, history = trainer.train(model, self.train_data, self.val_data)

            reward = self._calculate_reward(trained_model, history, config)

            return reward

        except Exception as e:
            print(f"КАТАСТРОФИЧЕСКАЯ ОШИБКА в цикле evaluate: {e}")
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

        # Добавьте здесь другие стратегии по мере необходимости
        # elif self.reward_strategy == "convergence_speed":
        #     ...

        else:
            raise ValueError(f"Неизвестная стратегия награды: {self.reward_strategy}")
