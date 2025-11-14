from typing import Dict, Any, Optional, List, Union
import warnings
from random import choice

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
        model_names: Union[str, List[str]],
        train_data: DataLoader,
        model_choice_strategy: str = "random",
        val_data: Optional[DataLoader] = None,
        reward_strategy: str = "neg_final_val_loss"
    ):
        self.model_names = model_names  # TODO накинуть проверки формата not implemented
        if model_choice_strategy in ["random", "sequential"]:  # перекинуть в регистры
            self.model_choice_strategy = model_choice_strategy
        else:
            raise NotImplementedError(f"model_choice_strategy should be random or sequential, not {model_choice_strategy}")
        warnings.warn("Different data for different models is not yet implemented, use unified task models", UserWarning)
        self.train_data = train_data  
        # TODO сделать подгрузку датасетов через dict-ы формата имя_модели: данные, в идеале что-то типа multi-key делать
        self.val_data = val_data
        self.reward_strategy = reward_strategy
        self.prev_model_idx = 0

    def evaluate(self, config: Dict[str, Any]) -> float:
        """ Убогое обозначение, переписать
        Оркестрирует весь процесс оценки одной конфигурации."""
        try:
            '''
            Убогая реализация, конфиги тут передавать мне не нравится, это как будто избыточно,
            я предлагаю парсеры вынести немного внаружу, хотя бы частично
            model_config = config.get("model", {})
            '''
            trainer_config = config.get("trainer", {})  # <-- Получаем весь блок "trainer"

            model_class = get_model_class(self._get_model_name())
            model = model_class()
            # model = model_class.from_config(model_config.get("params", {}))
            # удалить, мы избавляемся от гиперпараметров самих моделей, только оптимизаторы.
            # Теоретически потом можем вернуть, но пока что пробуем только гиперы оптимизаторов

            trainer = build_trainer(trainer_config)  # тут не уйти от конфига, так что с ним тусим

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
