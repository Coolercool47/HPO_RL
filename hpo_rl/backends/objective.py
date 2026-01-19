from typing import Callable, Dict, Any, Optional, List, Union
import numpy as np

from hpo_rl.backends.base import EvaluationBackend, CATASTROPHIC_FAILURE_REWARD
from hpo_rl.core.factory import get_model_class, build_trainer
from hpo_rl.models.base import BaseModel
from hpo_rl.trainers.base import BaseTrainer


class ObjectiveBackend(EvaluationBackend):

    def __init__(
        self,
        objective_function: Callable,
        hp_space: Dict[str, Any],
        num_epochs: int
    ):

        super().__init__(use_cache=False)  # кэш экономит много на повторных конфигах
        self.maximize = False  # минимизируем loss

        self.objective_function = objective_function
        self.hp_space = hp_space
        self.num_epochs = num_epochs
       
    def _evaluate(self, config: Dict[str, Any]):
        return self.objective_function(config, self.hp_space, self.num_epochs)

    def get_model_class():
        '''
        Получает название класса модели возвращает класс из YAML с дефолтными моделями. Возможно будет определять сразу тренер? 
        '''
        pass
    
    def get_trainer():
        '''
        Получает название класса тренера возвращает класс из YAML с дефолтными тренерами
        '''
        pass
