from typing import Callable, Dict, Any

from hpo_rl.backends.base import EvaluationBackend, CATASTROPHIC_FAILURE_REWARD
from hpo_rl.models.base import BaseModel
from hpo_rl.trainers.base import BaseTrainer



class RealTrainingBackend(EvaluationBackend):

    def __init__(
        self,
        model: BaseModel | str,
        trainer: BaseTrainer | str,
        hp_space: Dict[str, Any],
        data_processor: Callable, # включает в себя загрузку данных и их обработку
    ):
        '''
        objective_function:

        Для бэкенда real необходимы:

        модель (класс или название класса)

        тренер (class | str название класса из дефолтных) back = RTB(trainer="PyTorchTrainer") или back = RTB(trainer=TorchTrainer)

        данные для обучения (DataLoader или путь) - пути плохой вариант, надо обрабатывать, а там вариантов триллион

        данные для валидации (DataLoader или путь)
        ЛИБО
        Коэфициент размера выборки валидации - не вариант, потому что надо назад прокручивать фарш из train_data, чтобы потом делить. 

        Пространство гиперпараметров в виде:
        {
            model:
                param_name:
                    type: [float, int, categorical]
                    если float или int:
                        min:
                        max:
                    если categorical:
                        values: []
            optimizer:
                param_name:
                    type:
                    ...
            criterion:
                -.-
            

        }
        '''

        super().__init__(use_cache=True)  # кэш экономит много на повторных конфигах
        self.maximize = False  # минимизируем loss

        self.model_class = model if isinstance(model, type) and issubclass(model, BaseModel) else self.get_model_class(model)
        self.trainer = trainer(hp_space) if isinstance(trainer, type) and issubclass(trainer, BaseTrainer) else self.get_trainer(trainer)
        self.train_data, self.val_data = data_processor()
        self.hp_space = hp_space.copy()
       
    # получает что-то в виде словаря {"parameter_name": value}
    def _evaluate(self, config):
        # print(self.hp_space)
        # print(config)
        model, losses = self.trainer.train(config, self.model_class, self.train_data, self.val_data)
        return losses["val_loss_history"][-1]
        # типа обработали параметры из конфига

    def get_model_class(self, model_name):
        raise NotImplementedError("Method is yet to implement")
    
    def get_trainer(self, trainer_name):
        raise NotImplementedError("Method is yet to implement")