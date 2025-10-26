from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Dict, Any, Optional

ModelType = TypeVar("ModelType")
DataType = TypeVar("DataType")


class BaseTrainer(ABC, Generic[ModelType, DataType]):

    HYPERPARAMETERS: Dict[str, Dict[str, Any]] = {}

    def __init__(self, **kwargs):
        super().__init__()
        self._hparams = kwargs

    @abstractmethod
    def train(self, model: ModelType, train_loader: DataType, val_data: Optional[DataType] = None):
        """
        Метод обучения модели. Принимает модель и обучающую выборку, возвращает кортеж из обученной модели и истории обучения
        """
        pass

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        init_kwargs = {}
        for param_name, meta in cls.HYPERPARAMETERS.items():
            if param_name in config:
                init_kwargs[param_name] = config[param_name]
            elif "default" in meta:
                init_kwargs[param_name] = meta["default"]
            else:
                raise ValueError(
                    f"Обязательный параметр '{param_name}' отсутствует в конфиге для класса {cls.__name__}"
                )
        return cls(**init_kwargs)
