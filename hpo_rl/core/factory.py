from typing import Type, Dict, Any
import torch.optim as optim
import torch.nn as nn

from hpo_rl.models.base import BaseModel
from hpo_rl.trainers.base import BaseTrainer
from hpo_rl.core.builder import OptimizerBuilder, AdamBuilder, SGDBuilder
from hpo_rl.environments.base_env import BaseHPOEnv
from hpo_rl.backends.base import EvaluationBackend

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}
TRAINER_REGISTRY: Dict[str, Type[BaseTrainer]] = {}
ENV_REGISTRY: Dict[str, Type[BaseHPOEnv]] = {}
BACKEND_REGISTRY: Dict[str, Type[EvaluationBackend]] = {}

OPTIMIZER_REGISTRY: Dict[str, Type[optim.Optimizer]] = {
    "Adam": optim.Adam,
    "SGD": optim.SGD,
}

CRITERION_REGISTRY: Dict[str, Type[nn.Module]] = {
    "CrossEntropyLoss": nn.CrossEntropyLoss,
    "MSELoss": nn.MSELoss,
}

OPTIMIZER_BUILDER_REGISTRY: Dict[str, OptimizerBuilder] = {
    "Adam": AdamBuilder(),
    "SGD": SGDBuilder(),
}


def register_model(name: str, model_class: Type[BaseModel]):
    """Регистрирует класс модели в реестре."""
    MODEL_REGISTRY[name] = model_class


def register_trainer(name: str, trainer_class: Type[BaseTrainer]):
    """Регистрирует класс тренера в реестре."""
    TRAINER_REGISTRY[name] = trainer_class


def register_env(name: str, env_class: Type[BaseHPOEnv]):
    """Регистрирует класс среды в реестре."""
    ENV_REGISTRY[name] = env_class


def register_backend(name: str, backend_class: Type[EvaluationBackend]):
    """Регистрирует класс бэкенда в реестре."""
    BACKEND_REGISTRY[name] = backend_class


def get_model_class(name: str) -> Type[BaseModel]:
    """
    Возвращает класс модели по имени
    """
    try:
        return MODEL_REGISTRY[name]
    except KeyError:
        raise ValueError(f"""Модель не зарегестрирована в factory.py.
                         Доступные модели: {"\n".join(list(MODEL_REGISTRY.keys()))}""")


def get_optimizer_class(name: str) -> Type[optim.Optimizer]:
    """
    Возвращает класс оптимизатора по имени
    """
    try:
        return OPTIMIZER_REGISTRY["name"]
    except KeyError:
        raise ValueError(f"""Оптимизатор не зарегестрирован в factory.py.
                         Доступные оптимизаторы: {"\n".join(list(OPTIMIZER_REGISTRY.keys()))}""")


def get_criterion_instance(name: str) -> Type[nn.Module]:
    """
    Возвращает экземпляр функции потерь по имени
    """
    try:
        criterion_class = CRITERION_REGISTRY["name"]
        return criterion_class()
    except KeyError:
        raise ValueError(f"""Функция потерь не зарегестрирована в factory.py.
                         Доступные функции потерь: {"\n".join(list(CRITERION_REGISTRY.keys()))}""")


def build_optimizer(model: nn.Module, hparams: Dict[str, Any]) -> optim.Optimizer:
    """
    Создает экземпляр оптимизатора, используя систему строителей.
    """
    optimizer_name = hparams['optimizer']
    try:
        builder = OPTIMIZER_BUILDER_REGISTRY[optimizer_name]
        return builder.build(model.parameters(), hparams)
    except KeyError:
        raise ValueError(f"""Строитель оптимизатора не зарегестрирован в factory.py.
                         Доступные строители оптимизаторов: {"\n".join(list(CRITERION_REGISTRY.keys()))}""")


def build_trainer(config: Dict[str, Any]) -> BaseTrainer:
    """
    Создает экземпляр тренера на основе конфигурационного словаря.

    Args:
        config (Dict[str, Any]): Словарь с конфигурацией для тренера,
                                 обычно содержит 'name' и 'params'.
                                 Пример: {"name": "PyTorchTrainer", "params": {...}}

    Returns:
        Экземпляр тренера, готовый к использованию.
    """
    trainer_name = config.get("name")
    if not trainer_name:
        raise ValueError("В конфигурации тренера отсутствует ключ 'name'.")

    trainer_params = config.get("params", {})

    try:
        trainer_class = TRAINER_REGISTRY[trainer_name]
        # Используем универсальный фабричный метод самого класса
        return trainer_class.from_config(trainer_params)
    except KeyError:
        raise ValueError(f"Тренер '{trainer_name}' не зарегистрирован. Доступные: {list(TRAINER_REGISTRY.keys())}")


def build_backend(config: Dict[str, Any]) -> EvaluationBackend:
    """Создает экземпляр бэкенда из реестра."""
    backend_name = config.get("name")
    if not backend_name:
        raise ValueError("В конфигурации бэкенда отсутствует ключ 'name'.")

    backend_params = config.get("params", {})
    try:
        backend_class = BACKEND_REGISTRY[backend_name]
        return backend_class(**backend_params)
    except KeyError:
        raise ValueError(f"Бэкенд '{backend_name}' не зарегистрирован.")


def build_env(config: Dict[str, Any], backend: EvaluationBackend) -> BaseHPOEnv:
    """
    Создает экземпляр RL-среды на основе конфигурационного словаря.
    """
    env_name = config.get("name")
    if not env_name:
        raise ValueError("В конфигурации среды отсутствует ключ 'name'.")

    env_params = config.get("params", {})
    hp_space = config.get("hp_space", {})

    try:
        env_class = ENV_REGISTRY[env_name]
        # Передаем в конструктор бэкенд, пространство поиска и другие параметры
        return env_class(hp_space=hp_space, backend=backend, **env_params)
    except KeyError:
        raise ValueError(f"Среда '{env_name}' не зарегистрирована.")
