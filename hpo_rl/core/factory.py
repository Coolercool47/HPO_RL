
from __future__ import annotations

from typing import Type, Dict, Any, Optional, TYPE_CHECKING
import torch.optim as optim
import torch.nn as nn
from torch.optim import optimizer

from hpo_rl.models.base import BaseModel
from hpo_rl.trainers.base import BaseTrainer
from hpo_rl.core.builder import OptimizerBuilder, AdamBuilder, SGDBuilder


#from hpo_rl.environments.base_env import BaseHPOEnv
#from hpo_rl.backends.base import EvaluationBackend

if TYPE_CHECKING:
    from hpo_rl.environments.base_env import BaseHPOEnv
    from hpo_rl.backends.base import EvaluationBackend

'''
Это заглушка, которая решает проблему с циклическим импортом из-за автодокументации Sphinx. Лечится адекватно долго так что пока что так
'''


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
    MODEL_REGISTRY[name] = model_class


def register_trainer(name: str, trainer_class: Type[BaseTrainer]):
    TRAINER_REGISTRY[name] = trainer_class


def register_env(name: str, env_class: Type[BaseHPOEnv]):
    ENV_REGISTRY[name] = env_class


def register_backend(name: str, backend_class: Type[EvaluationBackend]):
    BACKEND_REGISTRY[name] = backend_class


def get_model_class(name: str) -> Type[BaseModel]:
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Модель '{name}' не зарегистрирована. Доступные: {available}")
    return MODEL_REGISTRY[name]


def get_optimizer_class(name: str) -> Type[optim.Optimizer]:
    if name not in OPTIMIZER_REGISTRY:
        available = ", ".join(OPTIMIZER_REGISTRY.keys())
        raise ValueError(f"Оптимизатор '{name}' не зарегистрирован. Доступные: {available}")
    return OPTIMIZER_REGISTRY[name]


def get_criterion_instance(name: str) -> nn.Module:
    print(name, CRITERION_REGISTRY)
    if name not in CRITERION_REGISTRY:
        available = ", ".join(CRITERION_REGISTRY.keys())
        raise ValueError(f"Функция потерь '{name}' не зарегистрирована. Доступные: {available}")
    return CRITERION_REGISTRY[name]()


def build_optimizer(model: nn.Module, hparams: Dict[str, Any]) -> optim.Optimizer:
    optimizer_name = hparams['optimizer']
    print(optimizer_name)
    if optimizer_name not in OPTIMIZER_BUILDER_REGISTRY:
        available = ", ".join(OPTIMIZER_BUILDER_REGISTRY.keys())
        raise ValueError(f"Строитель оптимизатора '{optimizer_name}' не зарегистрирован. Доступные: {available}")
    return OPTIMIZER_BUILDER_REGISTRY[optimizer_name].build(model.parameters(), hparams)


def build_trainer(config: Dict[str, Any]) -> BaseTrainer:
    """Создает тренер из конфига."""
    trainer_name = config.get("name")
    if not trainer_name:
        raise ValueError("В конфигурации тренера отсутствует ключ 'name'")

    if trainer_name not in TRAINER_REGISTRY:
        available = ", ".join(TRAINER_REGISTRY.keys())
        raise ValueError(f"Тренер '{trainer_name}' не зарегистрирован. Доступные: {available}")

    return TRAINER_REGISTRY[trainer_name].from_config(config.get("params", {}))


def build_backend(config: Dict[str, Any]) -> EvaluationBackend:
    """Создает бэкенд из конфига."""
    backend_name = config.get("name")
    if not backend_name:
        raise ValueError("В конфигурации бэкенда отсутствует ключ 'name'")

    if backend_name not in BACKEND_REGISTRY:
        available = ", ".join(BACKEND_REGISTRY.keys())
        raise ValueError(f"Бэкенд '{backend_name}' не зарегистрирован. Доступные: {available}")

    return BACKEND_REGISTRY[backend_name](**config.get("params", {}))


def build_env(
    config: Dict[str, Any],
    backend: EvaluationBackend,
    hp_space: Optional[Dict[str, Any]] = None
) -> BaseHPOEnv:
    """Создает RL-среду из конфига."""
    env_name = config.get("name")
    if not env_name:
        raise ValueError("В конфигурации среды отсутствует ключ 'name'")

    if hp_space is None:
        hp_space = config.get("hp_space", {})
    if not hp_space:
        raise ValueError("hp_space не может быть пустым")

    if env_name not in ENV_REGISTRY:
        available = ", ".join(ENV_REGISTRY.keys())
        raise ValueError(f"Среда '{env_name}' не зарегистрирована. Доступные: {available}")

    return ENV_REGISTRY[env_name](hp_space=hp_space, backend=backend, **config.get("params", {}))
