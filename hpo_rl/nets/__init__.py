"""Нейросетевые архитектуры для RL-агентов оптимизации гиперпараметров.

- :class:`BaseNet` — базовая полносвязная сеть
- :class:`RecurrentBaseNet` — рекуррентная базовая сеть
- :class:`MaskedNet` — сеть с маскированием действий
- :class:`MaskedRecurrentNet` — рекуррентная сеть с маскированием
- :class:`MaskedDiscreteActor` — дискретный актор с маскированием
- :class:`MaskedRecurrentDiscreteActor` — рекуррентный дискретный актор с маскированием
- :class:`RecurrentContinuousActorProbabilistic` — рекуррентный непрерывный актор
- :class:`RecurrentCritic` — рекуррентный критик
- :class:`RecurrentProbabilisticActorPolicy` — рекуррентная вероятностная политика
- :class:`RainbowNetWrapper` — обёртка Rainbow DQN
- :class:`ICMFeatureNet` — сеть признаков для ICM
- :class:`GradientMonitorMixin` — миксин мониторинга градиентов
- :class:`OptimizerStepMonitor` — монитор шагов оптимизатора
"""

from hpo_rl.nets.base_net import BaseNet
from hpo_rl.nets.recurrent_net import RecurrentBaseNet
from hpo_rl.nets.masked_net import MaskedNet
from hpo_rl.nets.masked_recurrent_net import MaskedRecurrentNet
from hpo_rl.nets.masked_actor import MaskedDiscreteActor
from hpo_rl.nets.recurrent_actor import MaskedRecurrentDiscreteActor
from hpo_rl.nets.recurrent_continuous_actor import RecurrentContinuousActorProbabilistic
from hpo_rl.nets.recurrent_critic import RecurrentCritic
from hpo_rl.nets.recurrent_policy import RecurrentProbabilisticActorPolicy
from hpo_rl.nets.rainbow_net import RainbowNetWrapper
from hpo_rl.nets.icm_feature_net import ICMFeatureNet
from hpo_rl.nets.gradient_monitor import (
    GradientMonitorMixin,
    OptimizerStepMonitor,
    GradientMonitoredNet,
    GradientMonitoredBaseNet,
    GradientMonitoredRecurrentBaseNet,
    GradientMonitoredRecurrentNet,
)

__all__ = [
    "BaseNet",
    "RecurrentBaseNet",
    "MaskedNet",
    "MaskedRecurrentNet",
    "MaskedDiscreteActor",
    "MaskedRecurrentDiscreteActor",
    "RecurrentContinuousActorProbabilistic",
    "RecurrentCritic",
    "RecurrentProbabilisticActorPolicy",
    "RainbowNetWrapper",
    "ICMFeatureNet",
    "GradientMonitorMixin",
    "OptimizerStepMonitor",
    "GradientMonitoredNet",
    "GradientMonitoredBaseNet",
    "GradientMonitoredRecurrentBaseNet",
    "GradientMonitoredRecurrentNet",
]
