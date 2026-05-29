"""Алгоритмы обучения с подкреплением для оптимизации гиперпараметров.

- :class:`ChunkedRNNPPO` — PPO с рекуррентной политикой и чанкованным буфером
- :class:`ChunkedRNNDQN` — DQN с рекуррентной сетью и чанкованным буфером
- :class:`RecurrentICMOnPolicyWrapper` — обёртка ICM для on-policy алгоритмов
"""

from hpo_rl.alg.recurrent_ppo import ChunkedRNNPPO
from hpo_rl.alg.recurrent_dqn import ChunkedRNNDQN
from hpo_rl.alg.recurrent_icm import RecurrentICMOnPolicyWrapper

__all__ = [
    "ChunkedRNNPPO",
    "ChunkedRNNDQN",
    "RecurrentICMOnPolicyWrapper",
]
