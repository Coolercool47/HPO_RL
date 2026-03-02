"""ICM-обёртка, совместимая с рекуррентным PPO (ChunkedRNNPPO).

Проблема
--------
Стандартный ``ICMOnPolicyWrapper`` вызывает ``_icm_preprocess_batch``
**до** того, как ``ChunkedRNNPPO._preprocess_batch`` перестроит
``batch.act`` из ``[T]`` в ``[num_chunks, seq_len]``.

В результате ``batch.policy.act_hat`` имеет форму ``[T, action_dim]``,
а ``batch.act`` — ``[num_chunks, seq_len]``.  При вычислении
``cross_entropy(act_hat, act)`` в ``_icm_update`` размерности не совпадают.

Решение
-------
``RecurrentICMOnPolicyWrapper`` переопределяет ``_wrapper_update_with_batch``
и **flatten-ит** ``batch.act`` перед вычислением ICM loss, чтобы размерности
``act_hat`` и ``act`` снова совпали.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np

from tianshou.algorithm.modelbased.icm import (
    ICMOnPolicyWrapper,
    ICMTrainingStats,
)
from tianshou.algorithm.algorithm_base import (
    OnPolicyAlgorithm,
    TrainingStats,
)
from tianshou.data import to_torch
from tianshou.data.types import RolloutBatchProtocol
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
from tianshou.algorithm.optim import OptimizerFactory


class RecurrentICMOnPolicyWrapper(ICMOnPolicyWrapper):
    """ICMOnPolicyWrapper с поддержкой рекуррентных алгоритмов.

    Единственное отличие от базового класса — в ``_wrapper_update_with_batch``
    ``batch.act`` и ``batch.policy.act_hat`` / ``mse_loss`` приводятся к
    одинаковой «плоской» размерности перед вычислением ICM loss.
    """

    def _wrapper_update_with_batch(
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        original_stats: TrainingStats,
    ) -> ICMTrainingStats:
        # ---- flatten act/act_hat/mse_loss для совместимости ----
        act_hat = batch.policy.act_hat          # [T, action_dim]
        mse_loss = batch.policy.mse_loss        # [T]

        act = batch.act                         # может быть [num_chunks, seq_len]
        if isinstance(act, torch.Tensor):
            act_flat = act.reshape(-1)
        else:
            act_flat = torch.as_tensor(act, dtype=torch.long).reshape(-1)

        act_flat = act_flat.to(dtype=torch.long, device=act_hat.device)

        # Убедимся, что размерности совпадают
        if act_hat.shape[0] != act_flat.shape[0]:
            raise ValueError(
                f"ICM act_hat batch ({act_hat.shape[0]}) != "
                f"act batch ({act_flat.shape[0]}) после flatten. "
                f"Проверьте seq_len и collection_step_num_env_steps."
            )

        inverse_loss = F.cross_entropy(act_hat, act_flat).mean()
        forward_loss = mse_loss.mean()
        loss = (
            (1 - self.forward_loss_weight) * inverse_loss
            + self.forward_loss_weight * forward_loss
        ) * self.lr_scale
        self.optim.step(loss)

        return ICMTrainingStats(
            original_stats,
            icm_loss=loss.item(),
            icm_forward_loss=forward_loss.item(),
            icm_inverse_loss=inverse_loss.item(),
        )
