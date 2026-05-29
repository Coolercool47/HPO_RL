"""ICM-обёртка для on-policy алгоритмов с рекуррентными батчами."""

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
    """ICM-обёртка для on-policy алгоритмов с рекуррентными батчами.

    Отличается от :class:`~tianshou.algorithm.modelbased.icm.ICMOnPolicyWrapper` тем,
    что в ``_wrapper_update_with_batch`` поля ``act`` и ``policy.act_hat`` /
    ``mse_loss`` выравниваются в плоский вид перед ICM loss.

    Args:
        wrapped_algorithm: базовый on-policy алгоритм (например, ChunkedRNNPPO).
        model: модуль :class:`~tianshou.utils.net.discrete.IntrinsicCuriosityModule`.
        optim: фабрика оптимизатора для ICM.
        lr_scale: множитель learning rate ICM относительно базового алгоритма.
        reward_scale: масштаб intrinsic reward при сборе rollout.
        forward_loss_weight: вес forward-loss в суммарном ICM loss.
    """

    def _wrapper_update_with_batch(
        self,
        batch: RolloutBatchProtocol,
        batch_size: int | None,
        repeat: int,
        original_stats: TrainingStats,
    ) -> ICMTrainingStats:
        """Шаг ICM: выравнивает ``act`` и ``act_hat``, считает inverse/forward loss.

        Args:
            batch: rollout с ``policy.act_hat`` и ``policy.mse_loss``.
            batch_size: размер мини-батча (не используется в переопределении).
            repeat: число повторов (не используется).
            original_stats: статистика базового алгоритма.

        Returns:
            ``ICMTrainingStats`` с суммарным и раздельными ICM-loss.
        """
        act_hat = batch.policy.act_hat       
        mse_loss = batch.policy.mse_loss       

        act = batch.act                       
        if isinstance(act, torch.Tensor):
            act_flat = act.reshape(-1)
        else:
            act_flat = torch.as_tensor(act, dtype=torch.long).reshape(-1)

        act_flat = act_flat.to(dtype=torch.long, device=act_hat.device)

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
