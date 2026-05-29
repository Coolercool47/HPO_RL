"""Обёртка политики: пробрасывает ``episode_reset`` в актор для RNN."""

from __future__ import annotations

from typing import Any, cast

from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.data import Batch
from tianshou.data.types import DistBatchProtocol, ObsBatchProtocol


class RecurrentProbabilisticActorPolicy(ProbabilisticActorPolicy):
    """ProbabilisticActorPolicy с ``episode_reset`` в ``info`` для RNN-актёра.

        Args:
            (наследуются от ProbabilisticActorPolicy)

        Note:
            Позволяет :class:`~hpo_rl.nets.recurrent_net.RecurrentBaseNet`
            сбрасывать hidden на границах чанка и эпизода.
    """

    def _info_for_actor(self, batch: ObsBatchProtocol) -> dict[str, Any] | None:
        base = batch.info
        if base is None:
            out: dict[str, Any] = {}
        elif isinstance(base, dict):
            out = dict(base)
        elif isinstance(base, Batch):
            out = {k: base[k] for k in base.keys()}
        if getattr(batch, "episode_reset", None) is not None:
            out["episode_reset"] = batch.episode_reset
        return out if out else None

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | Any | None = None,
    ) -> DistBatchProtocol:
        info = self._info_for_actor(batch)
        action_dist_input_BD, hidden_BH = self.actor(batch.obs, state=state, info=info)

        if isinstance(action_dist_input_BD, tuple):
            dist = self.dist_fn(*action_dist_input_BD)
        else:
            dist = self.dist_fn(action_dist_input_BD)

        act_B = (
            dist.mode
            if self.deterministic_eval and not self.is_within_training_step
            else dist.sample()
        )
        result = Batch(logits=action_dist_input_BD, act=act_B, state=hidden_BH, dist=dist)
        return cast(DistBatchProtocol, result)
