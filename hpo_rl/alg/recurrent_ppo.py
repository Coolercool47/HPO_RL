"""
ChunkedRNNPPO — PPO с рекуррентными сетями для tianshou 2.0.0.

Подход: Chunked BPTT (Truncated Backpropagation Through Time).
On-policy данные нарезаются на последовательности (chunks) длиной seq_len.
Actor и Critic получают 3D-входы [num_chunks, seq_len, obs_dim],
но loss считается по ВСЕМ шагам (не только по последнему).

Ключевые отличия от обычного PPO:
1. `_add_returns_and_advantages`: для рекуррентного критика значения V(s), V(s')
   считаются вдоль **непрерывных сегментов** траекторий (пер-env буферы + episode_start),
   а не через независимый one-step GRU.
2. `_preprocess_batch`: GAE на плоских данных, затем упаковка в chunks + `episode_reset`
   (границы эпизода **и** первый шаг каждого chunk для согласованности с truncated BPTT).
3. `_update_with_batch`: PPO-loss по 3D obs / act с учётом `episode_reset` (через policy info).

Рекомендуется использовать :class:`~hpo_rl.nets.recurrent_policy.RecurrentProbabilisticActorPolicy`,
чтобы маска ``episode_reset`` доходила до `RecurrentBaseNet`.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from tianshou.algorithm.algorithm_base import Algorithm
from tianshou.algorithm.modelfree.a2c import A2CTrainingStats
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.data import Batch, ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import (
    BatchWithAdvantagesProtocol,
    LogpOldProtocol,
    RolloutBatchProtocol,
)


class ChunkedRNNPPO(PPO):
    """PPO с поддержкой рекуррентных сетей через Chunked BPTT.

    Args:
        seq_len: длина окна (chunk) для RNN. Число шагов в выборке после сбора
            не должно быть меньше `seq_len`; «хвост» обрезается до кратности `seq_len`.
        collect_step_num_env_steps: если задано — проверяется, что делится на `seq_len`.
        **kwargs: аргументы :class:`~tianshou.algorithm.modelfree.ppo.PPO` (keyword-only).
    """

    def __init__(self, **kwargs: Any) -> None:
        seq_len = int(kwargs.pop("seq_len", 16))
        collect_step_num_env_steps = kwargs.pop("collect_step_num_env_steps", None)
        super().__init__(**kwargs)
        self.seq_len = seq_len
        if collect_step_num_env_steps is not None:
            c = int(collect_step_num_env_steps)
            if c % self.seq_len != 0:
                raise ValueError(
                    f"recurrent PPO: collect_step_num_env_steps={c} must be divisible by "
                    f"seq_len={self.seq_len} so each collect yields an integer number of chunks."
                )

    #  Утилиты для reshape / маски сброса

    @staticmethod
    def _reshape_to_chunks(x: Any, num_chunks: int, seq_len: int) -> Any:
        if isinstance(x, Batch):
            new_b = Batch()
            for k in x.keys():
                new_b[k] = ChunkedRNNPPO._reshape_to_chunks(x[k], num_chunks, seq_len)
            return new_b
        if isinstance(x, torch.Tensor):
            return x.reshape(num_chunks, seq_len, *x.shape[1:])
        if isinstance(x, np.ndarray):
            return x.reshape(num_chunks, seq_len, *x.shape[1:])
        return x

    @staticmethod
    def _flatten_chunks(x: Any) -> Any:
        if isinstance(x, Batch):
            new_b = Batch()
            for k in x.keys():
                new_b[k] = ChunkedRNNPPO._flatten_chunks(x[k])
            return new_b
        if isinstance(x, torch.Tensor):
            return x.reshape(-1, *x.shape[2:])
        if isinstance(x, np.ndarray):
            return x.reshape(-1, *x.shape[2:])
        return x

    @staticmethod
    def _buffer_subindices(buffer: ReplayBuffer, indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices)
        if hasattr(buffer, "_offset"):
            off = np.asarray(buffer._offset)
            return np.searchsorted(off, indices, side="right") - 1
        return np.zeros(len(indices), dtype=np.int64)

    @staticmethod
    def _episode_start_flags(buffer: ReplayBuffer, indices: np.ndarray, batch: Batch) -> np.ndarray:
        """True на шаге t, если RNN-контекст должен обнулиться перед обработкой s_t."""
        n = len(batch)
        term = np.asarray(batch.terminated)
        trunc = np.asarray(batch.truncated)
        done_prev = np.logical_or(term, trunc)
        buf_ids = ChunkedRNNPPO._buffer_subindices(buffer, indices)
        starts = np.zeros(n, dtype=bool)
        starts[0] = True
        for i in range(1, n):
            cross_env = buf_ids[i] != buf_ids[i - 1]
            starts[i] = cross_env or bool(done_prev[i - 1])
        return starts

    def _wrap_seq_obs(self, obs: Any, device: torch.device) -> Any:
        """obs: срез по времени [L, ...] или dict с ключом 'obs' (+ 'mask'). -> batch для RNN [1,L,...]."""
        if isinstance(obs, Batch) and "obs" in obs:
            d = {k: obs[k] for k in obs.keys()}
        elif isinstance(obs, dict):
            d = dict(obs)
        elif isinstance(obs, Batch):
            core = obs.__dict__.get("obs", obs)
            t = torch.as_tensor(np.asarray(core), dtype=torch.float32, device=device)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            return t.unsqueeze(0)
        else:
            d = None
        if d is not None:
            core = torch.as_tensor(np.asarray(d["obs"]), dtype=torch.float32, device=device)
            if core.dim() == 1:
                core = core.unsqueeze(0)
            b = Batch()
            b.obs = core.unsqueeze(0)
            if "mask" in d:
                m = torch.as_tensor(np.asarray(d["mask"]), dtype=torch.bool, device=device)
                if m.dim() == 1:
                    m = m.unsqueeze(0)
                b.mask = m.unsqueeze(0)
            return b
        t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.unsqueeze(0)

    def _critic_values_sequential(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray, *, field: str) -> torch.Tensor:
        """V(s) или V(s') по сегментам (корректный рекуррентный контекст)."""
        n = len(batch)
        starts = self._episode_start_flags(buffer, indices, batch)
        device = next(self.critic.parameters()).device
        pieces: list[torch.Tensor] = []
        seg_start = 0
        for i in range(1, n + 1):
            if i == n or starts[i]:
                sub = batch[seg_start:i]
                obs_field = sub.obs if field == "obs" else sub.obs_next
                obs_seq = self._wrap_seq_obs(obs_field, device)
                ep_reset = torch.zeros((1, len(sub)), dtype=torch.bool, device=device)
                ep_reset[0, 0] = True
                with torch.no_grad():
                    v_seg = self.critic(obs_seq, info={"episode_reset": ep_reset})
                pieces.append(v_seg.reshape(-1))
                seg_start = i
        return torch.cat(pieces, dim=0)

    def _add_returns_and_advantages(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithAdvantagesProtocol:
        if self.recompute_adv:
            self._buffer, self._indices = buffer, indices

        b = cast(Batch, batch)
        with torch.no_grad():
            v_s = self._critic_values_sequential(b, buffer, indices, field="obs")
            v_s_ = self._critic_values_sequential(b, buffer, indices, field="obs_next")

        batch.v_s = v_s.flatten()
        v_s_np = batch.v_s.cpu().numpy()
        v_s__np = v_s_.flatten().cpu().numpy() * Algorithm.value_mask(buffer, indices)

        if self.return_scaling:
            scale = np.sqrt(self.ret_rms.var + self._eps)
            v_s_np = v_s_np * scale
            v_s__np = v_s__np * scale

        unnormalized_returns, advantages = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_=v_s__np,
            v_s=v_s_np,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        if self.return_scaling:
            batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        batch.returns = to_torch_as(batch.returns, batch.v_s)
        batch.adv = to_torch_as(advantages, batch.v_s)
        return cast(BatchWithAdvantagesProtocol, batch)

    @staticmethod
    def _tensor_action_suffix(act: torch.Tensor) -> tuple[int, ...]:
        if act.dim() <= 1:
            return ()
        return tuple(int(x) for x in act.shape[1:])

    @staticmethod
    def _log_probs_for_dist(dist: torch.distributions.Distribution, acts: torch.Tensor) -> torch.Tensor:
        lp = dist.log_prob(acts)
        # For continuous actions with per-element log_prob (e.g. raw Normal without
        # Independent wrapper), lp has the same shape as acts including the action dim.
        # In chunked mode acts is [B, T] (discrete) or [B, T, A] (continuous).
        # Sum only when there's a genuine action dimension beyond batch+time (dim > 2).
        if lp.shape == acts.shape and lp.dim() > 2:
            lp = lp.sum(dim=-1)
        return lp

    #  Переопределение _preprocess_batch (tianshou 2.0.0 API)

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> LogpOldProtocol:
        if self.recompute_adv:
            self._buffer, self._indices = buffer, indices

        batch = self._add_returns_and_advantages(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)

        total_steps = len(batch)
        num_chunks = total_steps // self.seq_len
        valid_len = num_chunks * self.seq_len

        if num_chunks == 0:
            raise ValueError(
                f"Собрано {total_steps} шагов, что меньше seq_len={self.seq_len}! "
                f"Увеличьте collection_step_num_env_steps."
            )

        if valid_len < total_steps:
            batch = batch[:valid_len]

        starts = self._episode_start_flags(buffer, indices[:valid_len], batch)

        batch.obs = self._reshape_to_chunks(batch.obs, num_chunks, self.seq_len)
        batch.obs_next = self._reshape_to_chunks(batch.obs_next, num_chunks, self.seq_len)

        act_t = batch.act
        act_suffix = self._tensor_action_suffix(act_t)
        batch.act = act_t.reshape(num_chunks, self.seq_len, *act_suffix)

        batch.returns = batch.returns.reshape(num_chunks, self.seq_len)
        batch.adv = batch.adv.reshape(num_chunks, self.seq_len)
        batch.v_s = batch.v_s.reshape(num_chunks, self.seq_len)

        # episode_reset: границы эпизода + первый шаг каждого chunk (обязательный сброс для TBPTT)
        er = np.zeros((num_chunks, self.seq_len), dtype=bool)
        flat = starts[:valid_len].reshape(num_chunks, self.seq_len)
        er |= flat
        er[:, 0] = True
        batch.episode_reset = torch.as_tensor(er, dtype=torch.bool, device=batch.act.device)

        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                dist = self.policy(minibatch).dist
                logp_old.append(self._log_probs_for_dist(dist, minibatch.act))
        batch.logp_old = torch.cat(logp_old, dim=0)

        return cast(LogpOldProtocol, batch)

    def _update_with_batch(
        self,
        batch: LogpOldProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> A2CTrainingStats:
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        gradient_steps = 0
        split_batch_size = batch_size or -1

        for step in range(repeat):
            if self.recompute_adv and step > 0:
                flat_batch = self._flatten_to_1d_for_gae(batch)
                flat_batch = self._add_returns_and_advantages(flat_batch, self._buffer, self._indices)
                num_chunks = batch.act.shape[0]
                batch.returns = flat_batch.returns.reshape(num_chunks, self.seq_len)
                batch.adv = flat_batch.adv.reshape(num_chunks, self.seq_len)
                batch.v_s = flat_batch.v_s.reshape(num_chunks, self.seq_len)

            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1
                dist = self.policy(minibatch).dist

                advantages = minibatch.adv
                if self.advantage_normalization:
                    mean, std = advantages.mean(), advantages.std()
                    advantages = (advantages - mean) / (std + self._eps)

                new_logp = self._log_probs_for_dist(dist, minibatch.act)
                ratios = (new_logp - minibatch.logp_old).exp().float()

                surr1 = ratios * advantages
                surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

                if self.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * advantages)
                    clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                info_kw = {}
                if getattr(minibatch, "episode_reset", None) is not None:
                    info_kw["episode_reset"] = minibatch.episode_reset
                value = self.critic(minibatch.obs, info=info_kw or None)
                if value.shape != minibatch.returns.shape:
                    value = value.reshape(minibatch.returns.shape)

                if self.value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                        -self.eps_clip,
                        self.eps_clip,
                    )
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()

                ent_loss = dist.entropy().mean()
                if ent_loss.dim() > 0:
                    ent_loss = ent_loss.mean()

                loss = clip_loss + self.vf_coef * vf_loss - self.ent_coef * ent_loss
                self.optim.step(loss)

                clip_losses.append(clip_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                losses.append(loss.item())

        return A2CTrainingStats(
            loss=SequenceSummaryStats.from_sequence(losses),
            actor_loss=SequenceSummaryStats.from_sequence(clip_losses),
            vf_loss=SequenceSummaryStats.from_sequence(vf_losses),
            ent_loss=SequenceSummaryStats.from_sequence(ent_losses),
            gradient_steps=gradient_steps,
        )

    def _flatten_to_1d_for_gae(self, batch: LogpOldProtocol) -> Batch:
        """Восстанавливает плоский rollout-батч для пересчёта GAE."""
        flat = Batch()
        flat.obs = self._flatten_chunks(batch.obs)
        flat.obs_next = self._flatten_chunks(batch.obs_next)
        act = batch.act
        flat.act = act.reshape(-1, *act.shape[2:]) if act.dim() > 2 else act.reshape(-1)

        for key in ("rew", "terminated", "truncated", "done"):
            if hasattr(batch, key):
                val = getattr(batch, key)
                flat.__dict__[key] = np.asarray(val).reshape(-1)

        return flat
