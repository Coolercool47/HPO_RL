"""
ChunkedRNNPPO — PPO с рекуррентными сетями для tianshou 2.0.0.

Подход: Chunked BPTT (Truncated Backpropagation Through Time).
On-policy данные нарезаются на последовательности (chunks) длиной seq_len.
Actor и Critic получают 3D-входы [num_chunks, seq_len, obs_dim],
но loss считается по ВСЕМ шагам (не только по последнему).

Ключевые отличия от обычного PPO:
1. _preprocess_batch: GAE считается на ПЛОСКИХ данных (корректно),
   затем obs перестраивается в 3D chunks для RNN forward pass.
2. _update_with_batch: полностью переопределён для обработки 3D obs
   с 1D advantages/returns/logp_old.
"""

from typing import cast

from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.a2c import A2CTrainingStats
from tianshou.algorithm.optim import OptimizerFactory
from tianshou.data import Batch, ReplayBuffer, SequenceSummaryStats, to_torch_as
from tianshou.data.types import LogpOldProtocol, RolloutBatchProtocol
import torch
import numpy as np


class ChunkedRNNPPO(PPO):
    """PPO с поддержкой рекуррентных сетей через Chunked BPTT.

    On-policy данные (собранные коллектором) нарезаются на последовательности
    длиной ``seq_len``. Actor-RNN и Critic-RNN получают 3D-вход
    ``[num_chunks, seq_len, obs_dim]`` и выдают ``[num_chunks, seq_len, ...]``.

    GAE и returns вычисляются в стандартном 1D-формате (на плоских данных),
    т.к. ``compute_episodic_return`` требует знать границы эпизодов из буфера.
    Затем они reshape-аются в ``[num_chunks, seq_len]`` и используются для loss.

    Args:
        seq_len: длина окна (chunk) для RNN.
            ``collection_step_num_env_steps`` должно быть кратно ``seq_len``.
        **kwargs: все параметры стандартного PPO.
    """

    def __init__(self, seq_len: int = 16, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len

    # ------------------------------------------------------------------
    #  Утилиты для reshape
    # ------------------------------------------------------------------

    @staticmethod
    def _reshape_to_chunks(x, num_chunks: int, seq_len: int):
        """Reshape 1D/2D данные в [num_chunks, seq_len, ...].
        
        Работает рекурсивно для Batch.
        """
        if isinstance(x, Batch):
            new_b = Batch()
            for k, v in x.items():
                new_b[k] = ChunkedRNNPPO._reshape_to_chunks(v, num_chunks, seq_len)
            return new_b
        elif isinstance(x, torch.Tensor):
            return x.reshape(num_chunks, seq_len, *x.shape[1:])
        elif isinstance(x, np.ndarray):
            return x.reshape(num_chunks, seq_len, *x.shape[1:])
        return x

    @staticmethod
    def _flatten_chunks(x):
        """Flatten [num_chunks, seq_len, ...] → [num_chunks * seq_len, ...].
        
        Работает рекурсивно для Batch.
        """
        if isinstance(x, Batch):
            new_b = Batch()
            for k, v in x.items():
                new_b[k] = ChunkedRNNPPO._flatten_chunks(v)
            return new_b
        elif isinstance(x, torch.Tensor):
            return x.reshape(-1, *x.shape[2:])
        elif isinstance(x, np.ndarray):
            return x.reshape(-1, *x.shape[2:])
        return x

    # ------------------------------------------------------------------
    #  Переопределение _preprocess_batch (tianshou 2.0.0 API)
    # ------------------------------------------------------------------

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> LogpOldProtocol:
        """Вычисляет GAE на плоских данных, затем упаковывает в chunks.

        Порядок:
        1. _add_returns_and_advantages — GAE на 1D данных (v_s, returns, adv)
        2. Обрезка до кратности seq_len
        3. Вычисление logp_old на 3D obs (прогон actor-RNN по chunks)
        4. Сохранение всего в формате chunks [num_chunks, seq_len, ...]
        """
        if self.recompute_adv:
            self._buffer, self._indices = buffer, indices

        # ---- Шаг 1: GAE на плоских данных ----
        # Critic получает 2D obs → 1D v_s. Это нормально: при сборе данных
        # collector вызывает critic пошагово, GAE тоже считается пошагово.
        batch = self._add_returns_and_advantages(batch, buffer, indices)
        batch.act = to_torch_as(batch.act, batch.v_s)

        # ---- Шаг 2: обрезка до кратности seq_len ----
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

        # ---- Шаг 3: Reshape obs в 3D для RNN ----
        # obs: [valid_len, obs_dim] → [num_chunks, seq_len, obs_dim]
        # obs_next аналогично (нужен для recompute_adv)
        batch.obs = self._reshape_to_chunks(batch.obs, num_chunks, self.seq_len)
        batch.obs_next = self._reshape_to_chunks(batch.obs_next, num_chunks, self.seq_len)

        # act, returns, adv, v_s: [valid_len] → [num_chunks, seq_len]
        act = batch.act
        if act.dim() == 1:
            # Discrete: [valid_len] -> [num_chunks, seq_len]
            batch.act = act.reshape(num_chunks, self.seq_len)
        elif act.dim() == 2:
            # Continuous: [valid_len, action_dim] -> [num_chunks, seq_len, action_dim]
            batch.act = act.reshape(num_chunks, self.seq_len, act.shape[-1])
        else:
            raise ValueError(
                f"Unexpected batch.act shape {tuple(act.shape)}; expected 1D (discrete) "
                f"or 2D (continuous vector actions)."
            )
        batch.returns = batch.returns.reshape(num_chunks, self.seq_len)
        batch.adv = batch.adv.reshape(num_chunks, self.seq_len)
        batch.v_s = batch.v_s.reshape(num_chunks, self.seq_len)

        # ---- Шаг 4: logp_old на 3D данных ----
        # Прогоняем actor-RNN по полным chunks, чтобы logp_old соответствовал
        # тому же RNN-контексту, что будет при обновлении.
        logp_old = []
        with torch.no_grad():
            for minibatch in batch.split(self.max_batchsize, shuffle=False, merge_last=True):
                # minibatch.obs: [mini_chunks, seq_len, obs_dim]
                # policy forward → dist: [mini_chunks, seq_len, action_dim]
                dist = self.policy(minibatch).dist
                # log_prob(act): act [mini_chunks, seq_len] → [mini_chunks, seq_len]
                logp_old.append(dist.log_prob(minibatch.act))
        batch.logp_old = torch.cat(logp_old, dim=0)
        # logp_old: [num_chunks, seq_len]

        return cast(LogpOldProtocol, batch)

    # ------------------------------------------------------------------
    #  Переопределение _update_with_batch
    # ------------------------------------------------------------------

    def _update_with_batch(
        self,
        batch: LogpOldProtocol,
        batch_size: int | None,
        repeat: int,
    ) -> A2CTrainingStats:
        """PPO update с 3D obs для RNN.

        Данные в batch:
            obs:       [num_chunks, seq_len, obs_dim]  (или Batch с такими полями)
            act:       [num_chunks, seq_len]
            returns:   [num_chunks, seq_len]
            adv:       [num_chunks, seq_len]
            v_s:       [num_chunks, seq_len]
            logp_old:  [num_chunks, seq_len]

        batch.split(batch_size) делит по первой оси (num_chunks).
        Каждый minibatch имеет те же формы, но с mini_chunks вместо num_chunks.
        """
        losses, clip_losses, vf_losses, ent_losses = [], [], [], []
        gradient_steps = 0
        split_batch_size = batch_size or -1

        for step in range(repeat):
            if self.recompute_adv and step > 0:
                # Нужно пересчитать advantages. Для этого нужно вернуть данные в 1D,
                # вызвать _add_returns_and_advantages, и снова reshape.
                flat_batch = self._flatten_to_1d_for_gae(batch)
                flat_batch = self._add_returns_and_advantages(
                    flat_batch, self._buffer, self._indices
                )
                num_chunks = batch.act.shape[0]
                batch.returns = flat_batch.returns.reshape(num_chunks, self.seq_len)
                batch.adv = flat_batch.adv.reshape(num_chunks, self.seq_len)
                batch.v_s = flat_batch.v_s.reshape(num_chunks, self.seq_len)

            for minibatch in batch.split(split_batch_size, merge_last=True):
                gradient_steps += 1

                # -- Actor loss --
                # minibatch.obs: [mini_chunks, seq_len, obs_dim]
                # policy forward прогоняет RNN → dist: [mini_chunks, seq_len, act_dim]
                dist = self.policy(minibatch).dist

                # advantages: [mini_chunks, seq_len]
                advantages = minibatch.adv
                if self.advantage_normalization:
                    mean, std = advantages.mean(), advantages.std()
                    advantages = (advantages - mean) / (std + self._eps)

                # log_prob: [mini_chunks, seq_len] 
                new_logp = dist.log_prob(minibatch.act)
                ratios = (new_logp - minibatch.logp_old).exp().float()
                # ratios: [mini_chunks, seq_len]

                surr1 = ratios * advantages
                surr2 = ratios.clamp(1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages

                if self.dual_clip:
                    clip1 = torch.min(surr1, surr2)
                    clip2 = torch.max(clip1, self.dual_clip * advantages)
                    clip_loss = -torch.where(advantages < 0, clip2, clip1).mean()
                else:
                    clip_loss = -torch.min(surr1, surr2).mean()

                # -- Critic loss --
                # critic forward: obs [mini_chunks, seq_len, obs_dim] → value [mini_chunks, seq_len]
                value = self.critic(minibatch.obs)
                # RecurrentCritic возвращает [mini_chunks, seq_len] после squeeze(-1)
                # Убедимся что размерности совпадают с returns
                if value.shape != minibatch.returns.shape:
                    value = value.reshape(minibatch.returns.shape)

                if self.value_clip:
                    v_clip = minibatch.v_s + (value - minibatch.v_s).clamp(
                        -self.eps_clip, self.eps_clip,
                    )
                    vf1 = (minibatch.returns - value).pow(2)
                    vf2 = (minibatch.returns - v_clip).pow(2)
                    vf_loss = torch.max(vf1, vf2).mean()
                else:
                    vf_loss = (minibatch.returns - value).pow(2).mean()

                # -- Entropy --
                ent_loss = dist.entropy().mean()

                # -- Total loss --
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

    def _flatten_to_1d_for_gae(self, batch):
        """Вспомогательный метод для recompute_adv: flatten obs обратно в 1D."""
        flat = Batch()
        flat.obs = self._flatten_chunks(batch.obs)
        flat.obs_next = self._flatten_chunks(batch.obs_next)
        if batch.act.dim() == 3:
            flat.act = batch.act.reshape(
                batch.act.shape[0] * batch.act.shape[1],
                batch.act.shape[2],
            )
        else:
            flat.act = batch.act.reshape(-1)
        flat.rew = batch.rew.reshape(-1) if hasattr(batch, 'rew') else None
        flat.done = batch.done.reshape(-1) if hasattr(batch, 'done') else None
        flat.terminated = batch.terminated.reshape(-1) if hasattr(batch, 'terminated') else None
        flat.truncated = batch.truncated.reshape(-1) if hasattr(batch, 'truncated') else None
        return flat