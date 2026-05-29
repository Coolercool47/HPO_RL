"""DQN с episodic sequence sampling для рекуррентных Q-сетей (burn-in)."""

from tianshou.algorithm.modelfree.dqn import DQN
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.data.types import RolloutBatchProtocol, BatchWithReturnsProtocol
from tianshou.algorithm.modelfree.reinforce import SimpleLossTrainingStats
import torch
import numpy as np
from typing import cast


class ChunkedRNNDQN(DQN):
    """DQN с episodic sequence sampling для рекуррентных Q-сетей.

        Стандартный DQN сэмплирует отдельные переходы; для RNN нужны
        последовательности из одного эпизода (burn-in, аналог R2D2).

        Args:
            seq_len: длина последовательности (burn-in + целевой шаг)
            **kwargs: параметры :class:`~tianshou.algorithm.modelfree.dqn.DQN`
    """
    
    def __init__(self, seq_len: int = 10, *args, **kwargs):
        """Инициализирует ChunkedRNNDQN.

        Args:
            seq_len: длина последовательности (burn-in + целевой шаг).
            *args, **kwargs: параметры :class:`~tianshou.algorithm.modelfree.dqn.DQN`.
        """
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len

    def _build_sequences_from_buffer(
        self, buffer: ReplayBuffer, indices: np.ndarray
    ) -> np.ndarray:
        """Строит последовательности индексов длиной ``seq_len`` для каждого шага.

        Args:
            buffer: replay-буфер с методом ``prev()`` для навигации по эпизоду.
            indices: индексы целевых шагов в буфере.

        Returns:
            массив формы ``(len(indices), seq_len)`` — индексы в буфере.
            При коротком эпизоде ``prev()`` дублирует начальный индекс (корректно для RNN).
        """
        batch_size = len(indices)
        seq_indices = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        seq_indices[:, -1] = indices
        
        for t in range(self.seq_len - 2, -1, -1):
            seq_indices[:, t] = buffer.prev(seq_indices[:, t + 1])
        
        return seq_indices

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        """n-step returns и упаковка obs в последовательности для RNN.

        Args:
            batch: rollout-батч.
            buffer: replay-буфер.
            indices: индексы шагов для обучения.

        Returns:
            батч с n-step ``returns`` и ``obs`` формы ``[batch, seq_len, obs_dim]``.
        """
        batch = self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        
        seq_indices = self._build_sequences_from_buffer(buffer, indices)
        seq_batch = buffer[seq_indices.flatten()]
        batch_size = len(indices)
        
        def reshape_obs(obs):
            """Рекурсивно перестраивает obs в формат ``[batch, seq_len, ...]``.

            Args:
                obs: тензор, ndarray, dict или Batch.

            Returns:
                obs той же структуры с осями batch и seq_len.
            """
            if isinstance(obs, (dict, Batch)):
                new_obs = Batch()
                for k, v in obs.items():
                    new_obs[k] = reshape_obs(v)
                return new_obs
            elif isinstance(obs, torch.Tensor):
                return obs.view(batch_size, self.seq_len, *obs.shape[1:])
            elif isinstance(obs, np.ndarray):
                return obs.reshape(batch_size, self.seq_len, *obs.shape[1:])
            return obs
        
        seq_obs = reshape_obs(seq_batch.obs)
        batch.obs = seq_obs

        if isinstance(batch.obs, (dict, Batch)) and "mask" in batch.obs:
            mask_seq = batch.obs["mask"]
            if isinstance(mask_seq, np.ndarray) and mask_seq.ndim == 3:
                batch.obs["mask"] = mask_seq[:, -1, :]
            elif isinstance(mask_seq, torch.Tensor) and mask_seq.dim() == 3:
                batch.obs["mask"] = mask_seq[:, -1, :]
        
        return batch

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> SimpleLossTrainingStats:
        """Градиентный шаг DQN по Q-значениям последнего шага последовательности.

        Args:
            batch: батч с ``obs`` ``[batch, seq_len, ...]`` и полем ``returns``.

        Returns:
            статистика обучения с полем ``loss``.
        """
        self._periodically_update_lagged_network_weights()
        
        weight = batch.pop("weight", 1.0)
        
        result = self.policy(batch)
        q = result.logits  
        q = q[np.arange(len(q)), batch.act]
        
        returns = to_torch_as(batch.returns.flatten(), q)
        td_error = returns - q

        if self.huber_loss_delta is not None:
            y = q.reshape(-1, 1)
            t = returns.reshape(-1, 1)
            loss = torch.nn.functional.huber_loss(
                y, t, delta=self.huber_loss_delta, reduction="mean"
            )
        else:
            loss = (td_error.pow(2) * weight).mean()

        batch.weight = td_error  
        self.optim.step(loss)

        return SimpleLossTrainingStats(loss=loss.item())
