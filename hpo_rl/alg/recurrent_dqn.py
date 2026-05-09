from tianshou.algorithm.modelfree.dqn import DQN
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.data.types import RolloutBatchProtocol, BatchWithReturnsProtocol
from tianshou.algorithm.modelfree.reinforce import SimpleLossTrainingStats
from tianshou.algorithm.algorithm_base import Algorithm
import tianshou.algorithm.optim as opt
import torch
import numpy as np
from typing import cast


class ChunkedRNNDQN(DQN):
    """DQN с поддержкой рекуррентных сетей через episodic sequence sampling.
    
    Стандартный DQN сэмплирует из буфера отдельные переходы случайным образом,
    что несовместимо с RNN — скрытое состояние теряет смысл.
    
    ChunkedRNNDQN решает эту проблему:
    1. Сэмплирует из буфера случайные индексы
    2. Для каждого индекса восстанавливает последовательность из `seq_len` 
       предшествующих шагов (из того же эпизода)
    3. Прогоняет всю последовательность через RNN, но loss считает только 
       по последнему шагу (burn-in strategy)
    
    ИСПРАВЛЕННЫЕ БАГИ по сравнению с оригинальным DQN:
    - Gradient clipping через max_grad_norm (критично для BPTT через RNN)
    - _target_q вычисляет Q-targets с seq_len последовательным контекстом,
      а не по одному шагу с нулевым hidden state (несоответствие train/target)
    
    Это аналог подхода R2D2 (Recurrent Experience Replay in Distributed RL).
    
    Args:
        seq_len: длина последовательности для RNN (включая целевой шаг).
            Первые (seq_len - 1) шагов используются как "burn-in" для прогрева
            hidden state, loss считается только по последнему шагу.
        max_grad_norm: максимальная L2-норма градиентов (gradient clipping).
            Критично для стабильного обучения RNN через BPTT. По умолчанию 10.0.
        **kwargs: все параметры стандартного DQN.
    """
    
    def __init__(self, seq_len: int = 10, max_grad_norm: float = 10.0, *args, **kwargs):
        # Must be set BEFORE super().__init__() because _create_policy_optimizer
        # is called during parent __init__ and reads this attribute.
        self._rnn_max_grad_norm = max_grad_norm
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len

    def _create_policy_optimizer(self, optim_factory: opt.OptimizerFactory) -> Algorithm.Optimizer:
        """Override to enable gradient clipping — критично для стабильного BPTT через RNN."""
        return self._create_optimizer(self.policy, optim_factory, max_grad_norm=self._rnn_max_grad_norm)

    def _build_seq_obs_batch(self, buffer: ReplayBuffer, end_indices: np.ndarray) -> Batch:
        """Строит батч с 3D obs [batch, seq_len, obs_dim] для последовательностей,
        заканчивающихся на end_indices. Mask обрезается до 2D последнего шага.
        """
        batch_size = len(end_indices)
        seq_indices = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        seq_indices[:, -1] = end_indices
        for t in range(self.seq_len - 2, -1, -1):
            seq_indices[:, t] = buffer.prev(seq_indices[:, t + 1])

        seq_batch = buffer[seq_indices.flatten()]

        def reshape_obs(obs):
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
        # Обрезаем mask до 2D (только последний шаг): нужно для compute_q_value
        if isinstance(seq_obs, (dict, Batch)) and "mask" in seq_obs:
            mask_seq = seq_obs["mask"]
            if isinstance(mask_seq, np.ndarray) and mask_seq.ndim == 3:
                seq_obs["mask"] = mask_seq[:, -1, :]
            elif isinstance(mask_seq, torch.Tensor) and mask_seq.dim() == 3:
                seq_obs["mask"] = mask_seq[:, -1, :]
        return seq_obs

    def _target_q(self, buffer: ReplayBuffer, indices: np.ndarray) -> torch.Tensor:
        """Вычисляем target Q-values с полным последовательным контекстом.

        ИСПРАВЛЕНИЕ: Оригинальный DQN._target_q использует одиночный obs_next
        (2D, нулевой hidden state). Для RNN это порождает системное несоответствие:
        - Текущая Q-сеть обучается с seq_len шагами контекста
        - TD-таргеты вычисляются без контекста (hidden = zeros)
        → Сеть не может научиться использовать скрытое состояние, агент не обучается.

        Решение: строим последовательности длиной seq_len, заканчивающиеся на
        buffer.next(indices) — таким образом таргеты вычисляются с тем же
        контекстом, что и предсказания.
        """
        # buffer.next(i) = i для терминальных шагов (эпизод закончился),
        # что корректно — value_mask обнулит Q для терминальных состояний.
        next_indices = buffer.next(indices)
        seq_obs = self._build_seq_obs_batch(buffer, next_indices)

        obs_next_batch = Batch(obs=seq_obs, info=[None] * len(indices))

        result = self.policy(obs_next_batch)
        if self.use_target_network:
            target_q = self.policy(obs_next_batch, model=self.model_old).logits
        else:
            target_q = result.logits
        if self.is_double:
            return target_q[np.arange(len(result.act)), result.act]
        return target_q.max(dim=1)[0]

    def _preprocess_batch(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> BatchWithReturnsProtocol:
        """Вычисляем n-step returns (с контекстным _target_q), затем строим последовательности.
        
        1. compute_nstep_return — считает таргеты Q-learning с RNN-контекстом
        2. Из буфера извлекаем последовательности длиной seq_len для каждого индекса
        3. Упаковываем obs в 3D формат [batch, seq_len, obs_dim] для RNN
        """
        # Шаг 1: расчёт n-step return (теперь _target_q использует seq-контекст)
        batch = self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        
        # Шаг 2+3: строим последовательности obs [batch, seq_len, obs_dim].
        # Последний элемент последовательности — это целевой шаг (indices).
        batch.obs = self._build_seq_obs_batch(buffer, indices)
        
        return batch

    def _update_with_batch(
        self,
        batch: RolloutBatchProtocol,
    ) -> SimpleLossTrainingStats:
        """Обновление с учётом последовательностей.
        
        RNN получает последовательность [batch, seq_len, obs_dim],
        а loss считается по Q-значениям последнего шага (целевого).
        """
        self._periodically_update_lagged_network_weights()
        
        weight = batch.pop("weight", 1.0)
        
        # Forward pass: RNN получает полную последовательность
        # MaskedRecurrentNet сам определит is_sequence=True и обработает 3D вход
        result = self.policy(batch)
        q = result.logits  # Q-values для последнего шага последовательности
        
        # Действия — скаляры для каждого элемента батча (целевой шаг)
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

        batch.weight = td_error  # prio-buffer
        self.optim.step(loss)

        return SimpleLossTrainingStats(loss=loss.item())
