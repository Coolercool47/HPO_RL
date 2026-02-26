from tianshou.algorithm.modelfree.dqn import DQN
from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.data.types import RolloutBatchProtocol, BatchWithReturnsProtocol
from tianshou.algorithm.modelfree.reinforce import SimpleLossTrainingStats
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
    
    Это аналог подхода R2D2 (Recurrent Experience Replay in Distributed RL).
    
    Args:
        seq_len: длина последовательности для RNN (включая целевой шаг).
            Первые (seq_len - 1) шагов используются как "burn-in" для прогрева
            hidden state, loss считается только по последнему шагу.
        **kwargs: все параметры стандартного DQN.
    """
    
    def __init__(self, seq_len: int = 10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len

    def _build_sequences_from_buffer(
        self, buffer: ReplayBuffer, indices: np.ndarray
    ) -> np.ndarray:
        """Для каждого индекса строим последовательность из seq_len шагов назад.
        
        Использует buffer.prev() для навигации назад по эпизоду.
        Если эпизод короче seq_len, prev() упрётся в начало эпизода и 
        будет возвращать тот же индекс — это корректное поведение для RNN.
        
        Returns:
            seq_indices: массив формы (len(indices), seq_len) — индексы в буфере.
        """
        batch_size = len(indices)
        # seq_indices[i, -1] = indices[i] (целевой шаг)
        # seq_indices[i, -2] = prev(indices[i]) и т.д.
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
        """Вычисляем n-step returns стандартным способом, затем строим последовательности.
        
        1. compute_nstep_return — считает таргеты Q-learning (как в обычном DQN)
        2. Из буфера извлекаем последовательности длиной seq_len для каждого индекса
        3. Упаковываем obs в 3D формат [batch, seq_len, obs_dim] для RNN
        """
        # Шаг 1: стандартный расчёт n-step return для целевых индексов
        batch = self.compute_nstep_return(
            batch=batch,
            buffer=buffer,
            indices=indices,
            target_q_fn=self._target_q,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        
        # Шаг 2: строим последовательности из буфера
        seq_indices = self._build_sequences_from_buffer(buffer, indices)
        # seq_indices: (batch_size, seq_len)
        
        # Шаг 3: извлекаем наблюдения для всей последовательности
        # buffer[seq_indices.flatten()] даст нам все нужные obs
        seq_batch = buffer[seq_indices.flatten()]
        
        # Извлекаем obs и перестраиваем в 3D
        batch_size = len(indices)
        
        def reshape_obs(obs):
            """Рекурсивно перестраиваем obs в формат [batch, seq_len, ...]."""
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
        
        # Заменяем obs в батче на последовательность для RNN
        # obs.obs будет 3D [batch, seq_len, obs_dim] — для GRU
        batch.obs = seq_obs
        
        # ВАЖНО: mask должна быть 2D [batch, action_dim] (только последний шаг),
        # потому что DiscreteQLearningPolicy.compute_q_value() работает с 2D mask.
        # MaskedRecurrentNet внутри сам берёт mask[:, -1, :] из 3D, но policy-level
        # masking в compute_q_value получает mask напрямую из batch.obs.mask.
        # Поэтому оставляем mask только для целевого (последнего) шага.
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
