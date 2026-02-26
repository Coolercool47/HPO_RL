from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.data import Batch
import torch
import numpy as np

class ChunkedRNNPPO(PPO):
    def __init__(self, seq_len=16, *args, **kwargs):
        """
        seq_len: длина "окна" (последовательности), которую будет видеть RNN.
        """
        super().__init__(*args, **kwargs)
        self.seq_len = seq_len

    def process_fn(self, batch: Batch, buffer, indices: np.ndarray) -> Batch:
        # 1. Сначала считаем GAE и Returns как обычно (пока данные еще упорядочены)
        batch = super().process_fn(batch, buffer, indices)
        
        # 2. Формируем "окна" (chunks) для RNN
        total_steps = len(batch)
        num_chunks = total_steps // self.seq_len
        valid_len = num_chunks * self.seq_len
        
        if valid_len == 0:
            raise ValueError(f"Собрано {total_steps} шагов. Это меньше длины окна seq_len={self.seq_len}!")

        # Обрезаем хвостик (если количество шагов не делится на seq_len нацело)
        batch = batch[:valid_len]
        
        # 3. Переупаковываем плоский батч в 3D-формат [num_chunks, seq_len, ...]
        def reshape_batch(b):
            new_b = Batch()
            for k, v in b.items():
                if isinstance(v, torch.Tensor):
                    new_b[k] = v.view(num_chunks, self.seq_len, *v.shape[1:])
                elif isinstance(v, np.ndarray):
                    new_b[k] = v.reshape(num_chunks, self.seq_len, *v.shape[1:])
                elif isinstance(v, Batch):
                    new_b[k] = reshape_batch(v)
            return new_b
            
        # Теперь Tianshou будет перемешивать целые секвенции (chunks), а не отдельные шаги!
        return reshape_batch(batch)