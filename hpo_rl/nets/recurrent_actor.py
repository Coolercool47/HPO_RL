from tianshou.utils.net.discrete import DiscreteActor
import torch
from tianshou.data import Batch

class MaskedRecurrentDiscreteActor(DiscreteActor):
    """Actor для рекуррентных сетей с поддержкой action masking.
    
    Решает проблему: стандартный DiscreteActor использует MLP с flatten(1),
    что ломает 3D вход [batch, seq_len, hidden_dim] от RNN.
    
    Этот класс:
    1. Вызывает preprocess_net напрямую (а не через super().forward())
    2. Применяет финальный Linear per-timestep (без flatten)
    3. Накладывает action mask на логиты
    """
    def __init__(self, preprocess_net, action_shape, hidden_sizes=()):
        super().__init__(
            preprocess_net=preprocess_net, 
            action_shape=action_shape, 
            hidden_sizes=hidden_sizes, 
            softmax_output=False  # Важно! PPO работает с чистыми логитами
        )

    def forward(self, obs, state=None, info=None):
        # 1. RNN forward: [batch, seq_len, hidden_dim] или [batch, hidden_dim]
        x, hidden = self.preprocess(obs, state=state, info=info)
        
        # 2. Применяем финальный Linear per-timestep
        #    x может быть 3D [batch, seq_len, hidden] (обучение)
        #    или 2D [batch, hidden] (инференс)
        #    Linear обрабатывает оба случая корректно (работает по последней оси),
        #    но MLP внутри self.last делает flatten(1) — это ломает 3D.
        #    Поэтому для 3D: reshape → linear → reshape back.
        is_3d = x.dim() == 3
        if is_3d:
            batch_size, seq_len, feat_dim = x.shape
            x_flat = x.reshape(batch_size * seq_len, feat_dim)
            logits = self.last(x_flat)
            logits = logits.reshape(batch_size, seq_len, -1)
        else:
            logits = self.last(x)
        
        # 3. Action masking
        mask = None
        if isinstance(obs, (dict, Batch)) and "mask" in obs:
            mask = obs["mask"]
        elif hasattr(obs, "mask"):
            mask = obs.mask
            
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
            
            # Согласование размерностей маски и логитов
            # 3D obs → mask [batch, seq_len, action] и logits [batch, seq_len, action] — OK
            # 2D obs → mask [batch, action] и logits [batch, action] — OK
            if mask.dim() < logits.dim():
                mask = mask.unsqueeze(1)
                
            logits = logits.masked_fill(~mask, -1e8)
            
        return logits, hidden