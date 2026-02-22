from tianshou.utils.net.discrete import DiscreteActor
import torch
from tianshou.data import Batch

class MaskedRecurrentDiscreteActor(DiscreteActor):
    def __init__(self, preprocess_net, action_shape, hidden_sizes=()):
        # Отключаем softmax, чтобы получить сырые логиты
        super().__init__(
            preprocess_net=preprocess_net, 
            action_shape=action_shape, 
            hidden_sizes=hidden_sizes, 
            softmax_output=False
        )

    def forward(self, obs, state=None, info=None):
        # 1. Получаем сырые логиты (размерность может быть 2D или 3D)
        logits, hidden = super().forward(obs, state, info)
        
        # 2. Достаем маску
        mask = None
        if isinstance(obs, (dict, Batch)) and "mask" in obs:
            mask = obs["mask"]
        elif hasattr(obs, "mask"):
            mask = obs.mask
            
        # 3. Применяем маску
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
            
            # Если логиты 3D (B, Seq, Actions), а маска 2D (B, Actions) - подгоняем размер
            # (Хотя обычно Tianshou пакует маску тоже в 3D)
            if len(mask.shape) < len(logits.shape):
                mask = mask.unsqueeze(1)
                
            logits = logits.masked_fill(~mask, -1e8)
            
        # 4. Вручную считаем softmax для получения вероятностей
        probs = torch.softmax(logits, dim=-1)
            
        return probs, hidden