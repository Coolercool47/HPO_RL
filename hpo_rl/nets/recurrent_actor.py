from tianshou.utils.net.discrete import DiscreteActor
import torch
from tianshou.data import Batch

class MaskedRecurrentDiscreteActor(DiscreteActor):
    def __init__(self, preprocess_net, action_shape, hidden_sizes=()):
        super().__init__(
            preprocess_net=preprocess_net, 
            action_shape=action_shape, 
            hidden_sizes=hidden_sizes, 
            softmax_output=False # Важно!
        )

    def forward(self, obs, state=None, info=None):
        logits, hidden = super().forward(obs, state, info)
        
        mask = None
        if isinstance(obs, (dict, Batch)) and "mask" in obs:
            mask = obs["mask"]
        elif hasattr(obs, "mask"):
            mask = obs.mask
            
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
            
            # Согласование 2D маски и 3D логитов
            if len(mask.shape) < len(logits.shape):
                mask = mask.unsqueeze(1)
                
            logits = logits.masked_fill(~mask, -1e8)
            
        # Мы НЕ делаем softmax, возвращаем чистые логиты для PPO!
        return logits, hidden