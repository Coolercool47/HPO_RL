import torch
from tianshou.utils.net.discrete import DiscreteActor
from tianshou.data import Batch

class MaskedDiscreteActor(DiscreteActor):
    def __init__(self, preprocess_net, action_shape, hidden_sizes=()):
        super().__init__(
            preprocess_net=preprocess_net, 
            action_shape=action_shape, 
            hidden_sizes=hidden_sizes, 
            softmax_output=False
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
            # Применяем маску, но НЕ делаем softmax!
            logits = logits.masked_fill(~mask, -1e8)
            
        # Возвращаем ЛОГИТЫ! Categorical сам безопасно посчитает вероятности.
        return logits, hidden