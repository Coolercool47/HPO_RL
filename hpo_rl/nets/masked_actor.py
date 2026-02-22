import torch
from torch import nn
from tianshou.utils.net.discrete import DiscreteActor
from tianshou.data import Batch

class MaskedDiscreteActor(DiscreteActor):
    def __init__(self, preprocess_net, action_shape, hidden_sizes=()):
        # ВАЖНО: Отключаем автоматический softmax, чтобы получить сырые логиты!
        super().__init__(
            preprocess_net=preprocess_net, 
            action_shape=action_shape, 
            hidden_sizes=hidden_sizes, 
            softmax_output=False
        )

    def forward(self, obs, state=None, info=None):
        # 1. Получаем сырые логиты от базового класса
        logits, hidden = super().forward(obs, state, info)
        
        # 2. Извлекаем маску
        mask = None
        if isinstance(obs, (dict, Batch)) and "mask" in obs:
            mask = obs["mask"]
        elif hasattr(obs, "mask"):
            mask = obs.mask
            
        # 3. Накладываем маску на ЛОГИТЫ
        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
            # Запрещенным действиям ставим -1e8
            logits = logits.masked_fill(~mask, -1e8)
            
        # 4. Вручную применяем softmax, чтобы вернуть корректные вероятности.
        # Замаскированные действия (-1e8) превратятся в 0.0, и Categorical не упадет.
        probs = torch.softmax(logits, dim=-1)
            
        return probs, hidden