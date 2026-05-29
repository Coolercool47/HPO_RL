import torch
import numpy as np
from torch import nn
from tianshou.data import Batch
from tianshou.utils.net.common import ModuleWithVectorOutput

class MaskedNet(ModuleWithVectorOutput):
    """MLP с маскированием недопустимых действий в логитах.

    Args:
        state_shape: форма наблюдения.
        action_shape: форма пространства действий.
        hidden_sizes: размеры скрытых слоёв.
        device: устройство для тензоров.
    """

    def __init__(self, state_shape, action_shape, hidden_sizes=[128, 128], device='cpu'):
        """Инициализирует MaskedNet.

        Args:
            state_shape: форма наблюдения.
            action_shape: форма пространства действий.
            hidden_sizes: размеры скрытых слоёв.
            device: устройство.
        """

        out_dim = int(np.prod(action_shape))


        super().__init__(output_dim=out_dim)

        self.device = device
        input_dim = int(np.prod(state_shape))


        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = hidden_dim


        layers.append(nn.Linear(curr_dim, out_dim))
        self.model = nn.Sequential(*layers)


    def forward(self, obs, state=None, info=None):
        mask = None
        x = obs


        if isinstance(obs, (dict, Batch)):
            if "mask" in obs:
                mask = obs["mask"]
            if "obs" in obs:
                x = obs["obs"]

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        logits = self.model(x)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)


            min_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, min_value)

        return logits, state
