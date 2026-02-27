from tianshou.utils.net.common import ModuleWithVectorOutput
import numpy as np
import torch
from torch import nn
from tianshou.data import Batch

class BaseNet(ModuleWithVectorOutput):
    def __init__(self, state_shape, action_shape, hidden_sizes=[128, 128], device='cpu'):
        # Базовая сеть просто выдает размерность последнего скрытого слоя
        out_dim = hidden_sizes[-1]
        super().__init__(output_dim=out_dim)
        
        self.device = device
        input_dim = int(np.prod(state_shape))
        
        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = hidden_dim
            
        self.model = nn.Sequential(*layers)

    def forward(self, obs, state=None, info=None):
        x = obs.obs if isinstance(obs, (dict, Batch)) and "obs" in obs else obs
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return self.model(x), state