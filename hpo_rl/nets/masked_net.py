from torch import nn
import numpy as np
import torch

class MaskedNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape))
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs, state=None, info=None):
        if hasattr(obs, "obs"):
            x = obs.obs
        elif isinstance(obs, dict) and "obs" in obs:
            x = obs.get("obs")
        else:
            x = obs

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=next(self.model.parameters()).device)

        return self.model(x), state