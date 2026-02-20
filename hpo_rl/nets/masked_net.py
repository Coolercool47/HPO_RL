from torch import nn
import numpy as np
import torch

class MaskedNet(nn.Module):
    def __init__(self, state_shape, action_shape, mask_logits=False):
        super().__init__()
        input_dim = int(np.prod(state_shape))
        action_dim = int(np.prod(action_shape))
        self.mask_logits = mask_logits  # True для PPO, False для DQN
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, action_dim),
        )

    def forward(self, obs, state=None, info=None):
        mask = None
        if hasattr(obs, "mask"):
            mask = obs.mask
        elif isinstance(obs, dict) and "mask" in obs:
            mask = obs["mask"]

        if hasattr(obs, "obs"):
            x = obs.obs
        elif isinstance(obs, dict) and "obs" in obs:
            x = obs["obs"]
        else:
            x = obs

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=next(self.model.parameters()).device)

        logits = self.model(x)

        if self.mask_logits and mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask, -1e8)

        return logits, state