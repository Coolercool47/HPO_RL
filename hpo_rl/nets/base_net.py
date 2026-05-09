from tianshou.utils.net.common import ModuleWithVectorOutput
import numpy as np
import torch
from torch import nn
from tianshou.data import Batch
from tianshou.utils.torch_utils import torch_device


class BaseNet(ModuleWithVectorOutput):
    def __init__(
        self,
        state_shape,
        action_shape=0,
        hidden_sizes=[128, 128],
        device='cpu',
        concat: bool = False,
        norm_layer=None,
        **kwargs,
    ):
        _action_prod = int(np.prod(action_shape)) if action_shape is not None else 0

        if _action_prod == 0:
            out_dim = hidden_sizes[-1]
            self._has_output_head = False
        else:
            out_dim = _action_prod
            self._has_output_head = True

        effective_out_dim = hidden_sizes[-1] if not self._has_output_head else out_dim
        super().__init__(output_dim=effective_out_dim)

        self.device = device
        self._concat = concat
        input_dim = int(np.prod(state_shape))

        if concat and _action_prod > 0:
            input_dim += _action_prod

        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim

        if self._has_output_head:
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

        device = torch_device(self)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = x.flatten(1)

        logits = self.model(x)

        if mask is not None and self._has_output_head:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
            min_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, min_value)

        return logits, state