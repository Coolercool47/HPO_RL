import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

from tianshou.data import Batch
from tianshou.utils.net.common import ModuleWithVectorOutput

class MaskedRecurrentNet(ModuleWithVectorOutput):
    """GRU + MLP с маскированием логитов; отдаёт скрытое состояние RNN.

    Args:
        state_shape: форма наблюдения.
        action_shape: форма действий.
        hidden_sizes: [rnn_dim, ...] — первый элемент — hidden GRU.
        rnn_layers: число слоёв GRU.
    """

    def __init__(
        self, 
        state_shape: tuple, 
        action_shape: tuple, 
        hidden_sizes: list =[128, 128], 
        rnn_layers: int = 1
    ):
        """Инициализирует MaskedRecurrentNet.

        Args:
            state_shape: форма наблюдения.
            action_shape: форма действий.
            hidden_sizes: размеры GRU и MLP.
            rnn_layers: число слоёв GRU.
        """
        out_dim = int(np.prod(action_shape))
        super().__init__(output_dim=out_dim)
        
        input_dim = int(np.prod(state_shape))
        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_sizes[0]
        
        self.rnn = nn.GRU(
            input_size=input_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.rnn_layers, 
            batch_first=True
        )
        
        layers =[]
        curr_dim = self.hidden_dim
        for hidden_dim in hidden_sizes[1:]:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = hidden_dim
            
        layers.append(nn.Linear(curr_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, 
        obs: Union[np.ndarray, torch.Tensor, dict, Batch], 
        state: Optional[Union[dict, Batch, torch.Tensor]] = None, 
        info: Optional[Dict[str, Any]] = None
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, dict]]:
        
        mask = None
        x = obs

        if isinstance(obs, (dict, Batch)):
            mask = obs.get("mask", None)
            x = obs.get("obs", obs)
            
        device = next(self.parameters()).device
        
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        
        is_sequence = len(x.shape) == 3
        if not is_sequence:
            x = x.unsqueeze(1)

        if is_sequence:
            state = None  
        else:
            if state is not None:
                if isinstance(state, (dict, Batch)):
                    state = state.get("hidden", state)
                if not isinstance(state, torch.Tensor):
                    state = torch.as_tensor(state, dtype=torch.float32, device=device)
                
                if state.dim() == 3:
                    state = state.transpose(0, 1).contiguous()
                elif state.dim() == 2:
                    state = state.unsqueeze(0).contiguous()

        rnn_out, hidden_out = self.rnn(x, state)
        
        last_out = rnn_out[:, -1, :] 
        logits = self.mlp(last_out)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
            if mask.dim() == 3:
                mask = mask[:, -1, :]
            min_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, min_value)

        hidden_to_return = hidden_out.transpose(0, 1).detach()
        

        return logits, {"hidden": hidden_to_return}