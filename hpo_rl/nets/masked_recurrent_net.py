import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union

from tianshou.data import Batch
from tianshou.utils.net.common import ModuleWithVectorOutput

class MaskedRecurrentNet(ModuleWithVectorOutput):
    def __init__(
        self, 
        state_shape: tuple, 
        action_shape: tuple, 
        hidden_sizes: list =[128, 128], 
        rnn_layers: int = 1
    ):
        out_dim = int(np.prod(action_shape))
        super().__init__(output_dim=out_dim)
        
        input_dim = int(np.prod(state_shape))
        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_sizes[0]
        
        # rnn_layers = 10 (из вашего конфига)
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

        # --- ОБРАБОТКА STATE (ПРИЕМ ИЗ TIANSHOU) ---
        if is_sequence:
            state = None  # В режиме обучения игнорируем старые state
        else:
            if state is not None:
                if isinstance(state, (dict, Batch)):
                    state = state.get("hidden", state)
                if not isinstance(state, torch.Tensor):
                    state = torch.as_tensor(state, dtype=torch.float32, device=device)
                
                # Tianshou хранит (batch_size, num_layers, hidden_size).
                # GRU ожидает (num_layers, batch_size, hidden_size). Возвращаем обратно:
                if state.dim() == 3:
                    state = state.transpose(0, 1).contiguous()
                elif state.dim() == 2:
                    # Если вдруг пришел двумерный тензор, добавляем num_layers = 1
                    state = state.unsqueeze(0).contiguous()

        # rnn_out: (batch_size, seq_len, hidden_size)
        # hidden_out: (num_layers, batch_size, hidden_size) - !!!
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

        # --- ОБРАБОТКА STATE (ОТДАЧА В TIANSHOU) ---
        # Чтобы Tianshou Collector не падал с shape mismatch[10, 1, 128],
        # мы обязаны поставить batch_size на нулевое место: 
        # (num_layers, batch_size, hidden_size) -> (batch_size, num_layers, hidden_size)
        hidden_to_return = hidden_out.transpose(0, 1).detach()
        

        return logits, {"hidden": hidden_to_return}