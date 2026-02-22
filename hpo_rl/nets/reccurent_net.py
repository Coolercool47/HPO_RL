import torch
import torch.nn as nn
import numpy as np
from tianshou.data import Batch
from tianshou.utils.net.common import ModuleWithVectorOutput

class RecurrentBaseNet(ModuleWithVectorOutput):
    def __init__(self, state_shape, action_shape, hidden_layer_size=128, num_layers=1, device='cpu'):
        super().__init__(output_dim=hidden_layer_size)
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        input_dim = int(np.prod(state_shape))
        
        self.fc = nn.Linear(input_dim, hidden_layer_size)
        self.relu = nn.ReLU(inplace=True)
        
        self.rnn = nn.GRU(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True 
        )

    def forward(self, obs, state=None, info=None):
        x = obs.obs if isinstance(obs, (dict, Batch)) and "obs" in obs else obs
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            
        is_2d = False
        if len(x.shape) == 2:
            is_2d = True
            x = x.unsqueeze(1)
            
        x = self.relu(self.fc(x))
        
        # Подготовка состояния
        if state is None or (hasattr(state, "is_empty") and state.is_empty()):
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_layer_size, device=x.device)
        else:
            if isinstance(state, dict) and "hidden" in state:
                h_0 = state["hidden"]
            elif hasattr(state, "hidden"):
                h_0 = state.hidden
            else:
                h_0 = state
                
            if not isinstance(h_0, torch.Tensor):
                h_0 = torch.as_tensor(h_0, dtype=torch.float32, device=x.device)
                
            h_0 = h_0.transpose(0, 1).contiguous()
            
        # Проход через RNN
        out, h_n = self.rnn(x, h_0)
        
        if is_2d:
            out = out.squeeze(1)
            
        # Возвращаем размерности обратно
        h_n = h_n.transpose(0, 1).contiguous()
        
        # ВАЖНО: Добавляем .detach(), чтобы PyTorch позволил Tianshou скопировать этот тензор в буфер
        return out, {"hidden": h_n.detach()}