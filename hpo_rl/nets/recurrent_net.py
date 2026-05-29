import torch
import numpy as np
from torch import nn
from tianshou.data import Batch
from tianshou.utils.net.common import ModuleWithVectorOutput


class RecurrentBaseNet(ModuleWithVectorOutput):
    """GRU-backbone с пошаговым сбросом скрытого состояния.

    Args:
        state_shape: форма наблюдения.
        action_shape: не используется (совместимость с API Net).
        hidden_layer_size: размер скрытого слоя GRU.
        num_layers: число слоёв GRU.
        device: устройство вычислений.

    В ``info["episode_reset"]`` передаётся bool-тензор ``[B, T]``:
    перед шагом ``t``, где флаг True, hidden для элемента батча обнуляется.
    """

    def __init__(self, state_shape, action_shape, hidden_layer_size=128, num_layers=1, device="cpu"):
        """Инициализирует RecurrentBaseNet.

        Args:
            state_shape: форма наблюдения.
            action_shape: форма действий (не используется).
            hidden_layer_size: размерность GRU.
            num_layers: число слоёв GRU.
            device: устройство.
        """
        super().__init__(output_dim=hidden_layer_size)
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        input_dim = int(np.prod(state_shape))

        self.fc = nn.Linear(input_dim, hidden_layer_size)
        self.ln = nn.LayerNorm(hidden_layer_size)
        self.relu = nn.ReLU(inplace=True)

        self.rnn = nn.GRU(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
        )

    def _extract_x(self, obs):
        x = obs.obs if isinstance(obs, (dict, Batch)) and "obs" in obs else obs
        device = next(self.parameters()).device
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        else:
            x = x.to(device)
        return x

    def _init_h0(self, batch_size: int, x_device: torch.device, state):
        """Формирует начальное hidden ``[num_layers, B, H]`` из ``state``.

        Args:
            batch_size: размер батча
            x_device: устройство тензора
            state: предыдущее hidden или None

        Returns:
            тензор hidden для GRU
        """
        is_empty = state is None or (isinstance(state, dict) and not state) or (
            hasattr(state, "is_empty") and state.is_empty()
        )

        if is_empty:
            return torch.zeros(self.num_layers, batch_size, self.hidden_layer_size, device=x_device)

        if isinstance(state, dict) and "hidden" in state:
            h_0 = state["hidden"]
        elif hasattr(state, "hidden"):
            h_0 = state.hidden
        else:
            h_0 = state

        if not isinstance(h_0, torch.Tensor):
            h_0 = torch.as_tensor(h_0, dtype=torch.float32, device=x_device)


        if len(h_0.shape) == 4:
            h_0 = h_0[:, 0, :, :]

        if len(h_0.shape) == 3 and h_0.shape[1] == self.num_layers:
            h_0 = h_0.transpose(0, 1).contiguous()

        return h_0

    def _gru_with_step_resets(self, x, h, episode_reset):
        """Пошаговый GRU с обнулением hidden на ``episode_reset``.

        Args:
            x: вход ``[B, T, H]``
            h: hidden ``[num_layers, B, H]``
            episode_reset: bool-тензор ``[B, T]``

        Returns:
            кортеж (выход RNN, финальное hidden)
        """
        _bsz, time_steps, _h = x.shape
        outs = []
        for t in range(time_steps):
            reset_b = episode_reset[:, t]
            if reset_b.any():
                h = h.clone()
                h[:, reset_b] = 0.0
            out_t, h = self.rnn(x[:, t : t + 1, :], h)
            outs.append(out_t)
        return torch.cat(outs, dim=1), h

    def forward(self, obs, state=None, info=None):
        x = self._extract_x(obs)

        is_2d = False
        if len(x.shape) == 2:
            is_2d = True
            x = x.unsqueeze(1)

        x = self.relu(self.ln(self.fc(x)))

        episode_reset = None
        if info is not None and isinstance(info, dict):
            episode_reset = info.get("episode_reset")
        if episode_reset is not None:
            episode_reset = torch.as_tensor(episode_reset, dtype=torch.bool, device=x.device)
            if episode_reset.dim() == 1 and x.size(1) == 1:
                episode_reset = episode_reset.unsqueeze(1)
            elif episode_reset.dim() == 1:

                if episode_reset.shape[0] == x.size(1):
                    episode_reset = episode_reset.unsqueeze(0).expand(x.size(0), -1)

        bsz, time_steps, _ = x.shape
        h_0 = self._init_h0(bsz, x.device, state)

        if episode_reset is not None and bool(episode_reset.any().item()):
            out, h_n = self._gru_with_step_resets(x, h_0, episode_reset)
        else:
            out, h_n = self.rnn(x, h_0)

        if is_2d:
            out = out.squeeze(1)

        h_n = h_n.transpose(0, 1).contiguous()
        return out, {"hidden": h_n.detach()}
