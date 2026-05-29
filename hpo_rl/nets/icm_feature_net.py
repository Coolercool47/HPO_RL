"""Адаптер feature_net для ICM при Dict-наблюдениях ``Batch(obs, mask)``."""

from __future__ import annotations

import torch
import torch.nn as nn
from tianshou.data import Batch


class ICMFeatureNet(nn.Module):
    """Адаптер feature_net для ICM при Dict-наблюдениях.

        Args:
            net: внутренняя ``nn.Module``; принимает ``(obs)`` или
                 ``(obs, state, info)``, возвращает ``Tensor`` или ``(Tensor, state)``
    """

    def __init__(self, net: nn.Module) -> None:
        """Инициализирует ICMFeatureNet.

        Args:
            net: внутренняя сеть признаков.
        """
        super().__init__()
        self.net = net

    def forward(self, obs: torch.Tensor | dict | Batch) -> torch.Tensor:

        if isinstance(obs, (dict, Batch)):
            obs = obs.get("obs", obs) if hasattr(obs, "get") else obs["obs"]


        out = self.net(obs)


        if isinstance(out, tuple):
            out = out[0]

        return out
