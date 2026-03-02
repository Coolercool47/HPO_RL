"""Feature-net обёртка для ICM (Intrinsic Curiosity Module).

Модуль содержит :class:`ICMFeatureNet` — адаптер между средами
с Dict observation space (``Batch(obs=..., mask=...)``) и
``IntrinsicCuriosityModule`` из tianshou, который ожидает
``feature_net(obs) -> Tensor``.

Обёртка:
- извлекает ``obs`` из ``Batch``/``dict`` (игнорирует ``mask``),
- вызывает внутреннюю сеть,
- если сеть возвращает кортеж ``(logits, state)`` — берёт только ``logits``.
"""

import torch
import torch.nn as nn
from tianshou.data import Batch


class ICMFeatureNet(nn.Module):
    """Адаптер feature_net для ICM, совместимый с Dict observation space.

    Args:
        net: любая сеть (``nn.Module``).  Может принимать ``(obs)`` или
             ``(obs, state, info)`` и возвращать ``Tensor`` или
             ``(Tensor, state)``.
    """

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, obs: torch.Tensor | dict | Batch) -> torch.Tensor:
        # 1. Извлечь тензор obs из Batch/dict (Dict observation space)
        if isinstance(obs, (dict, Batch)):
            obs = obs.get("obs", obs) if hasattr(obs, "get") else obs["obs"]

        # 2. Вызвать внутреннюю сеть
        out = self.net(obs)

        # 3. Если сеть возвращает (logits, state) — взять только logits
        if isinstance(out, tuple):
            out = out[0]

        return out
