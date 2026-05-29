"""Обёртка Q-сети для Rainbow / C51.

Стандартная ``Net`` выдаёт плоский вектор размерности
``action_num * num_atoms``.  :class:`RainbowNetWrapper` преобразует
его в ``[Batch, action_num, num_atoms]``, как того ожидает
``RainbowPolicy`` / ``C51Policy``.
"""

import torch
import torch.nn as nn


class RainbowNetWrapper(nn.Module):
    """Reshape-обёртка для distributional Q-сетей (Rainbow, C51).

    Args:
        model: базовая сеть с выходом ``[Batch, action_num * num_atoms]``.
        action_num: число дискретных действий.
        num_atoms: число атомов распределения (обычно 51).
    """

    def __init__(self, model: nn.Module, action_num: int, num_atoms: int) -> None:
        super().__init__()
        self.model = model
        self.action_num = action_num
        self.num_atoms = num_atoms

    def forward(self, obs, state=None, info={}):

        logits, hidden = self.model(obs, state=state, info=info)

        logits = logits.view(-1, self.action_num, self.num_atoms)
        return logits, hidden
