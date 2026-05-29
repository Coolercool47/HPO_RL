"""Преобразование формы Q-сети для Rainbow / C51: ``[B, A*K]`` → ``[B, A, K]``."""

import torch
import torch.nn as nn


class RainbowNetWrapper(nn.Module):
    """Обёртка reshape для distributional Q-сетей (Rainbow, C51).

    Args:
        model: базовая сеть с выходом ``[Batch, action_num * num_atoms]``.
        action_num: число дискретных действий.
        num_atoms: число атомов распределения (обычно 51).
    """

    def __init__(self, model: nn.Module, action_num: int, num_atoms: int) -> None:
        """Инициализирует RainbowNetWrapper.

        Args:
            model: базовая Q-сеть.
            action_num: число действий.
            num_atoms: число атомов распределения.
        """
        super().__init__()
        self.model = model
        self.action_num = action_num
        self.num_atoms = num_atoms

    def forward(self, obs, state=None, info={}):

        logits, hidden = self.model(obs, state=state, info=info)

        logits = logits.view(-1, self.action_num, self.num_atoms)
        return logits, hidden
