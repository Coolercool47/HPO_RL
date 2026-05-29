"""Рекуррентный гауссовский актор для непрерывных действий (chunked BPTT)."""

from __future__ import annotations

import warnings
from typing import Any, Sequence

import numpy as np
import torch

from tianshou.utils.net.common import AbstractContinuousActorProbabilistic, MLP, ModuleWithVectorOutput

SIGMA_MIN = -20
SIGMA_MAX = 2


class RecurrentContinuousActorProbabilistic(AbstractContinuousActorProbabilistic):
    """Непрерывный актор: preprocess_net → (μ, σ); поддержка ``[B, T, H]``.

    Args:
        preprocess_net: backbone (в т.ч. рекуррентный).
        action_shape: форма действия.
        hidden_sizes: MLP для μ и σ.
        max_action: масштаб после tanh.
        unbounded: без ограничения действия по max_action.
        conditioned_sigma: σ от наблюдения, иначе обучаемый параметр.
    """

    def __init__(
        self,
        *,
        preprocess_net: ModuleWithVectorOutput,
        action_shape,
        hidden_sizes: Sequence[int] = (),
        max_action: float = 1.0,
        unbounded: bool = False,
        conditioned_sigma: bool = False,
    ) -> None:
        """Инициализирует RecurrentContinuousActorProbabilistic.

        Args:
            preprocess_net: сеть признаков.
            action_shape: форма действия.
            hidden_sizes: скрытые слои μ/σ.
            max_action: предел действия.
            unbounded: неограниченное действие.
            conditioned_sigma: условная дисперсия.
        """
        output_dim = int(np.prod(action_shape))
        super().__init__(output_dim)
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn("Note that max_action input will be discarded when unbounded is True.")
            max_action = 1.0
        self.preprocess = preprocess_net
        input_dim = preprocess_net.get_output_dim()
        self.mu = MLP(input_dim=input_dim, output_dim=self.output_dim, hidden_sizes=hidden_sizes)
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(input_dim=input_dim, output_dim=self.output_dim, hidden_sizes=hidden_sizes)
        else:
            self.sigma_param = torch.nn.Parameter(torch.zeros(output_dim, 1))
        self.max_action = max_action
        self._unbounded = unbounded

    def get_preprocess_net(self) -> ModuleWithVectorOutput:
        return self.preprocess

    def forward(
        self,
        obs: dict | torch.Tensor | np.ndarray,
        state: Any | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], Any | None]:
        feats, hidden = self.preprocess(obs, state=state, info=info)

        is_3d = feats.dim() == 3
        if is_3d:
            b, t, fdim = feats.shape
            z = feats.reshape(b * t, fdim)
            mu = self.mu(z).reshape(b, t, -1)
            if self._c_sigma:
                sigma = torch.clamp(self.sigma(z), min=SIGMA_MIN, max=SIGMA_MAX).exp().reshape(b, t, -1)
            else:
                shape = [1] * len(mu.shape)
                shape[-1] = -1
                sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
            if not self._unbounded:
                mu = self.max_action * torch.tanh(mu.reshape(b * t, -1)).reshape(b, t, -1)
        else:
            mu = self.mu(feats)
            if self._c_sigma:
                sigma = torch.clamp(self.sigma(feats), min=SIGMA_MIN, max=SIGMA_MAX).exp()
            else:
                shape = [1] * len(mu.shape)
                shape[1] = -1
                sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
            if not self._unbounded:
                mu = self.max_action * torch.tanh(mu)

        return (mu, sigma), hidden
