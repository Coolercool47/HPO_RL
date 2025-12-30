from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from hpo_rl.backends.base import EvaluationBackend


class BaseHPOEnv(gym.Env, ABC):
    """Базовый класс для RL-сред HPO, совместимый с Gymnasium."""

    metadata = {"render_modes": []}

    def __init__(self, hp_space: Dict[str, Any], backend: EvaluationBackend):
        super().__init__()

        if not isinstance(backend, EvaluationBackend):
            raise TypeError("backend должен быть экземпляром EvaluationBackend")
        if not hp_space:
            raise ValueError("hp_space не может быть пустым")

        self.backend = backend
        self.hp_space_config = hp_space
        self.hp_names = list(hp_space.keys())

    def _assemble_config(self, chosen_options: Dict[str, Any]) -> Dict[str, Any]:
        """Возможность обработки конфига перед evaluate()."""
        return chosen_options

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)

    @abstractmethod
    def step(self, action):
        raise NotImplementedError

    @abstractmethod
    def _get_obs(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        raise NotImplementedError

    def render(self):
        pass

    def close(self):
        """Очистка кэша бэкенда."""
        if hasattr(self.backend, 'clear_cache'):
            self.backend.clear_cache()
