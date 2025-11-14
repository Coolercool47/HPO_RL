from typing import Dict, Any, Optional
import math

import gymnasium as gym
import numpy as np

from hpo_rl.backends.base import EvaluationBackend
from hpo_rl.environments.base_env import BaseHPOEnv


class DiscretizedPipelineEnv(BaseHPOEnv):
    """
    Реализация среды-"конвейера" с дискретизированным непрерывным действием.
    """

    def __init__(
        self,
        hp_space: Dict[str, Any],
        backend: EvaluationBackend,
        num_bins: int = 1000
    ):
        super().__init__(hp_space, backend)

        self.num_bins = num_bins
        self.action_grid = np.linspace(0.0, 1.0, num=num_bins, dtype=np.float32)
        num_hyperparams = len(self.hp_names)

        self.param_types = []  # 'discrete' или 'continuous'

        for name in self.hp_names:
            param_info = self.hp_space_config[name]
            if 'discrete' in param_info:
                self.param_types.append('discrete')
            elif 'continuous' in param_info:
                self.param_types.append('continuous')

        self.action_space = gym.spaces.Discrete(num_bins)

        self.observation_space = gym.spaces.Dict({
            "step_count": gym.spaces.Box(
                low=0,
                high=num_hyperparams,
                shape=(1,),
                dtype=np.int32
            ),
            "chosen_values": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(num_hyperparams,),
                dtype=np.float32
            )
        })

        # Внутреннее состояние среды
        self.current_step = 0
        self.chosen_values = np.zeros(num_hyperparams, dtype=np.float32)
        self.final_config_options: Dict[str, Any] = {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.chosen_values.fill(0.0)
        self.final_config_options = {}
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        normalized_value = self.action_grid[action]

        param_name = self.hp_names[self.current_step]
        param_info = self.hp_space_config[param_name]
        param_type = self.param_types[self.current_step]

        if param_type == 'discrete':
            options = param_info['discrete']['choices']
            num_options = len(options)
            choice_index = min(math.floor(normalized_value * num_options), num_options - 1)

            # Сохраняем нормализованный индекс для observation
            self.chosen_values[self.current_step] = choice_index / max(num_options - 1, 1)
            self.final_config_options[param_name] = options[choice_index]

        elif 'continuous' in param_info:
            cont_info = param_info['continuous']
            low, high = cont_info['range']

            if cont_info.get('log', False):
                log_low, log_high = np.log(low), np.log(high)
                value = np.exp(log_low + normalized_value * (log_high - log_low))
            else:
                value = low + normalized_value * (high - low)
            if cont_info['type'] == 'int':
                value = int(round(value))

            self.chosen_values[self.current_step] = normalized_value
            self.final_config_options[param_name] = value

        self.current_step += 1
        terminated = self.current_step >= len(self.hp_names)
        reward = 0.0

        if terminated:
            final_config = self._assemble_config(self.final_config_options)
            reward = self.backend.evaluate(final_config)

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "step_count": np.array([self.current_step], dtype=np.int32),
            "chosen_values": self.chosen_values.copy()
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "param_types": self.param_types,
            "current_config": self.final_config_options.copy()
        }
