# rl_opt/environments/continuous_pipeline_env.py

from typing import Dict, Any, Optional
import math

import gymnasium as gym
import numpy as np

from hpo_rl.environments.base_env import BaseHPOEnv
from hpo_rl.backends.base import EvaluationBackend


class ContinuousPipelineEnv(BaseHPOEnv):
    """
    Реализация среды-"конвейера" с непрерывным пространством действий.

    На каждом шаге агент выдает одно число в диапазоне [0, 1].
    Это значение затем масштабируется в нужный диапазон для конкретного
    гиперпараметра (категориального или непрерывного).

    Предназначена для policy-gradient агентов (PPO, SAC, TD3), которые
    эффективно работают с `Box` пространством действий.
    """

    def __init__(self, hp_space: Dict[str, Any], backend: EvaluationBackend):
        """
        Инициализирует среду.
        """
        super().__init__(hp_space=hp_space, backend=backend)

        num_hyperparams = len(self.hp_names)

        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict({
            "step_count": gym.spaces.Box(low=0, high=num_hyperparams, shape=(1,), dtype=np.int32),
            "chosen_values_normalized": gym.spaces.Box(low=0.0, high=1.0, shape=(num_hyperparams,), dtype=np.float32)
        })

        # Внутреннее состояние среды
        self.current_step = 0
        self.chosen_normalized_values = np.zeros(num_hyperparams, dtype=np.float32)
        self.final_config_options: Dict[str, Any] = {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.chosen_normalized_values.fill(0.0)
        self.final_config_options = {}
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        normalized_value = float(action[0])
        self.chosen_normalized_values[self.current_step] = normalized_value

        param_name = self.hp_names[self.current_step]
        param_info = self.hp_space_config[param_name]

        if 'discrete' in param_info:
            options = param_info['discrete']['choices']
            num_options = len(options)
            choice_index = min(math.floor(normalized_value * num_options), num_options - 1)
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

            self.final_config_options[param_name] = value

        self.current_step += 1
        terminated = self.current_step >= len(self.hp_names)
        reward = 0.0

        if terminated:
            # 3. Если завершен, вызываем бэкенд
            final_config = self._assemble_config(self.final_config_options)
            reward = self.backend.evaluate(final_config)

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "step_count": np.array([self.current_step], dtype=np.int32),
            "chosen_values_normalized": self.chosen_normalized_values.copy()
        }

    def _get_info(self) -> Dict[str, Any]:
        # Для непрерывных действий маски не используются
        return {}
