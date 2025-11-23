from typing import Dict, Any, Optional
import math
import gymnasium as gym
import numpy as np

from hpo_rl.backends.base import EvaluationBackend
from hpo_rl.environments.base_env import BaseHPOEnv


class CyclicPipelineEnv(BaseHPOEnv):

    def __init__(
        self,
        hp_space: Dict[str, Any],
        backend: EvaluationBackend,
        num_bins: int = 20,
        max_steps: int = 200,
        sparse_reward: bool = False
    ):
        super().__init__(hp_space, backend)

        self.num_bins = num_bins
        self.max_steps_limit = max_steps
        self.sparse_reward = sparse_reward
        self.action_grid = np.linspace(0.0, 1.0, num=num_bins, dtype=np.float32)
        self.num_hyperparams = len(self.hp_names)

        self.param_types = []
        for name in self.hp_names:
            param_info = self.hp_space_config[name]
            if 'discrete' in param_info:
                self.param_types.append('discrete')
            elif 'continuous' in param_info:
                self.param_types.append('continuous')

        # ACTION SPACE
        # 0: Уменьшить (-1)
        # 1: Оставить (0)
        # 2: Увеличить (+1)
        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Dict({
            "active_param": gym.spaces.Box(low=0, high=1, shape=(self.num_hyperparams,), dtype=np.float32),

            "chosen_values": gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_hyperparams,), dtype=np.float32),

            "prev_delta": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        })

        self.cursor_idx = 0
        self.steps_total = 0
        self.current_indices = np.zeros(self.num_hyperparams, dtype=np.int32)

        self.current_metric = 0.0
        self.prev_delta = 0.0
        self.best_metric_so_far = -float('inf')
        self.best_config_so_far = {}
        self.final_config_options = {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.cursor_idx = 0
        self.steps_total = 0
        self.prev_delta = 0.0

        self.current_indices = self.np_random.integers(0, self.num_bins, size=self.num_hyperparams)

        self._update_config_from_indices()
        config = self._assemble_config(self.final_config_options)
        self.current_metric = self.backend.evaluate(config)

        self.best_metric_so_far = self.current_metric
        self.best_config_so_far = config.copy()

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        delta_idx = action - 1

        param_idx = self.cursor_idx
        old_idx = self.current_indices[param_idx]
        new_idx = np.clip(old_idx + delta_idx, 0, self.num_bins - 1)

        reward = 0.0

        if new_idx != old_idx:
            self.current_indices[param_idx] = new_idx

            self._update_config_from_indices()
            config = self._assemble_config(self.final_config_options)
            new_metric = self.backend.evaluate(config)

            diff = new_metric - self.current_metric

            scale_factor = 100.0
            if diff >= 0:
                reward = math.log(1 + diff * scale_factor)
            else:
                reward = -math.log(1 + abs(diff) * scale_factor)

            if new_metric > self.best_metric_so_far:
                reward += 1.0
                self.best_metric_so_far = new_metric
                self.best_config_so_far = config.copy()

            self.current_metric = new_metric
        else:
            if action != 1:
                reward = -0.01
            else:
                reward = -0.001

        self.prev_delta = reward
        self.steps_total += 1
        self.cursor_idx = (self.cursor_idx + 1) % self.num_hyperparams

        terminated = False
        truncated = self.steps_total >= self.max_steps_limit

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _update_config_from_indices(self):
        for i, name in enumerate(self.hp_names):
            idx = self.current_indices[i]
            norm_val = self.action_grid[idx]
            param_info = self.hp_space_config[name]

            val = None
            if 'discrete' in param_info:
                opts = param_info['discrete']['choices']
                c_idx = int(np.clip(math.floor(norm_val * len(opts)), 0, len(opts)-1))
                val = opts[c_idx]
            elif 'continuous' in param_info:
                c_info = param_info['continuous']
                l, h = c_info['range']
                if c_info.get('log', False):
                    val = np.exp(np.log(l) + norm_val * (np.log(h) - np.log(l)))
                else:
                    val = l + norm_val * (h - l)
                if c_info.get('type') == 'int':
                    val = int(round(val))

            self.final_config_options[name] = val

    def _get_obs(self) -> Dict[str, np.ndarray]:
        norm_values = self.current_indices.astype(np.float32) / (self.num_bins - 1)

        one_hot = np.zeros(self.num_hyperparams, dtype=np.float32)
        one_hot[self.cursor_idx] = 1.0

        return {
            "active_param": one_hot,
            "chosen_values": norm_values,
            "prev_delta": np.array([math.tanh(self.prev_delta)], dtype=np.float32)
        }

    def _get_info(self) -> Dict[str, Any]:
        return {
            "best_config": self.best_config_so_far,
            "best_metric": self.best_metric_so_far
        }