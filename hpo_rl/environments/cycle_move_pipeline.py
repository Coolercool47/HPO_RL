from typing import Dict, Any, Optional, List
import gymnasium as gym
import numpy as np

from hpo_rl.backends.base import EvaluationBackend
from hpo_rl.environments.base_env import BaseHPOEnv


# Попробовать сделать преобразования дискретных гиперпараметров в непрерывное при помощи ядра?

class CyclicPipelineEnv(BaseHPOEnv):
    """
    Среда для HPO с циклическим перебором параметров.
    Агент последовательно выбирает значения для каждого гиперпараметра.
    """

    def __init__(
        self,
        hp_space: Dict[str, Any],
        backend: EvaluationBackend,
        num_bins: int = 20,
        max_steps: int = 200,
        step_sizes: Optional[List[int]] = None,
        reward_mode: str = "per_step",   # per_step | per_cycle
        action_type: str = "discrete",   # discrete | continuous
        max_step_bins: Optional[int] = None,
        use_history: bool = True,
        history_cycles: int = 3
    ):
        super().__init__(hp_space, backend)

        self.num_bins = num_bins
        self.max_steps_limit = max_steps
        self.reward_mode = reward_mode
        self.action_type = action_type
        self.max_step_bins = max_step_bins
        self.use_history = use_history
        self.history_cycles = history_cycles

        if reward_mode not in ("per_step", "per_cycle"):
            raise ValueError(f"Invalid reward_mode: {reward_mode}")
        if action_type not in ("discrete", "continuous"):
            raise ValueError(f"Invalid action_type: {action_type}")

        if self.max_step_bins is None and action_type == "continuous":
            self.max_step_bins = max(1, num_bins // 10)

        self.action_grid = np.linspace(0.0, 1.0, num=num_bins, dtype=np.float32)
        self.num_hyperparams = len(self.hp_names)
        self.history_len = self.history_cycles * self.num_hyperparams

        # self.param_types = []
        # for name in self.hp_names:
        #     info = self.hp_space_config[name]
        #     self.param_types.append('discrete' if 'discrete' in info else 'continuous')
        # print("="*25)
        # print(info)
        self._init_action_space(step_sizes)
        self._init_observation_space()
        self._init_state()

    def _init_action_space(self, step_sizes):
        if self.action_type == "discrete":
            if step_sizes is None:
                self.step_sizes = [
                    max(1, self.num_bins // 5),
                    max(1, self.num_bins // 10),
                    max(1, self.num_bins // 20),
                    1
                ]
            else:
                # print(step_sizes)
                self.step_sizes = sorted(step_sizes, reverse=True)
            # шаги с минусом + на месте + шаги с плюсом
            self.num_actions = 2 * len(self.step_sizes) + 1
            self.action_space = gym.spaces.Discrete(self.num_actions)
        else:
            self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self.step_sizes = []

    def _init_observation_space(self):
        obs_spaces = {
            "active_param": gym.spaces.Box(low=0, high=1, shape=(self.num_hyperparams,), dtype=np.float32),
            "chosen_values": gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_hyperparams,), dtype=np.float32),
        }

        if self.use_history:
            obs_spaces["prev_values"] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.num_hyperparams,), dtype=np.float32)
            obs_spaces["reward_history"] = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.history_len,), dtype=np.float32)
            obs_spaces["action_history"] = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(self.history_len,), dtype=np.float32)
        else:
            # LSTM строит историю в скрытом виде из prev_reward + prev_action
            obs_spaces["prev_reward"] = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            obs_spaces["prev_action"] = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.observation_space = gym.spaces.Dict(obs_spaces)

    def _init_state(self):
        self.cursor_idx = 0
        self.steps_total = 0
        self.current_indices = np.zeros(self.num_hyperparams, dtype=np.int32)
        self.prev_cycle_indices = np.zeros(self.num_hyperparams, dtype=np.int32)

        self.current_metric = 0.0
        self.current_raw_metric = 0.0
        self.prev_reward = 0.0
        self.prev_action = 0.0

        self.reward_history_buffer = np.zeros(self.history_len, dtype=np.float32)
        self.action_history_buffer = np.zeros(self.history_len, dtype=np.float32)

        self.best_raw_metric = float('inf') if not self.backend.maximize else -float('inf')
        self.best_config_so_far = {}
        self.final_config_options = {}
        self.steps_without_improvement = 0

        self.cycle_metrics = []
        self.cycle_start_metric = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.cursor_idx = 0
        self.steps_total = 0
        self.prev_reward = 0.0
        self.prev_action = 0.0
        self.reward_history_buffer.fill(0.0)
        self.action_history_buffer.fill(0.0)

        self.current_indices = self.np_random.integers(0, self.num_bins, size=self.num_hyperparams)
        self.prev_cycle_indices = self.current_indices.copy()

        self._update_config_from_indices()
        config = self._assemble_config(self.final_config_options)
        raw = self.backend.evaluate(config)
        self.current_raw_metric = raw
        self.current_metric = self._to_reward(raw)

        self.best_raw_metric = raw
        self.best_config_so_far = config.copy()
        self.steps_without_improvement = 0
        self.cycle_metrics = []
        self.cycle_start_metric = self.current_metric

        return self._get_obs(), self._get_info()

    def step(self, action):
        # print(action)
        param_idx = self.cursor_idx
        old_idx = self.current_indices[param_idx]

        new_idx, step_size, delta_idx = self._apply_action(action, old_idx)
        reward = self._compute_reward(param_idx, old_idx, new_idx)
        normalized_action = self._normalize_action(delta_idx if self.action_type == "discrete" else new_idx - old_idx)

        self.steps_total += 1
        self.cursor_idx = (self.cursor_idx + 1) % self.num_hyperparams

        cycle_completed = (self.cursor_idx == 0 and self.steps_total > 0)
        if cycle_completed:
            self.prev_cycle_indices = self.current_indices.copy()
            if self.reward_mode == "per_cycle":
                reward = self._finalize_cycle_reward()

        self._update_history(reward, normalized_action)
        self.prev_reward = reward
        self.prev_action = normalized_action

        truncated = self.steps_total >= self.max_steps_limit
        info = self._get_info()
        info.update({
            'step_size': step_size,
            'steps_without_improvement': self.steps_without_improvement,
            'cycle_completed': cycle_completed
        })
        return self._get_obs(), reward, False, truncated, info

    def _apply_action(self, action, old_idx):
        """Возвращает (new_idx, step_size, delta_idx)."""
        if self.action_type == "discrete":
            # action = int(action)
            n = len(self.step_sizes)
            if action == n:
                # print("действие стоять на месте", action)
                return old_idx, 0, 0
            elif action < n:
                step = self.step_sizes[action]
                new_idx = np.clip(old_idx - step, 0, self.num_bins - 1)
                return new_idx, step, new_idx - old_idx
            else:
                step = self.step_sizes[action - n - 1]
                new_idx = np.clip(old_idx + step, 0, self.num_bins - 1)
                return new_idx, step, new_idx - old_idx
        else:
            val = float(np.clip(action[0] if isinstance(action, np.ndarray) else action, 0.0, 1.0))
            target = int(np.clip(val * (self.num_bins - 1), 0, self.num_bins - 1))

            if self.max_step_bins:
                delta = np.clip(target - old_idx, -self.max_step_bins, self.max_step_bins)
                new_idx = int(np.clip(old_idx + delta, 0, self.num_bins - 1))
            else:
                new_idx = target

            return new_idx, abs(new_idx - old_idx), new_idx - old_idx

    def _compute_reward(self, param_idx, old_idx, new_idx):
        # print("IDXs:",old_idx,new_idx)
        if new_idx == old_idx:
            self.steps_without_improvement += 1
            return -0.05 * self.steps_without_improvement

        self.current_indices[param_idx] = new_idx
        self._update_config_from_indices()
        config = self._assemble_config(self.final_config_options)
        raw = self.backend.evaluate(config)
        new_metric = self._to_reward(raw)

        if self._is_improvement(raw, self.best_raw_metric):
            self.best_raw_metric = raw
            self.best_config_so_far = config.copy()
            self.steps_without_improvement = 0
        else:
            self.steps_without_improvement += 1

        if self.reward_mode == "per_step":
            diff = new_metric - self.current_metric
            reward = np.sign(diff) * np.log(np.abs(diff) + 1.0)
        else:
            self.cycle_metrics.append(new_metric)
            reward = 0.0

        self.current_metric = new_metric
        self.current_raw_metric = raw
        
        return reward

    def _finalize_cycle_reward(self):
        if self.cycle_metrics:
            diff = self.cycle_metrics[-1] - self.cycle_start_metric
            reward = np.sign(diff) * np.log(np.abs(diff) + 1.0)
        else:
            reward = -0.05

        self.cycle_start_metric = self.current_metric
        self.cycle_metrics = []
        return reward

    def _normalize_action(self, delta):
        if self.action_type == "discrete":
            max_step = max(self.step_sizes) if self.step_sizes else 1
        else:
            max_step = self.max_step_bins or (self.num_bins - 1)
        return float(np.clip(delta / max(max_step, 1), -1.0, 1.0))

    def _update_history(self, reward, action):
        self.reward_history_buffer = np.roll(self.reward_history_buffer, -1)
        self.reward_history_buffer[-1] = reward / (1.0 + abs(reward))

        self.action_history_buffer = np.roll(self.action_history_buffer, -1)
        self.action_history_buffer[-1] = action

    def _update_config_from_indices(self):
        for i, name in enumerate(self.hp_names):
            idx = self.current_indices[i]
            norm = self.action_grid[idx]
            info = self.hp_space_config[name]
            # print(info)

            if info.get("type") == 'float' or info.get("type")== 'int':
                # c = info['type']
                lo, hi = info['min'], info['max']
                if info.get('log', False):
                    val = np.exp(np.log(lo) + norm * (np.log(hi) - np.log(lo)))
                else:
                    val = lo + norm * (hi - lo)
                if info.get('type') == 'int':
                    val = int(round(val))
            else: # Тут все плохо
                # print(info)
                opts = info['values']
                val = opts[int(np.clip(np.floor(norm * len(opts)), 0, len(opts) - 1))]

            self.final_config_options[name] = val

    def _get_obs(self) -> Dict[str, np.ndarray]:
        indices = np.array(self.current_indices, dtype=np.int32).flatten()
        norm_values = (indices.astype(np.float32) / (self.num_bins - 1))

        norm_values = np.atleast_1d(norm_values)

        active_param = np.zeros(self.num_hyperparams, dtype=np.float32)
        active_param[self.cursor_idx] = 1.0
        active_param = np.atleast_1d(active_param)

        obs = {
            "active_param": active_param,
            "chosen_values": norm_values,
        }

        if self.use_history:
            obs["prev_values"] = np.atleast_1d(
                (self.prev_cycle_indices.astype(np.float32) / (self.num_bins - 1)).flatten()
            )
            obs["reward_history"] = np.atleast_1d(self.reward_history_buffer.astype(np.float32))
            obs["action_history"] = np.atleast_1d(self.action_history_buffer.astype(np.float32))
        else:
            obs["prev_reward"] = np.array([self.prev_reward], dtype=np.float32)
            obs["prev_action"] = np.array([self.prev_action], dtype=np.float32)

        return obs

    def get_info(self) -> Dict[str, Any]:
        return self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        return {
            "best_config": self.best_config_so_far,
            "best_metric": self.best_raw_metric,
            "current_config": self.final_config_options.copy(),
            "current_metric": self.current_raw_metric
        }
