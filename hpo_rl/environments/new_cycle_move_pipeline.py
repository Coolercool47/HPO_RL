import gymnasium as gym
import numpy as np
import itertools

from typing import Dict, Any, Optional, List
from hpo_rl.backends.base import EvaluationBackend
from hpo_rl.environments.base_env import BaseHPOEnv

class CyclicPipelineEnvNew(BaseHPOEnv):
    def __init__(self,
        hp_space: Dict[str, Any],
        backend: EvaluationBackend,
        step_sizes: List[int],
        num_bins: int = 20,
        max_steps: int = 200,
        obs_mode: str = "index",  # "index" | "ohe"
        history_window: int = 0,  # 0 = no history, N = last N cycles
        reward_mode: str = "absolute",  # "absolute" | "delta"
        ):

        super().__init__(hp_space, backend)
        self._normalize_hp_space()

        self.num_bins = num_bins
        self.max_steps_limit = max_steps
        self.step_num_total = 0
        self.cur_step_num = 0
        self.obs_mode = obs_mode
        self.step_sizes = step_sizes
        self.history_window = history_window
        self.reward_mode = reward_mode

        if obs_mode not in ("index", "ohe"):
            raise ValueError(f"obs_mode must be 'index' or 'ohe', got '{obs_mode}'")

        self.num_hyperparams = len(self.hp_names)

        self.cur_idx_dict = {} 

        self.current_hyp_setup = {}
        self.best_config_so_far = {}
        self.best_raw_metric = float('-inf') if self.backend.maximize else float('inf')
        self.current_raw_metric = float('-inf') if self.backend.maximize else float('inf')
        self.raw_metric = 0
        self.reward = 0
        self._initial_raw_metric = None
        self._steps_without_change = 0

        self.hp_space_keys_iterator = itertools.cycle(self.hp_space_config)

        self.hp_lin_spaces = {} #инициализация np.linspace для гиперов 
        self._max_categorical_hyp_len = 0
        self._param_obs_slices = {}
        ohe_dim = 0
        for hp_name, values in self.hp_space_config.items():
            if values["type"] == "float":
                self.hp_lin_spaces[hp_name] = np.linspace(values["values"][0], values["values"][1], num=self.num_bins, dtype=np.float32)
                self._param_obs_slices[hp_name] = (ohe_dim, ohe_dim + 1)
                ohe_dim += 1
            elif values["type"] == "categorical":
                hyp_values_len = len(values["values"])
                if hyp_values_len > self._max_categorical_hyp_len:
                    self._max_categorical_hyp_len = hyp_values_len
                self._param_obs_slices[hp_name] = (ohe_dim, ohe_dim + hyp_values_len)
                ohe_dim += hyp_values_len

        self._ohe_param_dim = ohe_dim

        self._init_action_space()
        self._init_observation_space()

    def _normalize_hp_space(self):
        """Приводит hp_space к единому формату с ключом 'values'.

        function backend передаёт {"min": lo, "max": hi, "type": "float"},
        real/objective backend передаёт {"values": [lo, hi], "type": "float"}.
        После нормализации оба формата имеют ключ "values".
        """
        for hp_name in self.hp_space_config:
            entry = self.hp_space_config[hp_name]
            if entry["type"] in ("float", "int") and "values" not in entry:
                self.hp_space_config[hp_name]["values"] = [entry["min"], entry["max"]]

    def _init_action_space(self):
        self.step_sizes = [-i for i in self.step_sizes] + [0] + self.step_sizes
        self.step_sizes = sorted(self.step_sizes)

        num_actions = max(len(self.step_sizes), self._max_categorical_hyp_len)
        self.action_space = gym.spaces.Discrete(num_actions)

    def _init_observation_space(self):
        if self.obs_mode == "ohe":
            self._param_dim = self._ohe_param_dim
        else:
            self._param_dim = self.num_hyperparams

        flat_obs_dim = self._param_dim + 1 + self.num_hyperparams  # params + reward + active_param ohe

        if self.history_window > 0:
            self._snapshot_depth = self.history_window * self.num_hyperparams
            self._param_snapshot_len = self._snapshot_depth * self._param_dim
            flat_obs_dim += self._snapshot_depth + self._param_snapshot_len

        self.observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_obs_dim,), dtype=np.float32),
            "mask": gym.spaces.MultiBinary(int(self.action_space.n)),
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        self.current_hyp_setup = {}
        self.best_config_so_far = {}
        self.best_raw_metric = float('-inf') if self.backend.maximize else float('inf')
        self.current_raw_metric = float('-inf') if self.backend.maximize else float('inf')
        self.prev_raw_metric = None
        self._initial_raw_metric = None  # будет установлен в _compute_reward
        
        self.step_num_total = 0
        self.cur_step_num = 0
        self._steps_without_change = 0

        if self.history_window > 0:
            self._reward_history_buf = np.zeros(self._snapshot_depth, dtype=np.float32)
            self._param_snapshot_buf = np.zeros(
                (self._snapshot_depth, self._param_dim), dtype=np.float32
            )

        self.hp_space_keys_iterator = itertools.cycle(self.hp_space_config)

        self._spawn()
        # Evaluate initial metric BEFORE _compute_reward so best/prev are correct
        initial_val = self.backend.evaluate(self.current_hyp_setup)
        self._initial_raw_metric = initial_val
        self.raw_metric = initial_val
        self.current_raw_metric = initial_val
        self.best_raw_metric = initial_val
        self.best_config_so_far = self.current_hyp_setup.copy()
        self.prev_raw_metric = initial_val
        self._compute_reward()
        # First step reward should be 0 for delta modes (no change yet)
        if self.reward_mode in ("delta", "relative_delta"):
            self.reward = 0.0

        if self.history_window > 0:
            self._param_snapshot_buf[:] = self._current_param_vec()

        return self._get_obs(), self._get_info()

    def step(self, action):
        # --- Сохраняем ПРЕДЫДУЩЕЕ состояние в history ДО действия ---
        # Это гарантирует: history[-1] ≠ текущие obs (без дублирования)
        if self.history_window > 0:
            self._reward_history_buf = np.roll(self._reward_history_buf, -1)
            self._reward_history_buf[-1] = self.reward  # reward ПРЕДЫДУЩЕГО шага

            self._param_snapshot_buf = np.roll(self._param_snapshot_buf, -1, axis=0)
            self._param_snapshot_buf[-1] = self._current_param_vec()  # params ПРЕДЫДУЩЕГО шага

        hp_names_list = list(self.hp_space_config.keys())
        cur_hp_name = hp_names_list[self.cur_step_num]
        old_idx = self.cur_idx_dict[cur_hp_name]
        intended_move = self._intended_delta(action, self.hp_space_config[cur_hp_name])

        self._take_action(action)

        actually_stayed = (self.cur_idx_dict[cur_hp_name] == old_idx) and (intended_move == 0)
        if actually_stayed:
            self._steps_without_change += 1
        else:
            self._steps_without_change = 0

        self.step_num_total += 1
        self.cur_step_num = (self.cur_step_num + 1) % self.num_hyperparams
        
        reward = self._compute_reward()

        observation = self._get_obs()
        info = self._get_info()
        terminated = self._terminated_logic()
        truncated = self._truncated_logic()

        return observation, reward, terminated, truncated, info

    def _calculate_mask(self):
        mask = np.zeros(self.action_space.n, dtype=bool)

        # Определяем текущий параметр
        hp_names_list = list(self.hp_space_config.keys())
        cur_hp_name = hp_names_list[self.cur_step_num]
        info = self.hp_space_config[cur_hp_name]

        if info["type"] == "float":
            # step_sizes уже развёрнуты в [-5, -1, 0, 1, 5] — всё это валидные действия
            n_valid = len(self.step_sizes)  # после expand: 2*orig+1
            mask[:n_valid] = True
        elif info["type"] == "categorical":
            n_valid = len(info["values"])
            mask[:n_valid] = True

        return mask
    
    def _intended_delta(self, action, hp_info):
        if hp_info["type"] == "float":
            return self.step_sizes[action]
        return 0 if action == self.cur_idx_dict[list(self.hp_space_config.keys())[self.cur_step_num]] else 1

    def _take_action(self, action):
        cur_hp_name = next(self.hp_space_keys_iterator)
        hp_info = self.hp_space_config[cur_hp_name]

        if hp_info["type"] == "float": # мейби бейби но маски шоу
            self.cur_idx_dict[cur_hp_name] = int(np.clip(
                self.cur_idx_dict[cur_hp_name] + self.step_sizes[action],
                0, self.num_bins - 1
            ))
            self.current_hyp_setup[cur_hp_name] = self.hp_lin_spaces[cur_hp_name][self.cur_idx_dict[cur_hp_name]]

        elif hp_info["type"] == "categorical":
            self.cur_idx_dict[cur_hp_name] = action
            self.current_hyp_setup[cur_hp_name] = hp_info["values"][action]
    
    def _spawn(self):
        for hp_name, values in self.hp_space_config.items():
            if values["type"] == "float":
                self.cur_idx_dict[hp_name] = np.random.randint(0, self.num_bins)
                self.current_hyp_setup[hp_name] = self.hp_lin_spaces[hp_name][self.cur_idx_dict[hp_name]]

            elif values["type"] == "categorical":
                idx = np.random.randint(0, len(values["values"]))
                self.cur_idx_dict[hp_name] = idx
                self.current_hyp_setup[hp_name] = values["values"][idx]

    def _truncated_logic(self):
        return self.max_steps_limit <= self.step_num_total

    def _terminated_logic(self):
        return False

    def _symlog(self, x):
        return np.sign(x) * np.log1p(np.abs(x))

    def _compute_reward(self):
        self.raw_metric = self.backend.evaluate(self.current_hyp_setup)
        self.current_raw_metric = self.raw_metric

        # _initial_raw_metric is set in reset() before first _compute_reward call
        scale = abs(self._to_reward(self._initial_raw_metric)) + 1.0

        if self.reward_mode in ("delta", "relative_delta"):
            # Raw linear delta — split-proof by linearity.
            delta = self._to_reward(self.raw_metric) - self._to_reward(self.prev_raw_metric)
            self.reward = float(delta / scale)
            self.prev_raw_metric = self.raw_metric

        elif self.reward_mode == "rank_shaped":
            # Position-based: reward depends only on CURRENT position
            # relative to initial. symlog is safe here (no delta to split).
            improvement = self._to_reward(self.raw_metric) - self._to_reward(self._initial_raw_metric)
            self.reward = float(self._symlog(improvement / scale * 10.0))

        elif self.reward_mode == "best_improvement":
            # Bonus for new best, small penalty for distance from best.
            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                delta = self._to_reward(self.raw_metric) - self._to_reward(self.best_raw_metric)
                self.reward = float(delta / scale) + 0.5
            else:
                gap = self._to_reward(self.raw_metric) - self._to_reward(self.best_raw_metric)
                self.reward = float(np.clip(gap / scale, -0.5, 0.0))

        else:  # absolute
            self.reward = float(self._symlog(self._to_reward(self.raw_metric)))

        # Update best metric
        if self._is_improvement(self.raw_metric, self.best_raw_metric):
            self.best_raw_metric = self.raw_metric
            self.best_config_so_far = self.current_hyp_setup.copy()

        # Stagnation penalty
        if self._steps_without_change > 0:
            self.reward -= 0.02 * min(self._steps_without_change, 5)

        return self.reward

    def _current_param_vec(self):
        """Returns current param representation matching obs_mode."""
        if self.obs_mode == "ohe":
            return self._get_obs_ohe()
        return self._get_obs_index()

    def _get_obs(self):
        param_vec = self._current_param_vec()
        reward_val = np.array([self.reward], dtype=np.float32)
        active_param_ohe = np.zeros(self.num_hyperparams, dtype=np.float32)
        active_param_ohe[self.cur_step_num] = 1.0

        parts = [param_vec, reward_val, active_param_ohe]

        if self.history_window > 0:
            parts.append(self._reward_history_buf)
            parts.append(self._param_snapshot_buf.ravel())

        flat_obs = np.concatenate(parts)
        mask = self._calculate_mask()
        return {"obs": flat_obs, "mask": mask}

    def _get_obs_index(self):
        """Each param -> single normalized value in [0, 1]."""
        chosen_norm = np.zeros(self.num_hyperparams, dtype=np.float32)
        for i, hp_name in enumerate(self.hp_space_config):
            info = self.hp_space_config[hp_name]
            idx = self.cur_idx_dict[hp_name]
            if info["type"] == "float":
                chosen_norm[i] = idx / max(self.num_bins - 1, 1)
            elif info["type"] == "categorical":
                chosen_norm[i] = idx / max(len(info["values"]) - 1, 1)
        return chosen_norm

    def _get_obs_ohe(self):
        """Float params -> normalized [0,1]; categorical params -> one-hot vector."""
        param_vec = np.zeros(self._ohe_param_dim, dtype=np.float32)
        for hp_name, info in self.hp_space_config.items():
            start, end = self._param_obs_slices[hp_name]
            idx = self.cur_idx_dict[hp_name]
            if info["type"] == "float":
                param_vec[start] = idx / max(self.num_bins - 1, 1)
            elif info["type"] == "categorical":
                param_vec[start + idx] = 1.0
        return param_vec

    def _get_info(self) -> Dict[str, Any]:
        return {
            "best_config": self.best_config_so_far,
            "best_metric": self.best_raw_metric,
            "current_config": self.current_hyp_setup,
            "current_metric": self.current_raw_metric
        }

