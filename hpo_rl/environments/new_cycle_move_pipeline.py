import gymnasium as gym
import numpy as np
import itertools

from typing import Dict, Any, Optional, List
from hpo_rl.backends.base import EvaluationBackend
from hpo_rl.backends.sequential import SequentialBackend
from hpo_rl.environments.base_env import BaseHPOEnv

class CyclicPipelineEnvNew(BaseHPOEnv):
    """Циклическая дискретная среда HPO: за шаг изменяется один гиперпараметр.

    Float-параметры дискретизируются в ``num_bins``; categorical — прямой выбор.
    Наблюдение — dict с ``obs`` и маской допустимых действий ``mask``.

    Args:
        hp_space: пространство гиперпараметров (float/int/categorical).
        backend: бэкенд оценки метрики.
        step_sizes: размеры шагов по сетке для float-параметров.
        num_bins: число бинов дискретизации float.
        max_steps: лимит шагов эпизода.
        obs_mode: ``index`` (норм. индекс) или ``ohe`` (one-hot).
        history_window: окно истории в наблюдении (0 — без истории).
        reward_mode: схема награды (``absolute``, ``delta``, ``best_improvement`` и др.).

    Attributes:
        num_hyperparams: число параметров.
        cur_step_num: индекс активного параметра в цикле.
        best_config_so_far: лучшая конфигурация эпизода.
    """

    def __init__(self,
        hp_space: Dict[str, Any],
        backend: EvaluationBackend,
        step_sizes: List[int],
        num_bins: int = 20,
        max_steps: int = 200,
        obs_mode: str = "index",  
        history_window: int = 0, 
        reward_mode: str = "absolute", 
        ):
        """Инициализирует циклическую pipeline-среду.

        Args:
            hp_space: пространство гиперпараметров.
            backend: бэкенд оценки.
            step_sizes: шаги по сетке для float.
            num_bins: число бинов float.
            max_steps: максимум шагов.
            obs_mode: ``index`` или ``ohe``.
            history_window: окно истории.
            reward_mode: режим награды.
        """
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

        self.hp_lin_spaces = {} 
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

    def sync_bounds_to_backend(self) -> None:
        """Синхронизирует ``hp_lin_spaces`` с текущим дочерним бэкендом.

        Вызывается из ``reset()`` при использовании ``SequentialBackend``.
        Пересоздаёт linspace-сетки для float-параметров по native bounds
        текущей функции, чтобы агент работал в правильном масштабе
        (а не в объединённых merged bounds).
        """
        if not isinstance(self.backend, SequentialBackend):
            return
        cb = self.backend.current_backend
        if not hasattr(cb, 'bounds') or not hasattr(cb, 'dimensions'):
            return
        for i, hp_name in enumerate(self.hp_space_config):
            if i >= len(cb.bounds):
                break
            info = self.hp_space_config[hp_name]
            if info["type"] == "float":
                lo, hi = cb.bounds[i]
                self.hp_lin_spaces[hp_name] = np.linspace(
                    lo, hi, num=self.num_bins, dtype=np.float32
                )

    def _init_action_space(self):
        """Инициализирует дискретное action space с шагами по сетке и категориальными действиями."""
        self.step_sizes = [-i for i in self.step_sizes] + [0] + self.step_sizes
        self.step_sizes = sorted(self.step_sizes)

        num_actions = max(len(self.step_sizes), self._max_categorical_hyp_len)
        self.action_space = gym.spaces.Discrete(num_actions)

    def _init_observation_space(self):
        """Инициализирует Dict observation space: ``obs`` (flat vector) и ``mask`` (MultiBinary)."""
        if self.obs_mode == "ohe":
            self._param_dim = self._ohe_param_dim
        else:
            self._param_dim = self.num_hyperparams

        flat_obs_dim = self._param_dim + 1 + self.num_hyperparams 

        if self.history_window > 0:
            self._snapshot_depth = self.history_window * self.num_hyperparams
            self._param_snapshot_len = self._snapshot_depth * self._param_dim
            flat_obs_dim += self._snapshot_depth + self._param_snapshot_len

        self.observation_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_obs_dim,), dtype=np.float32),
            "mask": gym.spaces.MultiBinary(int(self.action_space.n)),
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Сбрасывает эпизод: случайная конфигурация, начальная оценка, obs+mask.

        Args:
            seed: seed генератора.
            options: опции Gymnasium.

        Returns:
            tuple: (observation, info).
        """
        super().reset(seed=seed, options=options)

        if isinstance(self.backend, SequentialBackend):
            self.backend.next_backend()
        self.sync_bounds_to_backend()

        self.current_hyp_setup = {}
        self.best_config_so_far = {}
        self.best_raw_metric = float('-inf') if self.backend.maximize else float('inf')
        self.current_raw_metric = float('-inf') if self.backend.maximize else float('inf')
        self.prev_raw_metric = None
        self._initial_raw_metric = None  
        
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
        initial_val = self.backend.evaluate(self.current_hyp_setup)
        self._initial_raw_metric = initial_val
        self.raw_metric = initial_val
        self.current_raw_metric = initial_val
        self.best_raw_metric = initial_val
        self.best_config_so_far = self.current_hyp_setup.copy()
        self.prev_raw_metric = initial_val
        self._compute_reward()
        if self.reward_mode in ("delta", "relative_delta"):
            self.reward = 0.0

        if self.history_window > 0:
            self._param_snapshot_buf[:] = self._current_param_vec()

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Сдвигает один параметр в цикле, оценивает конфигурацию.

        Args:
            action: индекс шага (float) или категории (categorical).

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        if self.history_window > 0:
            self._reward_history_buf = np.roll(self._reward_history_buf, -1)
            self._reward_history_buf[-1] = self.reward  

            self._param_snapshot_buf = np.roll(self._param_snapshot_buf, -1, axis=0)
            self._param_snapshot_buf[-1] = self._current_param_vec()  

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
        """Формирует маску допустимых действий для текущего активного параметра."""
        mask = np.zeros(self.action_space.n, dtype=bool)

        hp_names_list = list(self.hp_space_config.keys())
        cur_hp_name = hp_names_list[self.cur_step_num]
        info = self.hp_space_config[cur_hp_name]

        if info["type"] == "float":
            n_valid = len(self.step_sizes)  
            mask[:n_valid] = True
        elif info["type"] == "categorical":
            n_valid = len(info["values"])
            mask[:n_valid] = True

        return mask
    
    def _intended_delta(self, action, hp_info):
        """Возвращает задуманное изменение индекса параметра (0 для categorical без смены)."""
        if hp_info["type"] == "float":
            return self.step_sizes[action]
        return 0 if action == self.cur_idx_dict[list(self.hp_space_config.keys())[self.cur_step_num]] else 1

    def _take_action(self, action):
        """Применяет действие к текущему параметру в цикле и обновляет конфигурацию."""
        cur_hp_name = next(self.hp_space_keys_iterator)
        hp_info = self.hp_space_config[cur_hp_name]

        if hp_info["type"] == "float": 
            self.cur_idx_dict[cur_hp_name] = int(np.clip(
                self.cur_idx_dict[cur_hp_name] + self.step_sizes[action],
                0, self.num_bins - 1
            ))
            self.current_hyp_setup[cur_hp_name] = self.hp_lin_spaces[cur_hp_name][self.cur_idx_dict[cur_hp_name]]

        elif hp_info["type"] == "categorical":
            self.cur_idx_dict[cur_hp_name] = action
            self.current_hyp_setup[cur_hp_name] = hp_info["values"][action]
    
    def _spawn(self):
        """Сэмплирует случайную начальную конфигурацию гиперпараметров."""
        for hp_name, values in self.hp_space_config.items():
            if values["type"] == "float":
                self.cur_idx_dict[hp_name] = int(self.np_random.integers(0, self.num_bins))
                self.current_hyp_setup[hp_name] = self.hp_lin_spaces[hp_name][self.cur_idx_dict[hp_name]]

            elif values["type"] == "categorical":
                idx = int(self.np_random.integers(0, len(values["values"])))
                self.cur_idx_dict[hp_name] = idx
                self.current_hyp_setup[hp_name] = values["values"][idx]

    def _truncated_logic(self):
        """Проверяет достижение лимита шагов эпизода (truncated)."""
        return self.max_steps_limit <= self.step_num_total

    def _terminated_logic(self):
        """Проверяет условие терминации эпизода (всегда False для этой среды)."""
        return False

    def _symlog(self, x):
        """Симметричный log-преобразование: ``sign(x) * log1p(|x|)``."""
        return np.sign(x) * np.log1p(np.abs(x))

    def _compute_reward(self):
        """Оценивает конфигурацию через backend и вычисляет reward по ``reward_mode``."""
        self.raw_metric = self.backend.evaluate(self.current_hyp_setup)
        self.current_raw_metric = self.raw_metric

        scale = abs(self._to_reward(self._initial_raw_metric)) + 1.0

        if self.reward_mode in ("delta", "relative_delta"):
            delta = self._to_reward(self.raw_metric) - self._to_reward(self.prev_raw_metric)
            self.reward = float(delta / scale)
            self.prev_raw_metric = self.raw_metric

        elif self.reward_mode == "rank_shaped":
            improvement = self._to_reward(self.raw_metric) - self._to_reward(self._initial_raw_metric)
            self.reward = float(self._symlog(improvement / scale * 10.0))

        elif self.reward_mode == "best_improvement":
            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                delta = self._to_reward(self.raw_metric) - self._to_reward(self.best_raw_metric)
                self.reward = float(delta / scale) + 0.5
            else:
                gap = self._to_reward(self.raw_metric) - self._to_reward(self.best_raw_metric)
                self.reward = float(np.clip(gap / scale, -0.5, 0.0))

        else:  
            self.reward = float(self._symlog(self._to_reward(self.raw_metric)))

        if self._is_improvement(self.raw_metric, self.best_raw_metric):
            self.best_raw_metric = self.raw_metric
            self.best_config_so_far = self.current_hyp_setup.copy()

        if self._steps_without_change > 0:
            self.reward -= 0.02 * min(self._steps_without_change, 5)

        return self.reward

    def _current_param_vec(self):
        """Возвращает представление текущих параметров согласно ``obs_mode``."""
        if self.obs_mode == "ohe":
            return self._get_obs_ohe()
        return self._get_obs_index()

    def _get_obs(self):
        """Формирует dict-наблюдение: flat ``obs`` + ``mask`` допустимых действий."""
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
        """Кодирует каждый параметр одним нормализованным значением в [0, 1]."""
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
        """Float-параметры -> [0, 1]; categorical -> one-hot вектор."""
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
        """Возвращает служебную информацию: текущая и лучшая конфигурация, метрики."""
        return {
            "best_config": self.best_config_so_far,
            "best_metric": self.best_raw_metric,
            "current_config": self.current_hyp_setup,
            "current_metric": self.current_raw_metric
        }

