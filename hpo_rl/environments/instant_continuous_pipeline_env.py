import gymnasium as gym
import numpy as np
import warnings

from typing import Dict, Any, Optional
from hpo_rl.backends.base import EvaluationBackend
from hpo_rl.environments.base_env import BaseHPOEnv

from hpo_rl.backends.sequential import SequentialBackend


class InstantContinuousPipelineEnv(BaseHPOEnv):
    """Среда HPO с непрерывным многомерным действием: все параметры меняются за один шаг.

    Только float/int гиперпараметры. Вектор действий в ``[-1, 1]`` масштабируется
    в ``±max_delta`` и добавляется к текущим значениям с клипом в ``[lo, hi]``.

    Args:
        hp_space: пространство гиперпараметров (только float/int).
        backend: бэкенд оценки метрики.
        max_delta_frac: макс. относительное изменение за шаг (доля диапазона).
        max_steps: лимит шагов эпизода.
        obs_mode: режим наблюдения (``norm`` — нормализация в [0, 1]).
        history_window: число прошлых шагов в наблюдении (0 — без истории).
        reward_mode: схема награды (``bounded``, ``ternary``, ``delta``, ``absolute`` и др.).
        oob_penalty: штраф за выход за границы (масштабируется по величине нарушения).
        terminate_on_oob: завершать эпизод при OOB.
        oob_tolerance: подряд OOB-шагов до ``terminated``.

    Attributes:
        num_hyperparams: число оптимизируемых параметров.
        current_hyp_setup: текущая конфигурация.
        best_config_so_far: лучшая найденная конфигурация.

    Note:
        При ``SequentialBackend`` границы ``[lo, hi]`` синхронизируются с текущим
        дочерним бэкендом в ``reset()``.
    """

    def __init__(
        self,
        hp_space: Dict[str, Any],
        backend: EvaluationBackend,
        max_delta_frac: float = 0.1,
        max_steps: int = 200,
        obs_mode: str = "norm",
        history_window: int = 0,
        reward_mode: str = "absolute",
        oob_penalty: float = -10.0,
        terminate_on_oob: bool = False,
        oob_tolerance: int = 1,
    ):
        """Инициализирует непрерывную pipeline-среду.

        Args:
            hp_space: пространство гиперпараметров.
            backend: бэкенд оценки.
            max_delta_frac: доля диапазона на шаг изменения параметра.
            max_steps: максимум шагов в эпизоде.
            obs_mode: кодирование наблюдения.
            history_window: окно истории в obs.
            reward_mode: режим вычисления награды.
            oob_penalty: штраф за OOB.
            terminate_on_oob: флаг терминации по OOB.
            oob_tolerance: порог подряд OOB для terminated.
        """
        super().__init__(hp_space, backend)
        self._normalize_hp_space()
        self._validate_only_continuous()

        self._lo = np.array(
            [self.hp_space_config[n]["values"][0] for n in self.hp_names],
            dtype=np.float64,
        )
        self._hi = np.array(
            [self.hp_space_config[n]["values"][1] for n in self.hp_names],
            dtype=np.float64,
        )

        self.max_delta_frac = max_delta_frac
        self.max_steps_limit = max_steps
        self.obs_mode = obs_mode
        self.history_window = history_window
        self.reward_mode = reward_mode
        self.oob_penalty = oob_penalty
        self.terminate_on_oob = terminate_on_oob
        self.oob_tolerance = oob_tolerance
        
        self.num_hyperparams = len(self.hp_names)
        self._range = self._hi - self._lo
        self._max_delta = self.max_delta_frac * self._range  

        self._is_int = np.array(
            [self.hp_space_config[n]["type"] == "int" for n in self.hp_names],
            dtype=bool,
        )

        self.step_num_total = 0
        self.current_hyp_setup: Dict[str, float] = {}
        self.best_config_so_far: Dict[str, float] = {}
        self.best_raw_metric = float("-inf") if self.backend.maximize else float("inf")
        self.current_raw_metric = float("-inf") if self.backend.maximize else float("inf")
        self.raw_metric = 0.0
        self.reward = 0.0
        self._initial_metric: Optional[float] = None
        self._ema_abs_delta = 1.0
        self.prev_raw_metric: Optional[float] = None

        self._init_action_space()
        self._init_observation_space()

    def _normalize_hp_space(self):
        """Приводит hp_space к единому формату с ключом 'values'."""
        for hp_name in self.hp_space_config:
            entry = self.hp_space_config[hp_name]
            if entry["type"] in ("float", "int") and "values" not in entry:
                self.hp_space_config[hp_name]["values"] = [entry["min"], entry["max"]]

    def _validate_only_continuous(self):
        """Проверяет, что все параметры непрерывные."""
        for hp_name, info in self.hp_space_config.items():
            if info["type"] not in ("float", "int"):
                raise ValueError(
                    f"InstantContinuousPipelineEnv поддерживает только float/int "
                    f"гиперпараметры, но '{hp_name}' имеет type='{info['type']}'"
                )

    def _init_action_space(self):
        """Инициализирует многомерное action space: по одной delta на гиперпараметр."""
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_hyperparams,),
            dtype=np.float32,
        )

    def _init_observation_space(self):
        """Наблюдение: [normalized_params, gap_to_best, progress, reward, step_frac, (history)].

        Наблюдение включает:
        - ``normalized_params`` — позиция агента [0, 1] per dim
        - ``gap_to_best`` — tanh-compressed расстояние до лучшего, [-1, 0]
        - ``progress`` — tanh-compressed прогресс от начала, bounded
        - ``reward`` — reward текущего шага (bounded для bounded/ternary)
        - ``step_frac`` — прогресс эпизода [0, 1]

        Все метрические сигналы bounded через tanh — не зависят от масштаба
        функции.

        Если ``history_window > 0``, добавляются предыдущие параметры и
        reward для окна из последних N шагов.
        """
        self._obs_base_dim = self.num_hyperparams + 4

        flat_obs_dim = self._obs_base_dim

        if self.history_window > 0:
            self._hist_entry_dim = self.num_hyperparams + 1  
            flat_obs_dim += self.history_window * self._hist_entry_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_obs_dim,), dtype=np.float32
        )

    def sync_bounds_to_backend(self):
        """Синхронизирует ``_lo``, ``_hi``, ``_range``, ``_max_delta`` с текущим дочерним бэкендом.

        Вызывается из ``reset()`` при использовании ``SequentialBackend``.
        Env переходит на **native** bounds текущего дочернего бэкенда,
        чтобы агент работал в правильном масштабе для каждой функции.

        Поддерживает per-dimension bounds (e.g., bukin_n6 с асимметричными bounds).
        """
        if not isinstance(self.backend, SequentialBackend):
            return

        cb = self.backend.current_backend
        if not hasattr(cb, 'bounds') or not hasattr(cb, 'dimensions'):
            return

        for i in range(min(self.num_hyperparams, len(cb.bounds))):
            self._lo[i] = cb.bounds[i][0]
            self._hi[i] = cb.bounds[i][1]

        self._range = self._hi - self._lo
        self._max_delta = self.max_delta_frac * self._range

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Сбрасывает эпизод: случайная конфигурация, оценка, начальное наблюдение.

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
        self.best_raw_metric = float("-inf") if self.backend.maximize else float("inf")
        self.current_raw_metric = float("-inf") if self.backend.maximize else float("inf")
        self.prev_raw_metric = None
        self._initial_metric = None

        self.step_num_total = 0
        self._out_of_bounds = False
        self._oob_consecutive_count = 0
        self._oob_violation_frac = 0.0

        if self.history_window > 0:
            self._history_buf = np.zeros(
                (self.history_window, self._hist_entry_dim), dtype=np.float32
            )

        self._spawn()
        self._ema_abs_delta = 1.0   

        self._compute_reward()

        if self.reward_mode in ("delta", "relative_delta", "potential", "guided",
                                "bounded", "ternary", "abs_positive_delta"):
            self.reward = 0.0

        if self.history_window > 0:
            entry = np.concatenate([self._current_param_vec(), [self.reward]], dtype=np.float32)
            self._history_buf[:, :] = np.tile(entry, (self.history_window, 1))

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Применяет вектор действий ко всем параметрам, оценивает конфигурацию.

        Args:
            action: вектор в ``[-1, 1]`` размерности ``num_hyperparams``.

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        if self.history_window > 0:
            self._history_buf = np.roll(self._history_buf, -1, axis=0)
            entry = np.concatenate([self._current_param_vec(), [self.reward]])
            self._history_buf[-1] = entry

        self._take_action(action)

        self.step_num_total += 1

        reward = self._compute_reward()

        if self._out_of_bounds:
            self._oob_consecutive_count += 1
            reward += self.oob_penalty * self._oob_violation_frac
        else:
            self._oob_consecutive_count = 0
        self.reward = reward

        observation = self._get_obs()
        info = self._get_info()
        info["out_of_bounds"] = self._out_of_bounds
        info["oob_consecutive_count"] = self._oob_consecutive_count
        info["oob_violation_frac"] = self._oob_violation_frac
        terminated = self._terminated_logic()
        truncated = self._truncated_logic()

        return observation, reward, terminated, truncated, info

    def _take_action(self, action):
        """Применяет multi-dim действие ко всем гиперпараметрам одновременно.

        ``action[i]`` ∈ [-1, 1] масштабируется в ``[-max_delta_i, +max_delta_i]``
        и прибавляется к текущему значению параметра i.
        """
        action = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)

        current_vals = np.array(
            [self.current_hyp_setup[n] for n in self.hp_names], dtype=np.float64
        )

        deltas = action * self._max_delta
        raw_new_vals = current_vals + deltas

        violation_lo = np.clip(self._lo - raw_new_vals, 0, None)  
        violation_hi = np.clip(raw_new_vals - self._hi, 0, None)  
        violation = violation_lo + violation_hi  
        safe_max_delta = np.where(self._max_delta > 0, self._max_delta, 1.0)
        self._oob_violation_frac = float(np.max(violation / safe_max_delta))

        self._out_of_bounds = self._oob_violation_frac > 0

        new_vals = np.clip(raw_new_vals, self._lo, self._hi)

        if np.any(self._is_int):
            new_vals[self._is_int] = np.round(new_vals[self._is_int])

        for i, hp_name in enumerate(self.hp_names):
            self.current_hyp_setup[hp_name] = float(new_vals[i])

    def _spawn(self):
        """Сэмплирует случайную начальную конфигурацию в пределах [lo, hi]."""
        for i, hp_name in enumerate(self.hp_names):
            lo = float(self._lo[i])
            hi = float(self._hi[i])
            val = float(self.np_random.uniform(lo, hi))
            if self._is_int[i]:
                val = float(round(val))
            self.current_hyp_setup[hp_name] = val

    def _truncated_logic(self):
        """Проверяет достижение лимита шагов эпизода (truncated)."""
        return self.max_steps_limit <= self.step_num_total

    def _terminated_logic(self):
        """Проверяет терминацию по подряд OOB-шагам (если ``terminate_on_oob``)."""
        if self.terminate_on_oob and self._oob_consecutive_count >= self.oob_tolerance:
            return True
        return False

    def _symlog(self, x):
        """Симметричный log-преобразование: ``sign(x) * log1p(|x|)``."""
        return np.sign(x) * np.log1p(np.abs(x))

    def _compute_reward(self):
        """Оценивает конфигурацию через backend и вычисляет reward по ``reward_mode``."""
        self.raw_metric = self.backend.evaluate(self.current_hyp_setup)
        if self._initial_metric is None:
            self._initial_metric = self.raw_metric
        self.current_raw_metric = self.raw_metric
        if self.prev_raw_metric is None:
            self.prev_raw_metric = self.raw_metric

        scale = abs(self._to_reward(self._initial_metric)) + 1.0

        if self.reward_mode == "bounded":
            delta = self._to_reward(self.raw_metric) - self._to_reward(self.prev_raw_metric)
            self._ema_abs_delta = 0.95 * self._ema_abs_delta + 0.05 * abs(delta)
            normalized = delta / (self._ema_abs_delta + 1e-8)
            r = float(np.tanh(normalized))
            self.prev_raw_metric = self.raw_metric

            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                r += 0.5
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()

            self.reward = r

        elif self.reward_mode == "ternary":
            is_new_best = self._is_improvement(self.raw_metric, self.best_raw_metric)
            improved = self._is_improvement(self.raw_metric, self.prev_raw_metric)
            worsened = self._is_improvement(self.prev_raw_metric, self.raw_metric)

            if is_new_best:
                self.reward = 2.0
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()
            elif improved:
                self.reward = 1.0
            elif worsened:
                self.reward = -1.0
            else:
                self.reward = -0.1

            self.prev_raw_metric = self.raw_metric

        elif self.reward_mode == "guided":
            delta = self._to_reward(self.raw_metric) - self._to_reward(self.prev_raw_metric)
            r_delta_raw = float(delta / scale)
            
            r_delta = r_delta_raw if r_delta_raw >= 0 else max(r_delta_raw, -0.1)
            self.prev_raw_metric = self.raw_metric

            is_new_best = self._is_improvement(self.raw_metric, self.best_raw_metric)
            if is_new_best:
                improvement_over_best = (
                    self._to_reward(self.raw_metric) - self._to_reward(self.best_raw_metric)
                )
                r_best = float(improvement_over_best / scale) + 1.0
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()
            else:
                r_best = 0.0

            progress = self._to_reward(self.raw_metric) - self._to_reward(self._initial_metric)
            r_progress = float(np.clip(progress / scale, -1.0, 1.0))

            self.reward = r_delta + r_best + r_progress

        elif self.reward_mode == "potential":
            old_best_reward = self._to_reward(self.best_raw_metric)

            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()

            new_best_reward = self._to_reward(self.best_raw_metric)
            delta_potential = (new_best_reward - old_best_reward) / scale
            self.reward = float(delta_potential)

        elif self.reward_mode in ("delta", "relative_delta"):
            delta = self._to_reward(self.raw_metric) - self._to_reward(self.prev_raw_metric)
            self.reward = float(delta / scale)
            self.prev_raw_metric = self.raw_metric

            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()
        
        elif self.reward_mode == "abs_positive_delta":
            abs_reward = float(self._symlog(self._to_reward(self.raw_metric)))
            
            delta_raw = self._to_reward(self.raw_metric) - self._to_reward(self.prev_raw_metric)
            delta_reward = float(delta_raw / scale)
            
            delta_bonus = max(0.0, delta_reward)
            
            self.reward = abs_reward + 5.0 * delta_bonus
            
            self.prev_raw_metric = self.raw_metric
            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()

        elif self.reward_mode == "rank_shaped":
            improvement = self._to_reward(self.raw_metric) - self._to_reward(self._initial_metric)
            self.reward = float(self._symlog(improvement / scale * 10.0))

            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()

        elif self.reward_mode == "best_improvement":
            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                delta = self._to_reward(self.raw_metric) - self._to_reward(self.best_raw_metric)
                self.reward = float(delta / scale) + 0.5
            else:
                gap = self._to_reward(self.raw_metric) - self._to_reward(self.best_raw_metric)
                self.reward = float(np.clip(gap / scale, -0.5, 0.0))

            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()

        elif self.reward_mode == "best":
            self.reward = float(self._symlog(self._to_reward(self.best_raw_metric)))

            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()

        elif self.reward_mode == "absolute":  
            self.reward = float(self._symlog(self._to_reward(self.raw_metric)))

            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()
        else:
            warnings.warn("no mode selected")

        return self.reward

    def _current_param_vec(self):
        """Нормализованный вектор текущих параметров [0, 1]."""
        vec = np.zeros(self.num_hyperparams, dtype=np.float32)
        for i, hp_name in enumerate(self.hp_names):
            rng = float(self._range[i])
            if rng > 0:
                vec[i] = (self.current_hyp_setup[hp_name] - float(self._lo[i])) / rng
            else:
                vec[i] = 0.5
        return vec

    def _get_obs(self):
        """Формирует flat-наблюдение: параметры, gap к лучшему, прогресс, reward, step_frac, (history)."""
        param_vec = self._current_param_vec()

        init_scale = abs(self._to_reward(self._initial_metric)) + 1.0
        gap = self._to_reward(self.raw_metric) - self._to_reward(self.best_raw_metric)
        
        obs_gap = np.array(
            [np.tanh(gap / init_scale)], dtype=np.float32
        )

        progress = self._to_reward(self.raw_metric) - self._to_reward(self._initial_metric)
        obs_progress = np.array(
            [np.tanh(progress / init_scale)], dtype=np.float32
        )

        reward_val = np.array([self.reward], dtype=np.float32)
        step_frac = np.array(
            [self.step_num_total / max(self.max_steps_limit, 1)], dtype=np.float32
        )

        parts = [param_vec, obs_gap, obs_progress, reward_val, step_frac]

        if self.history_window > 0:
            parts.append(self._history_buf.ravel())

        return np.concatenate(parts)

    def _get_info(self) -> Dict[str, Any]:
        """Возвращает служебную информацию: текущая и лучшая конфигурация, метрики."""
        return {
            "best_config": self.best_config_so_far,
            "best_metric": self.best_raw_metric,
            "current_config": self.current_hyp_setup,
            "current_metric": self.current_raw_metric,
        }
