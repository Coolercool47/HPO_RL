import gymnasium as gym
import numpy as np

from typing import Dict, Any, Optional
from hpo_rl.backends.base import EvaluationBackend
from hpo_rl.environments.base_env import BaseHPOEnv

from hpo_rl.backends.sequential import SequentialBackend


class InstantContinuousPipelineEnv(BaseHPOEnv):
    """Continuous-action HPO environment with **multi-dimensional** action space.

    Работает только с непрерывными (float/int) гиперпараметрами.
    Агент на каждом шаге выдаёт вектор действий размерности
    ``num_hyperparams``, каждая компонента из ``[-1, 1]``.
    Каждая компонента масштабируется в ``[-max_delta, +max_delta]``
    соответствующего параметра и прибавляется к текущему значению.
    Результат клипается в допустимый диапазон ``[lo, hi]``.

    Никакого разбиения на бины — параметры хранятся как непрерывные числа.
    Никакого циклического переключения — все параметры изменяются
    одновременно за один шаг.

    Parameters
    ----------
    hp_space : Dict[str, Any]
        Пространство гиперпараметров.  Все параметры должны иметь
        ``type`` == ``"float"`` или ``"int"``.
    backend : EvaluationBackend
        Бэкенд для вычисления метрики.
    max_delta_frac : float
        Максимальное относительное изменение за один шаг, как доля от
        ``(hi - lo)``.  Например, 0.1 означает, что за шаг параметр
        может измениться максимум на 10 % диапазона.
    max_steps : int
        Максимальное число шагов в эпизоде.
    obs_mode : str
        ``"norm"`` — каждый параметр нормализуется в ``[0, 1]``.
    history_window : int
        0 — без истории, N — последние N шагов хранятся в наблюдении.
    reward_mode : str
        ``"guided"`` — **рекомендуемый для off-policy (SAC)**.
        Плотный reward: delta за шаг (нормализованный) + бонус +1 за новый best
        + proximity shaping (маленький бонус за близость к best-so-far).
        Совместим с replay buffer, даёт градиент на каждом шаге.
        ``"potential"`` — только при нахождении нового best (sparse).
        ``"delta"`` — пошаговое изменение метрики (нормализовано на initial).
        ``"relative_delta"`` — синоним ``"delta"``.
        ``"best_improvement"`` — бонус за новый best + penalty за gap.
        ``"absolute"`` — ``symlog(-f(x))``.
        ``"best"`` — ``symlog(-best_f)``.

    Note
    ----
    При использовании с ``SequentialBackend`` среда автоматически
    синхронизирует свои диапазоны ``[lo, hi]`` с текущим дочерним
    бэкендом при каждом ``reset()``.  Это означает, что
    ``max_delta_frac`` применяется к **native** диапазону текущей
    функции, а не к merged bounds.  Параметры в observation
    нормализованы в ``[0, 1]`` относительно текущих bounds.
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
    ):
        super().__init__(hp_space, backend)
        self._normalize_hp_space()
        self._validate_only_continuous()

        self.max_delta_frac = max_delta_frac
        self.max_steps_limit = max_steps
        self.obs_mode = obs_mode
        self.history_window = history_window
        self.reward_mode = reward_mode

        self.num_hyperparams = len(self.hp_names)
        self.step_num_total = 0

        # Текущие непрерывные значения параметров
        self.current_hyp_setup: Dict[str, float] = {}
        self.best_config_so_far: Dict[str, float] = {}
        self.best_raw_metric = float("-inf") if self.backend.maximize else float("inf")
        self.current_raw_metric = float("-inf") if self.backend.maximize else float("inf")
        self.raw_metric = 0.0
        self.reward = 0.0
        self._initial_metric = 0.0  # для improvement reward mode

        # Диапазоны
        self._lo = np.array(
            [self.hp_space_config[n]["values"][0] for n in self.hp_names],
            dtype=np.float64,
        )
        self._hi = np.array(
            [self.hp_space_config[n]["values"][1] for n in self.hp_names],
            dtype=np.float64,
        )
        self._range = self._hi - self._lo
        self._max_delta = self.max_delta_frac * self._range  # вектор max_delta

        # Типы параметров для округления int
        self._is_int = np.array(
            [self.hp_space_config[n]["type"] == "int" for n in self.hp_names],
            dtype=bool,
        )

        self._init_action_space()
        self._init_observation_space()

    # ------------------------------------------------------------------
    # Validation & normalization helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Spaces
    # ------------------------------------------------------------------
    def _init_action_space(self):
        """Multi-dimensional action: one delta per hyperparameter."""
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.num_hyperparams,),
            dtype=np.float32,
        )

    def _init_observation_space(self):
        """Observation: [normalized_params, raw_metric, best_metric, reward, step_frac, (history)].

        Наблюдение включает:
        - ``normalized_params`` — позиция агента [0, 1] per dim
        - ``raw_metric`` — текущая метрика (sign-flipped для minimize задач)
        - ``best_metric`` — лучшая найденная метрика (sign-flipped)
        - ``reward`` — reward текущего шага
        - ``step_frac`` — прогресс эпизода [0, 1]

        Все значения передаются **без ручной нормализации** —
        LayerNorm в сети сам приводит масштаб.

        Если ``history_window > 0``, добавляются предыдущие параметры и
        reward для окна из последних N шагов.
        """
        # params + raw_metric + best_metric + reward + step_frac
        self._obs_base_dim = self.num_hyperparams + 4

        flat_obs_dim = self._obs_base_dim

        if self.history_window > 0:
            self._hist_entry_dim = self.num_hyperparams + 1  # params + reward
            flat_obs_dim += self.history_window * self._hist_entry_dim

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_obs_dim,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Bounds sync (SequentialBackend)
    # ------------------------------------------------------------------
    def sync_bounds_to_backend(self):
        """Синхронизирует ``_lo``, ``_hi``, ``_range``, ``_max_delta`` с текущим дочерним бэкендом.

        Вызывается из ``reset()`` при использовании ``SequentialBackend``.
        Env переходит на **native** bounds текущего дочернего бэкенда,
        чтобы агент работал в правильном масштабе для каждой функции.
        Ремаппинг в ``SequentialBackend._evaluate()`` при этом не нужен —
        координаты уже в native bounds. Устанавливает ``backend.skip_remap = True``.
        """

        if not isinstance(self.backend, SequentialBackend):
            return

        cb = self.backend.current_backend
        if not hasattr(cb, 'bounds') or not hasattr(cb, 'dimensions'):
            return

        child_lo, child_hi = cb.bounds
        self._lo = np.full(self.num_hyperparams, child_lo, dtype=np.float64)
        self._hi = np.full(self.num_hyperparams, child_hi, dtype=np.float64)
        self._range = self._hi - self._lo
        self._max_delta = self.max_delta_frac * self._range

        # Отключаем ремаппинг — координаты уже в native bounds
        self.backend.skip_remap = True

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        # Синхронизируем bounds с текущим дочерним бэкендом (SequentialBackend)
        self.sync_bounds_to_backend()

        self.current_hyp_setup = {}
        self.best_config_so_far = {}
        self.best_raw_metric = float("-inf") if self.backend.maximize else float("inf")
        self.current_raw_metric = float("-inf") if self.backend.maximize else float("inf")
        self.prev_raw_metric = None

        self.step_num_total = 0

        if self.history_window > 0:
            self._history_buf = np.zeros(
                (self.history_window, self._hist_entry_dim), dtype=np.float32
            )

        self._spawn()
        # Evaluate initial metric BEFORE _compute_reward so scale is correct
        self._initial_metric = self.backend.evaluate(self.current_hyp_setup)
        self.raw_metric = self._initial_metric
        self.current_raw_metric = self._initial_metric
        # Set best to initial so best_improvement doesn't get inf delta
        self.best_raw_metric = self._initial_metric
        self.best_config_so_far = self.current_hyp_setup.copy()
        self.prev_raw_metric = self._initial_metric
        self._compute_reward()
        # First step reward should be 0 for delta-based modes (no change yet)
        if self.reward_mode in ("delta", "relative_delta", "potential", "guided"):
            self.reward = 0.0

        # Заполняем историю начальным состоянием
        if self.history_window > 0:
            entry = np.concatenate([self._current_param_vec(), [self.reward]])
            self._history_buf[:] = entry

        return self._get_obs(), self._get_info()

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        # Сохраняем в историю ДО действия
        if self.history_window > 0:
            self._history_buf = np.roll(self._history_buf, -1, axis=0)
            entry = np.concatenate([self._current_param_vec(), [self.reward]])
            self._history_buf[-1] = entry

        self._take_action(action)

        self.step_num_total += 1

        reward = self._compute_reward()

        observation = self._get_obs()
        info = self._get_info()
        terminated = self._terminated_logic()
        truncated = self._truncated_logic()

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def _take_action(self, action):
        """Применяет multi-dim действие ко всем гиперпараметрам одновременно.

        ``action[i]`` ∈ [-1, 1] масштабируется в ``[-max_delta_i, +max_delta_i]``
        и прибавляется к текущему значению параметра i.
        """
        action = np.asarray(action, dtype=np.float32).flatten()

        current_vals = np.array(
            [self.current_hyp_setup[n] for n in self.hp_names], dtype=np.float64
        )

        deltas = action * self._max_delta
        new_vals = np.clip(current_vals + deltas, self._lo, self._hi)

        # Округляем int параметры
        if np.any(self._is_int):
            new_vals[self._is_int] = np.round(new_vals[self._is_int])

        for i, hp_name in enumerate(self.hp_names):
            self.current_hyp_setup[hp_name] = float(new_vals[i])

    # ------------------------------------------------------------------
    # Spawn (random init)
    # ------------------------------------------------------------------
    def _spawn(self):
        for i, hp_name in enumerate(self.hp_names):
            lo = float(self._lo[i])
            hi = float(self._hi[i])
            val = np.random.uniform(lo, hi)
            if self._is_int[i]:
                val = float(round(val))
            self.current_hyp_setup[hp_name] = val

    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------
    def _truncated_logic(self):
        return self.max_steps_limit <= self.step_num_total

    def _terminated_logic(self):
        return False

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _symlog(self, x):
        return np.sign(x) * np.log1p(np.abs(x))

    def _compute_reward(self):
        self.raw_metric = self.backend.evaluate(self.current_hyp_setup)
        self.current_raw_metric = self.raw_metric

        scale = abs(self._to_reward(self._initial_metric)) + 1.0

        if self.reward_mode == "guided":
            # Dense reward for off-policy SAC:
            # 1) delta: step-wise improvement (positive = improved, negative = worsened)
            # 2) best_bonus: +1.0 if new best found (sparse but large)
            # 3) proximity: small continuous bonus for being close to best-so-far
            #
            # This gives gradient on EVERY step (delta), excitement for discovery
            # (best_bonus), and pull toward good regions (proximity).
            # All components are normalized by scale → comparable across functions.

            # (1) Step delta — gives per-step gradient signal
            delta = self._to_reward(self.raw_metric) - self._to_reward(self.prev_raw_metric)
            r_delta = float(delta / scale)
            self.prev_raw_metric = self.raw_metric

            # (2) Best bonus — reward discovery of new best
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

            # (3) Proximity to best — small continuous shaping signal
            # gap = how far current metric is from best (always <= 0 for reward)
            gap = self._to_reward(self.raw_metric) - self._to_reward(self.best_raw_metric)
            r_proximity = float(np.clip(gap / scale, -1.0, 0.0)) * 0.1

            self.reward = r_delta + r_best + r_proximity

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

        else:  # absolute
            self.reward = float(self._symlog(self._to_reward(self.raw_metric)))

            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()

        return self.reward

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
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
        param_vec = self._current_param_vec()

        # Raw metric signals — LayerNorm в сети сам нормализует масштаб
        raw_metric_val = np.array(
            [self._to_reward(self.raw_metric)], dtype=np.float32
        )
        best_metric_val = np.array(
            [self._to_reward(self.best_raw_metric)], dtype=np.float32
        )
        reward_val = np.array([self.reward], dtype=np.float32)
        step_frac = np.array(
            [self.step_num_total / max(self.max_steps_limit, 1)], dtype=np.float32
        )

        parts = [param_vec, raw_metric_val, best_metric_val, reward_val, step_frac]

        if self.history_window > 0:
            parts.append(self._history_buf.ravel())

        return np.concatenate(parts)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "best_config": self.best_config_so_far,
            "best_metric": self.best_raw_metric,
            "current_config": self.current_hyp_setup,
            "current_metric": self.current_raw_metric,
        }
