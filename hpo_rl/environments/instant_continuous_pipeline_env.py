import gymnasium as gym
import numpy as np
import warnings

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
    При выходе за границы применяется штрафной reward (``oob_penalty``),
    а при ``terminate_on_oob=True`` эпизод завершается после
    ``oob_tolerance`` подряд идущих OOB-шагов.

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
    oob_penalty : float
        Штраф (отрицательный reward), применяемый когда действие агента
        приводит к выходу за допустимые границы ``[lo, hi]``.
        По умолчанию ``-10.0``.  Штраф **масштабируется** пропорционально
        величине нарушения (``violation / max_delta``, от 0 до 1+) и
        **добавляется** к основному reward.  Это даёт агенту градиентный
        сигнал: чем сильнее он толкает в стену, тем больше штраф.
    terminate_on_oob : bool
        Если ``True``, эпизод завершается (``terminated=True``) при
        выходе за границы.  По умолчанию ``False``.
    oob_tolerance : int
        Количество **подряд идущих** шагов с выходом за границы,
        после которых срабатывает ``terminated``.  Действует только
        при ``terminate_on_oob=True``.  По умолчанию ``1`` —
        терминация при первом же OOB.  Если задать, например, ``3``,
        агент получит штраф на каждом OOB-шаге, но эпизод
        завершится только после 3 подряд OOB-шагов.
    reward_mode : str
        ``"guided"`` —
        Плотный reward: asymmetric delta за шаг (позитивная часть полная,
        негативная обрезана до -0.1 чтобы не наказывать за exploration через
        «долины») + бонус +1 за новый best + progress vs initial (маленький
        бонус за улучшение относительно начальной позиции).
        Совместим с replay buffer, даёт градиент на каждом шаге.
        НЕ содержит proximity-to-best, который ранее блокировал выход
        из локальных минимумов.
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
        oob_penalty: float = -10.0,
        terminate_on_oob: bool = False,
        oob_tolerance: int = 1,
    ):
        super().__init__(hp_space, backend)
        self._normalize_hp_space()
        self._validate_only_continuous()

        self.max_delta_frac = max_delta_frac
        self.max_steps_limit = max_steps
        self.obs_mode = obs_mode
        self.history_window = history_window
        self.reward_mode = reward_mode
        self.oob_penalty = oob_penalty
        self.terminate_on_oob = terminate_on_oob
        self.oob_tolerance = oob_tolerance

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
        """Observation: [normalized_params, symlog_metric, symlog_best, reward, step_frac, (history)].

        Наблюдение включает:
        - ``normalized_params`` — позиция агента [0, 1] per dim
        - ``symlog_metric`` — symlog(текущая метрика), сжатая в [-7, 0]
        - ``symlog_best`` — symlog(лучшая метрика), сжатая в [-7, 0]
        - ``reward`` — reward текущего шага
        - ``step_frac`` — прогресс эпизода [0, 1]

        Метрики сжимаются через ``symlog`` для выравнивания масштаба
        с нормализованными параметрами [0, 1].  Без сжатия метрики
        в [-1199, 0] доминируют вход сети (250x разница), и сеть
        игнорирует позицию агента.

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
        self._out_of_bounds = False
        self._oob_consecutive_count = 0
        self._oob_violation_frac = 0.0

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

        # Штраф за выход за границы — пропорционален величине нарушения,
        # чтобы агент получал градиент: чем сильнее толкает в стену, тем больше штраф.
        if self._out_of_bounds:
            self._oob_consecutive_count += 1
            reward += self.oob_penalty * self._oob_violation_frac
            self.reward = reward
        else:
            self._oob_consecutive_count = 0

        observation = self._get_obs()
        info = self._get_info()
        info["out_of_bounds"] = self._out_of_bounds
        info["oob_consecutive_count"] = self._oob_consecutive_count
        info["oob_violation_frac"] = self._oob_violation_frac
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
        raw_new_vals = current_vals + deltas

        # Проверяем выход за границы ДО клипа и запоминаем величину нарушения
        violation_lo = np.clip(self._lo - raw_new_vals, 0, None)  # >0 если ниже lo
        violation_hi = np.clip(raw_new_vals - self._hi, 0, None)  # >0 если выше hi
        violation = violation_lo + violation_hi  # суммарное нарушение per dim
        # Нормализуем нарушение по _max_delta (action=1 у стены → violation_frac=1)
        safe_max_delta = np.where(self._max_delta > 0, self._max_delta, 1.0)
        self._oob_violation_frac = float(np.max(violation / safe_max_delta))
        # self._oob_violation_frac = violation

        self._out_of_bounds = self._oob_violation_frac > 0

        new_vals = np.clip(raw_new_vals, self._lo, self._hi)

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
        if self.terminate_on_oob and self._oob_consecutive_count >= self.oob_tolerance:
            return True
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
            # Dense reward for off-policy SAC, designed to NOT trap in local minima.
            #
            # Previous design had r_proximity = clip(gap/scale, -1, 0) * 0.1
            # which PUNISHED any exploration away from best-so-far.  On Schwefel
            # (many local minima) the agent would find the nearest local min in
            # 1-2 steps and then receive negative reward for every exploratory step
            # toward the global optimum because the path crosses worse terrain.
            # Cumulative reward for the optimal path was -0.227 — the agent was
            # literally incentivised to stay put.
            #
            # New design:
            # 1) r_delta: asymmetric — full positive signal for improvements,
            #    but CLIPPED negative signal so exploration through worse regions
            #    costs little.  This allows crossing "valleys" between local minima.
            # 2) r_best: large bonus for discovering new best (unchanged).
            # 3) r_progress: small continuous bonus for being better than INITIAL
            #    position (not best!).  This gives a general "good region" signal
            #    without creating a rubber band to best-so-far.

            # (1) Step delta — asymmetric: rewards improvements more than penalises worsening
            delta = self._to_reward(self.raw_metric) - self._to_reward(self.prev_raw_metric)
            r_delta_raw = float(delta / scale)
            # Clip negative delta to reduce exploration penalty.
            # Positive delta is fully rewarded, negative is capped at -0.1
            # so agent pays small cost for crossing worse terrain.
            
            r_delta = r_delta_raw if r_delta_raw >= 0 else max(r_delta_raw, -0.1)
            # r_delta = r_delta_raw
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

            # (3) Progress vs initial — small continuous signal for being in a good region.
            # Unlike proximity-to-best, this does NOT penalise exploration away from
            # best-so-far.  It only rewards being better than the random starting point.
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
            # 1. Палка (Абсолютное положение)
            abs_reward = float(self._symlog(self._to_reward(self.raw_metric)))
            
            # 2. Морковка (Бонус только за шаг в правильном направлении)
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
            [self._symlog(self._to_reward(self.raw_metric))], dtype=np.float32
        )
        best_metric_val = np.array(
            [self._symlog(self._to_reward(self.best_raw_metric))], dtype=np.float32
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
