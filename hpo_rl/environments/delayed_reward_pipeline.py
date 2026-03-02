import gymnasium as gym
import numpy as np
import itertools

from typing import Dict, Any, Optional, List
from hpo_rl.backends.base import EvaluationBackend
from hpo_rl.environments.new_cycle_move_pipeline import CyclicPipelineEnvNew


class DelayedRewardPipelineEnv(CyclicPipelineEnvNew):
    """Среда на основе CyclicPipelineEnvNew с отложенной наградой.

    Агент по-прежнему циклически перебирает гиперпараметры (один шаг =
    одно изменение одного гиперпараметра).  Однако ``backend.evaluate()``
    вызывается **только** после того, как все гиперпараметры были
    пройдены (т.е. один полный цикл завершён).  На промежуточных шагах
    внутри цикла reward = 0.

    Это снижает число дорогостоящих вызовов бэкенда и заставляет агента
    планировать всю конфигурацию целиком, а не жадно оптимизировать
    каждый параметр по отдельности.
    """

    def __init__(
        self,
        hp_space: Dict[str, Any],
        backend: EvaluationBackend,
        step_sizes: List[int],
        num_bins: int = 20,
        max_steps: int = 200,
        obs_mode: str = "index",
        history_window: int = 0,
        reward_mode: str = "absolute",
    ):
        super().__init__(
            hp_space=hp_space,
            backend=backend,
            step_sizes=step_sizes,
            num_bins=num_bins,
            max_steps=max_steps,
            obs_mode=obs_mode,
            history_window=history_window,
            reward_mode=reward_mode,
        )

    # ------------------------------------------------------------------ #
    #  reset — первый evaluate после spawn, чтобы prev_raw_metric и
    #  best_raw_metric были корректны
    # ------------------------------------------------------------------ #
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        # После super().reset() уже был вызван _compute_reward (evaluate).
        # Это нормально — первая оценка нужна для инициализации baseline.
        return obs, info

    # ------------------------------------------------------------------ #
    #  step — reward только в конце цикла
    # ------------------------------------------------------------------ #
    def step(self, action):
        # --- history: сохраняем ПРЕДЫДУЩЕЕ состояние ДО действия ---
        if self.history_window > 0:
            self._reward_history_buf = np.roll(self._reward_history_buf, -1)
            self._reward_history_buf[-1] = self.reward

            self._param_snapshot_buf = np.roll(self._param_snapshot_buf, -1, axis=0)
            self._param_snapshot_buf[-1] = self._current_param_vec()

        # --- запоминаем старый индекс для penalty ---
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

        # --- Награда только когда цикл завершён (cur_step_num вернулся к 0) ---
        cycle_complete = (self.cur_step_num == 0)

        if cycle_complete:
            reward = self._compute_reward()
        else:
            # Промежуточный шаг — reward = 0, evaluate не вызывается
            self.reward = 0.0
            reward = 0.0

        observation = self._get_obs()
        info = self._get_info()
        terminated = self._terminated_logic()
        truncated = self._truncated_logic()

        return observation, reward, terminated, truncated, info
