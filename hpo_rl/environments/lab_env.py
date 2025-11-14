from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np

from hpo_rl.environments.base_env import BaseHPOEnv
from hpo_rl.backends.base import EvaluationBackend


class StrategyEnv(BaseHPOEnv):
    """
    Среда-"лаборатория" (Стратег), где агент управляет процессом HPO.

    Агент может выбирать, какой гиперпараметр настраивать следующим,
    и в какой момент завершить настройку и запустить оценку.
    Это требует маскирования действий и подходит для агентов типа MaskablePPO.
    """

    def __init__(self, hp_space: Dict[str, Any], backend: EvaluationBackend):
        super().__init__(hp_space=hp_space, backend=backend)

        self.hp_options = [hp_space[name]['discrete']['choices'] for name in self.hp_names]
        self.num_hyperparams = len(self.hp_names)
        self.max_options = max(len(options) for options in self.hp_options) if self.hp_options else 1

        # Определяем единое пространство действий
        # Действия 0..N-1: "Выбрать параметр N"
        # Действия N..N+M-1: "Выбрать значение M для текущего параметра"
        # Действие N+M: "СТОП и оценить"
        self.action_space_size = self.num_hyperparams + self.max_options + 1
        self.action_space = gym.spaces.Discrete(self.action_space_size)

        # Индексы-разделители в пространстве действий
        self._param_choice_offset = 0
        self._value_choice_offset = self.num_hyperparams
        self._stop_action_index = self.num_hyperparams + self.max_options

        self.observation_space = gym.spaces.Dict({
            "chosen_mask": gym.spaces.Box(low=0, high=1, shape=(self.num_hyperparams,), dtype=np.int8),
            "chosen_indices": gym.spaces.Box(low=0, high=self.max_options - 1, shape=(self.num_hyperparams,), dtype=np.int32),
        })

        self.reset()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Маска, показывающая, какие параметры УЖЕ были выбраны
        self.chosen_mask = np.zeros(self.num_hyperparams, dtype=np.int8)
        # Индексы выбранных значений
        self.chosen_indices = np.zeros(self.num_hyperparams, dtype=np.int32)
        # Фаза: сначала всегда выбираем параметр
        self.phase = "CHOOSE_PARAM"
        # Индекс параметра, для которого мы сейчас выбираем значение
        self.pending_param_idx: Optional[int] = None

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        reward = -0.01  # Небольшой штраф за каждый шаг, чтобы мотивировать к эффективности
        terminated = False

        # Действие СТОП
        if action == self._stop_action_index:
            terminated = True
            config_options = self._assemble_partial_config()
            final_config = self._assemble_config(config_options)
            reward += self.backend.evaluate(final_config)

        elif self.phase == "CHOOSE_PARAM":
            param_idx = action - self._param_choice_offset
            self.pending_param_idx = param_idx
            self.phase = "CHOOSE_VALUE"

        elif self.phase == "CHOOSE_VALUE":
            value_idx = action - self._value_choice_offset
            self.chosen_indices[self.pending_param_idx] = value_idx
            self.chosen_mask[self.pending_param_idx] = 1  # Отмечаем параметр как "настроенный"
            self.pending_param_idx = None  # Сбрасываем ожидание
            self.phase = "CHOOSE_PARAM"  # Возвращаемся к выбору параметра

        # Если все параметры настроены, эпизод завершается автоматически
        if np.all(self.chosen_mask == 1):
            if not terminated:  # Если СТОП не был нажат вручную
                terminated = True
                config_options = self._assemble_partial_config()
                final_config = self._assemble_config(config_options)
                reward += self.backend.evaluate(final_config)

        return self._get_obs(), reward, terminated, False, self._get_info()

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=np.int8)

        if self.phase == "CHOOSE_PARAM":
            # Разрешаем выбирать только те параметры, которые еще не были настроены
            available_params = 1 - self.chosen_mask
            mask[self._param_choice_offset: self._param_choice_offset + self.num_hyperparams] = available_params
            # Разрешаем остановиться, если хотя бы один параметр уже выбран
            if np.any(self.chosen_mask == 1):
                mask[self._stop_action_index] = 1

        elif self.phase == "CHOOSE_VALUE":
            # Разрешаем выбирать только значения для ожидаемого параметра
            num_options = len(self.hp_options[self.pending_param_idx])
            mask[self._value_choice_offset: self._value_choice_offset + num_options] = 1

        return mask

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "chosen_mask": self.chosen_mask.copy(),
            "chosen_indices": self.chosen_indices.copy(),
        }

    def _get_info(self) -> Dict[str, Any]:
        # SB3-Contrib ожидает маску в info, если используется VecEnv
        return {"action_mask": self.action_masks()}

    def _assemble_partial_config(self) -> Dict[str, Any]:
        """Собирает конфиг только из тех параметров, что были выбраны."""
        config = {}
        for i in range(self.num_hyperparams):
            if self.chosen_mask[i] == 1:
                param_name = self.hp_names[i]
                choice_index = self.chosen_indices[i]
                config[param_name] = self.hp_options[i][choice_index]
        return config
