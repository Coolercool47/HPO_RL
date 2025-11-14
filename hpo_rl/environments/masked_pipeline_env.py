from typing import Dict, Any, Optional

import gymnasium as gym
import numpy as np

from hpo_rl.environments.base_env import BaseHPOEnv
from hpo_rl.backends.base import EvaluationBackend


class DiscretePipelineEnv(BaseHPOEnv):
    """
    Реализация среды-"конвейера" с нативным дискретным пространством действий.

    На каждом шаге агент выбирает действие из пространства `Discrete(M)`,
    где M - максимальное количество опций среди всех гиперпараметров.
    Для корректной работы требуется маскирование невалидных действий.

    Предназначена для агентов, поддерживающих маскирование (например, MaskablePPO).
    """

    def __init__(self, hp_space: Dict[str, Any], backend: EvaluationBackend):
        """
        Инициализирует среду.
        """
        super().__init__(hp_space=hp_space, backend=backend)

        self.hp_options = []
        for name in self.hp_names:
            param_info = self.hp_space_config[name]
            if 'discrete' in param_info:
                self.hp_options.append(param_info['discrete']['choices'])
            else:
                raise TypeError(f"DiscretePipelineEnv поддерживает только 'discrete' параметры. "
                                f"Параметр '{name}' имеет другой тип.")

        num_hyperparams = len(self.hp_names)
        max_options = max(len(options) for options in self.hp_options) if self.hp_options else 1

        self.action_space = gym.spaces.Discrete(max_options)

        self.observation_space = gym.spaces.Dict({
            "step_count": gym.spaces.Box(low=0, high=num_hyperparams, shape=(1,), dtype=np.int32),
            "chosen_indices": gym.spaces.Box(low=0, high=max_options - 1, shape=(num_hyperparams,), dtype=np.int32)
        })

        self.current_step = 0
        self.chosen_indices = np.zeros(num_hyperparams, dtype=np.int32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.current_step = 0
        self.chosen_indices.fill(0)
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        num_valid_actions = len(self.hp_options[self.current_step])
        if action >= num_valid_actions:
            # за выбор неправильного состояния наказываем агента и не меняем состояние.
            return self._get_obs(), -1.0, False, False, self._get_info()

        self.chosen_indices[self.current_step] = action
        self.current_step += 1

        terminated = self.current_step >= len(self.hp_names)
        reward = 0.0

        if terminated:
            config_options = {
                self.hp_names[i]: self.hp_options[i][self.chosen_indices[i]]
                for i in range(len(self.hp_names))
            }
            final_config = self._assemble_config(config_options)
            reward = self.backend.evaluate(final_config)

        return self._get_obs(), reward, terminated, False, self._get_info()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "step_count": np.array([self.current_step], dtype=np.int32),
            "chosen_indices": self.chosen_indices.copy()
        }

    def _get_info(self) -> Dict[str, Any]:
        """Возвращает маску допустимых действий для текущего шага."""
        if self.current_step < len(self.hp_names):
            num_valid_actions = len(self.hp_options[self.current_step])
            mask = np.zeros(self.action_space.n, dtype=np.int8)
            mask[:num_valid_actions] = 1
            return {"action_mask": mask}
        # Возвращаем пустую маску, если эпизод завершен
        return {"action_mask": np.zeros(self.action_space.n, dtype=np.int8)}

    # sb3-contrib требует, чтобы этот метод был в среде
    def action_masks(self) -> np.ndarray:
        return self._get_info()["action_mask"]
