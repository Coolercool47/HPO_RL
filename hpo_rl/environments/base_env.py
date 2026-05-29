from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from hpo_rl.backends.base import EvaluationBackend


class BaseHPOEnv(gym.Env, ABC):
    """Базовый класс для RL-сред HPO, совместимый с Gymnasium."""

    metadata = {"render_modes": []}

    def __init__(self, hp_space: Dict[str, Any], backend: EvaluationBackend):
        """Инициализирует базовую HPO-среду.

        Args:
            hp_space: конфигурация пространства гиперпараметров.
            backend: бэкенд оценки метрики.
        """
        super().__init__()

        if not isinstance(backend, EvaluationBackend):
            raise TypeError("backend должен быть экземпляром EvaluationBackend")
        if not hp_space:
            raise ValueError("hp_space не может быть пустым")

        self.backend = backend
        self.hp_space_config = hp_space
        self.hp_names = list(hp_space.keys())

    def _to_reward(self, raw_metric: float) -> float:
        """Конвертирует сырое значение бэкенда в RL-reward (больше = лучше).

        При минимизации (``maximize=False``) инвертирует знак, чтобы
        RL-агент мог максимизировать reward.

        Args:
            raw_metric: Сырое значение от ``backend.evaluate()``.

        Returns:
            Значение reward для RL-агента.
        """
        return raw_metric if self.backend.maximize else -raw_metric

    def _is_improvement(self, new_raw: float, old_raw: float) -> bool:
        """Проверяет, является ли ``new_raw`` улучшением относительно ``old_raw``.

        Учитывает направление оптимизации бэкенда.

        Args:
            new_raw: Новое сырое значение метрики.
            old_raw: Старое сырое значение метрики.

        Returns:
            True, если ``new_raw`` строго лучше ``old_raw``.
        """
        if self.backend.maximize:
            return new_raw > old_raw
        return new_raw < old_raw

    def _assemble_config(self, chosen_options: Dict[str, Any]) -> Dict[str, Any]:
        """Возможность обработки конфига перед evaluate()."""
        return chosen_options

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Сбрасывает среду и возвращает начальное наблюдение.

        Args:
            seed: seed генератора случайных чисел.
            options: дополнительные опции Gymnasium.

        Returns:
            tuple: (observation, info).
        """
        super().reset(seed=seed)

    @abstractmethod
    def step(self, action):
        """Выполняет один шаг среды.

        Args:
            action: действие агента.

        Returns:
            tuple: (observation, reward, terminated, truncated, info).
        """
        raise NotImplementedError

    @abstractmethod
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Формирует текущее наблюдение.

        Returns:
            наблюдение для агента (ndarray или dict с ключом ``obs``).
        """
        raise NotImplementedError

    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """Формирует служебную информацию шага.

        Returns:
            dict: текущая и лучшая конфигурация, метрики.
        """
        raise NotImplementedError

    def render(self):
        """Заглушка render для Gymnasium (визуализация не реализована)."""
        pass

    def close(self):
        """Очистка кэша бэкенда."""
        if hasattr(self.backend, 'clear_cache'):
            self.backend.clear_cache()
