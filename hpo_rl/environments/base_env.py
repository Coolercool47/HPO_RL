from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np

from hpo_rl.backends.base import EvaluationBackend


class BaseHPOEnv(gym.Env, ABC):
    """
    Абстрактный базовый класс для всех RL-сред HPO в проекте.

    Определяет общий интерфейс, совместимый с Gymnasium, и инкапсулирует
    общую логику, такую как взаимодействие с EvaluationBackend и сборка

    финальной конфигурации.
    """
    # metadata для gymnasium, например, для рендеринга
    metadata = {"render_modes": []}

    def __init__(self, hp_space: Dict[str, Any], backend: EvaluationBackend):
        """
        Инициализирует базовую среду.

        Args:
            hp_space (Dict[str, Any]): Словарь, описывающий пространство
                поиска гиперпараметров.
            backend (EvaluationBackend): Экземпляр вычислительного бэкенда,
                который будет выполнять оценку конфигурации.
        """
        super().__init__()

        if not isinstance(backend, EvaluationBackend):
            raise TypeError("backend должен быть экземпляром EvaluationBackend.")
        self.backend = backend

        if not hp_space:
            raise ValueError("Пространство поиска (hp_space) не может быть пустым.")
        self.hp_space_config = hp_space
        self.hp_names = list(hp_space.keys())

    def _assemble_config(self, chosen_options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Собирает финальный, полный словарь конфигурации, который будет
        передан в `backend.evaluate()`.

        Эта функция-помощник берет словарь с выбранными гиперпараметрами
        и "заворачивает" его в структуру, которую ожидают наши
        фабрики и компоненты.

        Args:
            chosen_options (Dict[str, Any]): Словарь с конкретными
                значениями гиперпараметров, выбранными агентом.

        Returns:
            Dict[str, Any]: Полный конфигурационный словарь.
        """
        # Шизофункция, которая должна была разделять ответственность,
        # но фактически не делает буквально ничего - потому что разделять реально нечего :/
        return chosen_options

    @abstractmethod
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Сбрасывает среду в начальное состояние."""
        # gymnasium требует, чтобы seed и options были keyword-only
        super().reset(seed=seed)
        #  raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Выполняет один шаг в среде."""
        raise NotImplementedError

    @abstractmethod
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Собирает и возвращает текущее наблюдение."""
        raise NotImplementedError

    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """Собирает и возвращает дополнительную информацию."""
        raise NotImplementedError

    # --- Опциональные методы Gymnasium ---
    def render(self):
        """Рендеринг (в нашем случае не используется)."""
        pass

    def close(self):
        """Освобождение ресурсов (в нашем случае не используется)."""
        pass
