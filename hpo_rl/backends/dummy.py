from hpo_rl.backends.base import EvaluationBackend
from typing import Dict, Any
import math


class DummyBackend(EvaluationBackend):
    def __init__(self, optimum: Dict[str, Any]):
        super().__init__()
        if not optimum:
            raise ValueError("Для DummyBackend должен быть указан 'optimum'.")
        self.optimum = optimum
        print(f"DummyBackend инициализирован с оптимумом: {self.optimum}")

    def evaluate(self, config: Dict[str, Any]) -> float:
        # Вычисление награды по мере близости к "оптимуму" (заданному)
        # print(config)
        total_distance = 0.0
        num_params = 0
        print(config)
        for key, opt_value in self.optimum.items():
            if key in config:
                num_params += 1
                config_value = config[key]
                # Нормализуем расстояние для разных типов
                if isinstance(opt_value, (int, float)):
                    distance = abs(config_value - opt_value) / (abs(opt_value) + 1e-9)
                else:
                    distance = 0.0 if config_value == opt_value else 1.0

                total_distance += distance**2

        if num_params == 0:
            return 0.0

        mean_squared_distance = total_distance / num_params

        # Берем e^(-MSE) чтобы усилить сигнал награды
        reward = math.exp(-mean_squared_distance)
        return reward
