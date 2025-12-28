"""
Dummy backend для тестирования и отладки.

Этот backend предназначен для тестирования RL-агентов без необходимости
выполнения реальных вычислений. Он вычисляет награду на основе близости
к заданному оптимальному значению.
"""

from hpo_rl.backends.base import EvaluationBackend
from typing import Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class DummyBackend(EvaluationBackend):
    """
    Тестовый backend для отладки и тестирования RL-агентов.
    
    Вычисляет награду на основе близости к предопределенному оптимальному
    значению. Полезен для:
    - Отладки новых сред и агентов
    - Тестирования логики выбора действий
    - Быстрого прототипирования без дорогих вычислений
    
    Награда вычисляется как экспонента от отрицательного среднеквадратичного
    расстояния (MSE) между конфигурацией и оптимумом, нормализованного для
    разных типов значений.
    
    Attributes:
        optimum (Dict[str, Any]): Словарь оптимальных значений гиперпараметров.
            Ключи должны совпадать с именами гиперпараметров в конфигурации.
    
    Examples:
        >>> backend = DummyBackend(optimum={"x0": 5.0, "x1": -3.0})
        >>> reward = backend.evaluate({"x0": 5.0, "x1": -3.0})
        >>> assert abs(reward - 1.0) < 1e-6  # Точное совпадение
        >>> reward = backend.evaluate({"x0": 4.0, "x1": -3.0})
        >>> assert 0.0 < reward < 1.0  # Частичное совпадение
    """
    
    def __init__(self, optimum: Dict[str, Any]) -> None:
        """
        Инициализирует DummyBackend с заданным оптимумом.
        
        Args:
            optimum: Словарь оптимальных значений гиперпараметров.
                Ключи - имена параметров, значения - оптимальные значения
                (могут быть int, float, или другие типы для точного сравнения).
        
        Raises:
            ValueError: Если optimum пустой или None.
            TypeError: Если optimum не является словарем.
        
        Examples:
            >>> backend = DummyBackend(optimum={"learning_rate": 0.001, "batch_size": 32})
        """
        super().__init__()
        
        # Проверка на None должна быть первой, так как isinstance(None, dict) = False
        if optimum is None:
            raise ValueError("Для DummyBackend должен быть указан 'optimum' (не None).")
        
        # Проверка типа должна быть перед проверкой на пустоту
        if not isinstance(optimum, dict):
            raise TypeError(
                f"optimum должен быть словарем (dict), получен {type(optimum).__name__}"
            )
        
        # Проверка на пустоту должна быть последней
        if not optimum:
            raise ValueError("Для DummyBackend должен быть указан непустой 'optimum'.")
        
        self.optimum = optimum
        logger.info(
            f"DummyBackend initialized with {len(optimum)} optimal parameters: "
            f"{list(optimum.keys())}"
        )

    def evaluate(self, config: Dict[str, Any]) -> float:
        """
        Вычисляет награду на основе близости конфигурации к оптимуму.
        
        Награда вычисляется следующим образом:
        1. Для каждого параметра вычисляется нормализованное расстояние до оптимума
        2. Используется среднеквадратичное расстояние (MSE)
        3. Награда = exp(-MSE), что дает значение в диапазоне [0, 1]
        
        Для числовых значений (int, float):
        - Расстояние нормализуется относительно абсолютного значения оптимума
        - Используется относительная ошибка: |config - optimum| / (|optimum| + eps)
        
        Для нечисловых значений:
        - Расстояние = 0, если значения равны, иначе 1
        
        Args:
            config: Словарь с гиперпараметрами для оценки.
                Должен содержать ключи, соответствующие ключам в self.optimum.
                Параметры, отсутствующие в config, игнорируются.
        
        Returns:
            float: Награда в диапазоне [0, 1], где:
                - 1.0 означает точное совпадение со всеми оптимальными значениями
                - 0.0 означает полное несовпадение или отсутствие общих параметров
                - Значения между 0 и 1 указывают на степень близости
        
        Examples:
            >>> backend = DummyBackend(optimum={"x0": 0.0, "x1": 1.0})
            >>> backend.evaluate({"x0": 0.0, "x1": 1.0})
            1.0
            >>> backend.evaluate({"x0": 0.1, "x1": 1.0})
            0.990...
            >>> backend.evaluate({"x0": 10.0, "x1": 1.0})
            0.0...
        """
        if not isinstance(config, dict):
            logger.warning(
                f"config должен быть словарем, получен {type(config).__name__}. "
                "Возвращаем 0.0."
            )
            return 0.0
        
        total_distance = 0.0
        num_params = 0
        
        logger.debug(f"Evaluating config: {config} against optimum: {self.optimum}")
        
        for key, opt_value in self.optimum.items():
            if key not in config:
                logger.debug(f"Parameter '{key}' not found in config, skipping.")
                continue
            
            num_params += 1
            config_value = config[key]
            
            # Вычисляем нормализованное расстояние в зависимости от типа
            if isinstance(opt_value, (int, float)):
                # Для числовых значений используем относительную ошибку
                if not isinstance(config_value, (int, float)):
                    logger.warning(
                        f"Parameter '{key}': optimum is numeric ({opt_value}), "
                        f"but config value is {type(config_value).__name__}. "
                        "Using maximum distance (1.0)."
                    )
                    distance = 1.0
                else:
                    # Нормализуем относительно абсолютного значения оптимума
                    # Добавляем eps для избежания деления на ноль
                    abs_optimum = abs(opt_value)
                    if abs_optimum < 1e-9:
                        # Если оптимум близок к нулю, используем абсолютную разность
                        distance = abs(config_value - opt_value)
                    else:
                        # Относительная ошибка
                        distance = abs(config_value - opt_value) / abs_optimum
                
                # Ограничиваем расстояние сверху для числовой стабильности
                distance = min(distance, 10.0)
            else:
                # Для нечисловых значений используем точное сравнение
                distance = 0.0 if config_value == opt_value else 1.0
            
            total_distance += distance ** 2
        
        # Если нет общих параметров, возвращаем 0.0
        if num_params == 0:
            logger.warning(
                f"No matching parameters found between config ({list(config.keys())}) "
                f"and optimum ({list(self.optimum.keys())}). Returning 0.0."
            )
            return 0.0
        
        # Вычисляем среднеквадратичное расстояние
        mean_squared_distance = total_distance / num_params
        
        # Преобразуем в награду через экспоненту: exp(-MSE)
        # Это дает значение в диапазоне [0, 1], где 1 = полное совпадение
        reward = math.exp(-mean_squared_distance)
        
        logger.debug(
            f"Evaluation result: MSE={mean_squared_distance:.6f}, "
            f"reward={reward:.6f}, compared {num_params} parameters"
        )
        
        return reward
