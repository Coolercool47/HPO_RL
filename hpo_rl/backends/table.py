from typing import Dict, Any, List
import warnings

import pandas as pd
import numpy as np
from scipy.spatial import KDTree

from .base import EvaluationBackend, CATASTROPHIC_FAILURE_REWARD


class TableBackend(EvaluationBackend):
    """
    Бэкенд-симулятор, использующий предзаписанные данные из CSV.

    Для любой запрошенной конфигурации находит ближайшего соседа в данных
    и возвращает его результат.
    """

    def __init__(self, csv_path: str, param_names: List[str]):
        """
        Args:
            csv_path (str): Путь к CSV-файлу с данными.
            param_names (List[str]): Список имен колонок-гиперпараметров.
        """
        self.data = pd.read_csv(csv_path)
        self.param_names = param_names

        for col in self.param_names + ['reward']:
            if col not in self.data.columns:
                raise ValueError(f"В CSV-файле отсутствует колонка: '{col}'")

        # ВАЖНО: Текущая реализация корректно работает только с числовыми
        # параметрами. Для категориальных потребуется более сложная метрика.
        numeric_params = self.data[self.param_names].select_dtypes(include=np.number).columns.tolist()
        if len(numeric_params) != len(self.param_names):
            warnings.warn("TableBackend: В данных найдены нечисловые параметры. "
                          "Поиск ближайшего соседа может быть некорректным.", UserWarning)

        self.numeric_data = self.data[self.param_names].values
        self.kdtree = KDTree(self.numeric_data)

    def evaluate(self, config: Dict[str, Any]) -> float:
        """Находит ближайшего соседа и возвращает его награду."""
        try:
            query_point = [config[name] for name in self.param_names]
        except KeyError as e:
            print(f"ОШИБКА: В конфигурации отсутствует параметр '{e}', необходимый для TableBackend.")
            return CATASTROPHIC_FAILURE_REWARD

        _, index = self.kdtree.query(query_point)
        reward = self.data.iloc[index]['reward']

        return float(reward)
