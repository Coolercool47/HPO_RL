"""Оркестрация моделей для оптимизации гиперпараметров и backend'ов, а также выдача результатов и графиков. 

- :class:`controller` — оркестратор моделей и backend'ов
- :class:`plot_and_save` - обработка данных, выдача графиков и таблиц
- :func:`check` - проверка и обработка изначального config'а
"""

from hpo_rl.controller.controller import controller
from hpo_rl.controller.plot import plot_and_save
from hpo_rl.controller.check import check

__all__ = [
    "controller",
    "plot_and_save",
    "check",
]