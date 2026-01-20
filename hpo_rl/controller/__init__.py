"""Оркестрация моделей для оптимизации гиперпараметров и backend'ов. 

- :class:`controller` — оркестратор
"""

from hpo_rl.controller.controller import controller

__all__ = [
    "controller",
]