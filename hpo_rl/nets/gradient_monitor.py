"""
Миксин и обёртки для мониторинга градиентов в нейронных сетях.

Навешивает backward-хуки на все параметры модели, собирает статистику
(mean, max, norm) по градиентам каждого слоя и суммарно.
Логирует в wandb (если доступен) и/или в stdout.

Использование:
    # Вместо BaseNet — GradientMonitoredBaseNet
    # Вместо Net (tianshou) — GradientMonitoredNet

    config["full_args"]["net"]["net"] = GradientMonitoredBaseNet
    # или
    config["full_args"]["net"]["net"] = GradientMonitoredNet
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
from typing import Optional

try:
    import wandb
except ImportError:
    wandb = None


class GradientMonitorMixin:
    """Миксин, добавляющий мониторинг градиентов к любому nn.Module.

    Регистрирует full backward hooks на все параметры,
    накапливает статистику и предоставляет метод для логирования.

    Атрибуты:
        _grad_stats: dict[str, list[float]] — накопленные нормы градиентов по слоям
        _grad_hooks: list[RemovableHandle] — зарегистрированные хуки
        _grad_log_interval: int — каждые N backward-проходов логировать
        _grad_step_counter: int — счётчик backward-проходов
        _grad_logging_enabled: bool — включён ли мониторинг
    """

    _grad_instance_counter: dict[str, int] = {}  # class-level: отслеживание кол-ва инстансов по классам

    def _init_gradient_monitor(self, log_interval: int = 100, verbose: bool = False, 
                                monitor_name: str | None = None):
        """Инициализация мониторинга градиентов.

        Args:
            log_interval: каждые N backward-проходов выводить/логировать статистику.
            verbose: печатать ли в stdout при каждом логировании.
            monitor_name: имя для логов (если None — автоматически "ClassName_0", "ClassName_1", ...).
        """
        cls_name = type(self).__name__
        if monitor_name is None:
            count = GradientMonitorMixin._grad_instance_counter.get(cls_name, 0)
            GradientMonitorMixin._grad_instance_counter[cls_name] = count + 1
            self._grad_monitor_name = f"{cls_name}_{count}"
        else:
            self._grad_monitor_name = monitor_name
            
        self._grad_stats: dict[str, list[float]] = defaultdict(list)
        self._grad_hooks: list = []
        self._grad_log_interval = log_interval
        self._grad_step_counter = 0
        self._grad_verbose = verbose
        self._grad_logging_enabled = True

        self._register_grad_hooks()

    def _register_grad_hooks(self):
        """Регистрирует backward-хуки на все параметры с requires_grad=True."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(self._make_grad_hook(name))
                self._grad_hooks.append(hook)

    def _make_grad_hook(self, param_name: str):
        """Создаёт замыкание-хук для конкретного параметра."""
        def hook(grad):
            if not self._grad_logging_enabled:
                return
            if grad is None:
                return
            grad_data = grad.detach()
            grad_norm = grad_data.norm().item()
            grad_mean = grad_data.abs().mean().item()
            grad_max = grad_data.abs().max().item()

            self._grad_stats[f"{param_name}/norm"].append(grad_norm)
            self._grad_stats[f"{param_name}/mean"].append(grad_mean)
            self._grad_stats[f"{param_name}/max"].append(grad_max)

            # Увеличиваем счётчик только для первого параметра (чтобы не умножать)
            # Используем специальный маркер
            if not hasattr(self, '_grad_first_param_name'):
                self._grad_first_param_name = param_name
            if param_name == self._grad_first_param_name:
                self._grad_step_counter += 1
                if self._grad_step_counter % self._grad_log_interval == 0:
                    self._log_grad_stats()
        return hook

    def _log_grad_stats(self):
        """Агрегирует и логирует накопленную статистику градиентов."""
        if not self._grad_stats:
            return

        log_dict = {}
        all_norms = []

        # Получаем prefix из имени инстанса (уникальный: ClassName_0, ClassName_1, ...)
        net_prefix = self._grad_monitor_name

        for key, values in self._grad_stats.items():
            if not values:
                continue
            avg_val = np.mean(values)
            log_dict[f"grad/{net_prefix}/{key}"] = avg_val
            if key.endswith("/norm"):
                all_norms.extend(values)

        # Общая статистика
        if all_norms:
            log_dict[f"grad/{net_prefix}/total_norm_mean"] = np.mean(all_norms)
            log_dict[f"grad/{net_prefix}/total_norm_max"] = np.max(all_norms)
            log_dict[f"grad/{net_prefix}/total_norm_min"] = np.min(all_norms)

        # Логирование в wandb
        if wandb is not None and wandb.run is not None:
            wandb.log(log_dict, commit=False)

        # Логирование в stdout
        if self._grad_verbose:
            print(f"\n[GradMonitor] {self._grad_monitor_name} — step {self._grad_step_counter}")
            print(f"  Total grad norm: mean={np.mean(all_norms):.6f}, "
                  f"max={np.max(all_norms):.6f}, min={np.min(all_norms):.6f}")
            # Топ-5 слоёв по норме градиента
            layer_norms = {}
            for key, values in self._grad_stats.items():
                if key.endswith("/norm") and values:
                    layer_norms[key] = np.mean(values)
            sorted_layers = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)
            for layer_name, norm_val in sorted_layers[:5]:
                print(f"  {layer_name}: {norm_val:.6f}")

        # Очищаем буфер
        self._grad_stats.clear()

    def get_gradient_summary(self) -> dict:
        """Возвращает текущую накопленную статистику (без очистки)."""
        summary = {}
        for key, values in self._grad_stats.items():
            if values:
                summary[key] = {
                    "mean": np.mean(values),
                    "max": np.max(values),
                    "min": np.min(values),
                    "count": len(values),
                }
        return summary

    def remove_grad_hooks(self):
        """Удаляет все зарегистрированные хуки."""
        for hook in self._grad_hooks:
            hook.remove()
        self._grad_hooks.clear()
        self._grad_logging_enabled = False

    def set_grad_logging(self, enabled: bool):
        """Включает/выключает мониторинг."""
        self._grad_logging_enabled = enabled


# ──────────────────────────────────────────────────────────────────────
# Конкретные обёртки
# ──────────────────────────────────────────────────────────────────────

from hpo_rl.nets.base_net import BaseNet


class GradientMonitoredBaseNet(GradientMonitorMixin, BaseNet):
    """BaseNet с мониторингом градиентов.

    Полностью совместима с BaseNet — тот же интерфейс, те же параметры.
    Дополнительные kwargs:
        grad_log_interval (int): каждые N backward-проходов логировать. Default: 100.
        grad_verbose (bool): печатать ли в stdout. Default: False.
    """

    def __init__(self, state_shape, action_shape, hidden_sizes=[128, 128], device='cpu',
                 grad_log_interval: int = 100, grad_verbose: bool = False, **kwargs):
        # Убираем наши kwargs, чтобы не передавать в BaseNet
        BaseNet.__init__(self, state_shape=state_shape, action_shape=action_shape,
                         hidden_sizes=hidden_sizes, device=device)
        self._init_gradient_monitor(log_interval=grad_log_interval, verbose=grad_verbose)


from tianshou.utils.net.common import Net


class GradientMonitoredNet(GradientMonitorMixin, Net):
    """tianshou Net с мониторингом градиентов.

    Полностью совместима с Net — тот же интерфейс, те же параметры.
    Дополнительные kwargs:
        grad_log_interval (int): каждые N backward-проходов логировать. Default: 100.
        grad_verbose (bool): печатать ли в stdout. Default: False.
    """

    def __init__(self, *args, grad_log_interval: int = 100, grad_verbose: bool = False, **kwargs):
        # Извлекаем наши параметры до передачи в Net
        Net.__init__(self, *args, **kwargs)
        self._init_gradient_monitor(log_interval=grad_log_interval, verbose=grad_verbose)
