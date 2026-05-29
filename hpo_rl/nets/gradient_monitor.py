"""Мониторинг градиентов: mixin с хуками и лог после clip_grad_norm_.

``GradientMonitorMixin`` — статистики по слоям до clipping (post-accumulate hook).
``OptimizerStepMonitor`` — снимок ``.grad`` после ``clip_grad_norm_``.
Имена логов: ``<Class>/<role>#<id>``; target-сети из ``deepcopy`` без хуков.
"""

import copy
import numpy as np
import torch
from torch import nn
from tianshou.data import Batch
from tianshou.utils.net.common import ModuleWithVectorOutput
from tianshou.utils.torch_utils import torch_device


class GradientMonitorMixin:
    """Миксин мониторинга градиентов для любого ``nn.Module``.

        Вызывайте ``_init_grad_monitor(...)`` в конце ``__init__``.

        Использует ``register_post_accumulate_grad_hook`` на каждом параметре.
        После backward собирает скалярную статистику (без хранения полных тензоров).

        Args:
            grad_log_interval: период вывода лога (шаги backward)
            grad_verbose: печать в stdout
            grad_monitor_name: явная метка роли в логах
    """

    _instance_counter: int = 0

    @classmethod
    def reset_instance_counter(cls):
        """Сбрасывает счётчик экземпляров (между экспериментами)."""
        cls._instance_counter = 0

    def _init_grad_monitor(
        self,
        grad_log_interval: int = 500,
        grad_verbose: bool = False,
        grad_monitor_name: str | None = None,
    ):
        GradientMonitorMixin._instance_counter += 1
        self._gm_id = GradientMonitorMixin._instance_counter


        role = grad_monitor_name or self._auto_detect_role()
        cls_name = type(self).__name__
        self._gm_name = f"{cls_name}/{role}#{self._gm_id}"

        self._grad_log_interval = grad_log_interval
        self._grad_verbose = grad_verbose
        self._grad_monitor_name = grad_monitor_name


        self._grad_step = 0
        self._hook_handles: list = []


        self._grad_norms: dict = {}


        self._current_step_stats: dict = {}
        self._tracked_param_names: list = []

        self._register_grad_hooks()

    def _auto_detect_role(self) -> str:
        """Определяет роль сети по атрибутам конструктора."""
        if getattr(self, "_concat", False):
            return "critic"
        if getattr(self, "_has_output_head", False):
            return "q_net"
        return "preprocess"

    def _register_grad_hooks(self):
        """Регистрирует post-accumulate hook на обучаемых параметрах."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                self._tracked_param_names.append(name)
                handle = param.register_post_accumulate_grad_hook(
                    self._make_param_hook(name)
                )
                self._hook_handles.append(handle)

        self._num_tracked = len(self._tracked_param_names)

    def _make_param_hook(self, param_name: str):
        """Создаёт замыкание hook для одного параметра."""
        def hook(param):
            g = param.grad.detach()
            self._current_step_stats[param_name] = {
                "norm": g.norm().item(),
                "min": g.min().item(),
                "max": g.max().item(),
                "mean_abs": g.abs().mean().item(),
            }


            if len(self._current_step_stats) >= self._num_tracked:
                self._on_backward_complete()

        return hook

    def _on_backward_complete(self):
        """Вызывается, когда все параметры получили градиенты за шаг backward."""
        self._grad_step += 1


        for name, stats in self._current_step_stats.items():
            if name not in self._grad_norms:
                self._grad_norms[name] = []
            self._grad_norms[name].append(stats)


        self._current_step_stats = {}


        if self._grad_step % self._grad_log_interval == 0:
            self._log_grad_stats()

    def _log_grad_stats(self):
        """Агрегирует и выводит статистику градиентов за последний интервал."""
        if not self._grad_norms:
            return

        lines = [f"[GradMonitor] {self._gm_name} | step {self._grad_step}"]

        for param_name, stats_list in sorted(self._grad_norms.items()):
            norms = [s["norm"] for s in stats_list]
            mins = [s["min"] for s in stats_list]
            maxs = [s["max"] for s in stats_list]
            means = [s["mean_abs"] for s in stats_list]

            avg_norm = np.mean(norms)
            max_norm = np.max(norms)
            min_grad = np.min(mins)
            max_grad = np.max(maxs)
            avg_mean = np.mean(means)


            flag = ""
            if max_norm > 100:
                flag = " [EXPLODING]"
            elif avg_mean < 1e-7:
                flag = " [VANISHING]"

            lines.append(
                f"  {param_name:50s} | "
                f"norm: avg={avg_norm:.6f} max={max_norm:.6f} | "
                f"val: min={min_grad:.6f} max={max_grad:.6f} | "
                f"mean_abs={avg_mean:.6f}{flag}"
            )

        if self._grad_verbose:
            print("\n".join(lines))


        self._grad_norms.clear()


    def __deepcopy__(self, memo):
        """Создаёт копию без gradient hooks.

        tianshou создаёт target-сети через ``deepcopy``; для них хуки не нужны
        (обновление через Polyak averaging). Копия без мониторинга и хуков.
        """


        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k in ("_hook_handles", "_grad_norms", "_current_step_stats",
                      "_tracked_param_names"):

                setattr(result, k, [] if isinstance(v, list) else {})
            else:
                setattr(result, k, copy.deepcopy(v, memo))


        result._gm_name = f"{self._gm_name}:target(no-hooks)"
        result._hook_handles = []
        result._num_tracked = 0
        result._grad_step = 0
        result._grad_verbose = False
        return result


    def remove_hooks(self):
        """Удаляет все зарегистрированные hooks (предотвращает утечки памяти)."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._grad_norms.clear()
        self._current_step_stats.clear()


def _snapshot_net_grads(net: nn.Module) -> dict:
    """Снимает статистику ``.grad`` для всех обучаемых параметров сети.

    Args:
        net: сеть PyTorch

    Returns:
        словарь ``{имя_параметра: {norm, min, max, mean_abs}}``;
        параметры без градиента пропускаются
    """
    stats: dict = {}
    for name, param in net.named_parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad.detach()
            stats[name] = {
                "norm": g.norm().item(),
                "min": g.min().item(),
                "max": g.max().item(),
                "mean_abs": g.abs().mean().item(),
            }
    return stats


class OptimizerStepMonitor:
    """Логирует градиенты после ``clip_grad_norm_`` (патч ``algorithm.optim.step``).

    Args:
        algorithm: алгоритм tianshou (PPO, DQN, …).
        nets: сети для снимка ``.grad`` (желательно ``GradientMonitorMixin``).
        log_interval: период вывода в шагах оптимизатора.
        verbose: печать в stdout.
    """

    def __init__(
        self,
        algorithm,
        nets: list,
        log_interval: int = 200,
        verbose: bool = True,
    ):
        self.nets = nets
        self.log_interval = log_interval
        self.verbose = verbose


        self._step = 0

        self._accum: dict[int, dict[str, list]] = {i: {} for i in range(len(nets))}


        self._optim_wrapper = algorithm.optim
        self._original_step = self._optim_wrapper.step

        monitor = self

        def patched_step(loss, retain_graph=None, create_graph=False):

            monitor._optim_wrapper._optim.zero_grad()
            loss.backward(retain_graph=retain_graph, create_graph=create_graph)


            if monitor._optim_wrapper._max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    monitor._optim_wrapper._module.parameters(),
                    max_norm=monitor._optim_wrapper._max_grad_norm,
                )


            monitor._on_post_clip()


            monitor._optim_wrapper._optim.step()

        self._optim_wrapper.step = patched_step


    def _on_post_clip(self):
        self._step += 1

        for idx, net in enumerate(self.nets):
            snap = _snapshot_net_grads(net)
            for pname, stats in snap.items():
                self._accum[idx].setdefault(pname, []).append(stats)

        if self._step % self.log_interval == 0:
            self._log()

    def _log(self):
        for idx, net in enumerate(self.nets):
            name = getattr(net, "_gm_name", f"net_{idx}")
            acc = self._accum[idx]
            if not acc:
                continue

            lines = [f"[GradMonitor] {name} [post-clip] | step {self._step}"]

            for pname, stats_list in sorted(acc.items()):
                norms = [s["norm"] for s in stats_list]
                mins = [s["min"] for s in stats_list]
                maxs = [s["max"] for s in stats_list]
                means = [s["mean_abs"] for s in stats_list]

                avg_norm = np.mean(norms)
                max_norm = np.max(norms)
                min_grad = np.min(mins)
                max_grad = np.max(maxs)
                avg_mean = np.mean(means)

                flag = ""
                if max_norm > 100:
                    flag = " [EXPLODING]"
                elif avg_mean < 1e-7:
                    flag = " [VANISHING]"

                lines.append(
                    f"  {pname:50s} | "
                    f"norm: avg={avg_norm:.6f} max={max_norm:.6f} | "
                    f"val: min={min_grad:.6f} max={max_grad:.6f} | "
                    f"mean_abs={avg_mean:.6f}{flag}"
                )

            if self.verbose:
                print("\n".join(lines))


            self._accum[idx] = {}


    def remove(self):
        """Восстанавливает оригинальный ``optimizer.step``."""
        self._optim_wrapper.step = self._original_step
        self._accum.clear()


class GradientMonitoredNet(GradientMonitorMixin, ModuleWithVectorOutput):
    """MaskedNet с мониторингом градиентов.

        Args:
            state_shape: форма состояния
            action_shape: форма действий
            hidden_sizes: размеры скрытых слоёв
            device: устройство
            grad_log_interval: период лога
            grad_verbose: печать в stdout
            grad_monitor_name: метка в логах
            concat: для twin-critic SAC — конкатенация action к входу
    """

    def __init__(
        self,
        state_shape,
        action_shape=0,
        hidden_sizes=(128, 128),
        device="cpu",
        grad_log_interval: int = 500,
        grad_verbose: bool = False,
        grad_monitor_name: str | None = None,
        concat: bool = False,
        norm_layer=None,
        **kwargs,
    ):

        _action_prod = int(np.prod(action_shape)) if action_shape is not None else 0

        if _action_prod == 0:
            out_dim = hidden_sizes[-1]
            self._has_output_head = False
        else:
            out_dim = _action_prod
            self._has_output_head = True


        effective_out_dim = hidden_sizes[-1] if not self._has_output_head else out_dim
        super().__init__(output_dim=effective_out_dim)

        self.device = device
        self._concat = concat
        input_dim = int(np.prod(state_shape))

        if concat and _action_prod > 0:
            input_dim += int(np.prod(action_shape))

        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim

        if self._has_output_head:
            layers.append(nn.Linear(curr_dim, out_dim))

        self.model = nn.Sequential(*layers)


        self._init_grad_monitor(grad_log_interval, grad_verbose, grad_monitor_name)

    def forward(self, obs, state=None, info=None):
        mask = None
        x = obs

        if isinstance(obs, (dict, Batch)):
            if "mask" in obs:
                mask = obs["mask"]
            if "obs" in obs:
                x = obs["obs"]

        device = torch_device(self)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = x.flatten(1)

        logits = self.model(x)

        if mask is not None and self._has_output_head:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
            min_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, min_value)

        return logits, state


class GradientMonitoredBaseNet(GradientMonitorMixin, ModuleWithVectorOutput):
    """BaseNet с мониторингом градиентов.

        Args:
            state_shape: форма состояния
            action_shape: форма действий
            hidden_sizes: размеры скрытых слоёв
            device: устройство
            grad_log_interval: период лога
            grad_verbose: печать в stdout
            grad_monitor_name: метка в логах
            concat: для twin-critic SAC
    """

    def __init__(
        self,
        state_shape,
        action_shape=0,
        hidden_sizes=(128, 128),
        device="cpu",
        grad_log_interval: int = 500,
        grad_verbose: bool = False,
        grad_monitor_name: str | None = None,
        concat: bool = False,
        norm_layer=None,
        **kwargs,
    ):
        out_dim = hidden_sizes[-1]
        super().__init__(output_dim=out_dim)

        self.device = device
        self._concat = concat
        input_dim = int(np.prod(state_shape))

        if concat:
            input_dim += int(np.prod(action_shape))

        layers = []
        curr_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim

        self.model = nn.Sequential(*layers)


        self._init_grad_monitor(grad_log_interval, grad_verbose, grad_monitor_name)

    def forward(self, obs, state=None, info=None):
        x = obs.obs if isinstance(obs, (dict, Batch)) and "obs" in obs else obs
        device = torch_device(self)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = x.flatten(1)
        return self.model(x), state


class GradientMonitoredRecurrentBaseNet(GradientMonitorMixin, ModuleWithVectorOutput):
    """RecurrentBaseNet с мониторингом градиентов.

        Drop-in замена :class:`~hpo_rl.nets.recurrent_net.RecurrentBaseNet`.

        Args:
            state_shape: форма состояния
            action_shape: форма действий (совместимость API)
            hidden_layer_size: размер GRU
            num_layers: число слоёв GRU
            device: устройство
            grad_log_interval: период лога
            grad_verbose: печать в stdout
            grad_monitor_name: метка в логах
    """

    def __init__(
        self,
        state_shape,
        action_shape=0,
        hidden_layer_size: int = 128,
        num_layers: int = 1,
        device="cpu",
        grad_log_interval: int = 500,
        grad_verbose: bool = False,
        grad_monitor_name: str | None = None,
        **kwargs,
    ):
        super().__init__(output_dim=hidden_layer_size)
        self.device = device
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        input_dim = int(np.prod(state_shape))

        self.fc = nn.Linear(input_dim, hidden_layer_size)
        self.ln = nn.LayerNorm(hidden_layer_size)
        self.relu = nn.ReLU()

        self.rnn = nn.GRU(
            input_size=hidden_layer_size,
            hidden_size=hidden_layer_size,
            num_layers=num_layers,
            batch_first=True,
        )


        self._init_grad_monitor(grad_log_interval, grad_verbose, grad_monitor_name)

    def forward(self, obs, state=None, info=None):
        x = obs.obs if isinstance(obs, (dict, Batch)) and "obs" in obs else obs
        device = torch_device(self)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)

        is_2d = False
        if len(x.shape) == 2:
            is_2d = True
            x = x.unsqueeze(1)

        x = self.relu(self.ln(self.fc(x)))

        is_empty = (
            state is None
            or (isinstance(state, dict) and not state)
            or (hasattr(state, "is_empty") and state.is_empty())
        )

        if is_empty:
            h_0 = torch.zeros(
                self.num_layers, x.size(0), self.hidden_layer_size, device=x.device
            )
        else:
            if isinstance(state, dict) and "hidden" in state:
                h_0 = state["hidden"]
            elif hasattr(state, "hidden"):
                h_0 = state.hidden
            else:
                h_0 = state

            if not isinstance(h_0, torch.Tensor):
                h_0 = torch.as_tensor(h_0, dtype=torch.float32, device=x.device)

            if len(h_0.shape) == 4:
                h_0 = h_0[:, 0, :, :]

            if len(h_0.shape) == 3 and h_0.shape[1] == self.num_layers:
                h_0 = h_0.transpose(0, 1).contiguous()

        out, h_n = self.rnn(x, h_0)

        if is_2d:
            out = out.squeeze(1)

        h_n = h_n.transpose(0, 1).contiguous()
        return out, {"hidden": h_n.detach()}


class GradientMonitoredRecurrentNet(GradientMonitorMixin, ModuleWithVectorOutput):
    """MaskedRecurrentNet с мониторингом градиентов.

        Drop-in замена :class:`~hpo_rl.nets.masked_recurrent_net.MaskedRecurrentNet`.

        Args:
            state_shape: форма состояния
            action_shape: форма действий
            hidden_sizes: размеры скрытых слоёв
            rnn_layers: число слоёв GRU
            grad_log_interval: период лога
            grad_verbose: печать в stdout
            grad_monitor_name: метка в логах
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        hidden_sizes=(128, 128),
        rnn_layers: int = 1,
        grad_log_interval: int = 500,
        grad_verbose: bool = False,
        grad_monitor_name: str | None = None,
        **kwargs,
    ):
        out_dim = int(np.prod(action_shape))
        super().__init__(output_dim=out_dim)

        input_dim = int(np.prod(state_shape))
        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_sizes[0]

        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.rnn_layers,
            batch_first=True,
        )

        layers = []
        curr_dim = self.hidden_dim
        for hidden_dim in hidden_sizes[1:]:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim

        layers.append(nn.Linear(curr_dim, out_dim))
        self.mlp = nn.Sequential(*layers)


        self._init_grad_monitor(grad_log_interval, grad_verbose, grad_monitor_name)

    def forward(self, obs, state=None, info=None):
        mask = None
        x = obs

        if isinstance(obs, (dict, Batch)):
            mask = obs.get("mask", None)
            x = obs.get("obs", obs)

        device = next(self.parameters()).device

        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)

        is_sequence = len(x.shape) == 3
        if not is_sequence:
            x = x.unsqueeze(1)

        if is_sequence:
            state = None
        else:
            if state is not None:
                if isinstance(state, (dict, Batch)):
                    state = state.get("hidden", state)
                if not isinstance(state, torch.Tensor):
                    state = torch.as_tensor(state, dtype=torch.float32, device=device)
                if state.dim() == 3:
                    state = state.transpose(0, 1).contiguous()
                elif state.dim() == 2:
                    state = state.unsqueeze(0).contiguous()

        rnn_out, hidden_out = self.rnn(x, state)

        last_out = rnn_out[:, -1, :]
        logits = self.mlp(last_out)

        if mask is not None:
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
            if mask.dim() == 3:
                mask = mask[:, -1, :]
            min_value = torch.finfo(logits.dtype).min
            logits = logits.masked_fill(~mask, min_value)

        hidden_to_return = hidden_out.transpose(0, 1).detach()
        return logits, {"hidden": hidden_to_return}
