"""
Gradient monitoring wrappers for neural networks.

Provides mixin and ready-to-use Net classes that track gradient statistics
(min, max, mean norm per layer) using backward hooks.  The hooks are
registered lazily on the first forward pass and are designed to avoid
memory leaks:

* Gradients are `.detach()`-ed and reduced to scalars immediately —
  no full gradient tensors are kept alive.
* Hook handles are stored so they can be removed via `remove_hooks()`.

Naming convention for logs:
    Each net is labelled as ``<ClassName>/<role>#<id>``, where *role*
    is either the explicit ``grad_monitor_name`` you pass in, or
    auto-detected from constructor arguments:

      * ``concat=True``  → "critic"
      * ``action_shape > 0, concat=False`` → "actor"  (or "q_net" for value-based)
      * ``action_shape == 0`` → "preprocess"

    Target (lagged) networks created by tianshou via ``deepcopy`` are
    detected automatically: ``__deepcopy__`` disables hooks on the copy,
    so they never produce gradient logs.

Usage in config:
    "net": {
        "net": GradientMonitoredNet,       # or GradientMonitoredBaseNet
        "hidden_sizes": [256, 256],
        "grad_log_interval": 500,          # log every N backward steps
        "grad_verbose": True,              # print to stdout
        "grad_monitor_name": "actor",      # optional explicit label
    }

Post-clipping monitoring (OptimizerStepMonitor):
    ``register_post_accumulate_grad_hook`` fires AFTER ``loss.backward()``
    but BEFORE ``clip_grad_norm_`` — so the mixin sees raw (pre-clipping)
    gradients.  To also see post-clipping stats, wrap the tianshou
    algorithm's optimizer with ``OptimizerStepMonitor``::

        monitor = OptimizerStepMonitor(
            algorithm,                       # tianshou PPO / DQN / SAC …
            nets=[net_actor, net_critic],     # GradientMonitorMixin instances
            log_interval=200,
            verbose=True,
        )
        # … train as usual …
        monitor.remove()                    # cleanup when done

    The monitor patches ``algorithm.optim.step`` so that right after
    ``clip_grad_norm_`` (but before ``optim.step()``) it snapshots each
    net's ``.grad`` values and logs them with a ``[post-clip]`` tag.

Log format (each line per parameter):
    [GradMonitor] <NetName> | step <N>
      <param_name>                 | norm: avg=... max=... | val: min=... max=... | mean_abs=...

    norm avg/max  — L2-norm of the gradient tensor, averaged / max over the
                    last ``grad_log_interval`` backward steps.
    val min/max   — element-wise minimum / maximum gradient value (shows
                    sign and magnitude of the most extreme individual
                    gradient element over the interval).
    mean_abs      — mean of |grad| elements, averaged over the interval.
                    A quick proxy for "how large are gradients on average".
    [EXPLODING]   — printed when max_norm > 100.
    [VANISHING]   — printed when mean_abs < 1e-7.
"""

import copy
import numpy as np
import torch
from torch import nn
from tianshou.data import Batch
from tianshou.utils.net.common import ModuleWithVectorOutput
from tianshou.utils.torch_utils import torch_device


# ---------------------------------------------------------------------------
#  Mixin — core gradient monitoring logic
# ---------------------------------------------------------------------------

class GradientMonitorMixin:
    """Mixin that adds gradient monitoring to any nn.Module.

    Call ``_init_grad_monitor(...)`` at the END of your ``__init__``.

    Uses ``register_post_accumulate_grad_hook`` on each parameter to capture
    gradients at the exact moment they are computed.  When all tracked
    parameters have reported their gradient for the current backward pass,
    stats are aggregated and (if interval is reached) logged.

    No full gradient tensors are kept — only scalar statistics.
    """

    _instance_counter: int = 0  # class-level counter for naming

    @classmethod
    def reset_instance_counter(cls):
        """Reset the instance counter (useful between experiments)."""
        cls._instance_counter = 0

    def _init_grad_monitor(
        self,
        grad_log_interval: int = 500,
        grad_verbose: bool = False,
        grad_monitor_name: str | None = None,
    ):
        GradientMonitorMixin._instance_counter += 1
        self._gm_id = GradientMonitorMixin._instance_counter

        # Build human-readable name
        role = grad_monitor_name or self._auto_detect_role()
        cls_name = type(self).__name__
        self._gm_name = f"{cls_name}/{role}#{self._gm_id}"

        self._grad_log_interval = grad_log_interval
        self._grad_verbose = grad_verbose
        self._grad_monitor_name = grad_monitor_name

        # bookkeeping
        self._grad_step = 0
        self._hook_handles: list = []

        # per-interval accumulator:  param_name -> list of {norm, min, max, mean_abs}
        self._grad_norms: dict = {}

        # per-step accumulator: tracks which params reported in current step
        self._current_step_stats: dict = {}
        self._tracked_param_names: list = []

        self._register_grad_hooks()

    def _auto_detect_role(self) -> str:
        """Guess the role of this network from its constructor attributes."""
        if getattr(self, "_concat", False):
            return "critic"
        if getattr(self, "_has_output_head", False):
            return "q_net"
        return "preprocess"

    def _register_grad_hooks(self):
        """Register per-parameter gradient hooks."""
        for name, param in self.named_parameters():
            if param.requires_grad:
                self._tracked_param_names.append(name)
                handle = param.register_post_accumulate_grad_hook(
                    self._make_param_hook(name)
                )
                self._hook_handles.append(handle)

        self._num_tracked = len(self._tracked_param_names)

    def _make_param_hook(self, param_name: str):
        """Create a closure for per-parameter gradient hook."""
        def hook(param):
            g = param.grad.detach()
            self._current_step_stats[param_name] = {
                "norm": g.norm().item(),
                "min": g.min().item(),
                "max": g.max().item(),
                "mean_abs": g.abs().mean().item(),
            }

            # Check if all tracked params have reported
            if len(self._current_step_stats) >= self._num_tracked:
                self._on_backward_complete()

        return hook

    def _on_backward_complete(self):
        """Called when all parameters have received their gradients."""
        self._grad_step += 1

        # Move current step stats into interval accumulator
        for name, stats in self._current_step_stats.items():
            if name not in self._grad_norms:
                self._grad_norms[name] = []
            self._grad_norms[name].append(stats)

        # Reset per-step accumulator
        self._current_step_stats = {}

        # Log at interval
        if self._grad_step % self._grad_log_interval == 0:
            self._log_grad_stats()

    def _log_grad_stats(self):
        """Aggregate and print gradient statistics over the last interval."""
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

            # Warn about potential issues  (no emoji — ASCII only)
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

        # Reset accumulator
        self._grad_norms.clear()

    # ---- deepcopy support -------------------------------------------------

    def __deepcopy__(self, memo):
        """Create a copy WITHOUT gradient hooks.

        tianshou creates target (lagged) networks via ``deepcopy``.
        Target nets are not trained — they receive weight updates via
        Polyak averaging — so gradient hooks are useless and wasteful.
        This override produces a clean copy with no hooks and no
        monitoring overhead.
        """
        # 1. Shallow-copy the instance, then deep-copy everything except
        #    hook handles and accumulators.
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k in ("_hook_handles", "_grad_norms", "_current_step_stats",
                      "_tracked_param_names"):
                # Empty copies — no hooks on the target net
                setattr(result, k, [] if isinstance(v, list) else {})
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        # 2. Override monitoring attributes to mark this as a passive copy
        result._gm_name = f"{self._gm_name}:target(no-hooks)"
        result._hook_handles = []
        result._num_tracked = 0
        result._grad_step = 0
        result._grad_verbose = False
        return result

    # ---- cleanup ----------------------------------------------------------

    def remove_hooks(self):
        """Remove all registered hooks to avoid memory leaks."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._grad_norms.clear()
        self._current_step_stats.clear()


# ---------------------------------------------------------------------------
#  OptimizerStepMonitor — post-clipping gradient monitoring
# ---------------------------------------------------------------------------

def _snapshot_net_grads(net: nn.Module) -> dict:
    """Capture current .grad stats for every trainable parameter in *net*.

    Returns ``{param_name: {norm, min, max, mean_abs}}``.
    Parameters without a gradient are silently skipped.
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
    """Monitor gradients *after* ``clip_grad_norm_`` inside tianshou's
    ``Algorithm.Optimizer.step``.

    How it works
    ------------
    tianshou wraps the raw ``torch.optim.Optimizer`` in
    ``Algorithm.Optimizer`` whose ``step()`` does::

        zero_grad  →  loss.backward  →  clip_grad_norm_  →  optim.step

    ``register_post_accumulate_grad_hook`` fires right after
    ``loss.backward()`` — i.e. **before** clipping.  This class
    monkey-patches ``algorithm.optim.step`` to insert a snapshot of
    each tracked net's ``.grad`` **between** ``clip_grad_norm_`` and
    ``optim.step()`` so that logged values reflect the actual clipped
    gradients that drive weight updates.

    Parameters
    ----------
    algorithm : tianshou Algorithm
        The algorithm whose ``optim`` will be patched.
    nets : list[nn.Module]
        Networks whose parameters should be snapshotted after clipping.
        Each net should be a ``GradientMonitorMixin`` instance (for its
        ``_gm_name``), but plain ``nn.Module`` also works.
    log_interval : int
        Print / accumulate stats every N optimizer steps.
    verbose : bool
        Whether to print to stdout.
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

        # bookkeeping
        self._step = 0
        # net_idx -> param_name -> list[{norm, min, max, mean_abs}]
        self._accum: dict[int, dict[str, list]] = {i: {} for i in range(len(nets))}

        # ---- monkey-patch algorithm.optim.step ---------------------------
        self._optim_wrapper = algorithm.optim  # Algorithm.Optimizer instance
        self._original_step = self._optim_wrapper.step

        monitor = self  # closure reference

        def patched_step(loss, retain_graph=None, create_graph=False):
            # 1. zero_grad + backward
            monitor._optim_wrapper._optim.zero_grad()
            loss.backward(retain_graph=retain_graph, create_graph=create_graph)

            # 2. clip_grad_norm_ (same logic as original)
            if monitor._optim_wrapper._max_grad_norm is not None:
                nn.utils.clip_grad_norm_(
                    monitor._optim_wrapper._module.parameters(),
                    max_norm=monitor._optim_wrapper._max_grad_norm,
                )

            # 3. >>> snapshot post-clipping gradients <<<
            monitor._on_post_clip()

            # 4. actual optimizer step
            monitor._optim_wrapper._optim.step()

        self._optim_wrapper.step = patched_step

    # ---- internal --------------------------------------------------------

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

            # Reset this net's accumulator
            self._accum[idx] = {}

    # ---- cleanup ---------------------------------------------------------

    def remove(self):
        """Restore the original ``optimizer.step`` method."""
        self._optim_wrapper.step = self._original_step
        self._accum.clear()


# ---------------------------------------------------------------------------
#  GradientMonitoredNet  — drop-in replacement for MaskedNet
# ---------------------------------------------------------------------------

class GradientMonitoredNet(GradientMonitorMixin, ModuleWithVectorOutput):
    """MaskedNet + gradient monitoring.

    Accepts the same (state_shape, action_shape, hidden_sizes, device) as
    MaskedNet, plus ``grad_log_interval``, ``grad_verbose`` and optional
    ``grad_monitor_name`` for explicit labelling in logs.

    For SAC twin-critic, also accepts ``concat=True`` which prepends
    action_shape to the input dim (like tianshou's Net).
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
        **kwargs,  # absorb extra keys silently
    ):
        # Determine output dimensionality
        _action_prod = int(np.prod(action_shape)) if action_shape is not None else 0
        
        if _action_prod == 0:
            out_dim = hidden_sizes[-1]
            self._has_output_head = False
        else:
            out_dim = _action_prod
            self._has_output_head = True

        # For preprocess nets (no action head), output_dim = hidden_sizes[-1]
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

        # Initialize gradient monitoring (must be last)
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


# ---------------------------------------------------------------------------
#  GradientMonitoredBaseNet — drop-in replacement for BaseNet
# ---------------------------------------------------------------------------

class GradientMonitoredBaseNet(GradientMonitorMixin, ModuleWithVectorOutput):
    """BaseNet + gradient monitoring.

    Accepts the same (state_shape, action_shape, hidden_sizes, device) as
    BaseNet, plus ``grad_log_interval``, ``grad_verbose`` and optional
    ``grad_monitor_name``.

    For SAC twin-critic, also accepts ``concat=True``.
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

        # Initialize gradient monitoring (must be last)
        self._init_grad_monitor(grad_log_interval, grad_verbose, grad_monitor_name)

    def forward(self, obs, state=None, info=None):
        x = obs.obs if isinstance(obs, (dict, Batch)) and "obs" in obs else obs
        device = torch_device(self)
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        x = x.flatten(1)
        return self.model(x), state


# ---------------------------------------------------------------------------
#  Recurrent variants
# ---------------------------------------------------------------------------

class GradientMonitoredRecurrentBaseNet(GradientMonitorMixin, ModuleWithVectorOutput):
    """RecurrentBaseNet + gradient monitoring.

    Drop-in replacement for RecurrentBaseNet with gradient stats logging.
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

        # Initialize gradient monitoring (must be last)
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
    """MaskedRecurrentNet + gradient monitoring.

    Drop-in replacement for MaskedRecurrentNet with gradient stats logging.
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

        # Initialize gradient monitoring (must be last)
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
