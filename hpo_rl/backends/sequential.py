"""Multi-backend wrapper for training on multiple optimization tasks.

:class:`SequentialBackend` holds a list of child backends and switches
the active one between training epochs, improving the RL agent's
generalization across different optimization landscapes.
"""

import random
from typing import Any, Dict, List, Optional

from hpo_rl.backends.base import EvaluationBackend

_PROXIED_ATTRS = ("bounds", "dimensions", "global_optimum",
                  "global_optimum_value", "function_name")


def _backend_name(backend: EvaluationBackend) -> str:
    """Return a human-readable name for a backend."""
    if hasattr(backend, "function_name"):
        return backend.function_name
    if hasattr(backend, "objective_function"):
        fn = backend.objective_function
        return getattr(fn, "__name__", None) or type(backend).__name__
    return type(backend).__name__


class SequentialBackend(EvaluationBackend):
    """Wrapper that switches between child backends each training epoch.

    Switching modes:
        - ``"random"``     — uniform random choice (default).
        - ``"sequential"`` — strict round-robin.
        - ``"shuffle"``    — random permutation; each backend seen once per round.

    The controller calls :meth:`next_backend` once per epoch via
    ``periodic_train_hook``.

    Args:
        backends: Child :class:`EvaluationBackend` instances.
        mode: ``"random"`` | ``"sequential"`` | ``"shuffle"``.
        merged_bounds: Union of all child bounds ``{"x0": (lo, hi), ...}``
            for linear remapping.  Not needed when the env sets
            ``skip_remap = True`` via ``sync_bounds_to_backend``.
    """

    VALID_MODES = ("random", "sequential", "shuffle")

    def __init__(
        self,
        backends: List[EvaluationBackend],
        mode: str = "random",
        merged_bounds: Optional[Dict[str, tuple]] = None,
    ) -> None:
        if not backends:
            raise ValueError("backends list must not be empty")

        maximize_values = {b.maximize for b in backends}
        if len(maximize_values) > 1:
            raise ValueError(
                f"All backends must share the same optimization direction. "
                f"Got maximize={[b.maximize for b in backends]}"
            )
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")

        super().__init__(use_cache=False)

        self.backends = backends
        self.mode = mode
        self._switch_count = 0
        self._merged_bounds = merged_bounds
        self.skip_remap = False

        # Per-child bounds dicts for remapping (merged→child)
        self._child_bounds: List[Optional[Dict[str, tuple]]] = []
        for b in backends:
            if hasattr(b, "bounds") and hasattr(b, "dimensions"):
                self._child_bounds.append(
                    {f"x{i}": b.bounds for i in range(b.dimensions)}
                )
            else:
                self._child_bounds.append(None)

        # Shuffle state
        self._shuffle_order: List[int] = []
        self._shuffle_pos: int = 0

        # Init backend is only used for Tianshou's initial test step
        # (before epoch 1). Training starts from the first next_backend()
        # call, so we do NOT consume a shuffle slot here.
        self._current_idx = 0
        if mode == "shuffle":
            self._reshuffle()

        self._sync_attributes()

        names = [_backend_name(b) for b in backends]
        print(f"SequentialBackend: {len(backends)} backends ({', '.join(names)}), mode={mode}")

    # ------------------------------------------------------------------
    # Backend switching
    # ------------------------------------------------------------------

    @property
    def current_backend(self) -> EvaluationBackend:
        """Currently active child backend."""
        return self.backends[self._current_idx]

    def next_backend(self) -> None:
        """Switch to the next backend (called by controller once per epoch)."""
        self._switch_count += 1

        if self.mode == "sequential":
            self._current_idx = (self._switch_count - 1) % len(self.backends)
        elif self.mode == "shuffle":
            if self._shuffle_pos >= len(self._shuffle_order):
                self._reshuffle()
            self._current_idx = self._shuffle_order[self._shuffle_pos]
            self._shuffle_pos += 1
        else:  # random
            self._current_idx = random.randint(0, len(self.backends) - 1)

        self._sync_attributes()
        print(f"[SequentialBackend] Epoch {self._switch_count}: switched to '{self.name}'")

    def set_active_backend(self, idx: int) -> None:
        """Manually set active backend by index (used for per-child inference)."""
        if not 0 <= idx < len(self.backends):
            raise IndexError(
                f"Backend index {idx} out of range [0, {len(self.backends)})"
            )
        self._current_idx = idx
        self._sync_attributes()
        print(f"[SequentialBackend] Manually switched to '{self.name}' (idx={idx})")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, config: Dict[str, Any]) -> float:
        if self.skip_remap:
            return self.current_backend.evaluate(config)
        remapped = self._remap_config(config)
        return self.current_backend.evaluate(remapped)

    # ------------------------------------------------------------------
    # Cache delegation
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        for b in self.backends:
            b.clear_cache()

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total_hits = sum(b.cache_stats["hits"] for b in self.backends)
        total_misses = sum(b.cache_stats["misses"] for b in self.backends)
        total = total_hits + total_misses
        return {
            "hits": total_hits,
            "misses": total_misses,
            "size": sum(b.cache_stats["size"] for b in self.backends),
            "hit_rate": total_hits / total if total > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return _backend_name(self.current_backend)

    def __repr__(self) -> str:
        names = [_backend_name(b) for b in self.backends]
        return (
            f"SequentialBackend(backends=[{', '.join(names)}], "
            f"mode={self.mode!r}, current={names[self._current_idx]})"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sync_attributes(self) -> None:
        """Proxy key attributes from the current child backend.

        Clears stale attributes that don't exist on the new child
        (e.g. ``function_name`` when switching from a benchmark to
        a custom objective).
        """
        cb = self.current_backend
        self.maximize = cb.maximize
        for attr in _PROXIED_ATTRS:
            if hasattr(cb, attr):
                setattr(self, attr, getattr(cb, attr))
            elif hasattr(self, attr):
                delattr(self, attr)

    def _reshuffle(self) -> None:
        """Generate a new random permutation of backend indices.

        Prevents the first element from matching ``_current_idx`` so that
        the same backend never appears in two consecutive epochs at the
        boundary between shuffle rounds.
        """
        order = list(range(len(self.backends)))
        random.shuffle(order)
        if len(order) > 1 and order[0] == self._current_idx:
            swap_idx = random.randint(1, len(order) - 1)
            order[0], order[swap_idx] = order[swap_idx], order[0]
        self._shuffle_order = order
        self._shuffle_pos = 0
        names = [_backend_name(self.backends[i]) for i in order]
        print(f"[SequentialBackend] New shuffle order: {', '.join(names)}")

    def _remap_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Linearly remap values from merged bounds to current child bounds.

        ``merged [lo_m, hi_m] → child [lo_c, hi_c]`` per key.
        Returns config unchanged if merged or child bounds are absent.
        """
        child_bounds = self._child_bounds[self._current_idx]
        if self._merged_bounds is None or child_bounds is None:
            return config

        remapped: Dict[str, Any] = {}
        for key, val in config.items():
            if key in self._merged_bounds and key in child_bounds:
                m_lo, m_hi = self._merged_bounds[key]
                c_lo, c_hi = child_bounds[key]
                if m_hi - m_lo > 1e-12 and (m_lo != c_lo or m_hi != c_hi):
                    t = (val - m_lo) / (m_hi - m_lo)
                    remapped[key] = c_lo + t * (c_hi - c_lo)
                else:
                    remapped[key] = val
            else:
                remapped[key] = val
        return remapped
