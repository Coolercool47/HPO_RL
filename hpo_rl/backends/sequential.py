"""Multi-backend wrapper for domain randomization in HPO RL training.

:class:`SequentialBackend` holds a list of child backends and switches
the active one on each episode reset, improving the RL agent's
generalization across different optimization landscapes.
"""

import random
from typing import Any, Dict, List

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
    """Wrapper that switches between child backends for domain randomization.

    Each env.reset() triggers a switch to a (possibly different) backend,
    ensuring mixed training batches across multiple optimization landscapes.

    Switching modes:
        - ``"random"``     — uniform random choice (default).
        - ``"sequential"`` — strict round-robin.
        - ``"shuffle"``    — random permutation; each backend seen once per round.

    Args:
        backends: Child :class:`EvaluationBackend` instances.
        mode: ``"random"`` | ``"sequential"`` | ``"shuffle"``.
    """

    VALID_MODES = ("random", "sequential", "shuffle")

    def __init__(
        self,
        backends: List[EvaluationBackend],
        mode: str = "random",
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

        # Shuffle state
        self._shuffle_order: List[int] = []
        self._shuffle_pos: int = 0

        # Lock state — when True, next_backend() is a no-op.
        # Used during inference to keep a specific backend active.
        self._locked: bool = False

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
        """Switch to the next backend (called by env.reset() for domain randomization).

        No-op if backend is locked via set_active_backend().
        """
        if self._locked:
            return

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

    def set_active_backend(self, idx: int, lock: bool = True) -> None:
        """Manually set active backend by index (used for per-child inference).

        Args:
            idx: Index of the backend to activate.
            lock: If True (default), locks the backend so next_backend() is a no-op.
                  Call unlock() to re-enable automatic switching.
        """
        if not 0 <= idx < len(self.backends):
            raise IndexError(
                f"Backend index {idx} out of range [0, {len(self.backends)})"
            )
        self._current_idx = idx
        self._locked = lock
        self._sync_attributes()

    def unlock(self) -> None:
        """Re-enable automatic backend switching after set_active_backend()."""
        self._locked = False

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate(self, config: Dict[str, Any]) -> float:
        """Delegate evaluation to current backend (no remapping needed)."""
        return self.current_backend.evaluate(config)

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
