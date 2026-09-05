"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy (legacy global) and torch (if installed).

    The FMP class and Optuna samplers carry their own `seed`; this only pins the
    legacy global generators that third-party code (e.g. the synthetic backend's
    noise instance seed) may still consume.
    """
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:  # torch optional
        pass


def derive_seed(base_seed: int, *parts) -> int:
    """Stable sub-seed from a base seed and arbitrary hashable parts (e.g. method, task)."""
    h = np.uint64(1469598103934665603)
    for p in (base_seed, *parts):
        for ch in str(p).encode():
            h = np.uint64((int(h) ^ ch) * 1099511628211 % (2**64))
    return int(h % np.uint64(2**31 - 1))
