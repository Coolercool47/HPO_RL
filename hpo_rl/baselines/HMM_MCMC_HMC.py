"""H-MCMC-HMC: self-contained NUTS + GP surrogate + 3-state Gaussian HMM.

Samples a Boltzmann target pi(x) ~ exp(a_state(x) / T_state) over the unit
hypercube, where a_state is a GP-based acquisition (mu + kappa * sigma).
A compact Gaussian-emission HMM (EXPLOIT / EXPLORE / ESCAPE) modulates
kappa and temperature from recent improvement observations.

No imports from other hpo_rl.baselines.HMM_MCMC* modules.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
import warnings
from enum import IntEnum

import numpy as np
from tqdm.auto import tqdm

try:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.settings import cholesky_jitter, fast_pred_var

    _HAS_GP = True
except ImportError as _gp_import_err:  # pragma: no cover
    _HAS_GP = False
    _GP_IMPORT_ERR = _gp_import_err


def _require_gp() -> None:
    if not _HAS_GP:
        raise ImportError(
            "HMM_MCMC_HMC requires gpytorch and botorch. "
            "Install with: pip install gpytorch botorch"
        ) from _GP_IMPORT_ERR


def _logsumexp(a: np.ndarray) -> float:
    arr = np.asarray(a, dtype=np.float64)
    a_max = arr.max()
    if a_max == -np.inf:
        return -np.inf
    return float(a_max + np.log(np.sum(np.exp(arr - a_max))))


# ---------------------------------------------------------------------------
# HMM states (standalone)
# ---------------------------------------------------------------------------


class HMCState(IntEnum):
    EXPLOIT = 0
    EXPLORE = 1
    ESCAPE = 2


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------


def _float_bounds(info: dict) -> tuple[float, float, float, float, bool]:
    real_lo, real_hi = float(info["values"][0]), float(info["values"][1])
    if info.get("log"):
        return float(np.log10(real_lo)), float(np.log10(real_hi)), real_lo, real_hi, True
    return real_lo, real_hi, real_lo, real_hi, False


def _int_bounds(info: dict) -> tuple[float, float, int, int, bool]:
    real_lo, real_hi = int(info["values"][0]), int(info["values"][1])
    if info.get("log"):
        lo = float(np.log10(max(real_lo, 1)))
        hi = float(np.log10(real_hi))
        return lo, hi, real_lo, real_hi, True
    return float(real_lo), float(real_hi), real_lo, real_hi, False


def _reflect_scalar_unit(x: float) -> float:
    lo, hi = 0.0, 1.0
    for _ in range(32):
        if lo <= x <= hi:
            return float(x)
        if x < lo:
            x = lo + (lo - x)
        else:
            x = hi - (x - hi)
    return float(np.clip(x, lo, hi))


class _ParamSpace:
    """Parse dict_to_optimize and map configs to/from unit-cube continuous coords."""

    def __init__(self, dict_to_optimize: dict):
        self.dict_to_optimize = dict_to_optimize
        self.param_names = list(dict_to_optimize.keys())
        self._param_info: list[dict] = []
        self._continuous_indices: list[int] = []
        self._categorical_indices: list[int] = []
        self._los_cont: np.ndarray | None = None
        self._ranges_cont: np.ndarray | None = None

        for i, name in enumerate(self.param_names):
            info = dict_to_optimize[name]
            p_type = info["type"]
            values = info["values"]
            rec: dict = {"name": name, "type": p_type, "values": values}
            if p_type == "float":
                lo, hi, real_lo, real_hi, is_log = _float_bounds(info)
                rec.update(log=is_log, lo=lo, hi=hi, real_lo=real_lo, real_hi=real_hi)
                self._continuous_indices.append(i)
            elif p_type == "int":
                lo, hi, real_lo, real_hi, is_log = _int_bounds(info)
                rec.update(log=is_log, lo=lo, hi=hi, real_lo=real_lo, real_hi=real_hi)
                self._continuous_indices.append(i)
            elif p_type == "categorical":
                rec["choices"] = list(values)
                rec["n_categories"] = len(values)
                self._categorical_indices.append(i)
            self._param_info.append(rec)

        if self._continuous_indices:
            pis = [self._param_info[i] for i in self._continuous_indices]
            los = np.array([pi["lo"] for pi in pis], dtype=np.float64)
            his = np.array([pi["hi"] for pi in pis], dtype=np.float64)
            self._los_cont = los
            self._ranges_cont = his - los

    @property
    def continuous_dim(self) -> int:
        return len(self._continuous_indices)

    @property
    def has_categorical(self) -> bool:
        return len(self._categorical_indices) > 0

    def config_to_vector(self, cfg: dict) -> np.ndarray:
        return np.array(
            [float(cfg[self._param_info[i]["name"]]) for i in self._continuous_indices],
            dtype=np.float64,
        )

    def _reflect_scalar(self, x: float, lo: float, hi: float) -> float:
        width = hi - lo
        if width <= 0:
            return lo
        for _ in range(32):
            if lo <= x <= hi:
                return float(x)
            if x < lo:
                x = lo + (lo - x)
            else:
                x = hi - (x - hi)
        return float(np.clip(x, lo, hi))

    def vector_to_config(self, vec: np.ndarray, base_cfg: dict) -> dict:
        out = copy.deepcopy(base_cfg)
        for j, idx in enumerate(self._continuous_indices):
            pi = self._param_info[idx]
            name = pi["name"]
            lo, hi = pi["lo"], pi["hi"]
            val = self._reflect_scalar(float(vec[j]), lo, hi)
            if pi["type"] == "int" and not pi.get("log"):
                val = float(int(np.clip(round(val), int(lo), int(hi))))
            out[name] = val
        return out

    def to_unit(self, vec: np.ndarray) -> np.ndarray:
        if self._los_cont is None or self._ranges_cont is None:
            return np.asarray(vec, dtype=np.float64)
        return (np.asarray(vec, dtype=np.float64) - self._los_cont) / np.maximum(
            self._ranges_cont, 1e-12
        )

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        if self._los_cont is None or self._ranges_cont is None:
            return np.asarray(u, dtype=np.float64)
        return self._los_cont + np.asarray(u, dtype=np.float64) * self._ranges_cont

    def reflect_unit(self, u: np.ndarray) -> np.ndarray:
        out = np.asarray(u, dtype=np.float64).copy()
        for j in range(out.size):
            out[j] = _reflect_scalar_unit(float(out[j]))
        return out

    def decode_config(self, cfg: dict) -> dict:
        out = dict(cfg)
        for pi in self._param_info:
            name = pi["name"]
            if pi["type"] == "float" and pi.get("log"):
                out[name] = float(10.0 ** float(cfg[name]))
            elif pi["type"] == "int" and pi.get("log"):
                val = int(np.round(10.0 ** float(cfg[name])))
                out[name] = int(np.clip(val, pi["real_lo"], pi["real_hi"]))
            elif pi["type"] == "int":
                out[name] = int(round(float(cfg[name])))
        return out

    def sobol_configs(self, n: int, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        configs: list[dict] = []
        d = self.continuous_dim
        if d == 0:
            for _ in range(n):
                cfg = {}
                for idx in self._categorical_indices:
                    pi = self._param_info[idx]
                    cfg[pi["name"]] = rng.choice(pi["choices"])
                configs.append(cfg)
            return configs

        try:
            from scipy.stats.qmc import Sobol

            engine = Sobol(d=d, scramble=True, seed=seed)
            samples = engine.random(n)
        except Exception:
            samples = rng.random((n, d))

        for row in samples:
            u = np.asarray(row, dtype=np.float64)
            vec = self.from_unit(u)
            cfg = self.vector_to_config(vec, {})
            for idx in self._categorical_indices:
                pi = self._param_info[idx]
                cfg[pi["name"]] = rng.choice(pi["choices"])
            configs.append(cfg)
        return configs

    def random_categorical_step(self, cfg: dict, rng: np.random.Generator) -> dict:
        out = copy.deepcopy(cfg)
        for idx in self._categorical_indices:
            pi = self._param_info[idx]
            out[pi["name"]] = rng.choice(pi["choices"])
        return out

    @staticmethod
    def build_training_set(
        data: list[tuple[dict, float]],
        space: "_ParamSpace",
        max_obs: int,
        subsample: bool = True,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if space.continuous_dim == 0 or len(data) < 2:
            return None
        n = len(data)
        k = min(max_obs, n)
        if not subsample:
            indices = list(range(n - k, n)) if n > k else list(range(n))
        elif n <= k:
            indices = list(range(n))
        else:
            half = k // 2
            by_loss = sorted(range(n), key=lambda i: data[i][1])
            best_idx = by_loss[:half]
            recent_idx = list(range(n - (k - half), n))
            indices = []
            seen: set[int] = set()
            for idx in best_idx + recent_idx:
                if idx not in seen:
                    seen.add(idx)
                    indices.append(idx)
            if len(indices) < k:
                for idx in reversed(range(n)):
                    if idx not in seen:
                        seen.add(idx)
                        indices.append(idx)
                    if len(indices) >= k:
                        break
        rows = [space.to_unit(space.config_to_vector(data[i][0])) for i in indices]
        ys = [float(data[i][1]) for i in indices]
        X = np.vstack(rows)
        y = np.asarray(ys, dtype=np.float64)
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < 2:
            return None
        return X[mask], y[mask]


# ---------------------------------------------------------------------------
# GP surrogate
# ---------------------------------------------------------------------------


@dataclass
class _GPConfig:
    min_obs: int = 12
    max_obs: int = 256
    refit_every: int = 1
    jitter: float = 1e-3
    cholesky_jitter: float = 1e-3
    noise_floor: float = 1e-6


class _GPSurrogate:
    def __init__(self, dim: int, config: _GPConfig | None = None):
        _require_gp()
        self.dim = int(dim)
        self.config = config or _GPConfig()
        self.model: SingleTaskGP | None = None
        self._best_f: float | None = None
        self._dtype = torch.double
        self._device = torch.device("cpu")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def fit(self, X_unit: np.ndarray, y_loss: np.ndarray) -> None:
        X = np.asarray(X_unit, dtype=np.float64)
        y_arr = np.asarray(y_loss, dtype=np.float64).reshape(-1, 1)
        if X.shape[0] < 2 or X.shape[1] != self.dim:
            return
        mask = np.isfinite(y_arr.ravel()) & np.all(np.isfinite(X), axis=1)
        X = X[mask]
        y_arr = y_arr[mask]
        if X.shape[0] < 2:
            return
        if float(np.std(y_arr)) < 1e-12:
            y_arr = y_arr + self._rng_jitter(len(y_arr), scale=self.config.jitter)
        train_X = torch.as_tensor(X, dtype=self._dtype, device=self._device)
        train_Y = torch.as_tensor(-y_arr, dtype=self._dtype, device=self._device)
        gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        with cholesky_jitter(self.config.cholesky_jitter):
            fit_gpytorch_mll(mll)
        with torch.no_grad():
            noise = gp.likelihood.noise
            min_noise = self.config.noise_floor
            if noise.ndim == 0:
                gp.likelihood.noise.fill_(max(float(noise.item()), min_noise))
            else:
                gp.likelihood.noise.copy_(noise.clamp_min(min_noise))
        gp.eval()
        self.model = gp
        self._best_f = float(train_Y.max().item())

    def _posterior(self, x: torch.Tensor):
        with cholesky_jitter(self.config.cholesky_jitter), fast_pred_var(True):
            return self.model.posterior(x)

    @staticmethod
    def _rng_jitter(n: int, scale: float = 1e-6) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.normal(0.0, scale, size=n).reshape(-1, 1)

    def acq_value_and_grad(self, x_unit: np.ndarray, kappa: float) -> tuple[float, np.ndarray]:
        if self.model is None:
            raise RuntimeError("GP not fitted")
        x = torch.tensor(
            x_unit, dtype=self._dtype, device=self._device, requires_grad=True
        )
        post = self._posterior(x.reshape(1, -1))
        mu = post.mean.reshape(())
        sigma = post.variance.clamp_min(self.config.jitter).sqrt().reshape(())
        acq = mu + float(kappa) * sigma
        acq.backward()
        grad = x.grad.detach().cpu().numpy().astype(np.float64)
        acq_val = float(acq.detach().cpu().item())
        if not np.isfinite(acq_val):
            acq_val = 0.0
        grad = np.where(np.isfinite(grad), grad, 0.0)
        return acq_val, grad

    def posterior_mean_std(self, x_unit: np.ndarray) -> tuple[float, float]:
        if self.model is None:
            raise RuntimeError("GP not fitted")
        x = torch.as_tensor(
            np.asarray(x_unit, dtype=np.float64).reshape(1, -1),
            dtype=self._dtype,
            device=self._device,
        )
        with torch.no_grad():
            post = self._posterior(x)
            mu = float(post.mean.reshape(()).item())
            sigma = float(post.variance.clamp_min(self.config.jitter).sqrt().reshape(()).item())
        return mu, sigma

    def ard_scales(self) -> np.ndarray | None:
        if self.model is None:
            return None
        try:
            ls = self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
            ls = np.maximum(ls, 1e-6)
            return (1.0 / ls).astype(np.float64)
        except Exception:
            return None

    def argmax_point(
        self,
        bounds: np.ndarray | None = None,
        greedy: bool = False,
        use_greedy_pm: bool = False,
        n_restarts: int = 5,
        raw_samples: int = 64,
    ) -> np.ndarray:
        if self.model is None or self._best_f is None:
            raise RuntimeError("GP not fitted")
        from botorch.acquisition.analytic import LogExpectedImprovement, PosteriorMean
        from botorch.optim import optimize_acqf

        if bounds is None:
            lo = np.zeros(self.dim, dtype=np.float64)
            hi = np.ones(self.dim, dtype=np.float64)
        else:
            b = np.asarray(bounds, dtype=np.float64)
            lo = np.clip(b[0], 0.0, 1.0)
            hi = np.clip(b[1], 0.0, 1.0)
            hi = np.maximum(hi, lo + 1e-12)

        bounds_t = torch.stack([
            torch.as_tensor(lo, dtype=self._dtype, device=self._device),
            torch.as_tensor(hi, dtype=self._dtype, device=self._device),
        ])
        use_pm = bool(greedy and use_greedy_pm)
        acqf = PosteriorMean(model=self.model) if use_pm else LogExpectedImprovement(
            model=self.model, best_f=self._best_f
        )
        with cholesky_jitter(self.config.cholesky_jitter), fast_pred_var(True):
            candidate, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds_t,
                q=1,
                num_restarts=max(1, n_restarts),
                raw_samples=max(64, raw_samples),
            )
        return candidate.detach().cpu().numpy().reshape(-1)

    def explore_point(
        self,
        bounds: np.ndarray | None = None,
        beta: float = 4.0,
        n_restarts: int = 5,
        raw_samples: int = 64,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GP not fitted")
        from botorch.acquisition.analytic import UpperConfidenceBound
        from botorch.optim import optimize_acqf

        if bounds is None:
            lo = np.zeros(self.dim, dtype=np.float64)
            hi = np.ones(self.dim, dtype=np.float64)
        else:
            b = np.asarray(bounds, dtype=np.float64)
            lo = np.clip(b[0], 0.0, 1.0)
            hi = np.clip(b[1], 0.0, 1.0)
            hi = np.maximum(hi, lo + 1e-12)

        bounds_t = torch.stack([
            torch.as_tensor(lo, dtype=self._dtype, device=self._device),
            torch.as_tensor(hi, dtype=self._dtype, device=self._device),
        ])
        acqf = UpperConfidenceBound(model=self.model, beta=float(beta))
        with cholesky_jitter(self.config.cholesky_jitter), fast_pred_var(True):
            candidate, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds_t,
                q=1,
                num_restarts=max(1, n_restarts),
                raw_samples=max(64, raw_samples),
            )
        return candidate.detach().cpu().numpy().reshape(-1)

    def log_ei_at(self, u_unit: np.ndarray) -> float:
        if self.model is None or self._best_f is None:
            raise RuntimeError("GP not fitted")
        from botorch.acquisition.analytic import LogExpectedImprovement

        u = torch.as_tensor(
            np.asarray(u_unit, dtype=np.float64).reshape(1, -1),
            dtype=self._dtype,
            device=self._device,
        )
        acqf = LogExpectedImprovement(model=self.model, best_f=self._best_f)
        with cholesky_jitter(self.config.cholesky_jitter), fast_pred_var(True):
            val = acqf(u)
        return float(val.detach().cpu().item())


# ---------------------------------------------------------------------------
# Gaussian-emission HMM
# ---------------------------------------------------------------------------


class _GaussianHMM:
    """3-state HMM with Gaussian emissions over scalar improvement obs."""

    N_STATES = 3

    def __init__(
        self,
        window: int = 8,
        refit_every: int = 5,
        min_obs: int = 12,
        n_em_iters: int = 3,
        prior_strength: float = 25.0,
        switch_confirm: int = 2,
    ):
        self.window = int(window)
        self.refit_every = int(refit_every)
        self.min_obs = int(min_obs)
        self.n_em_iters = int(n_em_iters)
        self.prior_strength = float(prior_strength)
        self.switch_confirm = max(1, int(switch_confirm))
        self.state = HMCState.EXPLOIT
        self._buffer: list[float] = []
        self._step_count = 0
        self._pending_non_exploit: HMCState | None = None
        self._pending_count = 0
        self._init_params()

    def _init_params(self) -> None:
        # obs = (loss - best_loss)/scale: negative = beat best, 0 = flat, positive = worse than best
        self.pi0 = np.array([0.6, 0.3, 0.1], dtype=np.float64)
        self.A = np.array([
            [0.85, 0.10, 0.05],
            [0.20, 0.55, 0.25],
            [0.10, 0.30, 0.60],
        ], dtype=np.float64)
        self.mu = np.array([-0.05, 0.15, 0.45], dtype=np.float64)
        self.var = np.array([0.08, 0.15, 0.20], dtype=np.float64)

    def reset(self) -> None:
        self.state = HMCState.EXPLOIT
        self._buffer = []
        self._step_count = 0
        self._pending_non_exploit = None
        self._pending_count = 0
        self._init_params()

    def force_state(self, new_state: HMCState) -> None:
        self.state = new_state
        self._pending_non_exploit = None
        self._pending_count = 0

    def observe(self, obs_history: list[float]) -> HMCState:
        if obs_history:
            self._buffer.extend(obs_history[-1:])
            self._step_count += 1
            self._maybe_refit()
        if len(obs_history) < 2:
            return self.state
        seq = obs_history[-self.window:]
        candidate = HMCState(self._viterbi(seq))
        if self.state == HMCState.EXPLOIT and candidate != HMCState.EXPLOIT:
            if candidate == self._pending_non_exploit:
                self._pending_count += 1
            else:
                self._pending_non_exploit = candidate
                self._pending_count = 1
            if self._pending_count >= self.switch_confirm:
                self.state = candidate
                self._pending_non_exploit = None
                self._pending_count = 0
        else:
            self.state = candidate
            self._pending_non_exploit = None
            self._pending_count = 0
        return self.state

    def _log_emission(self, o: float) -> np.ndarray:
        var = np.maximum(self.var, 1e-6)
        return -0.5 * ((o - self.mu) ** 2 / var + np.log(2.0 * np.pi * var))

    def _viterbi(self, seq: list[float]) -> int:
        T = len(seq)
        log_A = np.log(np.maximum(self.A, 1e-12))
        log_pi0 = np.log(np.maximum(self.pi0, 1e-12))
        log_B = np.array([self._log_emission(o) for o in seq])
        delta = log_pi0 + log_B[0]
        psi = np.zeros((T, self.N_STATES), dtype=np.int64)
        for t in range(1, T):
            trans = delta[:, None] + log_A
            psi[t] = np.argmax(trans, axis=0)
            delta = np.max(trans, axis=0) + log_B[t]
        path = np.zeros(T, dtype=np.int64)
        path[-1] = int(np.argmax(delta))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        return int(path[-1])

    def _maybe_refit(self) -> None:
        if len(self._buffer) < self.min_obs:
            return
        if self._step_count % self.refit_every != 0:
            return
        obs = np.asarray(self._buffer[-256:], dtype=np.float64)
        self._baum_welch(obs)

    def _baum_welch(self, obs: np.ndarray) -> None:
        T = len(obs)
        log_B = np.array([self._log_emission(o) for o in obs])
        log_A = np.log(np.maximum(self.A, 1e-12))
        log_pi0 = np.log(np.maximum(self.pi0, 1e-12))

        for _ in range(self.n_em_iters):
            alpha = np.zeros((T, self.N_STATES))
            beta = np.zeros((T, self.N_STATES))
            alpha[0] = np.exp(log_pi0 + log_B[0])
            alpha[0] /= alpha[0].sum() + 1e-12
            for t in range(1, T):
                alpha[t] = alpha[t - 1] @ np.exp(log_A) * np.exp(log_B[t])
                alpha[t] /= alpha[t].sum() + 1e-12
            beta[-1] = 1.0
            for t in range(T - 2, -1, -1):
                beta[t] = np.exp(log_A) @ (beta[t + 1] * np.exp(log_B[t + 1]))
                beta[t] /= beta[t].sum() + 1e-12

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-12

            xi = np.zeros((T - 1, self.N_STATES, self.N_STATES))
            for t in range(T - 1):
                numer = (
                    alpha[t][:, None]
                    * np.exp(log_A)
                    * (np.exp(log_B[t + 1]) * beta[t + 1])[None, :]
                )
                xi[t] = numer / (numer.sum() + 1e-12)

            ps = self.prior_strength
            pi0_new = gamma[0] + ps * self.pi0
            pi0_new /= pi0_new.sum()
            A_new = xi.sum(axis=0) + ps * self.A
            A_new /= A_new.sum(axis=1, keepdims=True)

            mu_new = (gamma * obs[:, None]).sum(axis=0) / (gamma.sum(axis=0) + 1e-12)
            var_new = (gamma * (obs[:, None] - mu_new[None, :]) ** 2).sum(axis=0) / (
                gamma.sum(axis=0) + 1e-12
            )

            # Ordering prior: EXPLOIT < EXPLORE < ESCAPE
            order = np.argsort(mu_new)
            mu_ord = np.sort(mu_new)
            mu_ord[0] = min(mu_ord[0], -0.05)
            mu_ord[2] = max(mu_ord[2], 0.05)
            mu_new = mu_ord
            var_new = np.maximum(var_new[order][np.argsort(order)], 1e-4)

            self.pi0 = pi0_new
            self.A = A_new
            self.mu = mu_new
            self.var = var_new


# ---------------------------------------------------------------------------
# NUTS sampler
# ---------------------------------------------------------------------------


class _DualAveraging:
    def __init__(self, step_size_init: float, target_accept: float = 0.8):
        self.log_step = math.log(max(step_size_init, 1e-12))
        self.target_accept = target_accept
        self.h_bar = 0.0
        self.t = 0
        self.step_size = float(step_size_init)
        self._frozen = False

    def freeze(self) -> None:
        self._frozen = True

    def update(self, accept_prob: float) -> None:
        if self._frozen:
            return
        accept_prob = float(accept_prob)
        if not math.isfinite(accept_prob):
            accept_prob = 0.0
        self.t += 1
        eta = min(0.1, 1.0 / (self.t + 10.0))
        self.h_bar = (1.0 - eta) * self.h_bar + eta * (self.target_accept - accept_prob)
        w = 1.0 / (self.t + 1.0)
        self.log_step = w * self.log_step + (1.0 - w) * (
            self.log_step + eta * (accept_prob - self.target_accept)
        )
        self.step_size = float(math.exp(self.log_step))
        if not math.isfinite(self.step_size) or self.step_size <= 0.0:
            self.step_size = 1e-6
            self.log_step = math.log(self.step_size)


def _reflect_leapfrog(q: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q = q.copy()
    p = p.copy()
    for i in range(q.size):
        if q[i] < 0.0:
            q[i] = -q[i]
            p[i] = -p[i]
        elif q[i] > 1.0:
            q[i] = 2.0 - q[i]
            p[i] = -p[i]
    return q, p


class _NUTS:
    delta_max: float = 1000.0

    def __init__(
        self,
        gp: _GPSurrogate,
        space: _ParamSpace,
        step_size_init: float = 0.1,
        target_accept: float = 0.8,
        max_tree_depth: int = 10,
        rng: np.random.Generator | None = None,
    ):
        self.gp = gp
        self.space = space
        self.step_size = float(step_size_init)
        self.target_accept = float(target_accept)
        self.max_tree_depth = int(max_tree_depth)
        self.rng = rng or np.random.default_rng()
        self._adapt = _DualAveraging(step_size_init, target_accept)
        self._mass = np.ones(space.continuous_dim, dtype=np.float64)

    def set_mass_from_ard(self, ard: np.ndarray | None) -> None:
        d = self.space.continuous_dim
        if ard is None or ard.size != d:
            self._mass = np.ones(d, dtype=np.float64)
            return
        m = np.maximum(ard, 1e-6)
        m = m / np.mean(m)
        self._mass = m.astype(np.float64)

    def freeze_adaptation(self) -> None:
        self._adapt.freeze()
        self.step_size = self._adapt.step_size

    def _hamiltonian(self, q: np.ndarray, p: np.ndarray, kappa: float, T: float) -> float:
        log_pi, _ = self.gp.acq_value_and_grad(q, kappa)
        if not np.isfinite(log_pi):
            return float("inf")
        kinetic = 0.5 * np.sum((p ** 2) / np.maximum(self._mass, 1e-12))
        if not np.isfinite(kinetic):
            return float("inf")
        H = float(-log_pi / max(T, 1e-12) + kinetic)
        return H if math.isfinite(H) else float("inf")

    @staticmethod
    def _metropolis_alpha(H0: float, H1: float) -> float:
        if not math.isfinite(H0) or not math.isfinite(H1):
            return 0.0
        diff = H0 - H1
        if diff >= 0.0:
            return 1.0
        if diff < -700.0:
            return 0.0
        return float(math.exp(diff))

    def _leapfrog(
        self,
        q0: np.ndarray,
        p0: np.ndarray,
        kappa: float,
        T: float,
        step_size: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = q0.copy()
        p = p0.copy()
        eps = step_size
        inv_mass = 1.0 / np.maximum(self._mass, 1e-12)

        _, grad = self.gp.acq_value_and_grad(q, kappa)
        grad = np.where(np.isfinite(grad), grad, 0.0)
        p = p + 0.5 * eps * inv_mass * (grad / max(T, 1e-12))

        for _ in range(1):
            p_inv = p * inv_mass
            q = q + eps * p_inv
            q, p = _reflect_leapfrog(q, p)
            _, grad = self.gp.acq_value_and_grad(q, kappa)
            grad = np.where(np.isfinite(grad), grad, 0.0)
            p = p + 0.5 * eps * inv_mass * (grad / max(T, 1e-12))

        return self.space.reflect_unit(q), p

    def _build_tree(
        self,
        q: np.ndarray,
        p: np.ndarray,
        log_u: float,
        v: int,
        j: int,
        kappa: float,
        T: float,
        step_size: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool]:
        """Efficient NUTS subtree (Hoffman & Gelman 2014, Alg. 6).

        Returns (q_minus, p_minus, q_plus, p_plus, q_prime, p_prime, n_prime, s_prime).
        """
        if j == 0:
            q1, p1 = self._leapfrog(q, p, kappa, T, float(v) * step_size)
            H1 = self._hamiltonian(q1, p1, kappa, T)
            on_slice = math.isfinite(H1) and log_u <= -H1
            n_prime = 1 if on_slice else 0
            s_prime = bool(math.isfinite(H1) and log_u < self.delta_max - H1)
            return q1, p1, q1, p1, q1, p1, n_prime, s_prime

        q_minus, p_minus, q_plus, p_plus, q_prime, p_prime, n_prime, s_prime = self._build_tree(
            q, p, log_u, v, j - 1, kappa, T, step_size
        )
        if not s_prime:
            return q_minus, p_minus, q_plus, p_plus, q_prime, p_prime, n_prime, s_prime

        if v == -1:
            q_minus, p_minus, _, _, q2, p2, n2, s2 = self._build_tree(
                q_minus, p_minus, log_u, v, j - 1, kappa, T, step_size
            )
        else:
            _, _, q_plus, p_plus, q2, p2, n2, s2 = self._build_tree(
                q_plus, p_plus, log_u, v, j - 1, kappa, T, step_size
            )

        if s2 and n2 > 0 and self.rng.random() < n2 / max(n_prime + n2, 1):
            q_prime, p_prime = q2, p2

        n_prime += n2
        inv_mass = 1.0 / np.maximum(self._mass, 1e-12)
        p_vec = (q_plus - q_minus) * inv_mass
        s_prime = bool(
            s2
            and np.dot(p_vec, p_minus) >= 0.0
            and np.dot(p_vec, p_plus) >= 0.0
        )
        return q_minus, p_minus, q_plus, p_plus, q_prime, p_prime, n_prime, s_prime

    def sample(
        self,
        q0: np.ndarray,
        kappa: float,
        T: float,
        adapt: bool = True,
    ) -> tuple[np.ndarray, bool, int, float]:
        if self.space.continuous_dim == 0:
            return q0, True, 0, self.step_size

        p0 = self.rng.standard_normal(self.space.continuous_dim) * np.sqrt(self._mass)
        H0 = self._hamiltonian(q0, p0, kappa, T)
        if not math.isfinite(H0):
            return self.space.reflect_unit(q0), False, 0, self.step_size
        log_u = math.log(self.rng.random()) - H0

        q_minus = q_plus = qn = q0.copy()
        p_minus = p_plus = pn = p0.copy()
        j = 0
        n = 1
        s = True
        accept = False

        while s and j < self.max_tree_depth:
            v = 1 if self.rng.random() < 0.5 else -1
            if v == -1:
                q_minus, p_minus, _, _, q_prime, p_prime, n1, s1 = self._build_tree(
                    q_minus, p_minus, log_u, v, j, kappa, T, self.step_size
                )
            else:
                _, _, q_plus, p_plus, q_prime, p_prime, n1, s1 = self._build_tree(
                    q_plus, p_plus, log_u, v, j, kappa, T, self.step_size
                )

            if n1 > 0 and self.rng.random() < min(1.0, n1 / max(n, 1)):
                qn, pn = q_prime, p_prime
                accept = True

            n += n1
            s = s1
            j += 1
            if s:
                inv_mass = 1.0 / np.maximum(self._mass, 1e-12)
                p_vec = (q_plus - q_minus) * inv_mass
                s = bool(np.dot(p_vec, p_minus) >= 0.0 and np.dot(p_vec, p_plus) >= 0.0)

        q_out = self.space.reflect_unit(qn)
        if adapt:
            H1 = self._hamiltonian(q_out, pn, kappa, T)
            alpha = self._metropolis_alpha(H0, H1)
            self._adapt.update(alpha)
            self.step_size = self._adapt.step_size

        return q_out, accept, j, self.step_size


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class _RouteDecision:
    kernel: str
    burst_L: int = 0
    escape_deep: bool = False
    greedy_acq: bool = False


class HMM_MCMC_HMC:
    """NUTS over GP acquisition surface with 3-state Gaussian HMM control."""

    def __init__(
        self,
        objective_func,
        budget: int,
        dict_to_optimize: dict,
        n_init: int = 16,
        n_chains: int = 1,
        orchestrate_every: int = 1000,
        T_min: float | None = None,
        rejection_streak: int = 10,
        hmm_window: int = 8,
        hmm_refit_every: int = 5,
        hmm_min_obs: int = 12,
        hmm_n_em_iters: int = 3,
        hmm_prior_strength: float = 25.0,
        gp_min_obs: int = 12,
        gp_max_obs: int = 256,
        gp_refit_every: int = 1,
        gp_noise_floor: float = 1e-6,
        kappa_exploit: float = 1.0,
        kappa_explore: float = 3.0,
        kappa_escape: float = 5.0,
        T_exploit: float = 0.3,
        T_explore: float = 1.0,
        T_escape: float = 2.0,
        step_size_init: float = 0.1,
        target_accept: float = 0.8,
        nuts_max_tree_depth: int = 10,
        bo_warmup_frac: float = 0.25,
        exploit_argmax_every: int = 1,
        gp_argmax_restarts: int = 20,
        gp_argmax_raw_samples: int = 256,
        acq_greedy_after: float = 0.97,
        use_greedy_pm: bool = False,
        argmax_dedup_tol: float = 1e-3,
        explore_beta: float = 4.0,
        gp_subsample: bool = False,
        burst_steps_escape: int = 64,
        burst_steps_stuck: int = 32,
        hmm_switch_confirm: int = 2,
        improve_tol: float = 1e-12,
        p_cat_step: float = 0.0,
        use_hmm: bool = True,
        seed: int | None = None,
        show_progress: bool = True,
        progress_desc: str | None = None,
        verbose_history: bool = True,
        **kwargs,
    ):
        _require_gp()
        if n_chains != 1:
            raise ValueError("HMM_MCMC_HMC currently supports n_chains=1 only")
        if "inner_burst_steps" in kwargs:
            warnings.warn(
                "inner_burst_steps is deprecated; use burst_steps_escape",
                DeprecationWarning,
                stacklevel=2,
            )
            burst_steps_escape = int(kwargs.pop("inner_burst_steps"))
        if "explore_every" in kwargs:
            kwargs.pop("explore_every")
            warnings.warn("explore_every is removed and ignored", DeprecationWarning, stacklevel=2)
        if "warmup_fraction" in kwargs:
            kwargs.pop("warmup_fraction")
            warnings.warn("warmup_fraction is removed; use bo_warmup_frac", DeprecationWarning, stacklevel=2)
        _ = (orchestrate_every, T_min, kwargs)

        self.objective_func = objective_func
        self.budget = int(budget)
        self.dict_to_optimize = dict_to_optimize
        self.n_init = int(n_init)
        self.no_improve_limit = int(rejection_streak)
        self.gp_min_obs = int(gp_min_obs)
        self.gp_max_obs = max(int(gp_max_obs), self.budget)
        self.gp_refit_every = int(gp_refit_every)
        self.gp_noise_floor = float(gp_noise_floor)
        self.kappa_exploit = float(kappa_exploit)
        self.kappa_explore = float(kappa_explore)
        self.kappa_escape = float(kappa_escape)
        self.T_exploit = float(T_exploit)
        self.T_explore = float(T_explore)
        self.T_escape = float(T_escape)
        self.bo_warmup_frac = float(bo_warmup_frac)
        self.exploit_argmax_every = int(exploit_argmax_every)
        self.gp_argmax_restarts = int(gp_argmax_restarts)
        self.gp_argmax_raw_samples = int(gp_argmax_raw_samples)
        self.acq_greedy_after = float(acq_greedy_after)
        self.use_greedy_pm = bool(use_greedy_pm)
        self.argmax_dedup_tol = float(argmax_dedup_tol)
        self.explore_beta = float(explore_beta)
        self.gp_subsample = bool(gp_subsample)
        self.burst_steps_escape = int(burst_steps_escape)
        self.burst_steps_stuck = int(burst_steps_stuck)
        self.improve_tol = float(improve_tol)
        self.p_cat_step = float(p_cat_step)
        self.use_hmm = bool(use_hmm)
        self._seed = seed
        self.step_size_init = float(step_size_init)
        self.target_accept = float(target_accept)
        self.nuts_max_tree_depth = int(nuts_max_tree_depth)
        self.show_progress = bool(show_progress)
        self.progress_desc = progress_desc
        self.verbose_history = bool(verbose_history)

        self._space = _ParamSpace(dict_to_optimize)
        self._gp: _GPSurrogate | None = None
        self._gp_config = _GPConfig(
            min_obs=gp_min_obs,
            max_obs=self.gp_max_obs,
            refit_every=gp_refit_every,
            noise_floor=self.gp_noise_floor,
        )
        self._hmm = _GaussianHMM(
            window=hmm_window,
            refit_every=hmm_refit_every,
            min_obs=hmm_min_obs,
            n_em_iters=hmm_n_em_iters,
            prior_strength=hmm_prior_strength,
            switch_confirm=hmm_switch_confirm,
        )
        self._scale_window = max(16, hmm_window * 2)
        self._nuts: _NUTS | None = None
        self._rng = np.random.default_rng(seed)

        self.data: list[tuple[dict, float]] = []
        self.history_table: list[dict] = []
        self._evals_since_gp_refit = 0
        self._observations: list[float] = []
        self._no_improve_streak = 0
        self._scale_factor = 1.0
        self._current_x: dict = {}
        self._current_loss = float("inf")
        self._best_loss = float("inf")
        self._best_x: dict = {}
        self._exploit_step = 0
        self._nan_penalty = 1e6

    def _finite_data(self) -> list[tuple[dict, float]]:
        return [(c, l) for c, l in self.data if np.isfinite(l)]

    def _sanitize_loss(self, loss: float) -> float:
        v = float(loss)
        if np.isfinite(v):
            self._nan_penalty = max(self._nan_penalty, abs(v) + 1.0)
            return v
        return float(self._nan_penalty)

    def _eval_objective(self, cfg: dict) -> float:
        decoded = self._space.decode_config(cfg)
        return self._sanitize_loss(float(self.objective_func(decoded)))

    def reset(self) -> None:
        self.data = []
        self.history_table = []
        self._gp = None
        self._nuts = None
        self._hmm.reset()
        self._evals_since_gp_refit = 0
        self._observations = []
        self._no_improve_streak = 0
        self._current_x = {}
        self._current_loss = float("inf")
        self._best_loss = float("inf")
        self._best_x = {}
        self._exploit_step = 0
        self._nan_penalty = 1e6

    def _decode_best(self, result: tuple[dict, float]) -> tuple[dict, float]:
        cfg, loss = result
        return self._space.decode_config(cfg), loss

    def _compute_scale(self, losses: list[float]) -> float:
        finite = [float(l) for l in losses if np.isfinite(l)]
        if len(finite) < 2:
            return 1.0
        return float(max(np.std(finite), 1e-8))

    def _state_params(self, state: HMCState) -> tuple[float, float]:
        if state == HMCState.EXPLOIT:
            return self.kappa_exploit, self.T_exploit
        if state == HMCState.EXPLORE:
            return self.kappa_explore, self.T_explore
        return self.kappa_escape, self.T_escape

    def _training_unit_points(self) -> np.ndarray | None:
        finite = self._finite_data()
        train = _ParamSpace.build_training_set(
            finite, self._space, self.gp_max_obs, subsample=self.gp_subsample
        )
        if train is None:
            return None
        X, _ = train
        return np.asarray(X, dtype=np.float64)

    def _min_obs_distance(self, u: np.ndarray, X_obs: np.ndarray | None) -> float:
        if X_obs is None or X_obs.shape[0] == 0:
            return float("inf")
        u = np.asarray(u, dtype=np.float64).reshape(1, -1)
        return float(np.min(np.linalg.norm(X_obs - u, axis=1)))

    def _gp_argmax_point(self, greedy: bool = False) -> np.ndarray:
        if self._gp is None or not self._gp.is_ready:
            raise RuntimeError("GP not ready for argmax")
        X_obs = self._training_unit_points()
        use_greedy = bool(greedy and self.use_greedy_pm)
        u_star = self._gp.argmax_point(
            greedy=use_greedy,
            use_greedy_pm=self.use_greedy_pm,
            n_restarts=self.gp_argmax_restarts,
            raw_samples=self.gp_argmax_raw_samples,
        )
        if self._min_obs_distance(u_star, X_obs) < self.argmax_dedup_tol:
            u_star = self._gp.argmax_point(
                greedy=False,
                use_greedy_pm=self.use_greedy_pm,
                n_restarts=self.gp_argmax_restarts,
                raw_samples=self.gp_argmax_raw_samples,
            )
        if self._min_obs_distance(u_star, X_obs) < self.argmax_dedup_tol:
            u_star = self._gp.argmax_point(
                greedy=False,
                use_greedy_pm=self.use_greedy_pm,
                n_restarts=max(self.gp_argmax_restarts, 20),
                raw_samples=max(self.gp_argmax_raw_samples * 2, 512),
            )
        return u_star.astype(np.float64)

    def _gp_explore_point(self, beta_scale: float = 1.0) -> np.ndarray:
        if self._gp is None or not self._gp.is_ready:
            raise RuntimeError("GP not ready for explore")
        X_obs = self._training_unit_points()
        beta = self.explore_beta * float(beta_scale)
        u_star = self._gp.explore_point(
            beta=beta,
            n_restarts=self.gp_argmax_restarts,
            raw_samples=self.gp_argmax_raw_samples,
        )
        if self._min_obs_distance(u_star, X_obs) < self.argmax_dedup_tol:
            u_star = self._gp.explore_point(
                beta=beta * 2.0,
                n_restarts=self.gp_argmax_restarts,
                raw_samples=self.gp_argmax_raw_samples,
            )
        return u_star.astype(np.float64)

    def _run_nuts_burst(
        self,
        u_start: np.ndarray,
        kappa: float,
        T: float,
        L: int,
        adapt: bool,
    ) -> tuple[np.ndarray, int, int, float, float]:
        if self._gp is None or self._nuts is None:
            raise RuntimeError("GP/NUTS not ready for burst")
        u = np.asarray(u_start, dtype=np.float64).copy()
        log_ei0 = self._gp.log_ei_at(u)
        candidates: list[tuple[np.ndarray, float]] = [(u.copy(), float(log_ei0))]
        max_tree = 0
        step_size = self._nuts.step_size
        for _ in range(int(L)):
            u, _, tree_depth, step_size = self._nuts.sample(
                u, kappa=kappa, T=T, adapt=adapt
            )
            log_ei = self._gp.log_ei_at(u)
            candidates.append((u.copy(), float(log_ei)))
            max_tree = max(max_tree, int(tree_depth))
        candidates.sort(key=lambda item: item[1], reverse=True)
        X_obs = self._training_unit_points()
        u_star, max_log_ei = candidates[0]
        for u_cand, log_ei_cand in candidates:
            if self._min_obs_distance(u_cand, X_obs) >= self.argmax_dedup_tol:
                u_star, max_log_ei = u_cand, log_ei_cand
                break
        return u_star.astype(np.float64), int(L), max_tree, float(step_size), float(max_log_ei)

    def _route_kernel(
        self,
        state: HMCState,
        stuck_restart: bool,
        progress: float,
        bo_warmup_on: bool,
        do_cat: bool,
    ) -> _RouteDecision:
        if do_cat:
            return _RouteDecision(kernel="categorical")

        if self._gp is None or not self._gp.is_ready or self._space.continuous_dim == 0:
            return _RouteDecision(kernel="random")

        greedy_acq = (
            self.use_greedy_pm
            and state == HMCState.EXPLOIT
            and progress >= self.acq_greedy_after
            and not stuck_restart
        )

        if bo_warmup_on or progress >= self.acq_greedy_after:
            return _RouteDecision(kernel="argmax", greedy_acq=greedy_acq)

        if stuck_restart:
            if self.burst_steps_stuck > 0:
                return _RouteDecision(kernel="burst", burst_L=self.burst_steps_stuck)
            return _RouteDecision(kernel="argmax", greedy_acq=greedy_acq)

        if state == HMCState.EXPLOIT:
            return _RouteDecision(kernel="argmax", greedy_acq=greedy_acq)

        if state == HMCState.EXPLORE:
            return _RouteDecision(kernel="explore")

        if state == HMCState.ESCAPE:
            if self.burst_steps_escape > 0:
                return _RouteDecision(
                    kernel="burst",
                    burst_L=self.burst_steps_escape,
                    escape_deep=True,
                )
            return _RouteDecision(kernel="explore")

        return _RouteDecision(kernel="argmax", greedy_acq=greedy_acq)

    def _hmm_observation(self, loss_eval: float) -> float:
        return float((loss_eval - self._best_loss) / (self._scale_factor + 1e-8))

    def _maybe_sigma_nudge(self, u_eval: np.ndarray, improved_best: bool) -> None:
        if (
            not self.use_hmm
            or improved_best
            or self._gp is None
            or not self._gp.is_ready
            or self._hmm.state != HMCState.EXPLOIT
        ):
            return
        X_obs = self._training_unit_points()
        if X_obs is None or X_obs.shape[0] == 0:
            return
        sigmas = []
        for row in X_obs:
            _, sigma = self._gp.posterior_mean_std(row)
            sigmas.append(sigma)
        mean_sigma = float(np.mean(sigmas)) if sigmas else 1.0
        _, sigma_eval = self._gp.posterior_mean_std(u_eval)
        if mean_sigma > 0.0 and sigma_eval / mean_sigma > 1.5:
            self._observations.append(0.1)

    def _maybe_refit_gp(self) -> None:
        finite = self._finite_data()
        if len(finite) < self.gp_min_obs:
            return
        if self._evals_since_gp_refit < self.gp_refit_every and self._gp is not None:
            return
        train = _ParamSpace.build_training_set(
            finite, self._space, self.gp_max_obs, subsample=self.gp_subsample
        )
        if train is None:
            return
        X, y = train
        if self._gp is None:
            self._gp = _GPSurrogate(self._space.continuous_dim, self._gp_config)
        self._gp.fit(X, y)
        self._evals_since_gp_refit = 0
        if self._nuts is None:
            self._nuts = _NUTS(
                self._gp,
                self._space,
                step_size_init=self.step_size_init,
                target_accept=self.target_accept,
                max_tree_depth=self.nuts_max_tree_depth,
                rng=self._rng,
            )
        else:
            self._nuts.gp = self._gp
        ard = self._gp.ard_scales()
        if self._nuts is not None:
            self._nuts.set_mass_from_ard(ard)

    def _record_eval(
        self,
        cfg: dict,
        loss: float,
        state: HMCState,
        kernel: str,
        accepted: bool,
        tree_depth: int = 0,
        step_size: float = 0.0,
        burst_steps: int = 0,
        max_log_ei: float = 0.0,
    ) -> None:
        self.data.append((copy.deepcopy(cfg), float(loss)))
        self._evals_since_gp_refit += 1
        self.history_table.append({
            "Eval": len(self.data),
            "State": state.name,
            "Kernel": kernel,
            "Loss": float(loss),
            "Accepted": accepted,
            "TreeDepth": int(tree_depth),
            "StepSize": float(step_size),
            "BurstSteps": int(burst_steps),
            "MaxLogEI": float(max_log_ei),
            "NoImproveStreak": self._no_improve_streak,
            "Rej_Streak": self._no_improve_streak,
        })

    def main_loop(self) -> tuple[dict, float]:
        pbar = tqdm(
            total=self.budget,
            desc=self.progress_desc or "H-MCMC-HMC",
            disable=not self.show_progress,
        )

        init_seed = int(self._seed) if self._seed is not None else int(self._rng.integers(0, 2**31 - 1))
        init_cfgs = self._space.sobol_configs(self.n_init, seed=init_seed)
        init_scores: list[tuple[dict, float]] = []
        for cfg in init_cfgs:
            if len(self.data) >= self.budget:
                break
            loss = self._eval_objective(cfg)
            init_scores.append((cfg, loss))
            self._record_eval(cfg, loss, HMCState.EXPLOIT, "sobol", True)
            pbar.update(1)

        if len(self.data) >= self.budget:
            pbar.close()
            finite = self._finite_data()
            pool = finite or self.data
            cfg, loss = min(pool, key=lambda x: x[1])
            if not np.isfinite(loss):
                loss = float(self._nan_penalty)
            return self._decode_best((cfg, loss))

        init_scores.sort(key=lambda x: x[1])
        self._current_x, self._current_loss = copy.deepcopy(init_scores[0])
        self._best_x, self._best_loss = copy.deepcopy(init_scores[0])
        self._scale_factor = self._compute_scale([s for _, s in init_scores])

        self._maybe_refit_gp()
        if self._gp is not None and self._nuts is None:
            self._nuts = _NUTS(
                self._gp,
                self._space,
                step_size_init=self.step_size_init,
                target_accept=self.target_accept,
                max_tree_depth=self.nuts_max_tree_depth,
                rng=self._rng,
            )

        bo_warmup_evals = round(self.bo_warmup_frac * self.budget)
        adapt_frozen = False

        while len(self.data) < self.budget:
            self._maybe_refit_gp()
            if self._gp is None or self._nuts is None:
                x_eval = self._space.random_categorical_step(self._current_x, self._rng) if not self._space.continuous_dim else self._space.vector_to_config(
                    self._space.from_unit(self._rng.random(self._space.continuous_dim)),
                    self._current_x,
                )
                loss_eval = self._eval_objective(x_eval)
                state = HMCState.EXPLOIT if not self.use_hmm else self._hmm.state
                delta_obs = self._hmm_observation(loss_eval)
                improved_best = loss_eval < self._best_loss - self.improve_tol
                if improved_best:
                    self._best_x = copy.deepcopy(x_eval)
                    self._best_loss = loss_eval
                    self._no_improve_streak = 0
                else:
                    self._no_improve_streak += 1
                if np.isfinite(delta_obs):
                    self._observations.append(delta_obs)
                self._current_x = copy.deepcopy(x_eval)
                self._current_loss = loss_eval
                self._record_eval(x_eval, loss_eval, state, "random", True)
                pbar.update(1)
                continue

            progress = len(self.data) / self.budget
            stuck_restart = self._no_improve_streak >= self.no_improve_limit

            if self.use_hmm:
                if len(self._observations) >= 2:
                    state = self._hmm.observe(self._observations[-self._hmm.window :])
                else:
                    state = self._hmm.state
            else:
                state = HMCState.EXPLOIT

            if progress >= self.acq_greedy_after:
                state = HMCState.EXPLOIT
                if self.use_hmm:
                    self._hmm.force_state(state)

            kappa, T_state = self._state_params(state)
            post_init = max(0, len(self.data) - self.n_init)
            bo_warmup_on = post_init < bo_warmup_evals
            if post_init >= bo_warmup_evals and not adapt_frozen:
                self._nuts.freeze_adaptation()
                adapt_frozen = True

            effective_p_cat = self.p_cat_step if self._space.has_categorical else 0.0
            do_cat = effective_p_cat > 0.0 and self._rng.random() < effective_p_cat
            route = self._route_kernel(
                state=state,
                stuck_restart=stuck_restart,
                progress=progress,
                bo_warmup_on=bo_warmup_on,
                do_cat=do_cat,
            )

            tree_depth = 0
            step_size = self._nuts.step_size
            burst_steps = 0
            max_log_ei = 0.0
            kernel = route.kernel

            if route.kernel == "categorical":
                x_eval = self._space.random_categorical_step(self._current_x, self._rng)
                loss_eval = self._eval_objective(x_eval)
            elif route.kernel == "random":
                x_eval = self._space.vector_to_config(
                    self._space.from_unit(self._rng.random(self._space.continuous_dim)),
                    self._current_x,
                )
                loss_eval = self._eval_objective(x_eval)
            elif route.kernel == "burst":
                base_depth = self.nuts_max_tree_depth
                if route.escape_deep:
                    self._nuts.max_tree_depth = min(base_depth + 2, 15)
                u_start = self._space.to_unit(
                    self._space.config_to_vector(self._best_x)
                )
                u_star, burst_steps, tree_depth, step_size, max_log_ei = self._run_nuts_burst(
                    u_start,
                    kappa,
                    T_state,
                    route.burst_L,
                    adapt=bo_warmup_on,
                )
                if route.escape_deep:
                    self._nuts.max_tree_depth = base_depth
                x_eval = self._space.vector_to_config(
                    self._space.from_unit(u_star), self._current_x
                )
                loss_eval = self._eval_objective(x_eval)
            elif route.kernel == "explore":
                u_star = self._gp_explore_point()
                x_eval = self._space.vector_to_config(
                    self._space.from_unit(u_star), self._current_x
                )
                loss_eval = self._eval_objective(x_eval)
            else:
                u_star = self._gp_argmax_point(greedy=route.greedy_acq)
                x_eval = self._space.vector_to_config(
                    self._space.from_unit(u_star), self._current_x
                )
                loss_eval = self._eval_objective(x_eval)

            delta_obs = self._hmm_observation(loss_eval)
            improved_best = loss_eval < self._best_loss - self.improve_tol
            if improved_best:
                self._best_x = copy.deepcopy(x_eval)
                self._best_loss = loss_eval
                self._no_improve_streak = 0
            else:
                self._no_improve_streak += 1

            if np.isfinite(delta_obs):
                self._observations.append(delta_obs)
            if self._space.continuous_dim > 0:
                u_eval = self._space.to_unit(self._space.config_to_vector(x_eval))
                self._maybe_sigma_nudge(u_eval, improved_best)

            self._current_x = copy.deepcopy(x_eval)
            self._current_loss = loss_eval

            self._record_eval(
                x_eval,
                loss_eval,
                state,
                kernel,
                True,
                tree_depth=tree_depth,
                step_size=step_size,
                burst_steps=burst_steps,
                max_log_ei=max_log_ei,
            )
            pbar.update(1)

            recent_losses = [s for _, s in self.data[-self._scale_window :]]
            if recent_losses:
                self._scale_factor = self._compute_scale(recent_losses)

        pbar.close()
        if self.verbose_history:
            try:
                import pandas as pd

                df = pd.DataFrame(self.history_table)
                print("\n" + "=" * 60)
                print("HMM MCMC HMC State History (last run):")
                with pd.option_context("display.max_rows", 200, "display.max_columns", None):
                    print(df)
                print("=" * 60 + "\n")
            except ImportError:
                pass

        finite = self._finite_data()
        pool = finite or self.data
        cfg, loss = min(pool, key=lambda x: x[1])
        if not np.isfinite(loss):
            loss = float(self._nan_penalty)
        return self._decode_best((cfg, loss))


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------


def _self_check_gp_grad() -> None:
    _require_gp()
    rng = np.random.default_rng(0)
    X = rng.random((20, 2))
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    gp = _GPSurrogate(2)
    gp.fit(X, y)
    acq, grad = gp.acq_value_and_grad(X[0], kappa=1.0)
    assert np.isfinite(acq) and np.all(np.isfinite(grad))
    print(f"[self-check] GP grad: acq={acq:.4f}, ||grad||={float(np.linalg.norm(grad)):.4f}")


def _self_check_reflect_bounds() -> None:
    u = np.array([-0.2, 1.3, 0.5])
    out = _ParamSpace({"x": {"type": "float", "values": [0, 1]}}).reflect_unit(u[:1])
    assert 0.0 <= out[0] <= 1.0
    q, p = _reflect_leapfrog(np.array([1.1, 0.5]), np.array([0.3, -0.2]))
    assert np.all(q >= 0.0) and np.all(q <= 1.0)
    print("[self-check] reflective bounds: OK")


def _self_check_leapfrog_reversibility() -> None:
    _require_gp()
    rng = np.random.default_rng(1)
    space = _ParamSpace({"x0": {"type": "float", "values": [0, 1]}, "x1": {"type": "float", "values": [0, 1]}})
    X = rng.random((15, 2))
    y = np.zeros(15)
    gp = _GPSurrogate(2)
    gp.fit(X, y)
    nuts = _NUTS(gp, space, step_size_init=0.05, rng=rng)
    q = rng.random(2)
    p = rng.standard_normal(2)
    eps = 0.05
    q1, p1 = nuts._leapfrog(q, p, kappa=1.0, T=1.0, step_size=eps)
    # reverse leapfrog approximately recovers (flat acq -> grad ~0)
    p1r = -p1
    q0r, p0r = nuts._leapfrog(q1, p1r, kappa=1.0, T=1.0, step_size=eps)
    assert float(np.linalg.norm(q0r - q)) < 0.15
    print("[self-check] leapfrog reversibility (flat target): OK")


def _self_check_nuts_moves() -> None:
    """NUTS must return a candidate that differs from q0 and explores the surface."""
    _require_gp()
    rng = np.random.default_rng(7)
    space = _ParamSpace({
        "x0": {"type": "float", "values": [0.0, 1.0]},
        "x1": {"type": "float", "values": [0.0, 1.0]},
    })
    X = rng.uniform(0.0, 0.3, size=(25, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2).astype(np.float64)
    gp = _GPSurrogate(2)
    gp.fit(X, y)
    nuts = _NUTS(gp, space, step_size_init=0.08, max_tree_depth=6, rng=rng)
    q0 = np.array([0.85, 0.85], dtype=np.float64)
    moved = 0
    acq0, _ = gp.acq_value_and_grad(q0, kappa=1.0)
    initial_acq = acq0
    best_acq = acq0
    for _ in range(12):
        q_out, accept, _, _ = nuts.sample(q0, kappa=1.0, T=1.0, adapt=False)
        if float(np.linalg.norm(q_out - q0)) > 1e-3:
            moved += 1
        acq_out, _ = gp.acq_value_and_grad(q_out, kappa=1.0)
        best_acq = max(best_acq, acq_out)
        q0 = q_out.copy()
    assert moved >= 3, f"NUTS should move away from start, moved={moved}/12"
    assert best_acq >= initial_acq - 0.01, "NUTS should not drastically worsen acquisition"
    print(f"[self-check] NUTS moves: moved={moved}/12, acq_delta={best_acq - initial_acq:.4f}")


def _self_check_hmm_states() -> None:
    hmm = _GaussianHMM(window=6, min_obs=10_000, refit_every=10_000)
    improving = [-0.8, -0.5, -0.3, -0.2]
    flat = [0.35, 0.38, 0.4, 0.42]
    worsening = [0.4, 0.6, 0.8, 0.5]
    s_imp = HMCState(hmm._viterbi(improving))
    s_flat = HMCState(hmm._viterbi(flat))
    s_bad = HMCState(hmm._viterbi(worsening))
    assert s_imp == HMCState.EXPLOIT
    assert s_bad == HMCState.ESCAPE
    assert s_flat == HMCState.EXPLORE, f"flat deltas should map to EXPLORE, got {s_flat}"
    print(f"[self-check] HMM states: exploit={s_imp}, explore={s_flat}, escape={s_bad}")


def _self_check_dual_averaging() -> None:
    da = _DualAveraging(0.1, target_accept=0.8)
    for _ in range(50):
        da.update(0.85)
    assert np.isfinite(da.step_size) and da.step_size > 0.0
    print(f"[self-check] dual averaging step_size={da.step_size:.4f}")


def _self_check_exploit_concentration() -> None:
    _require_gp()
    rng = np.random.default_rng(2)
    space = _ParamSpace({"x": {"type": "float", "values": [-1.0, 1.0]}})
    X = rng.uniform(0.0, 0.2, size=(20, 1))
    y = (X[:, 0] ** 2).astype(np.float64)
    gp = _GPSurrogate(1)
    gp.fit(X, y)
    u_star = gp.argmax_point(greedy=False, n_restarts=20, raw_samples=256)
    assert float(u_star[0]) < 0.35, f"LogEI argmax should be near optimum, got {u_star}"
    mu_lo, _ = gp.posterior_mean_std(np.array([0.05]))
    mu_hi, _ = gp.posterior_mean_std(np.array([0.9]))
    assert mu_lo > mu_hi, "posterior mean should be higher near observed low-loss region"
    print(f"[self-check] exploit concentration argmax={float(u_star[0]):.3f}")


def _self_check_nan_training_filter() -> None:
    _require_gp()
    space = _ParamSpace({
        "x0": {"type": "float", "values": [0.0, 1.0]},
        "x1": {"type": "float", "values": [0.0, 1.0]},
    })
    data = [
        ({"x0": 0.1, "x1": 0.2}, 1.0),
        ({"x0": 0.3, "x1": 0.4}, float("nan")),
        ({"x0": 0.5, "x1": 0.6}, 0.5),
        ({"x0": 0.7, "x1": 0.8}, 0.2),
    ]
    train = _ParamSpace.build_training_set(data, space, 4)
    assert train is not None
    X, y = train
    assert X.shape[0] == 3 and np.all(np.isfinite(y))
    gp = _GPSurrogate(2)
    gp.fit(X, y)
    assert gp.is_ready
    print("[self-check] NaN training filter: OK")


def _self_check_nan_objective_run() -> None:
    dict_to_optimize = {
        "x0": {"type": "float", "values": [0.0, 1.0]},
        "x1": {"type": "float", "values": [0.0, 1.0]},
    }
    calls = {"n": 0}

    def objective(cfg: dict) -> float:
        calls["n"] += 1
        if calls["n"] % 5 == 0:
            return float("nan")
        return float(cfg["x0"] ** 2 + cfg["x1"] ** 2)

    alg = HMM_MCMC_HMC(
        objective_func=objective,
        budget=30,
        dict_to_optimize=dict_to_optimize,
        n_init=8,
        show_progress=False,
        verbose_history=False,
    )
    _, best_loss = alg.main_loop()
    losses = [loss for _, loss in alg.data]
    step_sizes = [row["StepSize"] for row in alg.history_table if row["Kernel"] == "burst"]
    assert np.isfinite(best_loss), f"best_loss must be finite, got {best_loss}"
    assert all(np.isfinite(loss) for loss in losses), "recorded losses must be finite"
    assert all(np.isfinite(step) and step > 0.0 for step in step_sizes) or not step_sizes, (
        "burst step sizes must be finite when burst runs"
    )
    print("[self-check] NaN objective run: OK")


def _self_check_decode_objective() -> None:
    """Objective must receive decoded real-space values for log-scale params."""
    dict_to_optimize = {
        "lr": {"type": "float", "values": [0.0001, 0.1], "log": True},
        "units": {"type": "int", "values": [64, 1024], "log": True},
        "dropout": {"type": "float", "values": [0.0, 1.0], "log": False},
    }
    received: list[dict] = []

    def objective(cfg: dict) -> float:
        received.append(dict(cfg))
        return float(cfg["lr"] + cfg["units"] * 1e-4 + cfg["dropout"])

    alg = HMM_MCMC_HMC(
        objective_func=objective,
        budget=12,
        dict_to_optimize=dict_to_optimize,
        n_init=6,
        show_progress=False,
        verbose_history=False,
    )
    _, best_loss = alg.main_loop()
    assert len(received) == 12, f"expected 12 objective calls, got {len(received)}"
    for cfg in received:
        assert 0.0001 <= cfg["lr"] <= 0.1, f"lr should be real-space, got {cfg['lr']}"
        assert 64 <= cfg["units"] <= 1024, f"units should be real-space int, got {cfg['units']}"
        assert 0.0 <= cfg["dropout"] <= 1.0
    stored_lrs = [c["lr"] for c, _ in alg.data]
    assert any(lr < 0 or lr > 1 for lr in stored_lrs), "internal data should keep log-space lr"
    assert np.isfinite(best_loss)
    print("[self-check] decode objective: OK")


def _self_check_no_improve_streak() -> None:
    """no_improve_streak tracks global-best improvement, not chain acceptance."""
    dict_to_optimize = {
        "x": {"type": "float", "values": [0.0, 1.0], "log": False},
    }
    calls = {"n": 0}

    def objective(cfg: dict) -> float:
        calls["n"] += 1
        x = float(cfg["x"])
        if calls["n"] == 1:
            return 1.0
        if calls["n"] == 2:
            return 0.5
        return 0.4

    alg = HMM_MCMC_HMC(
        objective_func=objective,
        budget=3,
        dict_to_optimize=dict_to_optimize,
        n_init=1,
        gp_min_obs=2,
        bo_warmup_frac=0.0,
        burst_steps_escape=0,
        burst_steps_stuck=0,
        show_progress=False,
        verbose_history=False,
        seed=11,
    )
    alg.main_loop()
    assert alg._no_improve_streak == 0, "global-best improvement should reset streak"
    print("[self-check] no-improve streak semantics: OK")


def _self_check_unified_routing() -> None:
    _require_gp()
    rng = np.random.default_rng(0)
    alg = HMM_MCMC_HMC(
        objective_func=lambda cfg: float(cfg["x0"] ** 2),
        budget=1,
        dict_to_optimize={"x0": {"type": "float", "values": [0.0, 1.0]}},
        n_init=1,
        show_progress=False,
        verbose_history=False,
        burst_steps_escape=64,
        burst_steps_stuck=32,
    )
    X = rng.random((12, 1))
    y = (X[:, 0] ** 2).astype(np.float64)
    alg._gp = _GPSurrogate(1)
    alg._gp.fit(X, y)
    exploit = alg._route_kernel(HMCState.EXPLOIT, False, 0.5, False, False)
    explore = alg._route_kernel(HMCState.EXPLORE, False, 0.5, False, False)
    escape = alg._route_kernel(HMCState.ESCAPE, False, 0.5, False, False)
    stuck = alg._route_kernel(HMCState.EXPLOIT, True, 0.5, False, False)
    warmup = alg._route_kernel(HMCState.EXPLORE, False, 0.5, True, False)
    endgame = alg._route_kernel(HMCState.ESCAPE, False, 0.98, False, False)
    assert exploit.kernel == "argmax"
    assert explore.kernel == "explore"
    assert escape.kernel == "burst" and escape.burst_L == 64
    assert stuck.kernel == "burst" and stuck.burst_L == 32
    assert warmup.kernel == "argmax"
    assert endgame.kernel == "argmax"
    print("[self-check] unified routing: OK")


def _self_check_burst_logei_selection() -> None:
    _require_gp()
    rng = np.random.default_rng(11)
    space = _ParamSpace({
        "x0": {"type": "float", "values": [0.0, 1.0]},
        "x1": {"type": "float", "values": [0.0, 1.0]},
    })
    X = rng.uniform(0.0, 0.3, size=(20, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2).astype(np.float64)
    gp = _GPSurrogate(2)
    gp.fit(X, y)
    nuts = _NUTS(gp, space, step_size_init=0.08, max_tree_depth=6, rng=rng)
    alg = HMM_MCMC_HMC(
        objective_func=lambda cfg: float(cfg["x0"] ** 2 + cfg["x1"] ** 2),
        budget=1,
        dict_to_optimize={
            "x0": {"type": "float", "values": [0.0, 1.0]},
            "x1": {"type": "float", "values": [0.0, 1.0]},
        },
        n_init=1,
        show_progress=False,
        verbose_history=False,
    )
    alg._gp = gp
    alg._nuts = nuts
    u0 = np.array([0.85, 0.85], dtype=np.float64)
    u_star, burst_steps, _, _, max_log_ei = alg._run_nuts_burst(u0, kappa=1.0, T=1.0, L=8, adapt=False)
    assert burst_steps == 8
    assert np.isfinite(max_log_ei)
    assert max_log_ei >= gp.log_ei_at(u0) - 1e-6
    print(f"[self-check] burst LogEI selection: max_log_ei={max_log_ei:.4f}")


def _self_check_hmm_hysteresis() -> None:
    hmm = _GaussianHMM(window=6, min_obs=10_000, refit_every=10_000, switch_confirm=2)
    hmm.state = HMCState.EXPLOIT
    flat = [0.35, 0.38, 0.4, 0.42, 0.39, 0.41]
    s1 = hmm.observe(flat)
    assert s1 == HMCState.EXPLOIT, "one noisy non-exploit vote should not leave EXPLOIT"
    s1b = hmm.observe(flat)
    assert s1b == HMCState.EXPLORE, "confirmed non-exploit votes should switch state"
    worsening = [0.4, 0.6, 0.8, 0.5, 0.7, 0.6]
    s2 = hmm.observe(worsening)
    assert s2 == HMCState.ESCAPE, "confirmed escape votes should switch to ESCAPE"
    print("[self-check] HMM exploit hysteresis: OK")


def _self_check_bo_arm_pure_argmax() -> None:
    _require_gp()
    dict_to_optimize = {
        "x0": {"type": "float", "values": [0.0, 1.0], "log": False},
        "x1": {"type": "float", "values": [0.0, 1.0], "log": False},
    }

    def objective(cfg: dict) -> float:
        return float(cfg["x0"] ** 2 + cfg["x1"] ** 2)

    alg = HMM_MCMC_HMC(
        objective_func=objective,
        budget=36,
        dict_to_optimize=dict_to_optimize,
        n_init=6,
        gp_min_obs=4,
        bo_warmup_frac=0.0,
        burst_steps_escape=0,
        burst_steps_stuck=0,
        use_hmm=False,
        show_progress=False,
        verbose_history=False,
        seed=13,
    )
    alg.main_loop()
    post = [row for row in alg.history_table if row["Eval"] > alg.n_init]
    assert post, "expected post-init rows"
    allowed = {"argmax", "sobol", "categorical", "random"}
    assert all(row["Kernel"] in allowed for row in post), (
        f"pure BO arm should only use argmax-like kernels, got { {row['Kernel'] for row in post} }"
    )
    assert all(row["Kernel"] == "argmax" for row in post), "post-init should be 100% argmax"
    print(f"[self-check] BO arm pure argmax: {len(post)}/{len(post)} post-init steps")


def _self_check_explore_routing() -> None:
    """EXPLORE should route to UCB explore, not burst."""
    _require_gp()
    dict_to_optimize = {
        "x0": {"type": "float", "values": [0.0, 1.0], "log": False},
        "x1": {"type": "float", "values": [0.0, 1.0], "log": False},
    }

    def objective(cfg: dict) -> float:
        return float(cfg["x0"] ** 2 + cfg["x1"] ** 2)

    alg = HMM_MCMC_HMC(
        objective_func=objective,
        budget=40,
        dict_to_optimize=dict_to_optimize,
        n_init=6,
        gp_min_obs=4,
        bo_warmup_frac=0.0,
        burst_steps_escape=64,
        burst_steps_stuck=32,
        acq_greedy_after=0.95,
        use_hmm=True,
        show_progress=False,
        verbose_history=False,
        seed=21,
    )
    alg.main_loop()
    explore_rows = [row for row in alg.history_table if row["State"] == "EXPLORE" and row["Eval"] > alg.n_init]
    if explore_rows:
        explore_ucb = sum(1 for row in explore_rows if row["Kernel"] == "explore")
        assert explore_ucb >= 1, "EXPLORE should use UCB explore kernel"
        print(f"[self-check] EXPLORE routing: explore={explore_ucb}/{len(explore_rows)}")
    else:
        print("[self-check] EXPLORE routing: no EXPLORE steps (skipped)")


def _self_check_burst_routing() -> None:
    """Hybrid mode should use burst for ESCAPE and stuck, not for EXPLORE."""
    _require_gp()
    dict_to_optimize = {
        "x0": {"type": "float", "values": [0.0, 1.0], "log": False},
        "x1": {"type": "float", "values": [0.0, 1.0], "log": False},
    }

    def objective(cfg: dict) -> float:
        return float(cfg["x0"] ** 2 + cfg["x1"] ** 2)

    alg = HMM_MCMC_HMC(
        objective_func=objective,
        budget=40,
        dict_to_optimize=dict_to_optimize,
        n_init=6,
        gp_min_obs=4,
        bo_warmup_frac=0.0,
        burst_steps_escape=32,
        burst_steps_stuck=32,
        acq_greedy_after=0.95,
        use_hmm=True,
        show_progress=False,
        verbose_history=False,
        seed=21,
    )
    alg.main_loop()
    post = [row for row in alg.history_table if row["Eval"] > alg.n_init]
    burst_rows = [row for row in post if row["Kernel"] == "burst"]
    explore_burst = [row for row in post if row["State"] == "EXPLORE" and row["Kernel"] == "burst"]
    assert burst_rows, "expected at least one burst step in hybrid mode"
    assert not explore_burst, "EXPLORE must not use burst kernel"
    print(f"[self-check] burst routing: {len(burst_rows)}/{len(post)} post-init steps, 0 EXPLORE bursts")


def _self_check_bo_warmup_routing() -> None:
    _require_gp()
    dict_to_optimize = {
        "x0": {"type": "float", "values": [0.0, 1.0], "log": False},
        "x1": {"type": "float", "values": [0.0, 1.0], "log": False},
    }

    def objective(cfg: dict) -> float:
        return float(cfg["x0"] ** 2 + cfg["x1"] ** 2)

    budget = 24
    n_init = 6
    alg = HMM_MCMC_HMC(
        objective_func=objective,
        budget=budget,
        dict_to_optimize=dict_to_optimize,
        n_init=n_init,
        gp_min_obs=4,
        bo_warmup_frac=0.25,
        exploit_argmax_every=5,
        show_progress=False,
        verbose_history=False,
        seed=5,
    )
    alg.main_loop()
    post_init = [row for row in alg.history_table if row["Eval"] > n_init]
    warmup_n = round(0.25 * budget)
    early = [row for row in post_init if row["Eval"] <= n_init + warmup_n]
    assert early, "expected post-init history"
    argmax_early = sum(1 for row in early if row["Kernel"] == "argmax")
    assert argmax_early >= max(1, len(early) // 2), (
        f"BO warmup should be argmax-dominant, got {argmax_early}/{len(early)}"
    )
    print(f"[self-check] BO warmup routing: argmax {argmax_early}/{len(early)} early steps")


def _self_check_ablation_parity() -> None:
    """With use_hmm=False, driver stays in EXPLOIT and completes without error."""
    dict_to_optimize = {
        "x0": {"type": "float", "values": [-5.0, 5.0], "log": False},
        "x1": {"type": "float", "values": [-5.0, 5.0], "log": False},
    }

    def objective(cfg: dict) -> float:
        return float(cfg["x0"] ** 2 + cfg["x1"] ** 2)

    alg = HMM_MCMC_HMC(
        objective_func=objective,
        budget=20,
        dict_to_optimize=dict_to_optimize,
        n_init=4,
        gp_min_obs=4,
        use_hmm=False,
        show_progress=False,
        verbose_history=False,
        seed=7,
    )
    _, best_loss = alg.main_loop()
    post_init = [row for row in alg.history_table if row["Eval"] > alg.n_init]
    states = {row["State"] for row in post_init}
    assert best_loss >= 0.0
    assert len(alg.data) == 20
    assert states <= {"EXPLOIT", "ESCAPE"}, f"no-HMM arm should not use EXPLORE, got {states}"
    print(f"[self-check] ablation parity: best_loss={best_loss:.4f}, states={sorted(states)}")
    print("[self-check] ablation parity: OK")


def _self_check_hmm_endgame_exploit() -> None:
    """Late budget should pin HMM to EXPLOIT when use_hmm=True."""
    _require_gp()
    dict_to_optimize = {
        "x0": {"type": "float", "values": [0.0, 1.0], "log": False},
        "x1": {"type": "float", "values": [0.0, 1.0], "log": False},
    }

    def objective(cfg: dict) -> float:
        return float(cfg["x0"] ** 2 + cfg["x1"] ** 2)

    budget = 40
    acq_greedy_after = 0.97
    alg = HMM_MCMC_HMC(
        objective_func=objective,
        budget=budget,
        dict_to_optimize=dict_to_optimize,
        n_init=6,
        gp_min_obs=4,
        acq_greedy_after=acq_greedy_after,
        use_hmm=True,
        show_progress=False,
        verbose_history=False,
        seed=3,
    )
    alg.main_loop()
    cutoff = int(np.ceil(acq_greedy_after * budget))
    late = [row for row in alg.history_table if row["Eval"] > cutoff]
    assert late, "expected late-budget history"
    late_states = {row["State"] for row in late}
    assert late_states == {"EXPLOIT"}, f"endgame should be EXPLOIT-only, got {late_states}"
    print(f"[self-check] HMM endgame exploit: {len(late)} late steps pinned to EXPLOIT")


def _self_check_budget_exactness() -> None:
    from hpo_rl.backends.function import OptimizationBenchmarkBackend

    backend = OptimizationBenchmarkBackend(function_name="schwefel", dimensions=2, noise_std=0)
    algo = HMM_MCMC_HMC(
        objective_func=backend.evaluate,
        budget=35,
        dict_to_optimize=backend.hp_space,
        n_init=5,
        gp_min_obs=5,
        show_progress=False,
        verbose_history=False,
    )
    algo.main_loop()
    assert len(algo.data) == 35
    print("[self-check] budget exactness: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("HMM_MCMC_HMC self-checks")
    print("=" * 60)
    _self_check_gp_grad()
    _self_check_reflect_bounds()
    _self_check_leapfrog_reversibility()
    _self_check_nuts_moves()
    _self_check_hmm_states()
    _self_check_dual_averaging()
    _self_check_exploit_concentration()
    _self_check_nan_training_filter()
    _self_check_nan_objective_run()
    _self_check_decode_objective()
    _self_check_no_improve_streak()
    _self_check_unified_routing()
    _self_check_burst_logei_selection()
    _self_check_hmm_hysteresis()
    _self_check_bo_warmup_routing()
    _self_check_ablation_parity()
    _self_check_hmm_endgame_exploit()
    _self_check_bo_arm_pure_argmax()
    _self_check_explore_routing()
    _self_check_burst_routing()
    _self_check_budget_exactness()
    print("=" * 60)
    print("All self-checks passed.")
    print("=" * 60)
