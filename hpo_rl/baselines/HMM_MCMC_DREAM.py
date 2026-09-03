"""H-MCMC-DREAM: self-contained DREAM(ZS) + soft HMM + Baum-Welch.

Minimal standalone implementation — no imports from other HMM_MCMC* modules.
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.stats.qmc import Sobol
from tqdm.auto import tqdm

SoftWeights = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]

_DEPRECATED_KWARGS = frozenset({
    "use_soft_states",
    "use_dream",
    "use_baum_welch",
    "hot_chain",
    "restart_patience",
    "dream_explore_threshold",
})


def _warn_deprecated_kwargs(kwargs: dict) -> None:
    for key in kwargs:
        if key in _DEPRECATED_KWARGS:
            warnings.warn(
                f"{key} is deprecated and ignored in the minimal DREAM implementation.",
                DeprecationWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# Utilities (from HMM_MCMC.py)
# ---------------------------------------------------------------------------

def _logsumexp(a):
    arr = np.asarray(a, dtype=np.float64)
    a_max = arr.max()
    if a_max == -np.inf:
        return -np.inf
    return float(a_max + np.log(np.sum(np.exp(arr - a_max))))


def decode_config(cfg: dict, dict_to_optimize: dict) -> dict:
    out = dict(cfg)
    for name, info in dict_to_optimize.items():
        if info.get("type") == "float" and info.get("log"):
            out[name] = float(10.0 ** float(cfg[name]))
        elif info.get("type") == "int" and info.get("log"):
            real_lo, real_hi = int(info["values"][0]), int(info["values"][1])
            val = int(np.round(10.0 ** float(cfg[name])))
            out[name] = int(np.clip(val, real_lo, real_hi))
        elif info.get("type") == "int":
            out[name] = int(round(float(cfg[name])))
    return out


def _int_param_bounds(info: dict) -> tuple[float, float, int, int, bool]:
    real_lo, real_hi = int(info["values"][0]), int(info["values"][1])
    if info.get("log"):
        lo = float(np.log10(max(real_lo, 1)))
        hi = float(np.log10(real_hi))
        return lo, hi, real_lo, real_hi, True
    return float(real_lo), float(real_hi), real_lo, real_hi, False


def _int_real_value(x, pi: dict) -> int:
    if pi.get("log"):
        return int(np.clip(round(10.0 ** float(x)), pi["real_lo"], pi["real_hi"]))
    return int(round(float(x)))


def _float_param_bounds(info: dict) -> tuple[float, float, float, float, bool]:
    real_lo, real_hi = float(info["values"][0]), float(info["values"][1])
    if info.get("log"):
        return float(np.log10(real_lo)), float(np.log10(real_hi)), real_lo, real_hi, True
    return real_lo, real_hi, real_lo, real_hi, False


_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
_SQRT2 = math.sqrt(2.0)


def _std_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _rational_approx(t: float) -> float:
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def _std_ppf(p: float) -> float:
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -_rational_approx(math.sqrt(-2.0 * math.log(p)))
    return _rational_approx(math.sqrt(-2.0 * math.log(1.0 - p)))


def _tn_rvs(a: float, b: float, loc: float, scale: float) -> float:
    cdf_a = _std_cdf(a)
    cdf_b = _std_cdf(b)
    u = np.random.uniform(cdf_a + 1e-15, cdf_b - 1e-15)
    return float(_std_ppf(u) * scale + loc)


def _tn_logpdf(x: float, a: float, b: float, loc: float, scale: float) -> float:
    z = (x - loc) / scale
    if z < a - 1e-10 or z > b + 1e-10:
        return -np.inf
    log_phi = -0.5 * z * z - _LOG_SQRT_2PI
    cdf_diff = _std_cdf(b) - _std_cdf(a)
    if cdf_diff < 1e-30:
        return -np.inf
    return float(log_phi - math.log(scale) - math.log(cdf_diff))


class HMMState(IntEnum):
    EXPLOIT = 0
    EXPLORE = 1
    TRAPPED = 2


# Per-state proposal weight tables (used by mix_soft_weights)
_FLOAT_WEIGHTS = {
    HMMState.EXPLOIT: (0.90, 0.10, 0.00),
    HMMState.EXPLORE: (0.00, 0.80, 0.20),
    HMMState.TRAPPED: (0.00, 0.50, 0.50),
}
_INT_WEIGHTS = {
    HMMState.EXPLOIT: (0.90, 0.10, 0.00),
    HMMState.EXPLORE: (0.00, 0.80, 0.20),
    HMMState.TRAPPED: (0.00, 0.50, 0.50),
}
_CATEGORICAL_WEIGHTS = {
    HMMState.EXPLOIT: (0.30, 0.00, 0.70),
    HMMState.EXPLORE: (0.10, 0.40, 0.50),
    HMMState.TRAPPED: (0.00, 0.80, 0.20),
}


def mix_soft_weights(posterior: np.ndarray) -> SoftWeights:
    fw = np.zeros(3, dtype=np.float64)
    iw = np.zeros(3, dtype=np.float64)
    cw = np.zeros(3, dtype=np.float64)
    for s in HMMState:
        p = float(posterior[int(s)])
        fw += p * np.array(_FLOAT_WEIGHTS[s], dtype=np.float64)
        iw += p * np.array(_INT_WEIGHTS[s], dtype=np.float64)
        cw += p * np.array(_CATEGORICAL_WEIGHTS[s], dtype=np.float64)
    return tuple(fw), tuple(iw), tuple(cw)


# ---------------------------------------------------------------------------
# Sobol initializer
# ---------------------------------------------------------------------------

class SobolInitializer:
    def __init__(self, dict_to_optimize: dict):
        self.dict_to_optimize = dict_to_optimize
        self.param_names = list(dict_to_optimize.keys())
        self.dim = len(self.param_names)

    def generate(self, n_points: int, seed: int = 42) -> list[dict]:
        unit_points = self._sobol_unit_cube(n_points, seed)
        return [self._unit_to_config(unit_points[i]) for i in range(n_points)]

    def _sobol_unit_cube(self, n: int, seed: int) -> np.ndarray:
        sampler = Sobol(d=self.dim, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n, 2))))
        raw = sampler.random_base2(m)
        return raw[:n]

    def _unit_to_config(self, u: np.ndarray) -> dict:
        config = {}
        for j, name in enumerate(self.param_names):
            info = self.dict_to_optimize[name]
            p_type = info["type"]
            values = info["values"]
            if p_type == "float":
                lo, hi, _, _, _ = _float_param_bounds(info)
                config[name] = lo + (hi - lo) * u[j]
            elif p_type == "int":
                real_lo, real_hi = int(values[0]), int(values[1])
                if info.get("log"):
                    lo, hi = np.log10(max(real_lo, 1)), np.log10(real_hi)
                    config[name] = float(lo + (hi - lo) * u[j])
                else:
                    config[name] = int(np.round(real_lo + (real_hi - real_lo) * u[j]))
            elif p_type == "categorical":
                idx = int(u[j] * len(values))
                idx = min(idx, len(values) - 1)
                config[name] = values[idx]
            else:
                raise ValueError(f"Unknown type: {p_type}")
        return config


# ---------------------------------------------------------------------------
# Soft Baum-Welch HMM (emissions + A-only BW + forward filter)
# ---------------------------------------------------------------------------

class _SoftBaumWelchHMM:
    """HMM with fixed emissions, online Baum-Welch for A, and soft forward filter."""

    N_STATES = 3

    def __init__(
        self,
        window: int = 8,
        obs_epsilon: float = 1e-8,
        lambda_noise: float = 0.01,
        refit_every: int = 5,
        min_obs: int = 12,
        n_em_iters: int = 3,
        prior_strength: float = 25.0,
        exploit_prior_scale: float = 3.0,
        bw_max_len: int = 64,
    ):
        self.window = window
        self.obs_epsilon = obs_epsilon
        self.lambda_noise = lambda_noise
        self.refit_every = refit_every
        self.min_obs = min_obs
        self.n_em_iters = n_em_iters
        self.prior_strength = prior_strength
        self.exploit_prior_scale = exploit_prior_scale
        self._bw_max_len = bw_max_len

        self.pi = np.array([1.0, 0.0, 0.0])
        self.A = np.array([
            [0.60, 0.40, 0.00],
            [0.50, 0.50, 0.00],
            [0.60, 0.20, 0.20],
        ])
        self._A_default = self.A.copy()

        self.emission_mu = np.array([-0.01, 0.10, 0.015])
        self.emission_sigma = np.array([0.06, 0.25, 0.020])
        self.c_noise = 1.0 / 20.0

        self.state = HMMState.EXPLOIT
        self._posterior = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._bw_buffer: list[float] = []
        self._step_count: int = 0

    def reset(self):
        self.state = HMMState.EXPLOIT
        self._posterior = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self._bw_buffer = []
        self._step_count = 0
        self.A = self._A_default.copy()

    def force_state(self, new_state: HMMState):
        self.state = new_state

    def posterior(self) -> np.ndarray:
        return self._posterior.copy()

    def observe(self, obs_history: list[float]) -> HMMState:
        if obs_history:
            self._update_buffer(obs_history)
            self._maybe_refit()

        if len(obs_history) < 2:
            return self.state

        obs_seq = obs_history[-self.window:]
        self._posterior = self._forward_filter(obs_seq)
        self.state = HMMState(int(np.argmax(self._posterior)))
        return self.state

    def _update_buffer(self, obs_history: list[float]) -> None:
        latest = float(obs_history[-1])
        if not self._bw_buffer or latest != self._bw_buffer[-1]:
            self._bw_buffer.append(latest)
            self._step_count += 1
        if len(self._bw_buffer) > self._bw_max_len:
            self._bw_buffer = self._bw_buffer[-self._bw_max_len:]

    def _maybe_refit(self) -> None:
        if self._step_count % self.refit_every != 0:
            return
        if len(self._bw_buffer) < self.min_obs:
            return
        self._fit_transitions(self._bw_buffer)

    def _log_emission(self, obs: float, state: int) -> float:
        mu = self.emission_mu[state]
        sigma = self.emission_sigma[state]
        z = (obs - mu) / sigma
        log_gauss = -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
        gauss = np.exp(log_gauss)
        density = (1.0 - self.lambda_noise) * gauss + self.lambda_noise * self.c_noise
        return np.log(density + 1e-30)

    def _forward_filter(self, obs_seq: list[float]) -> np.ndarray:
        obs = np.asarray(obs_seq, dtype=np.float64)
        T = len(obs)
        if T == 0:
            return self._posterior.copy()

        log_pi = np.log(self.pi + 1e-30)
        log_A = np.log(self.A + 1e-30)
        log_B = np.array(
            [self._log_emission(o, s) for o in obs for s in range(self.N_STATES)],
            dtype=np.float64,
        ).reshape(T, self.N_STATES)

        log_alpha = np.full((T, self.N_STATES), -np.inf, dtype=np.float64)
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            for j in range(self.N_STATES):
                log_alpha[t, j] = _logsumexp(log_alpha[t - 1] + log_A[:, j]) + log_B[t, j]

        log_post = log_alpha[T - 1]
        log_post -= _logsumexp(log_post)
        post = np.exp(log_post)
        post /= post.sum() + 1e-30
        return post

    def _fit_transitions(self, obs_seq: list[float]) -> None:
        obs = np.asarray(obs_seq, dtype=np.float64)
        T = len(obs)
        if T < 2:
            return

        log_A = np.log(self.A + 1e-30)
        log_pi = np.log(self.pi + 1e-30)
        log_B = np.array([self._log_emission(o, s) for o in obs for s in range(self.N_STATES)])
        log_B = log_B.reshape(T, self.N_STATES)

        A_new = self.A.copy()
        for _ in range(self.n_em_iters):
            log_alpha, log_beta, log_norm = self._forward_backward(log_pi, log_A, log_B)
            if not np.isfinite(log_norm):
                self.A = self._A_default.copy()
                return

            xi = np.zeros((T - 1, self.N_STATES, self.N_STATES), dtype=np.float64)
            gamma = np.zeros((T, self.N_STATES), dtype=np.float64)

            for t in range(T - 1):
                log_xi_t = (
                    log_alpha[t][:, None]
                    + log_A
                    + log_B[t + 1][None, :]
                    + log_beta[t + 1][None, :]
                    - log_norm
                )
                xi[t] = np.exp(log_xi_t)
                xi[t] /= xi[t].sum() + 1e-30

            for t in range(T):
                log_gamma_t = log_alpha[t] + log_beta[t] - log_norm
                gamma[t] = np.exp(log_gamma_t)
                gamma[t] /= gamma[t].sum() + 1e-30

            for i in range(self.N_STATES):
                row_prior = self.prior_strength * (
                    self.exploit_prior_scale if i == int(HMMState.EXPLOIT) else 1.0
                )
                denom = gamma[:-1, i].sum() + row_prior
                if denom < 1e-30:
                    A_new[i] = self._A_default[i]
                    continue
                for j in range(self.N_STATES):
                    num = xi[:, i, j].sum() + row_prior * self._A_default[i, j]
                    A_new[i, j] = num / denom

            row_sums = A_new.sum(axis=1, keepdims=True)
            if np.any(row_sums < 1e-30) or not np.all(np.isfinite(A_new)):
                self.A = self._A_default.copy()
                return
            A_new /= row_sums
            A_new = np.clip(A_new, 1e-6, 1.0)
            A_new /= A_new.sum(axis=1, keepdims=True)
            log_A = np.log(A_new + 1e-30)

        self.A = A_new

    def _forward_backward(
        self,
        log_pi: np.ndarray,
        log_A: np.ndarray,
        log_B: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        T, N = log_B.shape
        log_alpha = np.full((T, N), -np.inf)
        log_beta = np.full((T, N), -np.inf)

        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            for j in range(N):
                log_alpha[t, j] = _logsumexp(log_alpha[t - 1] + log_A[:, j]) + log_B[t, j]

        log_beta[T - 1] = 0.0
        for t in range(T - 2, -1, -1):
            for i in range(N):
                log_beta[t, i] = _logsumexp(log_A[i, :] + log_B[t + 1] + log_beta[t + 1])

        log_norm = _logsumexp(log_alpha[T - 1])
        return log_alpha, log_beta, log_norm


# ---------------------------------------------------------------------------
# DREAM proposal generator (soft weights only)
# ---------------------------------------------------------------------------

@dataclass
class DREAMConfig:
    n_pairs: int = 1
    cr: float = 0.9
    gamma1_prob: float = 0.1
    eps: float = 1e-3
    min_pop: int = 4
    z_pop_cap: int = 64
    diversity_frac: float = 0.0


class DREAMProposalGenerator:
    """Factorized proposals with soft HMM weights + DREAM(ZS) population kernel."""

    _EPS = 1e-30

    def __init__(
        self,
        dict_to_optimize: dict,
        sigma_fraction: float = 0.10,
        temperature: float = 0.60,
        wide_sigma_fraction: float = 0.40,
        kde_tau: float = 0.05,
        dream: DREAMConfig | None = None,
        z_archive_max: int = 512,
    ):
        self.dict_to_optimize = dict_to_optimize
        self.param_names: list[str] = list(dict_to_optimize.keys())
        self.sigma_fraction = sigma_fraction
        self.temperature = temperature
        self.dim = len(dict_to_optimize)
        self.wide_sigma_fraction = wide_sigma_fraction
        self._archive: list[tuple[dict, float]] = []
        self._archive_max = 200
        self._kde_tau = kde_tau
        self.dream = dream or DREAMConfig()
        self.z_archive_max = max(1, int(z_archive_max))
        self._z_archive: list[np.ndarray] = []
        self._Z: np.ndarray | None = None
        self._ranges: np.ndarray | None = None
        self._los: np.ndarray | None = None
        self._his: np.ndarray | None = None

        self._param_info: list[dict] = []
        for name in self.param_names:
            info = dict_to_optimize[name]
            p_type = info["type"]
            values = info["values"]
            rec: dict = {"name": name, "type": p_type, "values": values}
            if p_type == "float":
                lo, hi, real_lo, real_hi, is_log = _float_param_bounds(info)
                rec["log"] = is_log
                rec["real_lo"] = real_lo
                rec["real_hi"] = real_hi
                rec["lo"] = lo
                rec["hi"] = hi
                rec["range"] = rec["hi"] - rec["lo"]
                rec["sigma"] = self.sigma_fraction * rec["range"]
                rec["sigma_wide"] = self.wide_sigma_fraction * rec["range"]
            elif p_type == "int":
                lo, hi, real_lo, real_hi, is_log = _int_param_bounds(info)
                rec["log"] = is_log
                rec["real_lo"] = real_lo
                rec["real_hi"] = real_hi
                rec["lo"] = lo
                rec["hi"] = hi
                rec["range"] = rec["hi"] - rec["lo"]
                rec["sigma"] = self.sigma_fraction * rec["range"]
                rec["sigma_wide"] = self.wide_sigma_fraction * rec["range"]
            elif p_type == "categorical":
                rec["n_categories"] = len(values)
                rec["val_to_idx"] = {v: i for i, v in enumerate(values)}
            self._param_info.append(rec)

        self._category_history: dict[str, dict] = {}
        for pi in self._param_info:
            if pi["type"] == "categorical":
                self._category_history[pi["name"]] = {v: [] for v in pi["values"]}

        self._continuous_indices: list[int] = [
            i for i, pi in enumerate(self._param_info) if pi["type"] in ("float", "int")
        ]

        if self._continuous_indices:
            pis = [self._param_info[i] for i in self._continuous_indices]
            self._los = np.array([pi["lo"] for pi in pis], dtype=np.float64)
            self._his = np.array([pi["hi"] for pi in pis], dtype=np.float64)
            self._ranges = self._his - self._los

    @staticmethod
    def _w(kind: str, weights: SoftWeights) -> tuple[float, float, float]:
        fw, iw, cw = weights
        if kind == "float":
            return fw
        if kind == "int":
            return iw
        return cw

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

    def update_category_history(self, config: dict, loss: float) -> None:
        for pi in self._param_info:
            if pi["type"] == "categorical":
                name = pi["name"]
                val = config[name]
                self._category_history[name][val].append(loss)

        self._archive.append((config, loss))
        self._archive.sort(key=lambda x: x[1])
        if len(self._archive) > self._archive_max:
            self._archive = self._archive[: self._archive_max]

        if not self._continuous_indices:
            return
        z = self.config_to_vector(config)
        self._z_archive.append(z)
        if len(self._z_archive) > self.z_archive_max:
            self._z_archive = self._z_archive[-self.z_archive_max:]

    def refresh_population(self, chains) -> None:
        vectors: list[np.ndarray] = []
        for chain in chains:
            vectors.append(self.config_to_vector(chain.current_x))

        min_z = self.dream.min_pop
        z_cap = self.dream.z_pop_cap

        if len(self._z_archive) >= min_z:
            n_sample = min(len(self._z_archive), z_cap)
            if n_sample > 0:
                idx = np.random.choice(len(self._z_archive), size=n_sample, replace=False)
                for i in idx:
                    vectors.append(self._z_archive[int(i)].copy())
        else:
            for cfg, _ in self._archive:
                vectors.append(self.config_to_vector(cfg))

        n_div = int(self.dream.diversity_frac * z_cap)
        if n_div > 0 and self._ranges is not None and self._los is not None:
            rand = self._los + np.random.rand(n_div, self._los.size) * self._ranges
            for row in rand:
                vectors.append(row.copy())

        if vectors:
            self._Z = np.vstack(vectors)
        else:
            self._Z = None

    def _dream_pair_indices(self, x: np.ndarray, n_pairs: int) -> np.ndarray:
        if self._Z is None:
            raise ValueError("DREAM population Z is not initialized")
        n_pop = self._Z.shape[0]
        need = 2 * n_pairs
        if n_pop < need:
            return np.random.choice(n_pop, size=need, replace=n_pop < need)

        if self._ranges is not None and self._ranges.size > 0:
            scaled = np.abs(self._Z - x) / np.maximum(self._ranges, 1e-30)
            pool = np.where(np.max(scaled, axis=1) > 1e-9)[0]
        else:
            pool = np.arange(n_pop, dtype=np.int64)

        if pool.size < need:
            pool = np.arange(n_pop, dtype=np.int64)
        return np.random.choice(pool, size=need, replace=False)

    @property
    def population_size(self) -> int:
        return 0 if self._Z is None else int(self._Z.shape[0])

    def generate_proposal(
        self,
        current_x: dict,
        weights: SoftWeights,
        categorical_only: bool = False,
    ) -> dict:
        x_new: dict = {}
        for i, pi in enumerate(self._param_info):
            name = pi["name"]
            p_type = pi["type"]
            x_j = current_x[name]

            if p_type == "categorical":
                x_new[name] = self._sample_categorical(x_j, pi, weights)
            elif categorical_only:
                x_new[name] = x_j
            elif p_type == "float":
                x_new[name] = self._sample_continuous(x_j, pi, weights)
            elif p_type == "int":
                x_new[name] = self._sample_integer(x_j, pi, weights)

        return x_new

    def log_proposal_density(
        self,
        x_from: dict,
        x_to: dict,
        weights: SoftWeights,
    ) -> float:
        log_q_total = 0.0
        for pi in self._param_info:
            name = pi["name"]
            p_type = pi["type"]
            val_from = x_from[name]
            val_to = x_to[name]
            if p_type == "float":
                log_q_total += self._log_q_continuous(val_from, val_to, pi, weights)
            elif p_type == "int":
                log_q_total += self._log_q_integer(val_from, val_to, pi, weights)
            elif p_type == "categorical":
                log_q_total += self._log_q_categorical(val_from, val_to, pi, weights)
        return log_q_total

    def generate_dream_proposal(
        self,
        current_x: dict,
        cat_weights: tuple[float, float, float],
    ) -> dict:
        if self._Z is None or self._Z.shape[0] < self.dream.min_pop:
            soft = (_FLOAT_WEIGHTS[HMMState.EXPLORE], _INT_WEIGHTS[HMMState.EXPLORE], cat_weights)
            return self.generate_proposal(current_x, soft)

        x = self.config_to_vector(current_x)
        d = x.shape[0]

        crossover = np.random.rand(d) < self.dream.cr
        if not np.any(crossover):
            crossover[int(np.random.randint(0, d))] = True
        d_eff = int(np.sum(crossover))

        n_pairs = max(1, self.dream.n_pairs)
        delta = np.zeros(d, dtype=np.float64)
        pair_idx = self._dream_pair_indices(x, n_pairs)
        for k in range(n_pairs):
            i0, i1 = int(pair_idx[2 * k]), int(pair_idx[2 * k + 1])
            delta += self._Z[i0] - self._Z[i1]

        gamma = 2.38 / np.sqrt(2.0 * n_pairs * d_eff)
        if np.random.rand() < self.dream.gamma1_prob:
            gamma = 1.0

        x_new = x.copy()
        jitter_scale = self.dream.eps * self._ranges
        for j in range(d):
            if crossover[j]:
                noise = np.random.normal(0.0, jitter_scale[j])
                x_new[j] = x[j] + gamma * delta[j] + noise

        out = self.vector_to_config(x_new, current_x)
        soft_cat: SoftWeights = (
            _FLOAT_WEIGHTS[HMMState.EXPLOIT],
            _INT_WEIGHTS[HMMState.EXPLOIT],
            cat_weights,
        )
        for pi in self._param_info:
            if pi["type"] == "categorical":
                out[pi["name"]] = self._sample_categorical(current_x[pi["name"]], pi, soft_cat)
        return out

    def _archive_boltzmann_weights(self) -> np.ndarray:
        n = len(self._archive)
        if n == 0:
            return np.array([])
        losses = np.array([entry[1] for entry in self._archive])
        l_min, l_max = float(losses.min()), float(losses.max())
        if l_max - l_min < 1e-10:
            return np.ones(n) / n
        normalized = (losses - l_min) / (l_max - l_min)
        log_w = -normalized / self._kde_tau
        log_w -= _logsumexp(log_w)
        return np.exp(log_w)

    def _kde_bandwidth(self, pi: dict) -> float:
        return 0.03 * pi["range"]

    def _boltzmann_probs(self, param_name: str, categories: list) -> np.ndarray:
        C = len(categories)
        history = self._category_history[param_name]
        scores = np.zeros(C)
        observed = np.zeros(C, dtype=bool)

        for i, cat in enumerate(categories):
            losses = history[cat]
            n = len(losses)
            if n >= 4:
                scores[i] = np.percentile(losses, 10)
                observed[i] = True
            elif n >= 1:
                scores[i] = min(losses)
                observed[i] = True

        if not np.any(observed):
            return np.ones(C) / C

        best_score = np.min(scores[observed])
        scores[~observed] = best_score - 0.3 * (abs(best_score) + self._EPS)

        s_min = scores.min()
        s_max = scores.max()
        s_range = s_max - s_min
        if s_range < self._EPS:
            return np.ones(C) / C

        normalized = (scores - s_min) / s_range
        tau = self.temperature + self._EPS
        neg_scaled = -normalized / tau
        log_Z = _logsumexp(neg_scaled)
        log_probs = neg_scaled - log_Z
        probs = np.exp(log_probs)
        prob_sum = probs.sum()
        if prob_sum < self._EPS:
            return np.ones(C) / C
        return probs / prob_sum

    def _sample_continuous(self, x_j: float, pi: dict, weights: SoftWeights) -> float:
        w_narrow, w_kde, w_wide = self._w("float", weights)
        lo, hi = pi["lo"], pi["hi"]
        name = pi["name"]

        r = np.random.rand()
        if r < w_narrow:
            sigma = pi["sigma"]
            a, b = (lo - x_j) / sigma, (hi - x_j) / sigma
            return _tn_rvs(a, b, loc=x_j, scale=sigma)
        if r < w_narrow + w_kde and len(self._archive) >= 3:
            bw = self._archive_boltzmann_weights()
            idx = int(np.random.choice(len(self._archive), p=bw))
            center = float(self._archive[idx][0][name])
            h = self._kde_bandwidth(pi)
            a, b = (lo - center) / h, (hi - center) / h
            return _tn_rvs(a, b, loc=center, scale=h)

        sigma_wide = pi["sigma_wide"]
        a, b = (lo - x_j) / sigma_wide, (hi - x_j) / sigma_wide
        return _tn_rvs(a, b, loc=x_j, scale=sigma_wide)

    def _sample_integer(self, x_j, pi: dict, weights: SoftWeights):
        if pi.get("log"):
            return self._sample_continuous(float(x_j), pi, weights)

        w_local, w_kde, w_global = self._w("int", weights)
        lo, hi = int(pi["lo"]), int(pi["hi"])
        x_j = int(x_j)
        name = pi["name"]

        r = np.random.rand()
        if r < w_local:
            candidates = [v for v in [x_j - 1, x_j, x_j + 1] if lo <= v <= hi]
            return int(np.random.choice(candidates))
        if r < w_local + w_kde and len(self._archive) >= 3:
            bw = self._archive_boltzmann_weights()
            idx = int(np.random.choice(len(self._archive), p=bw))
            center = float(self._archive[idx][0][name])
            h = max(self._kde_bandwidth(pi), 1.0)
            val = _tn_rvs(
                (lo - 0.5 - center) / h,
                (hi + 0.5 - center) / h,
                loc=center,
                scale=h,
            )
            return int(np.clip(round(val), lo, hi))
        return int(np.random.randint(lo, hi + 1))

    def _sample_categorical(self, x_j, pi: dict, weights: SoftWeights):
        w_stay, w_uniform, w_history = self._w("cat", weights)
        categories = pi["values"]
        C = pi["n_categories"]
        name = pi["name"]

        p_boltz = self._boltzmann_probs(name, categories)
        pmf = np.zeros(C)
        for i, cat in enumerate(categories):
            stay = w_stay if cat == x_j else 0.0
            uniform = w_uniform / C
            history = w_history * p_boltz[i]
            pmf[i] = stay + uniform + history

        pmf_sum = pmf.sum()
        if pmf_sum > 0:
            pmf /= pmf_sum
        else:
            pmf = np.ones(C) / C

        idx = np.random.choice(C, p=pmf)
        return categories[idx]

    def _log_q_continuous(
        self, x_from: float, x_to: float, pi: dict, weights: SoftWeights
    ) -> float:
        w_narrow, w_kde, w_wide = self._w("float", weights)
        lo, hi = pi["lo"], pi["hi"]
        name = pi["name"]
        log_components = []

        if w_narrow > 0:
            sigma = pi["sigma"]
            a, b = (lo - x_from) / sigma, (hi - x_from) / sigma
            log_g = _tn_logpdf(x_to, a, b, loc=x_from, scale=sigma)
            log_components.append(np.log(w_narrow + self._EPS) + log_g)

        if w_kde > 0 and len(self._archive) >= 3:
            bw = self._archive_boltzmann_weights()
            h = self._kde_bandwidth(pi)
            n_use = min(len(self._archive), 50)
            kde_logpdfs = []
            for idx in range(n_use):
                center = float(self._archive[idx][0][name])
                a, b = (lo - center) / h, (hi - center) / h
                log_tn = _tn_logpdf(x_to, a, b, loc=center, scale=h)
                kde_logpdfs.append(np.log(bw[idx] + self._EPS) + log_tn)
            log_kde = _logsumexp(np.array(kde_logpdfs))
            log_components.append(np.log(w_kde + self._EPS) + log_kde)

        if w_wide > 0:
            sigma_wide = pi["sigma_wide"]
            a, b = (lo - x_from) / sigma_wide, (hi - x_from) / sigma_wide
            log_g_wide = _tn_logpdf(x_to, a, b, loc=x_from, scale=sigma_wide)
            log_components.append(np.log(w_wide + self._EPS) + log_g_wide)

        return float(_logsumexp(np.array(log_components)))

    def _log_q_integer(
        self, x_from, x_to, pi: dict, weights: SoftWeights
    ) -> float:
        if pi.get("log"):
            return self._log_q_continuous(float(x_from), float(x_to), pi, weights)

        w_local, w_kde, w_global = self._w("int", weights)
        lo, hi = int(pi["lo"]), int(pi["hi"])
        x_from = int(x_from)
        x_to = int(x_to)
        name = pi["name"]
        local_candidates = [v for v in [x_from - 1, x_from, x_from + 1] if lo <= v <= hi]
        n_local = len(local_candidates)
        n_global = hi - lo + 1

        log_components = []
        if x_to in local_candidates:
            log_components.append(np.log(w_local + self._EPS) - np.log(n_local))
        else:
            log_components.append(-np.inf)

        if w_kde > 0 and len(self._archive) >= 3:
            bw = self._archive_boltzmann_weights()
            h = max(self._kde_bandwidth(pi), 1.0)
            n_use = min(len(self._archive), 50)
            p_kde = 0.0
            for idx in range(n_use):
                center = float(self._archive[idx][0][name])
                p_val = _std_cdf((x_to + 0.5 - center) / h) - _std_cdf(
                    (x_to - 0.5 - center) / h
                )
                p_kde += bw[idx] * max(p_val, 0.0)
            if p_kde > 1e-30:
                log_components.append(np.log(w_kde + self._EPS) + np.log(p_kde))
            else:
                log_components.append(-np.inf)

        log_components.append(np.log(w_global + self._EPS) - np.log(n_global))
        return float(_logsumexp(np.array(log_components)))

    def _log_q_categorical(
        self, x_from, x_to, pi: dict, weights: SoftWeights
    ) -> float:
        w_stay, w_uniform, w_history = self._w("cat", weights)
        categories = pi["values"]
        C = pi["n_categories"]
        name = pi["name"]

        p_boltz = self._boltzmann_probs(name, categories)
        idx_to = pi["val_to_idx"][x_to]

        log_components = []
        if x_to == x_from:
            log_components.append(np.log(w_stay + self._EPS))
        else:
            log_components.append(-np.inf)
        log_components.append(np.log(w_uniform / C + self._EPS))
        log_components.append(np.log(w_history * p_boltz[idx_to] + self._EPS))
        return float(_logsumexp(np.array(log_components)))


# ---------------------------------------------------------------------------
# DREAM MCMC chain (always soft)
# ---------------------------------------------------------------------------

class DreamMCMCChain:
    """MCMC chain with DREAM(ZS) + soft factorized composite kernel."""

    def __init__(
        self,
        chain_id: int,
        x0: dict,
        loss0: float,
        proposal_gen: DREAMProposalGenerator,
        hmm: _SoftBaumWelchHMM,
        T_mcmc: float = 1.0,
        T_min: float | None = None,
        scale_factor: float = 1.0,
        p_cat_step: float = 0.30,
        anneal_T: bool = True,
        rejection_streak: int = 10,
        p_dream: float = 0.5,
    ):
        self.chain_id = chain_id
        self.current_x = copy.deepcopy(x0)
        self.current_loss = loss0
        self.current_loss_old_for_hmm = loss0
        self.best_x = copy.deepcopy(x0)
        self.best_loss = loss0
        self.T_mcmc_init = T_mcmc
        self.T_mcmc = T_mcmc
        self.T_min = T_min if T_min is not None else T_mcmc * 0.01
        self.scale_factor = scale_factor
        self.p_cat_step = p_cat_step
        self.anneal_T = anneal_T
        self.p_dream = p_dream

        self.proposal_gen = proposal_gen
        self.hmm = hmm
        self.state = HMMState.EXPLORE

        self._observations: list[float] = []
        self._rejection_streak: int = 0
        self._stagnation_limit: int = rejection_streak
        self._last_kernel: str = "factorized"
        self._last_soft_weights: SoftWeights | None = None
        self._steps_since_improve: int = 0

    def reset_at(self, x0: dict, loss0: float):
        self.current_x = copy.deepcopy(x0)
        self.current_loss = loss0
        self.best_x = copy.deepcopy(x0)
        self.best_loss = loss0
        self.hmm.reset()
        self.state = HMMState.EXPLORE
        self._observations = []
        self._rejection_streak = 0
        self._steps_since_improve = 0

    def _acceptance_probability_dream(self, loss_prime: float, T_effective: float) -> float:
        delta_normalized = (loss_prime - self.current_loss) / (self.scale_factor + 1e-8)
        log_likelihood_ratio = -delta_normalized / (T_effective + 1e-100)
        return float(np.exp(min(log_likelihood_ratio, 0.0)))

    def _acceptance_probability_soft(
        self,
        x_prime: dict,
        loss_prime: float,
        T_effective: float,
        weights: SoftWeights,
    ) -> float:
        delta_normalized = (loss_prime - self.current_loss) / (self.scale_factor + 1e-8)
        log_likelihood_ratio = -delta_normalized / (T_effective + 1e-100)

        log_q_reverse = self.proposal_gen.log_proposal_density(
            x_from=x_prime, x_to=self.current_x, weights=weights
        )
        log_q_forward = self.proposal_gen.log_proposal_density(
            x_from=self.current_x, x_to=x_prime, weights=weights
        )
        log_hastings_ratio = log_q_reverse - log_q_forward
        log_alpha = log_likelihood_ratio + log_hastings_ratio
        return float(np.exp(min(log_alpha, 0.0)))

    def step(
        self, objective_func, progress: float = 0.0, is_burnin: bool = False
    ) -> tuple[dict, float, bool]:
        if is_burnin:
            self.T_mcmc = self.T_mcmc_init
        elif self.anneal_T:
            self.T_mcmc = self.T_min + (self.T_mcmc_init - self.T_min) * (1.0 - progress)
        else:
            self.T_mcmc = self.T_mcmc_init

        if len(self._observations) >= 2:
            self.state = self.hmm.observe(self._observations[-self.hmm.window:])
        else:
            self.state = self.hmm.state

        if not is_burnin and self._rejection_streak >= self._stagnation_limit:
            if self.state != HMMState.TRAPPED:
                self.state = HMMState.TRAPPED
                self.hmm.force_state(HMMState.TRAPPED)
            n_over = self._rejection_streak - self._stagnation_limit + 1
            boost_factor = 1.0 + min(n_over * 3.0, 50.0)
            self.T_mcmc = self.T_mcmc_init * boost_factor

        T_effective = self.T_mcmc

        posterior = self.hmm.posterior()
        soft_weights = mix_soft_weights(posterior)
        self._last_soft_weights = soft_weights
        _, _, cw = soft_weights

        dream_ready = (
            isinstance(self.proposal_gen, DREAMProposalGenerator)
            and self.proposal_gen.population_size >= self.proposal_gen.dream.min_pop
        )

        use_dream_step = dream_ready and np.random.rand() < self.p_dream
        cat_only = (not use_dream_step) and (np.random.rand() < self.p_cat_step)

        if use_dream_step:
            self._last_kernel = "dream"
            x_prime = self.proposal_gen.generate_dream_proposal(self.current_x, cw)
        else:
            self._last_kernel = "factorized"
            x_prime = self.proposal_gen.generate_proposal(
                self.current_x, soft_weights, categorical_only=cat_only
            )

        loss_prime = objective_func(x_prime)
        self.current_loss_old_for_hmm = self.current_loss

        if use_dream_step:
            alpha = self._acceptance_probability_dream(loss_prime, T_effective)
        else:
            alpha = self._acceptance_probability_soft(
                x_prime, loss_prime, T_effective, soft_weights
            )

        accepted = False
        if np.random.rand() < alpha:
            self.current_x = copy.deepcopy(x_prime)
            self.current_loss = loss_prime
            self._rejection_streak = 0
            accepted = True
        else:
            self._rejection_streak += 1

        delta_loss = loss_prime - self.current_loss_old_for_hmm
        O_t = float(delta_loss / (self.scale_factor + 1e-8))
        self._observations.append(O_t)

        if loss_prime < self.best_loss:
            self.best_x = copy.deepcopy(x_prime)
            self.best_loss = loss_prime
            self._steps_since_improve = 0
        else:
            self._steps_since_improve += 1

        return x_prime, loss_prime, accepted


# ---------------------------------------------------------------------------
# DREAM orchestrator (rescue + collapse reseed only)
# ---------------------------------------------------------------------------

class DreamOrchestrator:
    """Orchestrator with stagnation-based rescue and collapse reseed."""

    def __init__(
        self,
        chains: list[DreamMCMCChain],
        proposal_gen: DREAMProposalGenerator,
        sobol_init: SobolInitializer,
        dict_to_optimize: dict,
        objective_func,
        clone_noise: float = 0.05,
        collapse_threshold: float = 0.01,
        reseed_fraction: float = 0.3,
        orchestrate_patience: int = 15,
    ):
        self.chains = chains
        self.proposal_gen = proposal_gen
        self.sobol_init = sobol_init
        self.dict_to_optimize = dict_to_optimize
        self.objective_func = objective_func
        self.clone_noise = clone_noise
        self.collapse_threshold = collapse_threshold
        self.reseed_fraction = reseed_fraction
        self.orchestrate_patience = orchestrate_patience

    def _is_rescue_candidate(self, chain: DreamMCMCChain, min_age: int) -> bool:
        stagnant = chain._steps_since_improve >= self.orchestrate_patience
        return (
            (chain.state == HMMState.TRAPPED or stagnant)
            and len(chain._observations) >= min_age
        )

    def orchestrate(self, data: list, budget: int = 1) -> list[tuple[dict, float]]:
        new_evals = []
        progress = len(data) / max(budget, 1)

        min_age = max(self.chains[0].hmm.window, 10) if self.chains else 10
        rescue_ids = [
            i for i, c in enumerate(self.chains)
            if self._is_rescue_candidate(c, min_age)
        ]
        if rescue_ids:
            exploit_chains = [
                i for i, c in enumerate(self.chains) if c.state == HMMState.EXPLOIT
            ]
            if exploit_chains:
                donor_idx = min(exploit_chains, key=lambda i: self.chains[i].best_loss)
            else:
                donor_idx = min(range(len(self.chains)), key=lambda i: self.chains[i].best_loss)

            donor = self.chains[donor_idx]
            for idx in rescue_ids:
                if idx == donor_idx:
                    continue
                new_x = self._add_noise(donor.best_x)
                new_loss = self.objective_func(new_x)
                new_evals.append((new_x, new_loss))
                self.chains[idx].reset_at(new_x, new_loss)
                self.proposal_gen.update_category_history(new_x, new_loss)

        if progress < 0.5 and self._detect_collapse():
            n_reseed = max(1, int(len(self.chains) * self.reseed_fraction))
            sorted_chains = sorted(
                range(len(self.chains)), key=lambda i: self.chains[i].best_loss
            )
            worst_ids = sorted_chains[-n_reseed:]
            new_configs = self.sobol_init.generate(
                n_reseed, seed=np.random.randint(0, 2**31)
            )
            for i, idx in enumerate(worst_ids):
                new_loss = self.objective_func(new_configs[i])
                new_evals.append((new_configs[i], new_loss))
                self.chains[idx].reset_at(new_configs[i], new_loss)
                self.proposal_gen.update_category_history(new_configs[i], new_loss)

        return new_evals

    def _add_noise(self, config: dict) -> dict:
        noisy = {}
        for name, info in self.dict_to_optimize.items():
            p_type = info["type"]
            values = info["values"]
            val = config[name]

            if p_type == "float":
                lo, hi = float(values[0]), float(values[1])
                if info.get("log"):
                    lo10, hi10 = float(np.log10(lo)), float(np.log10(hi))
                    sigma = (hi10 - lo10) * self.clone_noise
                    noisy[name] = float(np.clip(val + np.random.normal(0, sigma), lo10, hi10))
                else:
                    sigma = (hi - lo) * self.clone_noise
                    noisy[name] = float(np.clip(val + np.random.normal(0, sigma), lo, hi))
            elif p_type == "int":
                real_lo, real_hi = int(values[0]), int(values[1])
                if info.get("log"):
                    pi = {"log": True, "real_lo": real_lo, "real_hi": real_hi}
                    real = _int_real_value(val, pi)
                    step = int(np.random.choice([-1, 0, 1]))
                    noisy_real = int(np.clip(real + step, real_lo, real_hi))
                    noisy[name] = float(np.log10(max(noisy_real, 1)))
                else:
                    step = np.random.choice([-1, 0, 1])
                    noisy[name] = int(np.clip(int(val) + step, real_lo, real_hi))
            elif p_type == "categorical":
                if np.random.rand() < 0.2 and len(values) > 1:
                    choices = [v for v in values if v != val]
                    noisy[name] = np.random.choice(choices)
                else:
                    noisy[name] = val
            else:
                noisy[name] = val
        return noisy

    def _detect_collapse(self) -> bool:
        losses = [c.best_loss for c in self.chains]
        if len(losses) < 2:
            return False
        spread = np.std(losses) / (np.abs(np.mean(losses)) + 1e-30)
        return spread < self.collapse_threshold


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class HMM_MCMC_DREAM:
    """H-MCMC with DREAM(ZS), soft HMM filtering, and Baum-Welch transitions."""

    def __init__(
        self,
        objective_func,
        budget: int,
        dict_to_optimize: dict,
        n_init: int = 16,
        n_chains: int = 4,
        orchestrate_every: int = 5,
        orchestrate_patience: int = 15,
        T_mcmc: float = 1.0,
        T_min: float | None = None,
        anneal_T: bool = True,
        burnin_fraction: float = 0.10,
        sigma_fraction: float = 0.10,
        temperature: float = 0.60,
        hmm_window: int = 8,
        hmm_obs_epsilon: float = 1e-8,
        hmm_lambda_noise: float = 0.01,
        clone_noise: float = 0.05,
        wide_sigma_fraction: float = 0.40,
        p_cat_step: float = 0.30,
        kde_tau: float = 0.05,
        rejection_streak: int = 10,
        p_dream: float = 0.5,
        dream_n_pairs: int = 1,
        dream_cr: float = 0.9,
        dream_gamma1_prob: float = 0.1,
        dream_eps: float = 1e-3,
        dream_min_pop: int = 4,
        dream_z_archive_max: int = 512,
        dream_z_pop_cap: int = 64,
        dream_diversity_frac: float = 0.0,
        bw_refit_every: int = 5,
        bw_min_obs: int = 12,
        bw_n_em_iters: int = 3,
        bw_prior_strength: float = 25.0,
        bw_exploit_prior_scale: float = 3.0,
        bw_max_len: int = 64,
        show_progress: bool = True,
        progress_desc: str | None = None,
        verbose_history: bool = True,
        **kwargs,
    ):
        _warn_deprecated_kwargs(kwargs)

        self.dict_to_optimize = dict_to_optimize
        self._needs_decode = any(
            info.get("log")
            for info in dict_to_optimize.values()
            if info.get("type") in ("float", "int")
        )
        self._raw_objective_func = objective_func
        self.objective_func = self._wrap_objective(objective_func)
        self.budget = budget
        self.n_init = n_init
        self.n_chains = n_chains
        self.orchestrate_every = orchestrate_every
        self.orchestrate_patience = orchestrate_patience
        self.T_mcmc = T_mcmc
        self.T_min = T_min
        self.anneal_T = anneal_T
        self.burnin_fraction = burnin_fraction
        self.p_dream = p_dream
        self.dream_config = DREAMConfig(
            n_pairs=dream_n_pairs,
            cr=dream_cr,
            gamma1_prob=dream_gamma1_prob,
            eps=dream_eps,
            min_pop=dream_min_pop,
            z_pop_cap=dream_z_pop_cap,
            diversity_frac=dream_diversity_frac,
        )
        self.dream_z_archive_max = dream_z_archive_max
        self.bw_refit_every = bw_refit_every
        self.bw_min_obs = bw_min_obs
        self.bw_n_em_iters = bw_n_em_iters
        self.bw_prior_strength = bw_prior_strength
        self.bw_exploit_prior_scale = bw_exploit_prior_scale
        self.bw_max_len = bw_max_len
        self.show_progress = show_progress
        self.progress_desc = progress_desc
        self.verbose_history = verbose_history

        self._sigma_fraction = sigma_fraction
        self._temperature = temperature
        self._hmm_window = hmm_window
        self._hmm_obs_epsilon = hmm_obs_epsilon
        self._hmm_lambda_noise = hmm_lambda_noise
        self._clone_noise = clone_noise
        self._wide_sigma_fraction = wide_sigma_fraction
        self._p_cat_step = p_cat_step
        self._kde_tau = kde_tau
        self._stagnation_limit = rejection_streak

        self.data: list[tuple[dict, float]] = []
        self._chains: list[DreamMCMCChain] = []
        self._proposal_gen: DREAMProposalGenerator | None = None
        self._sobol_init: SobolInitializer | None = None
        self._orchestrator: DreamOrchestrator | None = None
        self.history_table: list[dict] = []

    def decode_config(self, cfg: dict) -> dict:
        return decode_config(cfg, self.dict_to_optimize)

    def _wrap_objective(self, objective_func):
        if not self._needs_decode:
            return objective_func

        def wrapped(cfg: dict) -> float:
            return objective_func(self.decode_config(cfg))

        return wrapped

    def _decode_best(self, result: tuple[dict, float]) -> tuple[dict, float]:
        cfg, loss = result
        return self.decode_config(cfg), loss

    def reset(self):
        self.data = []
        self._chains = []
        self._proposal_gen = None
        self._sobol_init = None
        self._orchestrator = None
        self.history_table = []

    @staticmethod
    def _compute_scale(losses: list[float]) -> float:
        n = len(losses)
        if n == 0:
            return 1.0
        if n == 1:
            return max(abs(losses[0]), 1.0)
        arr = np.array(losses)
        if n >= 10:
            q90, q10 = np.percentile(arr, [90, 10])
            scale = q90 - q10
        else:
            scale = float(arr.max() - arr.min())
        return max(scale, 1e-5)

    def _make_proposal_generator(self) -> DREAMProposalGenerator:
        return DREAMProposalGenerator(
            self.dict_to_optimize,
            sigma_fraction=self._sigma_fraction,
            temperature=self._temperature,
            wide_sigma_fraction=self._wide_sigma_fraction,
            kde_tau=self._kde_tau,
            dream=self.dream_config,
            z_archive_max=self.dream_z_archive_max,
        )

    def _make_hmm_controller(self) -> _SoftBaumWelchHMM:
        return _SoftBaumWelchHMM(
            window=self._hmm_window,
            obs_epsilon=self._hmm_obs_epsilon,
            lambda_noise=self._hmm_lambda_noise,
            refit_every=self.bw_refit_every,
            min_obs=self.bw_min_obs,
            n_em_iters=self.bw_n_em_iters,
            prior_strength=self.bw_prior_strength,
            exploit_prior_scale=self.bw_exploit_prior_scale,
            bw_max_len=self.bw_max_len,
        )

    def main_loop(self):
        self._sobol_init = SobolInitializer(self.dict_to_optimize)
        self._proposal_gen = self._make_proposal_generator()

        init_configs = self._sobol_init.generate(self.n_init)
        pbar = tqdm(
            total=self.budget,
            desc=self.progress_desc or "H-MCMC-DREAM",
            disable=not self.show_progress,
        )

        init_scores = []
        for cfg in init_configs:
            if len(self.data) >= self.budget:
                break
            score = self.objective_func(cfg)
            self.data.append((cfg, score))
            init_scores.append((cfg, score))
            self._proposal_gen.update_category_history(cfg, score)
            pbar.update(1)

        if len(self.data) >= self.budget:
            pbar.close()
            return self._decode_best(min(self.data, key=lambda x: x[1]))

        losses = [score for _, score in init_scores]
        scale_factor = self._compute_scale(losses)

        scaled_T_mcmc = self.T_mcmc
        scaled_T_min = self.T_min if self.T_min is not None else None
        self._proposal_gen.temperature = self._temperature

        init_scores.sort(key=lambda x: x[1])
        k = min(self.n_chains, len(init_scores))
        best_inits = init_scores[:k]

        self._chains = []
        for i, (cfg, score) in enumerate(best_inits):
            hmm = self._make_hmm_controller()
            chain = DreamMCMCChain(
                chain_id=i,
                x0=cfg,
                loss0=score,
                proposal_gen=self._proposal_gen,
                hmm=hmm,
                T_mcmc=scaled_T_mcmc,
                T_min=scaled_T_min,
                scale_factor=scale_factor,
                p_cat_step=self._p_cat_step,
                anneal_T=self.anneal_T,
                rejection_streak=self._stagnation_limit,
                p_dream=self.p_dream,
            )
            self._chains.append(chain)

        self._orchestrator = DreamOrchestrator(
            chains=self._chains,
            proposal_gen=self._proposal_gen,
            sobol_init=self._sobol_init,
            dict_to_optimize=self.dict_to_optimize,
            objective_func=self.objective_func,
            clone_noise=self._clone_noise,
            orchestrate_patience=self.orchestrate_patience,
        )

        step_counter = 0

        while len(self.data) < self.budget:
            self._proposal_gen.refresh_population(self._chains)

            for chain in self._chains:
                if len(self.data) >= self.budget:
                    break

                progress = len(self.data) / self.budget
                is_burnin = progress < self.burnin_fraction
                cfg, loss, accepted = chain.step(
                    self.objective_func, progress=progress, is_burnin=is_burnin
                )
                self.data.append((cfg, loss))
                self._proposal_gen.update_category_history(cfg, loss)

                kernel = chain._last_kernel
                self.history_table.append({
                    "Eval": len(self.data),
                    "Chain": chain.chain_id,
                    "State": chain.state.name,
                    "Kernel": kernel,
                    "Loss": float(loss),
                    "Accepted": accepted,
                    "Rej_Streak": chain._rejection_streak,
                })
                pbar.update(1)

            step_counter += 1

            if step_counter % self.orchestrate_every == 0:
                if len(self.data) < self.budget:
                    new_evals = self._orchestrator.orchestrate(
                        self.data, budget=self.budget
                    )
                    for cfg, loss in new_evals:
                        if len(self.data) >= self.budget:
                            break
                        self.data.append((cfg, loss))
                        self.history_table.append({
                            "Eval": len(self.data),
                            "Chain": "ORCHESTRATOR",
                            "State": "RESET/CLONE",
                            "Kernel": "orchestrator",
                            "Loss": float(loss),
                            "Accepted": True,
                            "Rej_Streak": 0,
                        })
                        pbar.update(1)

                if len(self.data) / self.budget < 0.5:
                    all_losses = [s for _, s in self.data]
                    scale_factor = self._compute_scale(all_losses)
                    for chain in self._chains:
                        chain.scale_factor = scale_factor

        pbar.close()
        if self.verbose_history:
            try:
                import pandas as pd

                df = pd.DataFrame(self.history_table)
                print("\n" + "=" * 60)
                print("HMM MCMC DREAM State History (last run):")
                with pd.option_context("display.max_rows", 200, "display.max_columns", None):
                    print(df)
                print("=" * 60 + "\n")
            except ImportError:
                pass

        return self._decode_best(min(self.data, key=lambda x: x[1]))


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_check_no_hmm_mcmc_imports() -> None:
    import pathlib
    import re

    src = pathlib.Path(__file__).read_text(encoding="utf-8")
    import_lines = [
        ln.strip()
        for ln in src.splitlines()
        if ln.strip().startswith(("import ", "from "))
    ]
    forbidden = re.compile(
        r"^(?:from|import)\s+hpo_rl\.baselines\.HMM_MCMC(?:_TEST)?(?:\b|\.)"
    )
    for line in import_lines:
        assert not forbidden.match(line), f"forbidden import found: {line}"
    print("[self-check] no HMM_MCMC* imports in source: OK")


def _self_check_soft_filter() -> None:
    ctrl = _SoftBaumWelchHMM(min_obs=1000, refit_every=10**9)
    obs = [-0.01, -0.02, 0.15, 0.12, 0.08]
    for i in range(2, len(obs) + 1):
        ctrl.observe(obs[:i])
    post = ctrl.posterior()
    assert post.shape == (3,)
    assert np.allclose(post.sum(), 1.0, atol=1e-6)
    assert np.all(post >= 0)
    print(f"[self-check] soft filter posterior={post}, state={ctrl.state.name}")
    print("[self-check] soft forward filter: OK")


def _self_check_soft_weights() -> None:
    post = np.array([0.7, 0.2, 0.1])
    fw, iw, cw = mix_soft_weights(post)
    assert len(fw) == len(iw) == len(cw) == 3
    assert all(w >= 0 for w in fw)
    print(f"[self-check] soft weights fw={fw}")
    print("[self-check] soft weight mixing: OK")


def _self_check_dream_vector_roundtrip() -> None:
    space = {
        "x0": {"type": "float", "values": [-5.0, 5.0]},
        "x1": {"type": "float", "values": [0.0, 1.0]},
    }
    gen = DREAMProposalGenerator(space)
    cfg = {"x0": 1.0, "x1": 0.5}
    vec = gen.config_to_vector(cfg)
    back = gen.vector_to_config(vec, cfg)
    assert abs(back["x0"] - 1.0) < 1e-9
    assert abs(back["x1"] - 0.5) < 1e-9
    print("[self-check] DREAM vector round-trip: OK")


def _self_check_dream_population() -> None:
    space = {"x": {"type": "float", "values": [-10.0, 10.0]}}
    gen = DREAMProposalGenerator(space, dream=DREAMConfig(min_pop=2, z_pop_cap=8))
    for v in [-8.0, -4.0, 0.0, 4.0, 8.0, 2.0, -2.0, 6.0]:
        gen.update_category_history({"x": v}, float(v * v))

    assert len(gen._z_archive) >= 8, "Z-archive should grow via update_category_history"

    class _FakeChain:
        def __init__(self, x):
            self.current_x = {"x": x}

    gen.refresh_population([_FakeChain(-1.0), _FakeChain(1.0)])
    assert gen.population_size >= 2
    assert gen._Z is not None
    deltas = gen._Z[:, 0] - gen._Z[0, 0]
    assert float(np.std(deltas)) > 1e-6, "DREAM Z should have non-trivial spread"

    prop = gen.generate_dream_proposal({"x": 0.0}, (0.1, 0.4, 0.5))
    assert "x" in prop
    assert -10.0 <= prop["x"] <= 10.0
    print("[self-check] DREAM population + Z-archive: OK")


def _self_check_soft_consistency() -> None:
    space = {
        "x0": {"type": "float", "values": [-5.0, 5.0]},
        "x1": {"type": "float", "values": [0.0, 1.0]},
    }
    gen = DREAMProposalGenerator(space)
    for i in range(12):
        cfg = {"x0": float(i % 5 - 2), "x1": float(i % 10) / 10.0}
        gen.update_category_history(cfg, float(i))

    post = np.array([0.05, 0.85, 0.10])
    soft = mix_soft_weights(post)
    current = {"x0": 0.0, "x1": 0.5}

    np.random.seed(0)
    changed_dims = 0
    for _ in range(30):
        prop = gen.generate_proposal(current, soft)
        if prop["x0"] != current["x0"]:
            changed_dims += 1
        if prop["x1"] != current["x1"]:
            changed_dims += 1

    assert changed_dims > 0, "soft path should perturb continuous dims"

    prop = gen.generate_proposal(current, soft)
    log_fwd = gen.log_proposal_density(current, prop, soft)
    log_rev = gen.log_proposal_density(prop, current, soft)
    assert np.isfinite(log_fwd), f"forward log_q not finite: {log_fwd}"
    assert np.isfinite(log_rev), f"reverse log_q not finite: {log_rev}"
    print("[self-check] soft-path density consistency: OK")


def _self_check_diversity_injection() -> None:
    space = {
        "x0": {"type": "float", "values": [-10.0, 10.0]},
        "x1": {"type": "float", "values": [-1.0, 1.0]},
    }
    clustered = [(-9.0, -0.9), (-8.8, -0.85), (-9.1, -0.88)]

    gen_base = DREAMProposalGenerator(
        space, dream=DREAMConfig(min_pop=2, z_pop_cap=16, diversity_frac=0.0)
    )
    gen_div = DREAMProposalGenerator(
        space, dream=DREAMConfig(min_pop=2, z_pop_cap=16, diversity_frac=0.5)
    )
    for cfg in [{"x0": a, "x1": b} for a, b in clustered]:
        loss = float((cfg["x0"] + 10) ** 2)
        gen_base.update_category_history(cfg, loss)
        gen_div.update_category_history(cfg, loss)

    class _FakeChain:
        def __init__(self, x0, x1):
            self.current_x = {"x0": x0, "x1": x1}

    chains = [_FakeChain(-9.0, -0.9), _FakeChain(-8.9, -0.88)]
    np.random.seed(0)
    gen_base.refresh_population(chains)
    np.random.seed(0)
    gen_div.refresh_population(chains)

    assert gen_base._Z is not None and gen_div._Z is not None
    spread_base = float(np.std(gen_base._Z))
    spread_div = float(np.std(gen_div._Z))
    assert spread_div > spread_base, f"diversity should widen Z: {spread_div} vs {spread_base}"

    x = gen_div.config_to_vector({"x0": -9.0, "x1": -0.9})
    pair_idx = gen_div._dream_pair_indices(x, n_pairs=2)
    deltas = gen_div._Z[pair_idx[0::2]] - gen_div._Z[pair_idx[1::2]]
    mean_abs_delta = float(np.mean(np.abs(deltas)))
    assert mean_abs_delta > 0.1, "diverse Z should yield non-trivial DREAM deltas"
    print(f"[self-check] diversity spread: base={spread_base:.3f}, div={spread_div:.3f}")
    print("[self-check] DREAM diversity injection: OK")


def _self_check_orchestrator_trigger() -> None:
    space = {
        "x": {"type": "float", "values": [-10.0, 10.0]},
        "y": {"type": "float", "values": [-10.0, 10.0]},
    }

    def flat_bowl(cfg):
        r2 = cfg["x"] ** 2 + cfg["y"] ** 2
        return float(1.0 + 0.01 * r2)

    np.random.seed(123)
    algo = HMM_MCMC_DREAM(
        objective_func=flat_bowl,
        budget=100,
        dict_to_optimize=space,
        n_init=6,
        n_chains=3,
        orchestrate_every=2,
        orchestrate_patience=6,
        burnin_fraction=0.0,
        T_mcmc=0.001,
        rejection_streak=100,
        p_dream=0.3,
        dream_min_pop=2,
        show_progress=False,
        verbose_history=False,
    )
    algo.main_loop()

    orch_rows = [r for r in algo.history_table if r["Chain"] == "ORCHESTRATOR"]
    assert len(orch_rows) >= 1, "orchestrator should fire on stagnation"
    for chain in algo._chains:
        assert chain._steps_since_improve < algo.orchestrate_patience, (
            "rescued chains should reset stagnation counter"
        )
    print(f"[self-check] orchestrator trigger: {len(orch_rows)} rescue eval(s)")
    print("[self-check] orchestrator stagnation trigger: OK")


def _self_check_collapse_reseed() -> None:
    """Collapse detection should trigger Sobol reseed when chains are too similar."""
    space = {"x": {"type": "float", "values": [-10.0, 10.0]}}

    def flat(cfg):
        return float(cfg["x"] ** 2)

    gen = DREAMProposalGenerator(space)
    sobol = SobolInitializer(space)

    chains: list[DreamMCMCChain] = []
    for i in range(3):
        hmm = _SoftBaumWelchHMM(min_obs=1000, refit_every=10**9)
        x0 = {"x": 0.5 + i * 1e-8}
        chain = DreamMCMCChain(
            chain_id=i,
            x0=x0,
            loss0=0.25,
            proposal_gen=gen,
            hmm=hmm,
            T_mcmc=1.0,
            scale_factor=1.0,
            p_cat_step=0.0,
            anneal_T=False,
        )
        chain.best_loss = 0.25 + i * 1e-12
        chains.append(chain)

    orch = DreamOrchestrator(
        chains=chains,
        proposal_gen=gen,
        sobol_init=sobol,
        dict_to_optimize=space,
        objective_func=flat,
        clone_noise=0.05,
        orchestrate_patience=100,
    )
    assert orch._detect_collapse(), "orchestrator should detect collapsed swarm"

    data = [(chains[0].current_x, 0.25)] * 20
    new_evals = orch.orchestrate(data, budget=100)
    assert len(new_evals) >= 1, "collapse reseed should produce new evals"
    print(f"[self-check] collapse reseed: {len(new_evals)} reseed eval(s)")
    print("[self-check] collapse reseed: OK")


def _self_check_schwefel_run() -> None:
    from hpo_rl.backends.function import OptimizationBenchmarkBackend

    backend = OptimizationBenchmarkBackend(
        function_name="schwefel", dimensions=2, noise_std=0
    )
    objective = backend.evaluate
    space = backend.hp_space

    algo = HMM_MCMC_DREAM(
        objective_func=objective,
        budget=40,
        dict_to_optimize=space,
        n_init=5,
        n_chains=2,
        orchestrate_every=1000,
        T_mcmc=0.01,
        sigma_fraction=0.0055,
        wide_sigma_fraction=0.5,
        temperature=0.3,
        hmm_window=4,
        burnin_fraction=0.0,
        p_cat_step=0.0,
        kde_tau=0.05,
        anneal_T=True,
        p_dream=0.5,
        dream_min_pop=2,
        bw_refit_every=3,
        bw_min_obs=8,
        show_progress=False,
        verbose_history=False,
    )
    best_cfg, best_loss = algo.main_loop()
    print(f"[self-check] Schwefel DREAM run: best_loss={best_loss:.4f}, cfg={best_cfg}")
    assert best_loss < 1e6, "Schwefel run returned unreasonable loss"
    print("[self-check] Schwefel DREAM run: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("HMM_MCMC_DREAM self-checks")
    print("=" * 60)
    _self_check_no_hmm_mcmc_imports()
    _self_check_soft_filter()
    _self_check_soft_weights()
    _self_check_dream_vector_roundtrip()
    _self_check_dream_population()
    _self_check_soft_consistency()
    _self_check_diversity_injection()
    _self_check_orchestrator_trigger()
    _self_check_collapse_reseed()
    _self_check_schwefel_run()
    print("=" * 60)
    print("All self-checks passed.")
    print("=" * 60)
