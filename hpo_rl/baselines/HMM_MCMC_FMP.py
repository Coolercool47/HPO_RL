"""H-MCMC-FMP: unified, seeded implementation of the Metropolis-type search with an
HMM regime controller, factorized mixture proposals (FMP) and an optional DREAM(ZS)
crossover kernel.

Every variant used in the rebuttal experiments is a *configuration* of this single
class, so ablations are single-factor:

    p_dream            0.0 -> FMP only; >0 -> DREAM kernel with that probability
    decoder            "viterbi" (hard state) | "soft" (forward-filter posterior mixing)
    explore_subsample  perturb only ceil(frac*P) continuous/int coords in EXPLORE
    controller         "hmm" | "fixed" | "random" | "rule"
    learn_transitions  online Baum-Welch on the transition matrix A
    learn_emissions    online Baum-Welch on emission means/variances as well
    dream_symmetric    True  -> DE move on float / log-int coords only (symmetric
                                kernel, likelihood-only acceptance is exact)
                       False -> legacy: int rounding + categorical resampling
    n_chains / orchestrate_every -> multi-chain orchestration (None/0 disables it)
    seed               every random draw goes through numpy.random.default_rng(seed)

Legacy modules HMM_MCMC_BW.py / HMM_MCMC_DREAM.py are kept unchanged as the record of
the submitted paper; this module supersedes them for new experiments.

Diagnostics kept on the instance after main_loop():
    data             list of (config_internal, loss) in evaluation order
    history          list of per-evaluation dict rows (state, kernel, T, scale,
                     posterior, observation, acceptance, config)
    A_history        list of (eval, chain, A) snapshots at every Baum-Welch refit
    emission_history list of (eval, chain, mu, sigma) if learn_emissions
    timing           dict with total wall time and objective time
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from scipy.stats.qmc import Sobol

# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)
_EPS = 1e-30


def _logsumexp(a) -> float:
    arr = np.asarray(a, dtype=np.float64)
    if arr.size == 0:
        return -np.inf
    m = arr.max()
    if not np.isfinite(m):
        return float(m)
    return float(m + np.log(np.sum(np.exp(arr - m))))


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


def _tn_rvs(rng: np.random.Generator, a: float, b: float, loc: float, scale: float) -> float:
    cdf_a, cdf_b = _std_cdf(a), _std_cdf(b)
    u = rng.uniform(cdf_a + 1e-15, cdf_b - 1e-15)
    return float(_std_ppf(u) * scale + loc)


def _tn_logpdf(x: float, a: float, b: float, loc: float, scale: float) -> float:
    z = (x - loc) / scale
    if z < a - 1e-10 or z > b + 1e-10:
        return -np.inf
    cdf_diff = _std_cdf(b) - _std_cdf(a)
    if cdf_diff < 1e-30:
        return -np.inf
    return float(-0.5 * z * z - _LOG_SQRT_2PI - math.log(scale) - math.log(cdf_diff))


# ---------------------------------------------------------------------------
# Search-space helpers (internal representation: log-scaled params live in log10)
# ---------------------------------------------------------------------------

def _param_record(name: str, info: dict) -> dict:
    p_type = info["type"]
    values = info["values"]
    rec: dict = {"name": name, "type": p_type, "values": list(values), "log": bool(info.get("log", False))}
    if p_type == "float":
        real_lo, real_hi = float(values[0]), float(values[1])
        if rec["log"]:
            lo, hi = math.log10(real_lo), math.log10(real_hi)
        else:
            lo, hi = real_lo, real_hi
        rec.update(real_lo=real_lo, real_hi=real_hi, lo=lo, hi=hi)
    elif p_type == "int":
        real_lo, real_hi = int(values[0]), int(values[1])
        if rec["log"]:
            lo, hi = math.log10(max(real_lo, 1)), math.log10(real_hi)
        else:
            lo, hi = float(real_lo), float(real_hi)
        rec.update(real_lo=real_lo, real_hi=real_hi, lo=lo, hi=hi)
    elif p_type == "categorical":
        rec["n_categories"] = len(values)
        rec["val_to_idx"] = {v: i for i, v in enumerate(values)}
    else:
        raise ValueError(f"Unknown parameter type {p_type!r} for {name!r}")
    if p_type in ("float", "int"):
        rec["range"] = rec["hi"] - rec["lo"]
    return rec


def decode_config(cfg: dict, params: list[dict]) -> dict:
    """Internal representation -> user-facing configuration."""
    out = {}
    for pi in params:
        name = pi["name"]
        v = cfg[name]
        if pi["type"] == "float":
            out[name] = float(10.0 ** float(v)) if pi["log"] else float(v)
        elif pi["type"] == "int":
            if pi["log"]:
                out[name] = int(np.clip(int(round(10.0 ** float(v))), pi["real_lo"], pi["real_hi"]))
            else:
                out[name] = int(np.clip(int(round(float(v))), pi["real_lo"], pi["real_hi"]))
        else:
            out[name] = v
    return out


class SobolInitializer:
    def __init__(self, params: list[dict]):
        self.params = params
        self.dim = len(params)

    def generate(self, n_points: int, seed: int) -> list[dict]:
        sampler = Sobol(d=self.dim, scramble=True, seed=int(seed))
        m = int(math.ceil(math.log2(max(n_points, 2))))
        raw = sampler.random_base2(m)[:n_points]
        return [self._unit_to_config(raw[i]) for i in range(n_points)]

    def _unit_to_config(self, u: np.ndarray) -> dict:
        cfg = {}
        for j, pi in enumerate(self.params):
            if pi["type"] == "float":
                cfg[pi["name"]] = float(pi["lo"] + pi["range"] * u[j])
            elif pi["type"] == "int":
                if pi["log"]:
                    cfg[pi["name"]] = float(pi["lo"] + pi["range"] * u[j])
                else:
                    cfg[pi["name"]] = int(round(pi["real_lo"] + (pi["real_hi"] - pi["real_lo"]) * u[j]))
            else:
                idx = min(int(u[j] * pi["n_categories"]), pi["n_categories"] - 1)
                cfg[pi["name"]] = pi["values"][idx]
        return cfg


# ---------------------------------------------------------------------------
# Regime states and proposal weight tables
# ---------------------------------------------------------------------------

class HMMState(IntEnum):
    EXPLOIT = 0
    EXPLORE = 1
    TRAPPED = 2


STATE_NAMES = ("EXPLOIT", "EXPLORE", "TRAPPED")

DEFAULT_FLOAT_WEIGHTS = {  # (narrow / local, archive-KDE, wide / global)
    HMMState.EXPLOIT: (0.90, 0.10, 0.00),
    HMMState.EXPLORE: (0.00, 0.80, 0.20),
    HMMState.TRAPPED: (0.00, 0.50, 0.50),
}
DEFAULT_CAT_WEIGHTS = {  # (stay, uniform, history)
    HMMState.EXPLOIT: (0.30, 0.00, 0.70),
    HMMState.EXPLORE: (0.10, 0.40, 0.50),
    HMMState.TRAPPED: (0.00, 0.80, 0.20),
}
DEFAULT_TRANSITION_PRIOR = (
    (0.60, 0.40, 0.00),
    (0.50, 0.50, 0.00),
    (0.60, 0.20, 0.20),
)
DEFAULT_EMISSION_MU = (-0.01, 0.10, 0.015)
DEFAULT_EMISSION_SIGMA = (0.06, 0.25, 0.020)
DEFAULT_INITIAL_STATE = (1.0, 0.0, 0.0)

Weights = tuple[tuple[float, float, float], tuple[float, float, float]]  # (float/int, categorical)


def _normalize_weight_table(table, default: dict) -> dict:
    """Accept dict keyed by HMMState, by state name, or by int; fall back to defaults."""
    if table is None:
        return dict(default)
    out = dict(default)
    for k, v in table.items():
        if isinstance(k, str):
            k = HMMState[k]
        out[HMMState(int(k))] = tuple(float(x) for x in v)
    return out


def _mix_weights(posterior: np.ndarray, float_table: dict, cat_table: dict) -> Weights:
    fw = np.zeros(3)
    cw = np.zeros(3)
    for s in HMMState:
        p = float(posterior[int(s)])
        if p > 0.0:
            fw += p * np.asarray(float_table[s], dtype=np.float64)
            cw += p * np.asarray(cat_table[s], dtype=np.float64)
    return (tuple(float(x) for x in fw), tuple(float(x) for x in cw))


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

class BaseController:
    """Maps the recent observation window to a regime posterior over the 3 states."""

    name = "base"

    def __init__(self, rng: np.random.Generator, window: int):
        self.rng = rng
        self.window = int(window)
        self.state = HMMState.EXPLOIT
        self.posterior = np.array(DEFAULT_INITIAL_STATE, dtype=np.float64)

    def reset(self) -> None:
        self.state = HMMState.EXPLOIT
        self.posterior = np.array(DEFAULT_INITIAL_STATE, dtype=np.float64)

    def force_state(self, s: HMMState) -> None:
        self.state = s
        self.posterior = np.zeros(3)
        self.posterior[int(s)] = 1.0

    def observe(self, obs_history: list[float]) -> HMMState:  # pragma: no cover - abstract
        raise NotImplementedError

    def refit_snapshot(self):
        return None


class FixedController(BaseController):
    name = "fixed"

    def __init__(self, rng, window, fixed_state: HMMState = HMMState.EXPLOIT):
        super().__init__(rng, window)
        self.fixed_state = HMMState(fixed_state)
        self.force_state(self.fixed_state)

    def reset(self):
        self.force_state(self.fixed_state)

    def observe(self, obs_history):
        self.force_state(self.fixed_state)
        return self.state


class RandomController(BaseController):
    """Uniformly random EXPLOIT / EXPLORE at every step (TRAPPED only via the counter)."""

    name = "random"

    def observe(self, obs_history):
        s = HMMState(int(self.rng.integers(0, 2)))
        self.force_state(s)
        return self.state


class RuleController(BaseController):
    """Threshold rule on the same normalized increments the HMM sees:
    EXPLOIT if mean(O_window) <= threshold else EXPLORE."""

    name = "rule"

    def __init__(self, rng, window, threshold: float = 0.05):
        super().__init__(rng, window)
        self.threshold = float(threshold)

    def observe(self, obs_history):
        if len(obs_history) < 2:
            return self.state
        w = np.asarray(obs_history[-self.window:], dtype=np.float64)
        s = HMMState.EXPLOIT if float(np.mean(w)) <= self.threshold else HMMState.EXPLORE
        self.force_state(s)
        return self.state


class HMMController(BaseController):
    """3-state HMM over normalized loss increments with robust Gaussian emissions.

    decoder="viterbi": hard MAP path over the last `window` observations.
    decoder="soft":    filtered posterior (forward pass) over the same window.
    Online Baum-Welch (MAP with Dirichlet / conjugate smoothing toward the prior)
    updates A and optionally the emission parameters.
    """

    name = "hmm"
    N = 3

    def __init__(
        self,
        rng,
        window=8,
        decoder="soft",
        lambda_noise=0.01,
        emission_mu=DEFAULT_EMISSION_MU,
        emission_sigma=DEFAULT_EMISSION_SIGMA,
        transition_prior=DEFAULT_TRANSITION_PRIOR,
        initial_state=DEFAULT_INITIAL_STATE,
        learn_transitions=True,
        learn_emissions=False,
        refit_every=5,
        min_obs=12,
        n_em_iters=3,
        prior_strength=25.0,
        exploit_prior_scale=3.0,
        bw_max_len=64,
        noise_support=20.0,
    ):
        super().__init__(rng, window)
        if decoder not in ("viterbi", "soft"):
            raise ValueError("decoder must be 'viterbi' or 'soft'")
        self.decoder = decoder
        self.lambda_noise = float(lambda_noise)
        self.c_noise = 1.0 / float(noise_support)
        self.mu0 = np.asarray(emission_mu, dtype=np.float64).copy()
        self.sigma0 = np.asarray(emission_sigma, dtype=np.float64).copy()
        self.A0 = np.asarray(transition_prior, dtype=np.float64).copy()
        self.pi0 = np.asarray(initial_state, dtype=np.float64).copy()
        if self.A0.shape != (3, 3) or self.mu0.shape != (3,) or self.sigma0.shape != (3,):
            raise ValueError("emission_mu/emission_sigma need 3 entries, transition_prior must be 3x3")
        self.learn_transitions = bool(learn_transitions)
        self.learn_emissions = bool(learn_emissions)
        self.refit_every = int(refit_every)
        self.min_obs = int(min_obs)
        self.n_em_iters = int(n_em_iters)
        self.prior_strength = float(prior_strength)
        self.exploit_prior_scale = float(exploit_prior_scale)
        self.bw_max_len = int(bw_max_len)
        self._init_params()

    def _init_params(self):
        self.A = self.A0.copy()
        self.mu = self.mu0.copy()
        self.sigma = self.sigma0.copy()
        self.pi = self.pi0.copy()
        self._buffer: list[float] = []
        self._n_seen = 0
        self._refits = 0
        self._last_refit_changed = False

    def reset(self):
        super().reset()
        self._init_params()

    # -- emissions ---------------------------------------------------------
    def _log_B(self, obs: np.ndarray, mu=None, sigma=None) -> np.ndarray:
        """(T, N) log emission matrix."""
        mu = self.mu if mu is None else mu
        sigma = self.sigma if sigma is None else sigma
        z = (obs[:, None] - mu[None, :]) / sigma[None, :]
        log_gauss = -0.5 * z * z - np.log(sigma)[None, :] - _LOG_SQRT_2PI
        dens = (1.0 - self.lambda_noise) * np.exp(log_gauss) + self.lambda_noise * self.c_noise
        return np.log(dens + 1e-300)

    # -- decoding ----------------------------------------------------------
    def observe(self, obs_history: list[float]) -> HMMState:
        if obs_history:
            self._buffer.append(float(obs_history[-1]))
            if len(self._buffer) > self.bw_max_len:
                self._buffer = self._buffer[-self.bw_max_len:]
            self._n_seen += 1
            self._maybe_refit()
        if len(obs_history) < 2:
            return self.state
        obs = np.asarray(obs_history[-self.window:], dtype=np.float64)
        log_B = self._log_B(obs)
        log_A = np.log(self.A + 1e-300)
        log_pi = np.log(self.pi + 1e-300)
        if self.decoder == "soft":
            post = self._forward_posterior(log_pi, log_A, log_B)
        else:
            s = self._viterbi(log_pi, log_A, log_B)
            post = np.zeros(self.N)
            post[s] = 1.0
        self.posterior = post
        self.state = HMMState(int(np.argmax(post)))
        return self.state

    def _forward_posterior(self, log_pi, log_A, log_B) -> np.ndarray:
        T = log_B.shape[0]
        la = log_pi + log_B[0]
        for t in range(1, T):
            la = np.array([_logsumexp(la + log_A[:, j]) for j in range(self.N)]) + log_B[t]
        la = la - _logsumexp(la)
        post = np.exp(la)
        return post / (post.sum() + _EPS)

    def _viterbi(self, log_pi, log_A, log_B) -> int:
        """Last state of the MAP path (only the last state is needed downstream)."""
        T = log_B.shape[0]
        delta = log_pi + log_B[0]
        for t in range(1, T):
            trans = delta[:, None] + log_A
            delta = np.max(trans, axis=0) + log_B[t]
        return int(np.argmax(delta))

    # -- Baum-Welch --------------------------------------------------------
    def _maybe_refit(self):
        if not (self.learn_transitions or self.learn_emissions):
            return
        if self._n_seen % self.refit_every != 0 or len(self._buffer) < self.min_obs:
            return
        self._baum_welch(np.asarray(self._buffer, dtype=np.float64))

    def _forward_backward(self, log_pi, log_A, log_B):
        T, N = log_B.shape
        log_alpha = np.full((T, N), -np.inf)
        log_beta = np.zeros((T, N))
        log_alpha[0] = log_pi + log_B[0]
        for t in range(1, T):
            for j in range(N):
                log_alpha[t, j] = _logsumexp(log_alpha[t - 1] + log_A[:, j]) + log_B[t, j]
        for t in range(T - 2, -1, -1):
            for i in range(N):
                log_beta[t, i] = _logsumexp(log_A[i, :] + log_B[t + 1] + log_beta[t + 1])
        log_norm = _logsumexp(log_alpha[T - 1])
        return log_alpha, log_beta, log_norm

    def _baum_welch(self, obs: np.ndarray):
        T = len(obs)
        if T < 2:
            return
        A = self.A.copy()
        mu = self.mu.copy()
        sigma = self.sigma.copy()
        log_pi = np.log(self.pi + 1e-300)
        for _ in range(self.n_em_iters):
            log_A = np.log(A + 1e-300)
            log_B = self._log_B(obs, mu, sigma)
            log_alpha, log_beta, log_norm = self._forward_backward(log_pi, log_A, log_B)
            if not np.isfinite(log_norm):
                self.A, self.mu, self.sigma = self.A0.copy(), self.mu0.copy(), self.sigma0.copy()
                return
            gamma = np.exp(log_alpha + log_beta - log_norm)
            gamma /= gamma.sum(axis=1, keepdims=True) + _EPS
            xi = np.exp(
                log_alpha[:-1, :, None] + log_A[None, :, :] + log_B[1:, None, :] + log_beta[1:, None, :] - log_norm
            )
            xi /= xi.sum(axis=(1, 2), keepdims=True) + _EPS
            if self.learn_transitions:
                for i in range(self.N):
                    kappa = self.prior_strength * (self.exploit_prior_scale if i == int(HMMState.EXPLOIT) else 1.0)
                    denom = gamma[:-1, i].sum() + kappa
                    A[i] = (xi[:, i, :].sum(axis=0) + kappa * self.A0[i]) / denom
                A = np.clip(A, 1e-6, 1.0)
                A /= A.sum(axis=1, keepdims=True)
            if self.learn_emissions:
                kappa = self.prior_strength
                g = gamma.sum(axis=0)
                mu_new = ((gamma * obs[:, None]).sum(axis=0) + kappa * self.mu0) / (g + kappa)
                var_new = ((gamma * (obs[:, None] - mu_new[None, :]) ** 2).sum(axis=0)
                           + kappa * self.sigma0 ** 2) / (g + kappa)
                mu, sigma = mu_new, np.sqrt(np.maximum(var_new, 1e-8))
        self.A = A
        self.mu = mu
        self.sigma = sigma
        self._refits += 1
        self._last_refit_changed = True

    def refit_snapshot(self):
        if self._last_refit_changed:
            self._last_refit_changed = False
            return {"A": self.A.copy(), "mu": self.mu.copy(), "sigma": self.sigma.copy()}
        return None


# ---------------------------------------------------------------------------
# Factorized mixture proposal generator (+ DREAM population kernel)
# ---------------------------------------------------------------------------

@dataclass
class DREAMConfig:
    n_pairs: int = 1
    cr: float = 0.9
    gamma1_prob: float = 0.1
    eps: float = 1e-3
    min_pop: int = 4
    z_pop_cap: int = 64
    z_archive_max: int = 512
    diversity_frac: float = 0.0
    symmetric: bool = True


class ProposalGenerator:
    def __init__(
        self,
        rng: np.random.Generator,
        params: list[dict],
        sigma_fraction=0.10,
        wide_sigma_fraction=0.40,
        kde_tau=0.05,
        kde_bandwidth_frac=0.03,
        cat_tau=0.60,
        archive_max=200,
        kde_max_terms=50,
        float_weights=None,
        cat_weights=None,
        dream: DREAMConfig | None = None,
    ):
        self.rng = rng
        self.params = params
        self.sigma_fraction = float(sigma_fraction)
        self.wide_sigma_fraction = float(wide_sigma_fraction)
        self.kde_tau = float(kde_tau)
        self.kde_bandwidth_frac = float(kde_bandwidth_frac)
        self.cat_tau = float(cat_tau)
        self.archive_max = int(archive_max)
        self.kde_max_terms = int(kde_max_terms)
        self.float_weights = _normalize_weight_table(float_weights, DEFAULT_FLOAT_WEIGHTS)
        self.cat_weights = _normalize_weight_table(cat_weights, DEFAULT_CAT_WEIGHTS)
        self.dream = dream or DREAMConfig()

        self._archive: list[tuple[dict, float]] = []
        self._cat_history: dict[str, dict] = {
            pi["name"]: {v: [] for v in pi["values"]} for pi in params if pi["type"] == "categorical"
        }
        self._fi_indices = [i for i, pi in enumerate(params) if pi["type"] in ("float", "int")]
        # coordinates the DE kernel may move in symmetric mode: float + log-int only
        self._de_sym = set(i for i in self._fi_indices if params[i]["type"] == "float" or params[i]["log"])
        self._z_archive: list[np.ndarray] = []
        self._Z: np.ndarray | None = None
        if self._fi_indices:
            self._los = np.array([params[i]["lo"] for i in self._fi_indices])
            self._his = np.array([params[i]["hi"] for i in self._fi_indices])
            self._ranges = self._his - self._los
        else:
            self._los = self._his = self._ranges = None

    # -- archive -----------------------------------------------------------
    def update(self, config: dict, loss: float) -> None:
        for pi in self.params:
            if pi["type"] == "categorical":
                self._cat_history[pi["name"]][config[pi["name"]]].append(float(loss))
        self._archive.append((config, float(loss)))
        self._archive.sort(key=lambda t: t[1])
        del self._archive[self.archive_max:]
        if self._fi_indices:
            self._z_archive.append(self.config_to_vector(config))
            if len(self._z_archive) > self.dream.z_archive_max:
                self._z_archive = self._z_archive[-self.dream.z_archive_max:]

    def _archive_weights(self) -> np.ndarray:
        losses = np.array([l for _, l in self._archive])
        n = len(losses)
        if n == 0:
            return np.array([])
        lo, hi = losses.min(), losses.max()
        if hi - lo < 1e-12:
            return np.ones(n) / n
        lw = -(losses - lo) / (hi - lo) / self.kde_tau
        lw -= _logsumexp(lw)
        return np.exp(lw)

    def _bandwidth(self, pi: dict) -> float:
        return self.kde_bandwidth_frac * pi["range"]

    def _cat_probs(self, pi: dict) -> np.ndarray:
        cats = pi["values"]
        C = len(cats)
        hist = self._cat_history[pi["name"]]
        scores = np.zeros(C)
        observed = np.zeros(C, dtype=bool)
        for i, c in enumerate(cats):
            ls = hist[c]
            if len(ls) >= 4:
                scores[i] = np.percentile(ls, 10)
                observed[i] = True
            elif len(ls) >= 1:
                scores[i] = min(ls)
                observed[i] = True
        if not observed.any():
            return np.ones(C) / C
        best = scores[observed].min()
        scores[~observed] = best - 0.3 * (abs(best) + _EPS)
        rng_s = scores.max() - scores.min()
        if rng_s < _EPS:
            return np.ones(C) / C
        ls = -((scores - scores.min()) / rng_s) / (self.cat_tau + _EPS)
        ls -= _logsumexp(ls)
        p = np.exp(ls)
        return p / p.sum()

    # -- vectors -----------------------------------------------------------
    def config_to_vector(self, cfg: dict) -> np.ndarray:
        return np.array([float(cfg[self.params[i]["name"]]) for i in self._fi_indices], dtype=np.float64)

    @staticmethod
    def _reflect(x: float, lo: float, hi: float) -> float:
        if hi <= lo:
            return lo
        for _ in range(32):
            if lo <= x <= hi:
                return float(x)
            x = lo + (lo - x) if x < lo else hi - (x - hi)
        return float(np.clip(x, lo, hi))

    # -- factorized proposal ----------------------------------------------
    def propose(self, x: dict, weights: Weights, state: HMMState, categorical_only=False,
                explore_subsample=False, explore_frac=0.2) -> dict:
        fw, cw = weights
        subset = None
        if explore_subsample and state == HMMState.EXPLORE and self._fi_indices:
            n_mod = max(1, int(math.ceil(explore_frac * len(self._fi_indices))))
            subset = set(int(i) for i in self.rng.choice(
                self._fi_indices, size=min(n_mod, len(self._fi_indices)), replace=False))
        out = {}
        for i, pi in enumerate(self.params):
            name = pi["name"]
            xj = x[name]
            if pi["type"] == "categorical":
                out[name] = self._sample_cat(xj, pi, cw)
            elif categorical_only or (subset is not None and i not in subset):
                out[name] = xj
            elif pi["type"] == "float" or pi["log"]:
                out[name] = self._sample_cont(float(xj), pi, fw)
            else:
                out[name] = self._sample_int(int(xj), pi, fw)
        return out

    def log_q(self, x_from: dict, x_to: dict, weights: Weights) -> float:
        fw, cw = weights
        total = 0.0
        for pi in self.params:
            name = pi["name"]
            if pi["type"] == "categorical":
                total += self._log_q_cat(x_from[name], x_to[name], pi, cw)
            elif pi["type"] == "float" or pi["log"]:
                total += self._log_q_cont(float(x_from[name]), float(x_to[name]), pi, fw)
            else:
                total += self._log_q_int(int(x_from[name]), int(x_to[name]), pi, fw)
        return total

    def _sample_cont(self, xj, pi, fw):
        w_n, w_k, w_w = fw
        lo, hi = pi["lo"], pi["hi"]
        r = self.rng.random()
        if r < w_n:
            s = self.sigma_fraction * pi["range"]
            return _tn_rvs(self.rng, (lo - xj) / s, (hi - xj) / s, xj, s)
        if r < w_n + w_k and len(self._archive) >= 3:
            aw = self._archive_weights()
            idx = int(self.rng.choice(len(self._archive), p=aw))
            c = float(self._archive[idx][0][pi["name"]])
            h = self._bandwidth(pi)
            return _tn_rvs(self.rng, (lo - c) / h, (hi - c) / h, c, h)
        s = self.wide_sigma_fraction * pi["range"]
        return _tn_rvs(self.rng, (lo - xj) / s, (hi - xj) / s, xj, s)

    def _sample_int(self, xj, pi, fw):
        w_l, w_k, w_g = fw
        lo, hi = int(pi["lo"]), int(pi["hi"])
        r = self.rng.random()
        if r < w_l:
            cands = [v for v in (xj - 1, xj, xj + 1) if lo <= v <= hi]
            return int(self.rng.choice(cands))
        if r < w_l + w_k and len(self._archive) >= 3:
            aw = self._archive_weights()
            idx = int(self.rng.choice(len(self._archive), p=aw))
            c = float(self._archive[idx][0][pi["name"]])
            h = max(self._bandwidth(pi), 1.0)
            v = _tn_rvs(self.rng, (lo - 0.5 - c) / h, (hi + 0.5 - c) / h, c, h)
            return int(np.clip(round(v), lo, hi))
        return int(self.rng.integers(lo, hi + 1))

    def _sample_cat(self, xj, pi, cw):
        w_s, w_u, w_h = cw
        cats = pi["values"]
        C = len(cats)
        ph = self._cat_probs(pi)
        pmf = np.array([(w_s if c == xj else 0.0) + w_u / C + w_h * ph[i] for i, c in enumerate(cats)])
        if pmf.sum() <= 0:
            pmf = np.ones(C)
        pmf /= pmf.sum()
        return cats[int(self.rng.choice(C, p=pmf))]

    def _log_q_cont(self, xf, xt, pi, fw):
        w_n, w_k, w_w = fw
        lo, hi = pi["lo"], pi["hi"]
        comps = []
        if w_n > 0:
            s = self.sigma_fraction * pi["range"]
            comps.append(math.log(w_n + _EPS) + _tn_logpdf(xt, (lo - xf) / s, (hi - xf) / s, xf, s))
        if w_k > 0 and len(self._archive) >= 3:
            aw = self._archive_weights()
            h = self._bandwidth(pi)
            n_use = min(len(self._archive), self.kde_max_terms)
            terms = []
            for idx in range(n_use):
                c = float(self._archive[idx][0][pi["name"]])
                terms.append(math.log(aw[idx] + _EPS) + _tn_logpdf(xt, (lo - c) / h, (hi - c) / h, c, h))
            comps.append(math.log(w_k + _EPS) + _logsumexp(terms))
        if w_w > 0:
            s = self.wide_sigma_fraction * pi["range"]
            comps.append(math.log(w_w + _EPS) + _tn_logpdf(xt, (lo - xf) / s, (hi - xf) / s, xf, s))
        return _logsumexp(comps) if comps else -np.inf

    def _log_q_int(self, xf, xt, pi, fw):
        w_l, w_k, w_g = fw
        lo, hi = int(pi["lo"]), int(pi["hi"])
        cands = [v for v in (xf - 1, xf, xf + 1) if lo <= v <= hi]
        comps = [math.log(w_l + _EPS) - math.log(len(cands)) if xt in cands else -np.inf]
        if w_k > 0 and len(self._archive) >= 3:
            aw = self._archive_weights()
            h = max(self._bandwidth(pi), 1.0)
            n_use = min(len(self._archive), self.kde_max_terms)
            p = 0.0
            for idx in range(n_use):
                c = float(self._archive[idx][0][pi["name"]])
                p += aw[idx] * max(_std_cdf((xt + 0.5 - c) / h) - _std_cdf((xt - 0.5 - c) / h), 0.0)
            comps.append(math.log(w_k + _EPS) + math.log(p + _EPS))
        comps.append(math.log(w_g + _EPS) - math.log(hi - lo + 1))
        return _logsumexp(comps)

    def _log_q_cat(self, xf, xt, pi, cw):
        w_s, w_u, w_h = cw
        C = pi["n_categories"]
        ph = self._cat_probs(pi)
        comps = [math.log(w_s + _EPS) if xt == xf else -np.inf,
                 math.log(w_u / C + _EPS),
                 math.log(w_h * ph[pi["val_to_idx"][xt]] + _EPS)]
        return _logsumexp(comps)

    # -- DREAM(ZS) population kernel ----------------------------------------
    def refresh_population(self, chain_configs: list[dict]) -> None:
        if not self._fi_indices:
            self._Z = None
            return
        vecs = [self.config_to_vector(c) for c in chain_configs]
        if len(self._z_archive) >= self.dream.min_pop:
            n = min(len(self._z_archive), self.dream.z_pop_cap)
            idx = self.rng.choice(len(self._z_archive), size=n, replace=False)
            vecs.extend(self._z_archive[int(i)].copy() for i in idx)
        else:
            vecs.extend(self.config_to_vector(c) for c, _ in self._archive)
        n_div = int(self.dream.diversity_frac * self.dream.z_pop_cap)
        if n_div > 0:
            vecs.extend(self._los + self.rng.random((n_div, self._los.size)) * self._ranges)
        self._Z = np.vstack(vecs) if vecs else None

    @property
    def population_size(self) -> int:
        return 0 if self._Z is None else int(self._Z.shape[0])

    def dream_ready(self) -> bool:
        return len(self._fi_indices) > 0 and self.population_size >= self.dream.min_pop

    def _pair_indices(self, x: np.ndarray, n_pairs: int) -> np.ndarray:
        n_pop = self._Z.shape[0]
        need = 2 * n_pairs
        if n_pop < need:
            return self.rng.choice(n_pop, size=need, replace=True)
        scaled = np.abs(self._Z - x) / np.maximum(self._ranges, _EPS)
        pool = np.where(np.max(scaled, axis=1) > 1e-9)[0]
        if pool.size < need:
            pool = np.arange(n_pop)
        return self.rng.choice(pool, size=need, replace=False)

    def propose_dream(self, x: dict, cat_weights: tuple[float, float, float]) -> dict:
        d = self.dream
        xv = self.config_to_vector(x)
        dim = xv.size
        movable = np.array([(i in self._de_sym) if d.symmetric else True for i in self._fi_indices], dtype=bool)
        if not movable.any():
            fw = self.float_weights[HMMState.EXPLORE]
            return self.propose(x, (fw, cat_weights), HMMState.EXPLORE)
        cross = (self.rng.random(dim) < d.cr) & movable
        if not cross.any():
            cross[int(self.rng.choice(np.where(movable)[0]))] = True
        d_eff = int(cross.sum())
        n_pairs = max(1, d.n_pairs)
        delta = np.zeros(dim)
        idx = self._pair_indices(xv, n_pairs)
        for k in range(n_pairs):
            delta += self._Z[int(idx[2 * k])] - self._Z[int(idx[2 * k + 1])]
        gamma = 1.0 if self.rng.random() < d.gamma1_prob else 2.38 / math.sqrt(2.0 * n_pairs * d_eff)
        noise = self.rng.normal(0.0, d.eps * self._ranges)
        x_new = xv.copy()
        x_new[cross] = xv[cross] + gamma * delta[cross] + noise[cross]
        out = copy.deepcopy(x)
        for k, i in enumerate(self._fi_indices):
            if not cross[k]:
                continue
            pi = self.params[i]
            v = self._reflect(float(x_new[k]), pi["lo"], pi["hi"])
            if pi["type"] == "int" and not pi["log"]:
                v = int(np.clip(round(v), int(pi["lo"]), int(pi["hi"])))
            out[pi["name"]] = v
        if not d.symmetric:
            for pi in self.params:
                if pi["type"] == "categorical":
                    out[pi["name"]] = self._sample_cat(x[pi["name"]], pi, cat_weights)
        return out


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

class Chain:
    def __init__(self, chain_id, x0, loss0, gen: ProposalGenerator, ctrl: BaseController, rng,
                 T0, T_min, scale, p_cat_step, anneal_T, stagnation_limit, boost_coef, boost_cap,
                 p_dream, decoder_soft, explore_subsample, explore_frac):
        self.chain_id = chain_id
        self.gen = gen
        self.ctrl = ctrl
        self.rng = rng
        self.T0 = float(T0)
        self.T = float(T0)
        self.T_min = float(T_min) if T_min is not None else 0.01 * float(T0)
        self.scale = float(scale)
        self.p_cat_step = float(p_cat_step)
        self.anneal_T = bool(anneal_T)
        self.stagnation_limit = int(stagnation_limit)
        self.boost_coef = float(boost_coef)
        self.boost_cap = float(boost_cap)
        self.p_dream = float(p_dream)
        self.decoder_soft = bool(decoder_soft)
        self.explore_subsample = bool(explore_subsample)
        self.explore_frac = float(explore_frac)
        self.reset_at(x0, loss0)

    def reset_at(self, x0, loss0):
        self.current_x = copy.deepcopy(x0)
        self.current_loss = float(loss0)
        self.best_x = copy.deepcopy(x0)
        self.best_loss = float(loss0)
        self.ctrl.reset()
        self.state = HMMState.EXPLOIT
        self.observations: list[float] = []
        self.rejection_streak = 0
        self.steps_since_improve = 0
        self.last_kernel = "factorized"
        self.last_posterior = self.ctrl.posterior.copy()
        self.last_obs = float("nan")

    def step(self, objective, progress: float, is_burnin: bool):
        # temperature schedule
        if is_burnin or not self.anneal_T:
            self.T = self.T0
        else:
            self.T = self.T_min + (self.T0 - self.T_min) * (1.0 - progress)
        # regime
        if len(self.observations) >= 2:
            self.state = self.ctrl.observe(self.observations[-self.ctrl.window:])
        else:
            self.state = self.ctrl.state
        forced = False
        if not is_burnin and self.rejection_streak >= self.stagnation_limit:
            if self.state != HMMState.TRAPPED:
                self.ctrl.force_state(HMMState.TRAPPED)
                self.state = HMMState.TRAPPED
            n_over = self.rejection_streak - self.stagnation_limit + 1
            self.T = self.T0 * (1.0 + min(n_over * self.boost_coef, self.boost_cap))
            forced = True
        posterior = self.ctrl.posterior.copy()
        if not self.decoder_soft:
            posterior = np.zeros(3)
            posterior[int(self.state)] = 1.0
        self.last_posterior = posterior
        weights = _mix_weights(posterior, self.gen.float_weights, self.gen.cat_weights)
        # kernel
        use_dream = self.p_dream > 0.0 and self.gen.dream_ready() and self.rng.random() < self.p_dream
        if use_dream:
            self.last_kernel = "dream"
            x_prime = self.gen.propose_dream(self.current_x, weights[1])
        else:
            self.last_kernel = "factorized"
            cat_only = self.rng.random() < self.p_cat_step
            x_prime = self.gen.propose(self.current_x, weights, self.state, categorical_only=cat_only,
                                       explore_subsample=self.explore_subsample, explore_frac=self.explore_frac)
        loss_prime = float(objective(x_prime))
        loss_old = self.current_loss
        # acceptance
        d_norm = (loss_prime - loss_old) / (self.scale + 1e-8)
        log_alpha = -d_norm / (self.T + 1e-100)
        if not use_dream:
            log_alpha += self.gen.log_q(x_prime, self.current_x, weights) - self.gen.log_q(self.current_x, x_prime, weights)
        if np.isnan(log_alpha):
            alpha = 0.0
        else:
            alpha = float(np.exp(min(log_alpha, 0.0)))
        accepted = bool(self.rng.random() < alpha)
        if accepted:
            self.current_x = copy.deepcopy(x_prime)
            self.current_loss = loss_prime
            self.rejection_streak = 0
        else:
            self.rejection_streak += 1
        obs = float((loss_prime - loss_old) / (self.scale + 1e-8))
        self.observations.append(obs)
        self.last_obs = obs
        if loss_prime < self.best_loss:
            self.best_x = copy.deepcopy(x_prime)
            self.best_loss = loss_prime
            self.steps_since_improve = 0
        else:
            self.steps_since_improve += 1
        return x_prime, loss_prime, accepted, alpha, forced


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    def __init__(self, chains, gen, sobol, params, objective, rng, clone_noise=0.05,
                 collapse_threshold=0.01, reseed_fraction=0.3, patience=15, reseed_until=0.5):
        self.chains = chains
        self.gen = gen
        self.sobol = sobol
        self.params = params
        self.objective = objective
        self.rng = rng
        self.clone_noise = float(clone_noise)
        self.collapse_threshold = float(collapse_threshold)
        self.reseed_fraction = float(reseed_fraction)
        self.patience = int(patience)
        self.reseed_until = float(reseed_until)
        self.n_rescues = 0
        self.n_reseeds = 0

    def run(self, progress: float):
        new_evals = []
        min_age = max(self.chains[0].ctrl.window, 10)
        rescue = [i for i, c in enumerate(self.chains)
                  if (c.state == HMMState.TRAPPED or c.steps_since_improve >= self.patience)
                  and len(c.observations) >= min_age]
        if rescue:
            exploit = [i for i, c in enumerate(self.chains) if c.state == HMMState.EXPLOIT]
            pool = exploit if exploit else list(range(len(self.chains)))
            donor = min(pool, key=lambda i: self.chains[i].best_loss)
            for i in rescue:
                if i == donor:
                    continue
                x = self._noisy_clone(self.chains[donor].best_x)
                l = float(self.objective(x))
                new_evals.append((x, l, "rescue"))
                self.chains[i].reset_at(x, l)
                self.gen.update(x, l)
                self.n_rescues += 1
        if progress < self.reseed_until and self._collapsed():
            n = max(1, int(len(self.chains) * self.reseed_fraction))
            worst = sorted(range(len(self.chains)), key=lambda i: self.chains[i].best_loss)[-n:]
            cfgs = self.sobol.generate(n, seed=int(self.rng.integers(0, 2**31 - 1)))
            for k, i in enumerate(worst):
                l = float(self.objective(cfgs[k]))
                new_evals.append((cfgs[k], l, "reseed"))
                self.chains[i].reset_at(cfgs[k], l)
                self.gen.update(cfgs[k], l)
                self.n_reseeds += 1
        return new_evals

    def _noisy_clone(self, cfg: dict) -> dict:
        out = {}
        for pi in self.params:
            v = cfg[pi["name"]]
            if pi["type"] == "float" or (pi["type"] == "int" and pi["log"]):
                s = pi["range"] * self.clone_noise
                out[pi["name"]] = float(np.clip(float(v) + self.rng.normal(0.0, s), pi["lo"], pi["hi"]))
            elif pi["type"] == "int":
                out[pi["name"]] = int(np.clip(int(v) + int(self.rng.choice([-1, 0, 1])), pi["real_lo"], pi["real_hi"]))
            else:
                if self.rng.random() < 0.2 and pi["n_categories"] > 1:
                    others = [c for c in pi["values"] if c != v]
                    out[pi["name"]] = others[int(self.rng.integers(0, len(others)))]
                else:
                    out[pi["name"]] = v
        return out

    def _collapsed(self) -> bool:
        losses = np.array([c.best_loss for c in self.chains])
        if losses.size < 2:
            return False
        return float(np.std(losses) / (abs(float(np.mean(losses))) + _EPS)) < self.collapse_threshold


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def compute_scale(losses) -> float:
    arr = np.asarray(losses, dtype=np.float64)
    n = arr.size
    if n == 0:
        return 1.0
    if n == 1:
        return max(abs(float(arr[0])), 1.0)
    if n >= 10:
        q90, q10 = np.percentile(arr, [90, 10])
        s = float(q90 - q10)
    else:
        s = float(arr.max() - arr.min())
    return max(s, 1e-5)


class HMM_MCMC_FMP:
    """See module docstring. All arguments are keyword-only except the first three."""

    CONTROLLERS = ("hmm", "fixed", "random", "rule")

    def __init__(
        self,
        objective_func,
        budget: int,
        dict_to_optimize: dict,
        *,
        seed: int | None = None,
        # initial design and chains
        n_init: int = 16,
        n_chains: int = 4,
        # orchestration (orchestrate_every=None or 0 disables it)
        orchestrate_every: int | None = 5,
        orchestrate_patience: int = 15,
        clone_noise: float = 0.05,
        collapse_threshold: float = 0.01,
        reseed_fraction: float = 0.3,
        reseed_until: float = 0.5,
        # temperature / acceptance
        T_mcmc: float = 1.0,
        T_min: float | None = None,
        anneal_T: bool = True,
        burnin_fraction: float = 0.10,
        rejection_streak: int = 10,
        boost_coef: float = 3.0,
        boost_cap: float = 50.0,
        scale_refresh_until: float = 0.5,
        # proposals
        sigma_fraction: float = 0.10,
        wide_sigma_fraction: float = 0.40,
        kde_tau: float = 0.05,
        kde_bandwidth_frac: float = 0.03,
        temperature: float = 0.60,        # categorical Boltzmann temperature (paper: tau)
        p_cat_step: float = 0.30,
        archive_max: int = 200,
        explore_subsample: bool = False,
        explore_subsample_frac: float = 0.2,
        float_weights: dict | None = None,
        cat_weights: dict | None = None,
        # controller
        controller: str = "hmm",
        decoder: str = "soft",
        hmm_window: int = 8,
        hmm_lambda_noise: float = 0.01,
        emission_mu=DEFAULT_EMISSION_MU,
        emission_sigma=DEFAULT_EMISSION_SIGMA,
        transition_prior=DEFAULT_TRANSITION_PRIOR,
        learn_transitions: bool = True,
        learn_emissions: bool = False,
        bw_refit_every: int = 5,
        bw_min_obs: int = 12,
        bw_n_em_iters: int = 3,
        bw_prior_strength: float = 25.0,
        bw_exploit_prior_scale: float = 3.0,
        bw_max_len: int = 64,
        rule_threshold: float = 0.05,
        fixed_state: str = "EXPLOIT",
        # DREAM
        p_dream: float = 0.0,
        dream_n_pairs: int = 1,
        dream_cr: float = 0.9,
        dream_gamma1_prob: float = 0.1,
        dream_eps: float = 1e-3,
        dream_min_pop: int = 4,
        dream_z_archive_max: int = 512,
        dream_z_pop_cap: int = 64,
        dream_diversity_frac: float = 0.0,
        dream_symmetric: bool = True,
        # io
        show_progress: bool = False,
        progress_desc: str | None = None,
        record_configs: bool = True,
    ):
        if controller not in self.CONTROLLERS:
            raise ValueError(f"controller must be one of {self.CONTROLLERS}")
        if decoder not in ("viterbi", "soft"):
            raise ValueError("decoder must be 'viterbi' or 'soft'")
        if not (0.0 <= p_dream <= 1.0):
            raise ValueError("p_dream must be in [0, 1]")
        if n_init < 1 or n_chains < 1:
            raise ValueError("n_init and n_chains must be >= 1")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.dict_to_optimize = dict_to_optimize
        self.params = [_param_record(k, v) for k, v in dict_to_optimize.items()]
        self._raw_objective = objective_func
        self.budget = int(budget)
        self.n_init = int(n_init)
        self.n_chains = int(n_chains)
        self.orchestrate_every = int(orchestrate_every) if orchestrate_every else 0
        self.orchestrate_patience = int(orchestrate_patience)
        self.clone_noise = clone_noise
        self.collapse_threshold = collapse_threshold
        self.reseed_fraction = reseed_fraction
        self.reseed_until = reseed_until
        self.T_mcmc = float(T_mcmc)
        self.T_min = T_min
        self.anneal_T = anneal_T
        self.burnin_fraction = float(burnin_fraction)
        self.rejection_streak = int(rejection_streak)
        self.boost_coef = boost_coef
        self.boost_cap = boost_cap
        self.scale_refresh_until = float(scale_refresh_until)
        self.p_cat_step = float(p_cat_step)
        self.explore_subsample = bool(explore_subsample)
        self.explore_subsample_frac = float(explore_subsample_frac)
        self.controller = controller
        self.decoder = decoder
        self.hmm_window = int(hmm_window)
        self.rule_threshold = rule_threshold
        self.fixed_state = HMMState[fixed_state] if isinstance(fixed_state, str) else HMMState(fixed_state)
        self.p_dream = float(p_dream)
        self.show_progress = show_progress
        self.progress_desc = progress_desc
        self.record_configs = record_configs
        self._ctrl_kwargs = dict(
            window=hmm_window, decoder=decoder, lambda_noise=hmm_lambda_noise,
            emission_mu=emission_mu, emission_sigma=emission_sigma, transition_prior=transition_prior,
            learn_transitions=learn_transitions, learn_emissions=learn_emissions,
            refit_every=bw_refit_every, min_obs=bw_min_obs, n_em_iters=bw_n_em_iters,
            prior_strength=bw_prior_strength, exploit_prior_scale=bw_exploit_prior_scale, bw_max_len=bw_max_len,
        )
        self._gen_kwargs = dict(
            sigma_fraction=sigma_fraction, wide_sigma_fraction=wide_sigma_fraction, kde_tau=kde_tau,
            kde_bandwidth_frac=kde_bandwidth_frac, cat_tau=temperature, archive_max=archive_max,
            float_weights=float_weights, cat_weights=cat_weights,
            dream=DREAMConfig(n_pairs=dream_n_pairs, cr=dream_cr, gamma1_prob=dream_gamma1_prob, eps=dream_eps,
                              min_pop=dream_min_pop, z_pop_cap=dream_z_pop_cap, z_archive_max=dream_z_archive_max,
                              diversity_frac=dream_diversity_frac, symmetric=dream_symmetric),
        )
        self.reset()

    # -- public helpers ------------------------------------------------------
    def decode_config(self, cfg: dict) -> dict:
        return decode_config(cfg, self.params)

    def reset(self):
        self.data: list[tuple[dict, float]] = []
        self.history: list[dict] = []
        self.A_history: list[dict] = []
        self.emission_history: list[dict] = []
        self.timing = {"total": 0.0, "objective": 0.0}
        self.n_objective_calls = 0
        self._chains: list[Chain] = []
        self._gen: ProposalGenerator | None = None
        self._orch: Orchestrator | None = None
        self.scale = 1.0

    @property
    def history_table(self):  # backwards-compatible name
        return self.history

    # -- internals -----------------------------------------------------------
    def _objective(self, cfg: dict) -> float:
        t = time.perf_counter()
        v = float(self._raw_objective(self.decode_config(cfg)))
        self.timing["objective"] += time.perf_counter() - t
        self.n_objective_calls += 1
        return v

    def _make_controller(self) -> BaseController:
        if self.controller == "hmm":
            return HMMController(self.rng, **self._ctrl_kwargs)
        if self.controller == "fixed":
            return FixedController(self.rng, self.hmm_window, self.fixed_state)
        if self.controller == "random":
            return RandomController(self.rng, self.hmm_window)
        return RuleController(self.rng, self.hmm_window, self.rule_threshold)

    def _record(self, cfg, loss, chain=None, kernel="init", accepted=True, alpha=1.0, forced=False):
        row = {
            "eval": len(self.data),
            "chain": -1 if chain is None else chain.chain_id,
            "state": "INIT" if chain is None else chain.state.name,
            "kernel": kernel,
            "loss": float(loss),
            "accepted": bool(accepted),
            "alpha": float(alpha),
            "forced_trapped": bool(forced),
            "rej_streak": 0 if chain is None else chain.rejection_streak,
            "T": float("nan") if chain is None else chain.T,
            "scale": self.scale,
            "obs": float("nan") if chain is None else chain.last_obs,
            "post_exploit": float("nan") if chain is None else float(chain.last_posterior[0]),
            "post_explore": float("nan") if chain is None else float(chain.last_posterior[1]),
            "post_trapped": float("nan") if chain is None else float(chain.last_posterior[2]),
        }
        if self.record_configs:
            row["config"] = self.decode_config(cfg)
        self.history.append(row)

    def _snapshot_controllers(self):
        for c in self._chains:
            snap = c.ctrl.refit_snapshot()
            if snap is not None:
                self.A_history.append({"eval": len(self.data), "chain": c.chain_id, "A": snap["A"].tolist()})
                if self._ctrl_kwargs["learn_emissions"]:
                    self.emission_history.append({"eval": len(self.data), "chain": c.chain_id,
                                                  "mu": snap["mu"].tolist(), "sigma": snap["sigma"].tolist()})

    # -- main loop -------------------------------------------------------------
    def main_loop(self):
        t_start = time.perf_counter()
        self.reset()
        sobol = SobolInitializer(self.params)
        self._gen = ProposalGenerator(self.rng, self.params, **self._gen_kwargs)
        pbar = None
        if self.show_progress:
            from tqdm.auto import tqdm
            pbar = tqdm(total=self.budget, desc=self.progress_desc or "H-MCMC-FMP")

        init_cfgs = sobol.generate(min(self.n_init, self.budget), seed=int(self.rng.integers(0, 2**31 - 1)))
        init = []
        for cfg in init_cfgs:
            loss = self._objective(cfg)
            self.data.append((cfg, loss))
            init.append((cfg, loss))
            self._gen.update(cfg, loss)
            self._record(cfg, loss)
            if pbar:
                pbar.update(1)
        if len(self.data) >= self.budget:
            return self._finish(t_start, pbar)

        self.scale = compute_scale([l for _, l in init])
        init.sort(key=lambda t: t[1])
        self._chains = []
        for i, (cfg, loss) in enumerate(init[: self.n_chains]):
            self._chains.append(Chain(
                i, cfg, loss, self._gen, self._make_controller(), self.rng,
                T0=self.T_mcmc, T_min=self.T_min, scale=self.scale, p_cat_step=self.p_cat_step,
                anneal_T=self.anneal_T, stagnation_limit=self.rejection_streak,
                boost_coef=self.boost_coef, boost_cap=self.boost_cap, p_dream=self.p_dream,
                decoder_soft=(self.decoder == "soft" and self.controller == "hmm"),
                explore_subsample=self.explore_subsample, explore_frac=self.explore_subsample_frac,
            ))
        self._orch = Orchestrator(self._chains, self._gen, sobol, self.params, self._objective, self.rng,
                                  clone_noise=self.clone_noise, collapse_threshold=self.collapse_threshold,
                                  reseed_fraction=self.reseed_fraction, patience=self.orchestrate_patience,
                                  reseed_until=self.reseed_until)
        sweep = 0
        while len(self.data) < self.budget:
            if self.p_dream > 0.0:
                self._gen.refresh_population([c.current_x for c in self._chains])
            for chain in self._chains:
                if len(self.data) >= self.budget:
                    break
                progress = len(self.data) / self.budget
                is_burnin = progress < self.burnin_fraction
                cfg, loss, acc, alpha, forced = chain.step(self._objective, progress, is_burnin)
                self.data.append((cfg, loss))
                self._gen.update(cfg, loss)
                self._record(cfg, loss, chain, chain.last_kernel, acc, alpha, forced)
                self._snapshot_controllers()
                if pbar:
                    pbar.update(1)
            sweep += 1
            if self.orchestrate_every and sweep % self.orchestrate_every == 0 and len(self.data) < self.budget:
                progress = len(self.data) / self.budget
                for cfg, loss, kind in self._orch.run(progress):
                    if len(self.data) >= self.budget:
                        break
                    self.data.append((cfg, loss))
                    self._record(cfg, loss, None, kind)
                    if pbar:
                        pbar.update(1)
                if len(self.data) / self.budget < self.scale_refresh_until:
                    self.scale = compute_scale([l for _, l in self.data])
                    for c in self._chains:
                        c.scale = self.scale
        return self._finish(t_start, pbar)

    def _finish(self, t_start, pbar):
        if pbar:
            pbar.close()
        self.timing["total"] = time.perf_counter() - t_start
        best_cfg, best_loss = min(self.data, key=lambda t: t[1])
        return self.decode_config(best_cfg), float(best_loss)

    # -- summaries for analysis ----------------------------------------------
    def summary(self) -> dict:
        rows = [r for r in self.history if r["chain"] >= 0]
        n = max(len(rows), 1)
        out = {
            "n_evals": len(self.data),
            "best_loss": float(min(l for _, l in self.data)) if self.data else float("nan"),
            "acceptance_rate": float(np.mean([r["accepted"] for r in rows])) if rows else float("nan"),
            "kernel_fraction_dream": float(np.mean([r["kernel"] == "dream" for r in rows])) if rows else 0.0,
            "n_rescues": self._orch.n_rescues if self._orch else 0,
            "n_reseeds": self._orch.n_reseeds if self._orch else 0,
            "n_bw_refits": len(self.A_history),
            "time_total": self.timing["total"],
            "time_objective": self.timing["objective"],
            "time_overhead": self.timing["total"] - self.timing["objective"],
        }
        for s in STATE_NAMES:
            sr = [r for r in rows if r["state"] == s]
            out[f"frac_{s.lower()}"] = len(sr) / n
            out[f"acc_{s.lower()}"] = float(np.mean([r["accepted"] for r in sr])) if sr else float("nan")
        return out
