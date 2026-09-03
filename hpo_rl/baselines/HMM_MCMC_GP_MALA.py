"""H-MCMC-GP-MALA: Baum-Welch HMM + gpytorch/botorch GP + preconditioned MALA + RAM.

Extends :class:`HMM_MCMC_BW` with a Gaussian-process surrogate over continuous
dimensions.  The GP is fit on ``-loss`` (maximization convention).  EXPLOIT uses
TuRBO-style trust-region botorch LogExpectedImprovement argmax; EXPLORE uses
high-kappa UCB or pathwise Thompson sampling via MALA.  Diagonal RAM adapts the
Langevin step-size / preconditioner.  TRAPPED, burn-in, and early archive phases
fall back to the standard factorized mixture.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
from tqdm.auto import tqdm

from hpo_rl.baselines.HMM_MCMC import (
    HMMState,
    FactorizedProposalGenerator,
    SobolInitializer,
    MCMCChain,
    GlobalOrchestrator,
)
from hpo_rl.baselines.HMM_MCMC_BW import HMM_MCMC_BW
from hpo_rl.baselines.HMM_MCMC_RAM import DiagRAMConfig, DiagRAMPreconditioner
from hpo_rl.baselines.HMM_MCMC_TEST import BaumWelchHMMController

try:
    import torch
    from botorch.models import SingleTaskGP
    from botorch.models.transforms import Standardize
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    _HAS_GP = True
except ImportError as _gp_import_err:  # pragma: no cover - exercised via self-check guard
    _HAS_GP = False
    _GP_IMPORT_ERR = _gp_import_err


def _require_gp() -> None:
    if not _HAS_GP:
        raise ImportError(
            "HMM_MCMC_GP_MALA requires gpytorch and botorch. "
            "Install with: pip install 'hpo_rl[experiments]' "
            "or pip install gpytorch botorch"
        ) from _GP_IMPORT_ERR


# ---------------------------------------------------------------------------
# GP surrogate configuration and model
# ---------------------------------------------------------------------------


@dataclass
class GPConfig:
    matern_nu: float = 2.5
    refit_every: int = 1
    min_obs: int = 12
    max_obs: int = 128
    kappa_exploit: float = 1.0
    kappa_explore: float = 4.0
    explore_mode: str = "thompson"  # "thompson" | "ucb"
    jitter: float = 1e-4


class GPSurrogate:
    """Shared gpytorch/botorch GP over unit-cube continuous coordinates."""

    def __init__(self, dim: int, config: GPConfig | None = None):
        _require_gp()
        self.dim = int(dim)
        self.config = config or GPConfig()
        self.model: SingleTaskGP | None = None
        self._n_fit: int = 0
        self._best_f: float | None = None
        self._dtype = torch.double
        self._device = torch.device("cpu")

    @property
    def is_ready(self) -> bool:
        return self.model is not None

    def fit(self, X_unit: np.ndarray, y: np.ndarray) -> None:
        """Fit ARD Matérn GP on unit-cube inputs and scalar losses (stored as ``-loss``)."""
        _require_gp()
        X = np.asarray(X_unit, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        if X.shape[0] < 2 or X.shape[1] != self.dim:
            return

        train_X = torch.as_tensor(X, dtype=self._dtype, device=self._device)
        train_Y = torch.as_tensor(-y_arr, dtype=self._dtype, device=self._device)

        gp = SingleTaskGP(
            train_X,
            train_Y,
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        gp.eval()

        self.model = gp
        self._best_f = float(train_Y.max().item())
        self._n_fit += 1

    def _ucb_and_grad(self, x_unit: np.ndarray, kappa: float) -> tuple[float, np.ndarray]:
        if self.model is None:
            raise RuntimeError("GP is not fitted")
        x = torch.tensor(x_unit, dtype=self._dtype, device=self._device, requires_grad=True)
        post = self.model.posterior(x.reshape(1, -1))
        mu = post.mean.reshape(())
        sigma = post.variance.clamp_min(self.config.jitter).sqrt().reshape(())
        ucb = mu + kappa * sigma
        ucb.backward()
        grad = x.grad.detach().cpu().numpy().astype(np.float64)
        return float(ucb.detach().cpu().item()), grad

    def _sample_thompson(self, x_unit: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("GP is not fitted")
        try:
            from botorch.sampling.pathwise import draw_matheron_paths
            from botorch.utils.sampling import get_sampler

            sampler = get_sampler(self.model)
            paths = draw_matheron_paths(
                model=self.model, sample_shape=torch.Size([1]), sampler=sampler
            )
            return paths(x_unit.reshape(1, -1)).reshape(())
        except Exception:
            post = self.model.posterior(x_unit.reshape(1, -1))
            mu = post.mean.reshape(())
            sigma = post.variance.clamp_min(self.config.jitter).sqrt().reshape(())
            eps = torch.randn((), dtype=self._dtype, device=self._device)
            return mu + sigma * eps

    def select_explore_point(
        self,
        candidates_unit: np.ndarray,
        kappa: float,
    ) -> np.ndarray:
        """Pick candidate maximizing ``mu + kappa * sigma`` (maximize / -loss space)."""
        if self.model is None:
            raise RuntimeError("GP is not fitted")
        cands = np.asarray(candidates_unit, dtype=np.float64)
        if cands.ndim == 1:
            cands = cands.reshape(1, -1)
        x = torch.as_tensor(cands, dtype=self._dtype, device=self._device)
        with torch.no_grad():
            post = self.model.posterior(x)
            mu = post.mean.reshape(-1)
            sigma = post.variance.clamp_min(self.config.jitter).sqrt().reshape(-1)
        score = mu + float(kappa) * sigma
        best_idx = int(torch.argmax(score).item())
        return cands[best_idx].copy()

    def optimize_ei(
        self,
        n_restarts: int = 5,
        raw_samples: int = 50,
        bounds: np.ndarray | None = None,
        greedy: bool = False,
    ) -> np.ndarray:
        """Multi-start botorch EI (or PosteriorMean when greedy) optimization."""
        if self.model is None or self._best_f is None:
            raise RuntimeError("GP is not fitted")
        from botorch.acquisition.analytic import LogExpectedImprovement, PosteriorMean
        from botorch.optim import optimize_acqf

        if bounds is None:
            lo = np.zeros(self.dim, dtype=np.float64)
            hi = np.ones(self.dim, dtype=np.float64)
        else:
            b = np.asarray(bounds, dtype=np.float64)
            if b.shape != (2, self.dim):
                raise ValueError(f"bounds must have shape (2, {self.dim}), got {b.shape}")
            lo = np.clip(b[0], 0.0, 1.0)
            hi = np.clip(b[1], 0.0, 1.0)
            hi = np.maximum(hi, lo + 1e-12)

        bounds_t = torch.stack([
            torch.as_tensor(lo, dtype=self._dtype, device=self._device),
            torch.as_tensor(hi, dtype=self._dtype, device=self._device),
        ])
        if greedy:
            acqf = PosteriorMean(model=self.model)
        else:
            acqf = LogExpectedImprovement(model=self.model, best_f=self._best_f)
        candidate, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds_t,
            q=1,
            num_restarts=max(1, n_restarts),
            raw_samples=max(64, raw_samples),
        )
        return candidate.detach().cpu().numpy().reshape(-1)

    def _predict(self, x_unit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("GP is not fitted")
        x = x_unit.reshape(1, -1).to(dtype=self._dtype, device=self._device)
        with torch.no_grad():
            post = self.model.posterior(x)
            mu = post.mean.reshape(())
            sigma = post.variance.clamp_min(self.config.jitter).sqrt().reshape(())
        return mu, sigma

    def acq_and_grad(
        self,
        x_unit: np.ndarray,
        state: HMMState,
        temperature: float,
    ) -> tuple[float, np.ndarray]:
        """Return ``a(x)`` and ``grad a(x)`` for the state-specific acquisition (maximize)."""
        _require_gp()
        if self.model is None:
            raise RuntimeError("GP is not fitted")

        _ = max(float(temperature), 1e-12)  # reserved; surrogate T applied in chain
        x = torch.tensor(x_unit, dtype=self._dtype, device=self._device, requires_grad=True)

        if state == HMMState.EXPLOIT:
            kappa = self.config.kappa_exploit
            post = self.model.posterior(x.reshape(1, -1))
            mu = post.mean.reshape(())
            sigma = post.variance.clamp_min(self.config.jitter).sqrt().reshape(())
            acq = mu + kappa * sigma
        elif state == HMMState.EXPLORE and self.config.explore_mode == "thompson":
            acq = self._sample_thompson(x)
        else:
            kappa = self.config.kappa_explore
            post = self.model.posterior(x.reshape(1, -1))
            mu = post.mean.reshape(())
            sigma = post.variance.clamp_min(self.config.jitter).sqrt().reshape(())
            acq = mu + kappa * sigma

        acq.backward()
        grad = x.grad.detach().cpu().numpy().astype(np.float64)
        return float(acq.detach().cpu().item()), grad

    def ard_scales(self) -> np.ndarray | None:
        """Per-dimension ARD lengthscales mapped to RAM scale units."""
        if self.model is None:
            return None
        try:
            ls = self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().reshape(-1)
            ls = np.maximum(ls, 1e-6)
            return (1.0 / ls).astype(np.float64)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Unit-cube helpers
# ---------------------------------------------------------------------------


def _reflect_scalar_unit(x: float) -> float:
    lo, hi = 0.0, 1.0
    width = hi - lo
    for _ in range(32):
        if lo <= x <= hi:
            return float(x)
        if x < lo:
            x = lo + (lo - x)
        else:
            x = hi - (x - hi)
    return float(np.clip(x, lo, hi))


def reflect_unit(u: np.ndarray) -> np.ndarray:
    """Reflect ``u`` into the unit hypercube ``[0, 1]^d``."""
    out = np.asarray(u, dtype=np.float64).copy()
    for j in range(out.size):
        out[j] = _reflect_scalar_unit(float(out[j]))
    return out


# ---------------------------------------------------------------------------
# Proposal generator with unit-cube helpers
# ---------------------------------------------------------------------------


class GpProposalGenerator(FactorizedProposalGenerator):
    """Factorized mixture plus continuous vectorization for GP / MALA."""

    def __init__(self, dict_to_optimize: dict, **kwargs):
        super().__init__(dict_to_optimize, **kwargs)
        self._continuous_indices: list[int] = [
            i for i, pi in enumerate(self._param_info) if pi["type"] in ("float", "int")
        ]
        self._los_cont: np.ndarray | None = None
        self._ranges_cont: np.ndarray | None = None
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
        return any(pi["type"] == "categorical" for pi in self._param_info)

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
        return reflect_unit(u)

    def default_initial_scales_unit(self) -> np.ndarray:
        if self.continuous_dim == 0:
            return np.array([], dtype=np.float64)
        return np.full(self.continuous_dim, self.sigma_fraction, dtype=np.float64)

    def elite_marginal_std_unit(
        self, top_k: int = 50, min_std_frac: float = 0.01
    ) -> np.ndarray:
        if not self._continuous_indices or len(self._archive) < 2:
            return self.default_initial_scales_unit()
        k = min(top_k, len(self._archive))
        rows = np.array(
            [self.to_unit(self.config_to_vector(entry[0])) for entry in self._archive[:k]],
            dtype=np.float64,
        )
        std = np.std(rows, axis=0)
        floor = np.full(std.shape, min_std_frac, dtype=np.float64)
        return np.maximum(std, floor)

    def default_initial_scales(self) -> np.ndarray:
        if self._ranges_cont is None:
            return np.array([], dtype=np.float64)
        return self.sigma_fraction * self._ranges_cont

    def elite_marginal_std(self, top_k: int = 50, min_std_frac: float = 0.01) -> np.ndarray:
        if not self._continuous_indices or len(self._archive) < 2:
            return self.default_initial_scales()
        k = min(top_k, len(self._archive))
        rows = np.array(
            [self.config_to_vector(entry[0]) for entry in self._archive[:k]],
            dtype=np.float64,
        )
        std = np.std(rows, axis=0)
        floor = min_std_frac * self._ranges_cont
        return np.maximum(std, floor)

    def gp_training_data(self, max_obs: int) -> tuple[np.ndarray, np.ndarray] | None:
        if not self._continuous_indices or len(self._archive) < 2:
            return None
        k = min(max_obs, len(self._archive))
        rows = []
        ys = []
        for cfg, loss in self._archive[:k]:
            rows.append(self.to_unit(self.config_to_vector(cfg)))
            ys.append(float(loss))
        return np.vstack(rows), np.asarray(ys, dtype=np.float64)

    def gp_training_data_from(
        self,
        data: list[tuple[dict, float]],
        max_obs: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Build GP training set from full evaluation history (not elite-only archive)."""
        if not self._continuous_indices or len(data) < 2:
            return None

        n = len(data)
        k = min(max_obs, n)
        if n <= k:
            indices = list(range(n))
        else:
            half = k // 2
            by_loss = sorted(range(n), key=lambda i: data[i][1])
            best_idx = by_loss[:half]
            recent_idx = list(range(n - (k - half), n))
            indices: list[int] = []
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

        rows = [self.to_unit(self.config_to_vector(data[i][0])) for i in indices]
        ys = [float(data[i][1]) for i in indices]
        return np.vstack(rows), np.asarray(ys, dtype=np.float64)


# ---------------------------------------------------------------------------
# MALA helpers
# ---------------------------------------------------------------------------


def _log_mala_density(
    x_from: np.ndarray,
    x_to: np.ndarray,
    grad_logpi_from: np.ndarray,
    s: np.ndarray,
    eps: float,
) -> float:
    """Log density of preconditioned MALA proposal q(x_to | x_from)."""
    s_safe = np.maximum(s, 1e-12)
    var = (eps * s_safe) ** 2
    mean = x_from + 0.5 * (eps ** 2) * (s_safe ** 2) * grad_logpi_from
    diff = x_to - mean
    d = x_from.size
    return float(
        -0.5 * np.sum(diff * diff / var)
        - np.sum(np.log(eps * s_safe))
        - 0.5 * d * np.log(2.0 * np.pi)
    )


def _mala_propose(
    x: np.ndarray,
    grad_logpi: np.ndarray,
    s: np.ndarray,
    eps: float,
    rng: np.random.Generator,
) -> np.ndarray:
    s_safe = np.maximum(s, 1e-12)
    drift = 0.5 * (eps ** 2) * (s_safe ** 2) * grad_logpi
    noise = eps * s_safe * rng.standard_normal(size=x.shape)
    return x + drift + noise


# ---------------------------------------------------------------------------
# GP-MALA MCMC chain
# ---------------------------------------------------------------------------


class GpMalaChain(MCMCChain):
    """MCMC chain with GP-guided preconditioned MALA in EXPLOIT / EXPLORE."""

    def __init__(
        self,
        *args,
        precond: DiagRAMPreconditioner | None = None,
        use_gp_mala: bool = True,
        mala_step_size: float = 0.1,
        gp_target_temperature: float = 1.0,
        gp_argmax_every: int = 1,
        gp_argmax_restarts: int = 5,
        gp_argmax_steps: int = 50,
        gp_surrogate: GPSurrogate | None = None,
        tr_enable: bool = True,
        tr_length_init: float = 0.8,
        tr_length_min: float = 0.05,
        tr_length_max: float = 1.6,
        tr_success_tol: int = 3,
        tr_failure_tol: int = 4,
        tr_improve_tol: float = 1e-3,
        acq_anneal: bool = True,
        acq_greedy_after: float = 0.85,
        tr_restart_candidates: int = 64,
        tr_restart_kappa_start: float = 3.0,
        tr_restart_kappa_end: float = 0.5,
        bo_warmup_active: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.precond = precond
        self.use_gp_mala = use_gp_mala
        self.mala_step_size = float(mala_step_size)
        self.gp_target_temperature = float(gp_target_temperature)
        self.gp_argmax_every = int(gp_argmax_every)
        self.gp_argmax_restarts = int(gp_argmax_restarts)
        self.gp_argmax_steps = int(gp_argmax_steps)
        self.gp_surrogate = gp_surrogate
        self.tr_enable = bool(tr_enable)
        self.tr_length_init = float(tr_length_init)
        self.tr_length_min = float(tr_length_min)
        self.tr_length_max = float(tr_length_max)
        self.tr_success_tol = int(tr_success_tol)
        self.tr_failure_tol = int(tr_failure_tol)
        self.tr_improve_tol = float(tr_improve_tol)
        self.acq_anneal = bool(acq_anneal)
        self.acq_greedy_after = float(acq_greedy_after)
        self.tr_restart_candidates = int(tr_restart_candidates)
        self.tr_restart_kappa_start = float(tr_restart_kappa_start)
        self.tr_restart_kappa_end = float(tr_restart_kappa_end)
        self.bo_warmup_active = bool(bo_warmup_active)
        self.tr_length = float(tr_length_init)
        self.tr_success = 0
        self.tr_failure = 0
        self.tr_center_x = copy.deepcopy(self.current_x)
        self.tr_center_loss = float(self.current_loss)
        self._progress: float = 0.0
        self._last_kernel: str = "factorized"
        self._eval_kernels: list[str] = []
        self._exploit_steps: int = 0
        self._rng = np.random.default_rng()

    def _anneal(self, start: float, end: float) -> float:
        """Linear interpolation from ``start`` (early) to ``end`` (late budget)."""
        if not self.acq_anneal:
            return float(start)
        p = float(np.clip(self._progress, 0.0, 1.0))
        return float(start + (end - start) * p)

    def _prepare_step(self, progress: float, is_burnin: bool) -> float:
        if is_burnin:
            self.T_mcmc = self.T_mcmc_init
        elif self.anneal_T:
            self.T_mcmc = self.T_min + (self.T_mcmc_init - self.T_min) * (1.0 - progress)
        else:
            self.T_mcmc = self.T_mcmc_init

        if len(self._observations) >= 2:
            self.state = self.hmm.observe(self._observations[-self.hmm.window :])
        else:
            self.state = self.hmm.state

        if not is_burnin and self._rejection_streak >= self._stagnation_limit:
            if self.state != HMMState.TRAPPED:
                self.state = HMMState.TRAPPED
                self.hmm.force_state(HMMState.TRAPPED)
            n_over = self._rejection_streak - self._stagnation_limit + 1
            boost_factor = 1.0 + min(n_over * 3.0, 50.0)
            self.T_mcmc = self.T_mcmc_init * boost_factor

        return self.T_mcmc

    def _effective_p_cat_step(self) -> float:
        gen = self.proposal_gen
        if isinstance(gen, GpProposalGenerator) and not gen.has_categorical:
            return 0.0
        return self.p_cat_step

    def _factorized_step(
        self, objective_func, T_effective: float, is_burnin: bool
    ) -> tuple[list[tuple[dict, float]], bool]:
        self._last_kernel = "factorized"
        cat_only = np.random.rand() < self._effective_p_cat_step()
        x_prime = self.proposal_gen.generate_proposal(
            self.current_x, self.state, categorical_only=cat_only
        )
        loss_prime = objective_func(x_prime)
        self.current_loss_old_for_hmm = self.current_loss

        if self.state == HMMState.EXPLORE and not is_burnin:
            alpha = 1.0 if loss_prime <= self.current_loss else 0.0
        else:
            alpha = self._acceptance_probability(x_prime, loss_prime, T_effective)

        accepted = np.random.rand() < alpha
        if accepted:
            self.current_x = copy.deepcopy(x_prime)
            self.current_loss = loss_prime
            self._rejection_streak = 0
        else:
            self._rejection_streak += 1

        delta_loss = loss_prime - self.current_loss_old_for_hmm
        O_t = float(delta_loss / (self.scale_factor + 1e-8))
        self._observations.append(O_t)

        if loss_prime < self.best_loss:
            self.best_x = copy.deepcopy(x_prime)
            self.best_loss = loss_prime

        return [(x_prime, loss_prime)], accepted

    def _gp_mala_step(
        self,
        objective_func,
        T_effective: float,
        gp: GPSurrogate,
        mala_state: HMMState,
    ) -> tuple[list[tuple[dict, float]], bool]:
        gen = self.proposal_gen
        assert isinstance(gen, GpProposalGenerator)
        assert self.precond is not None

        self._last_kernel = (
            "gp_mala_exploit" if mala_state == HMMState.EXPLOIT else "gp_mala_explore"
        )

        if np.random.rand() < self._effective_p_cat_step():
            x_prime = gen.generate_proposal(
                self.current_x, mala_state, categorical_only=True
            )
            loss_prime = objective_func(x_prime)
            self.current_loss_old_for_hmm = self.current_loss
            alpha = self._acceptance_probability(x_prime, loss_prime, T_effective)
            accepted = np.random.rand() < alpha
            if accepted:
                self.current_x = copy.deepcopy(x_prime)
                self.current_loss = loss_prime
                self._rejection_streak = 0
            else:
                self._rejection_streak += 1
            delta_loss = loss_prime - self.current_loss_old_for_hmm
            self._observations.append(float(delta_loss / (self.scale_factor + 1e-8)))
            if loss_prime < self.best_loss:
                self.best_x = copy.deepcopy(x_prime)
                self.best_loss = loss_prime
            return [(x_prime, loss_prime)], accepted

        x_before = copy.deepcopy(self.current_x)
        u_before = gen.to_unit(gen.config_to_vector(x_before))
        T_gp = max(self.gp_target_temperature, 1e-12)

        acq_before, grad_acq_before = gp.acq_and_grad(u_before, mala_state, T_gp)
        grad_logpi_before = grad_acq_before / T_gp

        s = self.precond.s.copy()
        eps = self.mala_step_size
        u_raw = _mala_propose(u_before, grad_logpi_before, s, eps, self._rng)
        u_prime = gen.reflect_unit(u_raw)

        x_prime = gen.vector_to_config(gen.from_unit(u_prime), x_before)
        loss_prime = objective_func(x_prime)
        self.current_loss_old_for_hmm = self.current_loss

        acq_prime, grad_acq_prime = gp.acq_and_grad(u_prime, mala_state, T_gp)
        grad_logpi_prime = grad_acq_prime / T_gp

        logpi_before = acq_before / T_gp
        logpi_prime = acq_prime / T_gp

        log_q_fwd = _log_mala_density(u_before, u_prime, grad_logpi_before, s, eps)
        log_q_rev = _log_mala_density(u_prime, u_before, grad_logpi_prime, s, eps)
        log_alpha = (logpi_prime - logpi_before) + (log_q_rev - log_q_fwd)
        alpha = float(np.exp(min(log_alpha, 0.0)))

        accepted = self._rng.random() < alpha
        if accepted:
            self.current_x = copy.deepcopy(x_prime)
            self.current_loss = loss_prime
            self._rejection_streak = 0
        else:
            self._rejection_streak += 1

        s_safe = np.maximum(s, 1e-12)
        z = (u_prime - u_before) / s_safe
        self.precond.update(alpha, z)

        delta_loss = loss_prime - self.current_loss_old_for_hmm
        O_t = float(delta_loss / (self.scale_factor + 1e-8))
        self._observations.append(O_t)

        if loss_prime < self.best_loss:
            self.best_x = copy.deepcopy(x_prime)
            self.best_loss = loss_prime

        return [(x_prime, loss_prime)], accepted

    def _tr_bounds(self, gp: GPSurrogate, gen: GpProposalGenerator) -> np.ndarray:
        """ARD-weighted trust-region box in unit-cube coordinates, shape (2, d)."""
        d = gen.continuous_dim
        if not self.tr_enable or d == 0:
            return np.vstack([np.zeros(d), np.ones(d)])

        u_c = gen.to_unit(gen.config_to_vector(self.tr_center_x))
        ard = gp.ard_scales()
        if ard is not None and ard.size == d:
            ls = 1.0 / np.maximum(ard, 1e-12)
            log_w = np.log(np.maximum(ls, 1e-12))
            log_w -= np.mean(log_w)
            w = np.exp(log_w)
        else:
            w = np.ones(d, dtype=np.float64)

        half_w = 0.5 * self.tr_length * w
        lo = np.clip(u_c - half_w, 0.0, 1.0)
        hi = np.clip(u_c + half_w, 0.0, 1.0)
        hi = np.maximum(hi, lo + 1e-12)
        return np.vstack([lo, hi])

    def _tr_restart(
        self,
        objective_func,
        gen: GpProposalGenerator,
        gp: GPSurrogate | None,
    ) -> tuple[dict, float]:
        """Relocate trust region after collapse; returns one evaluated config."""
        d = gen.continuous_dim
        if (
            gp is not None
            and gp.is_ready
            and self.tr_enable
            and self.tr_restart_candidates > 0
        ):
            cands = self._rng.random((self.tr_restart_candidates, d))
            kappa = self._anneal(self.tr_restart_kappa_start, self.tr_restart_kappa_end)
            u_new = gp.select_explore_point(cands, kappa)
        else:
            u_new = self._rng.random(d)
        x_new = gen.vector_to_config(gen.from_unit(u_new), self.current_x)
        loss_new = objective_func(x_new)

        self.tr_center_x = copy.deepcopy(x_new)
        self.tr_center_loss = float(loss_new)
        self.tr_length = self.tr_length_init
        self.tr_success = 0
        self.tr_failure = 0
        if self.precond is not None:
            self.precond.init_from_std(gen.default_initial_scales_unit())

        if loss_new < self.best_loss:
            self.best_x = copy.deepcopy(x_new)
            self.best_loss = loss_new

        return x_new, loss_new

    def _tr_update(self, x_prime: dict, loss_prime: float) -> None:
        """Update trust-region length and local incumbent after an argmax eval."""
        tol = self.tr_improve_tol * max(abs(self.tr_center_loss), 1e-12)
        improved = loss_prime < self.tr_center_loss - tol

        if improved:
            self.tr_center_x = copy.deepcopy(x_prime)
            self.tr_center_loss = float(loss_prime)
            self.tr_success += 1
            self.tr_failure = 0
            if self.tr_success >= self.tr_success_tol:
                self.tr_length = min(self.tr_length * 2.0, self.tr_length_max)
                self.tr_success = 0
        else:
            self.tr_failure += 1
            self.tr_success = 0
            if self.tr_failure >= self.tr_failure_tol:
                self.tr_length *= 0.5
                self.tr_failure = 0

    def _gp_argmax_step(
        self,
        objective_func,
        gp: GPSurrogate,
    ) -> tuple[list[tuple[dict, float]], bool]:
        gen = self.proposal_gen
        assert isinstance(gen, GpProposalGenerator)

        self._last_kernel = "gp_argmax"
        bounds = self._tr_bounds(gp, gen)
        greedy = self.acq_anneal and self._progress >= self.acq_greedy_after
        u_star = gp.optimize_ei(
            n_restarts=self.gp_argmax_restarts,
            raw_samples=self.gp_argmax_steps,
            bounds=bounds,
            greedy=greedy,
        )
        x_prime = gen.vector_to_config(gen.from_unit(u_star), self.current_x)
        loss_prime = objective_func(x_prime)
        self.current_loss_old_for_hmm = self.current_loss

        accepted = loss_prime <= self.current_loss
        if accepted:
            self.current_x = copy.deepcopy(x_prime)
            self.current_loss = loss_prime
            self._rejection_streak = 0

        delta_loss = loss_prime - self.current_loss_old_for_hmm
        O_t = float(delta_loss / (self.scale_factor + 1e-8))
        self._observations.append(O_t)

        if loss_prime < self.best_loss:
            self.best_x = copy.deepcopy(x_prime)
            self.best_loss = loss_prime

        evals: list[tuple[dict, float]] = [(x_prime, loss_prime)]
        self._eval_kernels = ["gp_argmax"]

        if self.tr_enable:
            self._tr_update(x_prime, loss_prime)
            if self.tr_length < self.tr_length_min:
                restart_cfg, restart_loss = self._tr_restart(objective_func, gen, gp)
                evals.append((restart_cfg, restart_loss))
                self._last_kernel = "gp_tr_restart"
                self._eval_kernels.append("gp_tr_restart")

        return evals, accepted

    def step(
        self,
        objective_func,
        progress: float = 0.0,
        is_burnin: bool = False,
    ) -> tuple[list[tuple[dict, float]], bool]:
        self._progress = float(progress)
        T_effective = self._prepare_step(progress, is_burnin)

        gp_ready = (
            self.gp_surrogate is not None
            and self.gp_surrogate.is_ready
        )
        gen = self.proposal_gen
        has_cont = isinstance(gen, GpProposalGenerator) and gen.continuous_dim > 0

        if (
            self.bo_warmup_active
            and self.use_gp_mala
            and not is_burnin
            and gp_ready
            and has_cont
        ):
            assert self.gp_surrogate is not None
            return self._gp_argmax_step(objective_func, self.gp_surrogate)

        use_mala_now = (
            self.use_gp_mala
            and not is_burnin
            and gp_ready
            and self.state in (HMMState.EXPLOIT, HMMState.EXPLORE)
            and self.precond is not None
            and has_cont
        )

        if use_mala_now:
            assert self.gp_surrogate is not None
            if self.state == HMMState.EXPLOIT:
                self._exploit_steps += 1
                if (
                    self.gp_argmax_every > 0
                    and self._exploit_steps % self.gp_argmax_every == 0
                ):
                    return self._gp_argmax_step(objective_func, self.gp_surrogate)
            return self._gp_mala_step(
                objective_func, T_effective, self.gp_surrogate, self.state
            )
        return self._factorized_step(objective_func, T_effective, is_burnin)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class HMM_MCMC_GP_MALA(HMM_MCMC_BW):
    """H-MCMC-FMP-BW with GP surrogate + preconditioned MALA + diagonal RAM."""

    def __init__(
        self,
        objective_func,
        budget: int,
        dict_to_optimize: dict,
        n_init: int = 12,
        n_chains: int = 4,
        orchestrate_every: int = 5,
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
        use_baum_welch: bool = True,
        bw_refit_every: int = 5,
        bw_min_obs: int = 12,
        bw_n_em_iters: int = 3,
        bw_prior_strength: float = 25.0,
        bw_exploit_prior_scale: float = 3.0,
        bw_max_len: int = 64,
        use_gp_mala: bool = True,
        ram_target_accept: float = 0.574,
        ram_gamma: float = 0.7,
        ram_s_min: float = 0.02,
        ram_s_max: float = 0.5,
        ram_precond_init: str = "archive",
        gp_refit_every: int = 1,
        gp_min_obs: int = 12,
        gp_max_obs: int = 256,
        gp_matern_nu: float = 2.5,
        gp_kappa_exploit: float = 1.0,
        gp_kappa_explore: float = 2.0,
        bo_warmup_frac: float = 0.25,
        gp_explore_mode: str = "thompson",
        gp_target_temperature: float = 1.0,
        gp_argmax_every: int = 1,
        gp_argmax_restarts: int = 5,
        gp_argmax_steps: int = 50,
        mala_step_size: float = 0.1,
        tr_enable: bool = True,
        tr_length_init: float = 0.8,
        tr_length_min: float = 0.05,
        tr_length_max: float = 1.6,
        tr_success_tol: int = 3,
        tr_failure_tol: int = 4,
        tr_improve_tol: float = 1e-3,
        acq_anneal: bool = True,
        acq_greedy_after: float = 0.85,
        tr_restart_candidates: int = 64,
        tr_restart_kappa_start: float = 3.0,
        tr_restart_kappa_end: float = 0.5,
        show_progress: bool = True,
        progress_desc: str | None = None,
        verbose_history: bool = True,
    ):
        _require_gp()
        super().__init__(
            objective_func=objective_func,
            budget=budget,
            dict_to_optimize=dict_to_optimize,
            n_init=n_init,
            n_chains=n_chains,
            orchestrate_every=orchestrate_every,
            T_mcmc=T_mcmc,
            T_min=T_min,
            anneal_T=anneal_T,
            burnin_fraction=burnin_fraction,
            sigma_fraction=sigma_fraction,
            temperature=temperature,
            hmm_window=hmm_window,
            hmm_obs_epsilon=hmm_obs_epsilon,
            hmm_lambda_noise=hmm_lambda_noise,
            clone_noise=clone_noise,
            wide_sigma_fraction=wide_sigma_fraction,
            p_cat_step=p_cat_step,
            kde_tau=kde_tau,
            rejection_streak=rejection_streak,
            use_baum_welch=use_baum_welch,
            bw_refit_every=bw_refit_every,
            bw_min_obs=bw_min_obs,
            bw_n_em_iters=bw_n_em_iters,
            bw_prior_strength=bw_prior_strength,
            bw_exploit_prior_scale=bw_exploit_prior_scale,
            bw_max_len=bw_max_len,
            show_progress=show_progress,
            progress_desc=progress_desc,
            verbose_history=verbose_history,
        )
        self.use_gp_mala = use_gp_mala
        self.ram_target_accept = ram_target_accept
        self.ram_gamma = ram_gamma
        self.ram_s_min = ram_s_min
        self.ram_s_max = ram_s_max
        self.ram_precond_init = ram_precond_init
        self.gp_refit_every = gp_refit_every
        self.gp_min_obs = gp_min_obs
        self.gp_max_obs = gp_max_obs
        self.gp_matern_nu = gp_matern_nu
        self.gp_kappa_exploit = gp_kappa_exploit
        self.gp_kappa_explore = gp_kappa_explore
        self.gp_explore_mode = gp_explore_mode
        self.gp_target_temperature = gp_target_temperature
        self.gp_argmax_every = gp_argmax_every
        self.gp_argmax_restarts = gp_argmax_restarts
        self.gp_argmax_steps = gp_argmax_steps
        self.mala_step_size = mala_step_size
        self.tr_enable = tr_enable
        self.tr_length_init = tr_length_init
        self.tr_length_min = tr_length_min
        self.tr_length_max = tr_length_max
        self.tr_success_tol = tr_success_tol
        self.tr_failure_tol = tr_failure_tol
        self.tr_improve_tol = tr_improve_tol
        self.acq_anneal = acq_anneal
        self.acq_greedy_after = acq_greedy_after
        self.tr_restart_candidates = tr_restart_candidates
        self.tr_restart_kappa_start = tr_restart_kappa_start
        self.tr_restart_kappa_end = tr_restart_kappa_end
        self.bo_warmup_frac = float(bo_warmup_frac)
        self._ram_config = DiagRAMConfig(
            target_accept=ram_target_accept,
            gamma=ram_gamma,
            s_min_frac=ram_s_min,
            s_max_frac=ram_s_max,
        )
        self._gp_config = GPConfig(
            matern_nu=gp_matern_nu,
            refit_every=gp_refit_every,
            min_obs=gp_min_obs,
            max_obs=gp_max_obs,
            kappa_exploit=gp_kappa_exploit,
            kappa_explore=gp_kappa_explore,
            explore_mode=gp_explore_mode,
        )
        self._gp_surrogate: GPSurrogate | None = None
        self._evals_since_gp_refit: int = 0

    def _make_proposal_generator(self) -> GpProposalGenerator:
        return GpProposalGenerator(
            self.dict_to_optimize,
            sigma_fraction=self._sigma_fraction,
            temperature=self._temperature,
            wide_sigma_fraction=self._wide_sigma_fraction,
            kde_tau=self._kde_tau,
        )

    def _make_preconditioner(self, gen: GpProposalGenerator) -> DiagRAMPreconditioner | None:
        if gen.continuous_dim == 0:
            return None

        unit_ranges = np.ones(gen.continuous_dim, dtype=np.float64)
        if self.ram_precond_init == "archive" and len(gen._archive) >= 2:
            initial_s = gen.elite_marginal_std_unit()
        else:
            initial_s = gen.default_initial_scales_unit()

        return DiagRAMPreconditioner(
            dim=gen.continuous_dim,
            ranges=unit_ranges,
            config=self._ram_config,
            initial_s=initial_s,
        )

    def _maybe_refit_gp(self, gen: GpProposalGenerator) -> None:
        if not self.use_gp_mala:
            return
        if len(self.data) < self.gp_min_obs:
            return
        if self._evals_since_gp_refit < self.gp_refit_every and self._gp_surrogate is not None:
            return

        train = gen.gp_training_data_from(self.data, self.gp_max_obs)
        if train is None:
            return
        X, y = train
        if self._gp_surrogate is None:
            self._gp_surrogate = GPSurrogate(gen.continuous_dim, self._gp_config)
        self._gp_surrogate.fit(X, y)
        self._evals_since_gp_refit = 0

        if self.ram_precond_init == "ard" and self._gp_surrogate.is_ready:
            ard = self._gp_surrogate.ard_scales()
            if ard is not None:
                scaled = np.clip(ard, self.ram_s_min, self.ram_s_max)
                for chain in self._chains:
                    if chain.precond is not None:
                        chain.precond.init_from_std(scaled)

    def main_loop(self):
        """Main loop with GP-MALA proposals and Baum-Welch HMM."""
        self._sobol_init = SobolInitializer(self.dict_to_optimize)
        self._proposal_gen = self._make_proposal_generator()
        self.gp_max_obs = max(int(self.gp_max_obs), int(self.budget))
        self._gp_config.max_obs = self.gp_max_obs

        pbar = tqdm(
            total=self.budget,
            desc=self.progress_desc or "H-MCMC-GP-MALA",
            disable=not self.show_progress,
        )

        init_scores = []
        for cfg in self._sobol_init.generate(self.n_init):
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
            precond = self._make_preconditioner(self._proposal_gen)
            chain = GpMalaChain(
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
                precond=precond,
                use_gp_mala=self.use_gp_mala,
                mala_step_size=self.mala_step_size,
                gp_target_temperature=self.gp_target_temperature,
                gp_argmax_every=self.gp_argmax_every,
                gp_argmax_restarts=self.gp_argmax_restarts,
                gp_argmax_steps=self.gp_argmax_steps,
                gp_surrogate=self._gp_surrogate,
                tr_enable=self.tr_enable,
                tr_length_init=self.tr_length_init,
                tr_length_min=self.tr_length_min,
                tr_length_max=self.tr_length_max,
                tr_success_tol=self.tr_success_tol,
                tr_failure_tol=self.tr_failure_tol,
                tr_improve_tol=self.tr_improve_tol,
                acq_anneal=self.acq_anneal,
                acq_greedy_after=self.acq_greedy_after,
                tr_restart_candidates=self.tr_restart_candidates,
                tr_restart_kappa_start=self.tr_restart_kappa_start,
                tr_restart_kappa_end=self.tr_restart_kappa_end,
            )
            self._chains.append(chain)

        self._maybe_refit_gp(self._proposal_gen)
        for chain in self._chains:
            chain.gp_surrogate = self._gp_surrogate

        self._orchestrator = GlobalOrchestrator(
            chains=self._chains,
            proposal_gen=self._proposal_gen,
            sobol_init=self._sobol_init,
            dict_to_optimize=self.dict_to_optimize,
            objective_func=self.objective_func,
            clone_noise=self._clone_noise,
        )

        step_counter = 0
        bo_warmup_evals = round(self.bo_warmup_frac * self.budget)

        while len(self.data) < self.budget:
            self._maybe_refit_gp(self._proposal_gen)
            for chain in self._chains:
                chain.gp_surrogate = self._gp_surrogate

            post_init = max(0, len(self.data) - self.n_init)
            bo_warmup_on = post_init < bo_warmup_evals

            for chain in self._chains:
                if len(self.data) >= self.budget:
                    break

                chain.bo_warmup_active = bo_warmup_on
                progress = len(self.data) / self.budget
                is_burnin = progress < self.burnin_fraction

                evals, accepted = chain.step(
                    self.objective_func,
                    progress=progress,
                    is_burnin=is_burnin,
                )

                for i, (cfg, loss) in enumerate(evals):
                    if len(self.data) >= self.budget:
                        break
                    self.data.append((cfg, loss))
                    self._proposal_gen.update_category_history(cfg, loss)
                    self._evals_since_gp_refit += 1

                    eval_kernels = getattr(chain, "_eval_kernels", None)
                    if eval_kernels and i < len(eval_kernels):
                        kernel = eval_kernels[i]
                    else:
                        kernel = getattr(chain, "_last_kernel", "factorized")
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
                        self._proposal_gen.update_category_history(cfg, loss)
                        self._evals_since_gp_refit += 1

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
                print("HMM MCMC GP-MALA State History (last run):")
                with pd.option_context("display.max_rows", 200, "display.max_columns", None):
                    print(df)
                print("=" * 60 + "\n")
            except ImportError:
                pass

        return self._decode_best(min(self.data, key=lambda x: x[1]))


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------


def _self_check_gp_fit_and_grad() -> None:
    _require_gp()
    rng = np.random.default_rng(0)
    X = rng.random((20, 2))
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    gp = GPSurrogate(2, GPConfig(min_obs=5, max_obs=20))
    gp.fit(X, y)
    assert gp.is_ready
    assert gp._best_f is not None
    acq, grad = gp.acq_and_grad(X[0], HMMState.EXPLOIT, temperature=1.0)
    assert np.isfinite(acq), "acquisition must be finite"
    assert grad.shape == (2,)
    assert np.all(np.isfinite(grad)), "gradient must be finite"
    print(f"[self-check] GP fit+grad: acq={acq:.4f}, ||grad||={float(np.linalg.norm(grad)):.4f}")


def _self_check_mala_step() -> None:
    _require_gp()
    space = {
        "x0": {"type": "float", "values": [-5.0, 5.0]},
        "x1": {"type": "float", "values": [0.0, 1.0]},
    }
    gen = GpProposalGenerator(space, sigma_fraction=0.1)
    rng = np.random.default_rng(1)
    for i in range(15):
        cfg = {
            "x0": float(rng.uniform(-4, 4)),
            "x1": float(rng.uniform(0.05, 0.95)),
        }
        gen.update_category_history(cfg, float(cfg["x0"] ** 2 + cfg["x1"]))

    train = gen.gp_training_data(15)
    assert train is not None
    gp = GPSurrogate(2, GPConfig(min_obs=5))
    gp.fit(*train)

    precond = DiagRAMPreconditioner(2, np.ones(2), initial_s=0.1 * np.ones(2))

    def loss_fn(cfg):
        return float(cfg["x0"] ** 2 + cfg["x1"])

    chain = GpMalaChain(
        chain_id=0,
        x0={"x0": 0.5, "x1": 0.5},
        loss0=0.75,
        proposal_gen=gen,
        hmm=BaumWelchHMMController(min_obs=10**9, refit_every=10**9),
        T_mcmc=1.0,
        scale_factor=1.0,
        precond=precond,
        use_gp_mala=True,
        mala_step_size=0.1,
        gp_target_temperature=1.0,
        gp_argmax_every=100,
        gp_surrogate=gp,
    )
    chain.state = HMMState.EXPLOIT
    chain.hmm.force_state(HMMState.EXPLOIT)

    evals, _ = chain.step(loss_fn, progress=0.5, is_burnin=False)
    assert chain._last_kernel in {"gp_mala_exploit", "gp_argmax"}
    assert len(evals) == 1
    cfg, _ = evals[0]
    assert -5.0 <= cfg["x0"] <= 5.0
    assert 0.0 <= cfg["x1"] <= 1.0

    chain.state = HMMState.EXPLORE
    chain.hmm.force_state(HMMState.EXPLORE)
    evals2, _ = chain.step(loss_fn, progress=0.5, is_burnin=False)
    assert chain._last_kernel == "gp_mala_explore"
    assert len(evals2) == 1
    print("[self-check] MALA step exploit/explore: OK")


def _self_check_hastings_symmetry() -> None:
    rng = np.random.default_rng(2)
    s = np.array([0.1, 0.2])
    eps = 0.05
    x = rng.random(2)
    grad = np.zeros(2)
    y_raw = _mala_propose(x, grad, s, eps, rng)
    y = reflect_unit(y_raw)
    log_fwd = _log_mala_density(x, y, grad, s, eps)
    log_rev = _log_mala_density(y, x, grad, s, eps)
    assert abs(log_fwd - log_rev) < 1e-6, "flat target should give symmetric Hastings"
    print("[self-check] Hastings symmetry (flat target): OK")


def _self_check_argmax_in_bounds() -> None:
    _require_gp()
    rng = np.random.default_rng(3)
    X = rng.random((20, 2))
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    gp = GPSurrogate(2, GPConfig(min_obs=5))
    gp.fit(X, y)
    u_star = gp.optimize_ei(n_restarts=3, raw_samples=64)
    assert u_star.shape == (2,)
    assert np.all(u_star >= 0.0) and np.all(u_star <= 1.0)

    # Quadratic bowl: EI argmax should land near the unit-cube minimum (0, 0)
    assert float(np.linalg.norm(u_star)) < 0.35, (
        f"EI argmax should be near optimum, got u={u_star}"
    )

    space = {"x": {"type": "float", "values": [-1.0, 1.0]}}
    gen = GpProposalGenerator(space, sigma_fraction=0.1)
    data = []
    for i in range(12):
        cfg = {"x": float(rng.uniform(-0.9, 0.9))}
        loss = float(cfg["x"] ** 2)
        gen.update_category_history(cfg, loss)
        data.append((cfg, loss))
    train = gen.gp_training_data_from(data, 12)
    assert train is not None
    gp2 = GPSurrogate(1, GPConfig(min_obs=5))
    gp2.fit(*train)

    precond = DiagRAMPreconditioner(1, np.ones(1), initial_s=np.array([0.1]))
    chain = GpMalaChain(
        chain_id=0,
        x0={"x": 0.0},
        loss0=0.0,
        proposal_gen=gen,
        hmm=BaumWelchHMMController(min_obs=10**9, refit_every=10**9),
        T_mcmc=1.0,
        scale_factor=1.0,
        precond=precond,
        use_gp_mala=True,
        gp_argmax_every=1,
        gp_surrogate=gp2,
    )
    chain.state = HMMState.EXPLOIT
    chain.hmm.force_state(HMMState.EXPLOIT)
    chain._exploit_steps = 0
    evals, _ = chain.step(lambda c: float(c["x"] ** 2), progress=0.5, is_burnin=False)
    assert chain._last_kernel == "gp_argmax"
    assert len(evals) == 1
    print("[self-check] argmax in-bounds: OK")


def _self_check_thompson_freshness() -> None:
    _require_gp()
    rng = np.random.default_rng(4)
    X = rng.random((25, 2))
    y = (X[:, 0] ** 2 + 0.5 * X[:, 1]).astype(np.float64)
    gp = GPSurrogate(2, GPConfig(min_obs=5, explore_mode="thompson"))
    gp.fit(X, y)
    u = rng.random(2)
    a1, _ = gp.acq_and_grad(u, HMMState.EXPLORE, temperature=1.0)
    a2, _ = gp.acq_and_grad(u, HMMState.EXPLORE, temperature=1.0)
    assert a1 != a2, "EXPLORE Thompson samples should differ per call"
    print("[self-check] Thompson freshness: OK")


def _self_check_optimize_ei_bounds() -> None:
    _require_gp()
    rng = np.random.default_rng(5)
    X = rng.random((20, 2))
    y = (X[:, 0] ** 2 + X[:, 1]).astype(np.float64)
    gp = GPSurrogate(2, GPConfig(min_obs=5))
    gp.fit(X, y)
    tight = np.array([[0.1, 0.2], [0.4, 0.5]], dtype=np.float64)
    u_star = gp.optimize_ei(n_restarts=3, raw_samples=64, bounds=tight)
    assert np.all(u_star >= tight[0] - 1e-6) and np.all(u_star <= tight[1] + 1e-6)
    print("[self-check] optimize_ei bounds: OK")


def _self_check_tr_shrink_expand() -> None:
    _require_gp()
    rng = np.random.default_rng(6)
    space = {
        "x0": {"type": "float", "values": [0.0, 1.0]},
        "x1": {"type": "float", "values": [0.0, 1.0]},
    }
    gen = GpProposalGenerator(space, sigma_fraction=0.1)
    data = []
    for _ in range(20):
        cfg = {"x0": float(rng.uniform(0.2, 0.8)), "x1": float(rng.uniform(0.2, 0.8))}
        loss = float((cfg["x0"] - 0.2) ** 2 + (cfg["x1"] - 0.3) ** 2)
        gen.update_category_history(cfg, loss)
        data.append((cfg, loss))

    train = gen.gp_training_data_from(data, 20)
    assert train is not None
    gp = GPSurrogate(2, GPConfig(min_obs=5))
    gp.fit(*train)

    precond = DiagRAMPreconditioner(2, np.ones(2), initial_s=0.1 * np.ones(2))

    def loss_fn(cfg):
        return float((cfg["x0"] - 0.2) ** 2 + (cfg["x1"] - 0.3) ** 2)

    chain = GpMalaChain(
        chain_id=0,
        x0={"x0": 0.7, "x1": 0.7},
        loss0=0.5,
        proposal_gen=gen,
        hmm=BaumWelchHMMController(min_obs=10**9, refit_every=10**9),
        T_mcmc=1.0,
        scale_factor=1.0,
        precond=precond,
        use_gp_mala=True,
        gp_surrogate=gp,
        tr_enable=True,
        tr_length_init=0.2,
        tr_length_min=0.08,
        tr_length_max=0.4,
        tr_success_tol=10,
        tr_failure_tol=1,
        tr_improve_tol=1e-6,
    )
    chain.state = HMMState.EXPLOIT
    chain.hmm.force_state(HMMState.EXPLOIT)

    length_before = chain.tr_length
    evals, _ = chain._gp_argmax_step(loss_fn, gp)
    assert chain.tr_length < length_before, "TR should shrink after non-improving argmax"
    assert len(evals) == 1

    chain.tr_length = chain.tr_length_min * 0.4
    evals2, _ = chain._gp_argmax_step(loss_fn, gp)
    assert len(evals2) == 2, "collapse should append restart eval"
    assert chain._last_kernel == "gp_tr_restart"
    assert "gp_tr_restart" in chain._eval_kernels
    print("[self-check] TR shrink/expand/restart: OK")


def _self_check_restart_uncertainty() -> None:
    _require_gp()
    rng = np.random.default_rng(7)
    X = rng.uniform(0.75, 0.95, size=(12, 2))
    y = ((X[:, 0] - 0.85) ** 2 + (X[:, 1] - 0.85) ** 2).astype(np.float64)
    gp = GPSurrogate(2, GPConfig(min_obs=5))
    gp.fit(X, y)

    cands = rng.random((128, 2))
    u_explore = gp.select_explore_point(cands, kappa=20.0)
    u_exploit = gp.select_explore_point(cands, kappa=0.0)

    def _sigma(u: np.ndarray) -> float:
        x = torch.as_tensor(u.reshape(1, -1), dtype=gp._dtype, device=gp._device)
        with torch.no_grad():
            post = gp.model.posterior(x)
            return float(post.variance.clamp_min(gp.config.jitter).sqrt().item())

    assert _sigma(u_explore) >= _sigma(u_exploit), (
        f"high kappa should favor higher-uncertainty points: "
        f"sigma_explore={_sigma(u_explore):.4f}, sigma_exploit={_sigma(u_exploit):.4f}"
    )
    assert np.any(np.linalg.norm(cands - u_explore, axis=1) < 1e-9), (
        "select_explore_point must return one of the candidates"
    )
    print("[self-check] restart uncertainty: OK")


def _self_check_greedy_argmax() -> None:
    _require_gp()
    rng = np.random.default_rng(8)
    X = rng.uniform(0.0, 0.15, size=(20, 2))
    y = (X[:, 0] ** 2 + X[:, 1] ** 2).astype(np.float64)
    gp = GPSurrogate(2, GPConfig(min_obs=5))
    gp.fit(X, y)

    u_greedy = gp.optimize_ei(n_restarts=3, raw_samples=64, greedy=True)
    u_ei = gp.optimize_ei(n_restarts=3, raw_samples=64, greedy=False)
    assert float(np.linalg.norm(u_greedy)) < 0.5, (
        f"greedy argmax should stay near observed optimum, got {u_greedy}"
    )
    assert not np.allclose(u_greedy, u_ei, atol=0.05), (
        "greedy and EI argmax should differ on this setup"
    )
    print("[self-check] greedy argmax: OK")


def _self_check_bo_warmup_routing() -> None:
    _require_gp()
    rng = np.random.default_rng(9)
    space = {"x": {"type": "float", "values": [-1.0, 1.0]}}
    gen = GpProposalGenerator(space, sigma_fraction=0.1)
    data = []
    for i in range(12):
        cfg = {"x": float(rng.uniform(-0.9, 0.9))}
        loss = float(cfg["x"] ** 2)
        gen.update_category_history(cfg, loss)
        data.append((cfg, loss))
    train = gen.gp_training_data_from(data, 12)
    assert train is not None
    gp = GPSurrogate(1, GPConfig(min_obs=5))
    gp.fit(*train)

    precond = DiagRAMPreconditioner(1, np.ones(1), initial_s=0.1 * np.ones(1))
    chain = GpMalaChain(
        chain_id=0,
        x0={"x": 0.0},
        loss0=0.0,
        proposal_gen=gen,
        hmm=BaumWelchHMMController(min_obs=10**9, refit_every=10**9),
        T_mcmc=1.0,
        scale_factor=1.0,
        precond=precond,
        use_gp_mala=True,
        gp_surrogate=gp,
        bo_warmup_active=True,
    )
    chain.state = HMMState.EXPLORE
    chain.hmm.force_state(HMMState.EXPLORE)
    evals, _ = chain.step(lambda c: float(c["x"] ** 2), progress=0.1, is_burnin=False)
    assert chain._last_kernel == "gp_argmax", (
        f"warmup should force gp_argmax, got {chain._last_kernel}"
    )
    assert len(evals) >= 1
    assert not gen.has_categorical
    print("[self-check] BO warmup routing: OK")


def _self_check_budget_exactness() -> None:
    from hpo_rl.backends.function import OptimizationBenchmarkBackend

    backend = OptimizationBenchmarkBackend(
        function_name="schwefel", dimensions=2, noise_std=0
    )
    algo = HMM_MCMC_GP_MALA(
        objective_func=backend.evaluate,
        budget=35,
        dict_to_optimize=backend.hp_space,
        n_init=5,
        n_chains=2,
        orchestrate_every=1000,
        burnin_fraction=0.0,
        gp_min_obs=5,
        gp_refit_every=1,
        use_baum_welch=False,
        use_gp_mala=True,
        tr_enable=True,
        show_progress=False,
        verbose_history=False,
    )
    algo.main_loop()
    assert len(algo.data) == 35, f"expected 35 evals, got {len(algo.data)}"
    print("[self-check] budget exactness: OK")


def _self_check_ablation_parity() -> None:
    space = {"x": {"type": "float", "values": [-5.0, 5.0]}}

    def quad(cfg):
        return float(cfg["x"] ** 2)

    algo = HMM_MCMC_GP_MALA(
        objective_func=quad,
        budget=20,
        dict_to_optimize=space,
        n_init=4,
        n_chains=1,
        orchestrate_every=1000,
        burnin_fraction=0.0,
        use_baum_welch=False,
        use_gp_mala=False,
        show_progress=False,
        verbose_history=False,
    )
    best_cfg, best_loss = algo.main_loop()
    assert best_loss >= 0.0
    assert len(algo.data) == 20
    kernels = {r.get("Kernel") for r in algo.history_table}
    assert kernels == {"factorized"}, f"expected only factorized kernel, got {kernels}"
    print(f"[self-check] ablation parity: best_loss={best_loss:.4f}")
    print("[self-check] ablation parity: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("HMM_MCMC_GP_MALA self-checks")
    print("=" * 60)
    _self_check_gp_fit_and_grad()
    _self_check_mala_step()
    _self_check_hastings_symmetry()
    _self_check_argmax_in_bounds()
    _self_check_optimize_ei_bounds()
    _self_check_tr_shrink_expand()
    _self_check_restart_uncertainty()
    _self_check_greedy_argmax()
    _self_check_thompson_freshness()
    _self_check_bo_warmup_routing()
    _self_check_budget_exactness()
    _self_check_ablation_parity()
    print("=" * 60)
    print("All self-checks passed.")
    print("=" * 60)
