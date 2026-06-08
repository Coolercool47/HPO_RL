"""H-MCMC-FMP (TEST): Baum-Welch transition learning + spline proposal distributions.

Расширяет :class:`HMM_MCMC` двумя экспериментальными компонентами:

1. **Baum-Welch (A-only)** — онлайн переоценка матрицы переходов ``A`` по цепочке
   наблюдений каждой MCMC-цепи. Эмиссии ``μ/σ`` и ``π`` фиксированы (семантика
   EXPLOIT / EXPLORE / TRAPPED сохраняется).

2. **Monotone-CDF spline proposals** — предлагающие распределения на [lo, hi]
   через монотонную кусочно-линейную CDF (I-spline порядка 1) с inverse-transform
   sampling и точной плотностью для корректного Hastings ratio.
"""

from __future__ import annotations

import numpy as np
from tqdm.auto import tqdm

from hpo_rl.baselines.HMM_MCMC import (
    HMM_MCMC,
    HMMController,
    HMMState,
    FactorizedProposalGenerator,
    SobolInitializer,
    MCMCChain,
    GlobalOrchestrator,
    _logsumexp,
    _tn_rvs,
    _tn_logpdf,
)


# ---------------------------------------------------------------------------
# Baum-Welch HMM controller (A-only, per-chain online)
# ---------------------------------------------------------------------------

class BaumWelchHMMController(HMMController):
    """HMM-контроллер с онлайн Baum-Welch для матрицы переходов ``A``.

    Эмиссионные параметры и ``π`` не переобучаются — только ``A`` с MAP-сглаживанием
    к априору (дефолтная матрица из :class:`HMMController`).
    """

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
        super().__init__(window=window, obs_epsilon=obs_epsilon, lambda_noise=lambda_noise)
        self.refit_every = refit_every
        self.min_obs = min_obs
        self.n_em_iters = n_em_iters
        self.prior_strength = prior_strength
        self.exploit_prior_scale = exploit_prior_scale
        self._bw_max_len = bw_max_len

        self._A_default = self.A.copy()
        self._bw_buffer: list[float] = []
        self._step_count: int = 0

    def reset(self):
        super().reset()
        self._bw_buffer = []
        self._step_count = 0
        self.A = self._A_default.copy()

    def observe(self, obs_history: list[float]) -> HMMState:
        if obs_history:
            self._update_buffer(obs_history)
            self._maybe_refit()

        if len(obs_history) < 2:
            return self.state

        obs_seq = obs_history[-self.window:]
        self.state = HMMState(self._viterbi(obs_seq))
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

    def _fit_transitions(self, obs_seq: list[float]) -> None:
        """Forward-backward EM: обновляет только ``A`` с Dirichlet MAP к ``_A_default``."""
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
                log_beta[t, i] = _logsumexp(
                    log_A[i, :] + log_B[t + 1] + log_beta[t + 1]
                )

        log_norm = _logsumexp(log_alpha[T - 1])
        return log_alpha, log_beta, log_norm


# ---------------------------------------------------------------------------
# Spline proposal generator (monotone CDF, per HMM state)
# ---------------------------------------------------------------------------

class SplineProposalGenerator(FactorizedProposalGenerator):
    """Предложения через монотонную кусочно-линейную CDF на [lo, hi].

    Для каждого состояния HMM строится CDF из:
      - Boltzmann-взвешенной гистограммы архива,
      - локального бампа вокруг текущего ``center``,
      - равномерного пола (эргодичность MH).

    При недостатке архивных точек — фоллбэк на гауссову смесь родителя.
    """

    SPLINE_CONFIG = {
        HMMState.EXPLOIT: {"locality_weight": 0.70, "uniform_floor": 0.05, "archive_weight": 0.25},
        HMMState.EXPLORE: {"locality_weight": 0.10, "uniform_floor": 0.40, "archive_weight": 0.50},
        HMMState.TRAPPED: {"locality_weight": 0.05, "uniform_floor": 0.20, "archive_weight": 0.75},
    }
    SPLINE_MIX_WEIGHT = {
        HMMState.EXPLOIT: 0.15,
        HMMState.EXPLORE: 0.80,
        HMMState.TRAPPED: 0.90,
    }

    def __init__(
        self,
        dict_to_optimize: dict,
        sigma_fraction: float = 0.10,
        temperature: float = 0.60,
        wide_sigma_fraction: float = 0.40,
        kde_tau: float = 0.05,
        spline_knots: int = 60,
        spline_floor: float = 0.05,
        spline_min_archive: int = 5,
        locality_sigma_fraction: float = 0.08,
        spline_mix_scale: float = 1.0,
    ):
        super().__init__(
            dict_to_optimize,
            sigma_fraction=sigma_fraction,
            temperature=temperature,
            wide_sigma_fraction=wide_sigma_fraction,
            kde_tau=kde_tau,
        )
        self.spline_knots = max(4, spline_knots)
        self.spline_floor = float(np.clip(spline_floor, 1e-4, 0.5))
        self.spline_min_archive = spline_min_archive
        self.locality_sigma_fraction = locality_sigma_fraction
        self.spline_mix_scale = float(max(spline_mix_scale, 0.0))
        self._cdf_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    def update_category_history(self, config: dict, loss: float) -> None:
        super().update_category_history(config, loss)
        self._cdf_cache.clear()

    def _mix_weight(self, state: HMMState) -> float:
        base = self.SPLINE_MIX_WEIGHT[state]
        return float(np.clip(base * self.spline_mix_scale, 0.0, 1.0))

    def _sample_narrow_tn(self, x_j: float, pi: dict) -> float:
        lo, hi = pi["lo"], pi["hi"]
        sigma = pi["sigma"]
        a, b = (lo - x_j) / sigma, (hi - x_j) / sigma
        return _tn_rvs(a, b, loc=x_j, scale=sigma)

    def _log_narrow_tn(self, x_from: float, x_to: float, pi: dict) -> float:
        lo, hi = pi["lo"], pi["hi"]
        sigma = pi["sigma"]
        a, b = (lo - x_from) / sigma, (hi - x_from) / sigma
        return _tn_logpdf(x_to, a, b, loc=x_from, scale=sigma)

    def _local_integer_candidates(self, x_j: int, pi: dict) -> list[int]:
        lo, hi = pi["lo"], pi["hi"]
        return [v for v in [x_j - 1, x_j, x_j + 1] if lo <= v <= hi]

    def _sample_local_integer(self, x_j: int, pi: dict) -> int:
        candidates = self._local_integer_candidates(x_j, pi)
        return int(np.random.choice(candidates))

    def _log_local_integer(self, x_from: int, x_to: int, pi: dict) -> float:
        candidates = self._local_integer_candidates(x_from, pi)
        if x_to in candidates:
            return float(-np.log(len(candidates)))
        return float(-np.inf)

    def _log_spline_integer(self, x_from: int, x_to: int, pi: dict, state: HMMState) -> float:
        x_knots, F = self._cdf_knots(pi, state, float(x_from))
        lo_f, hi_f = float(x_to) - 0.5, float(x_to) + 0.5
        masses = np.diff(F)
        prob = 0.0
        for k in range(len(masses)):
            seg_lo, seg_hi = x_knots[k], x_knots[k + 1]
            overlap_lo = max(seg_lo, lo_f)
            overlap_hi = min(seg_hi, hi_f)
            if overlap_hi > overlap_lo:
                width = seg_hi - seg_lo
                if width > 1e-30:
                    prob += masses[k] * (overlap_hi - overlap_lo) / width
        return float(np.log(max(prob, 1e-300)))

    def _spline_ready(self) -> bool:
        return len(self._archive) >= self.spline_min_archive

    def _knot_grid(self, lo: float, hi: float) -> np.ndarray:
        return np.linspace(lo, hi, self.spline_knots + 1)

    def _segment_masses(self, pi: dict, state: HMMState, center: float) -> np.ndarray:
        """Невозвратные массы m_k для сегментов [x_k, x_{k+1}]."""
        lo, hi = pi["lo"], pi["hi"]
        name = pi["name"]
        knots = self._knot_grid(lo, hi)
        n_seg = len(knots) - 1
        cfg = self.SPLINE_CONFIG[state]

        hist = np.full(n_seg, 1.0 / n_seg)
        if len(self._archive) >= 3:
            weights = self._archive_boltzmann_weights()
            seg_mass = np.zeros(n_seg)
            for idx, (cfg_entry, _) in enumerate(self._archive):
                w = weights[idx]
                val = float(cfg_entry[name])
                val = float(np.clip(val, lo, hi))
                seg_idx = int(np.searchsorted(knots, val, side="right") - 1)
                seg_idx = int(np.clip(seg_idx, 0, n_seg - 1))
                seg_mass[seg_idx] += w
            if seg_mass.sum() > 1e-30:
                hist = seg_mass / seg_mass.sum()

        locality = np.zeros(n_seg)
        sigma_loc = max(self.locality_sigma_fraction * pi["range"], 1e-8)
        for k in range(n_seg):
            mid = 0.5 * (knots[k] + knots[k + 1])
            z = (mid - center) / sigma_loc
            locality[k] = np.exp(-0.5 * z * z)
        if locality.sum() > 1e-30:
            locality /= locality.sum()

        uniform = np.ones(n_seg) / n_seg
        w_arch = cfg["archive_weight"]
        w_loc = cfg["locality_weight"]
        w_uni = cfg["uniform_floor"]
        w_rem = max(0.0, 1.0 - w_arch - w_loc - w_uni)

        masses = (
            w_arch * hist
            + w_loc * locality
            + w_uni * uniform
            + w_rem * uniform
        )
        masses = np.maximum(masses, 1e-12)
        return masses / masses.sum()

    def _cdf_knots(self, pi: dict, state: HMMState, center: float) -> tuple[np.ndarray, np.ndarray]:
        """Возвращает (knot_positions, F_values) монотонной CDF."""
        cache_key = (pi["name"], int(state), float(center))
        cached = self._cdf_cache.get(cache_key)
        if cached is not None:
            return cached

        lo, hi = pi["lo"], pi["hi"]
        x_knots = self._knot_grid(lo, hi)
        masses = self._segment_masses(pi, state, center)
        F = np.zeros(len(x_knots))
        F[1:] = np.cumsum(masses)
        F[-1] = 1.0
        result = (x_knots, F)
        self._cdf_cache[cache_key] = result
        return result

    def _invert_cdf(self, u: float, x_knots: np.ndarray, F: np.ndarray) -> float:
        """Inverse-transform: u ∈ [0,1] → x через бисекцию на кусочно-линейной CDF."""
        u = float(np.clip(u, 0.0, 1.0))
        if u <= 0.0:
            return float(x_knots[0])
        if u >= 1.0:
            return float(x_knots[-1])

        seg = int(np.searchsorted(F, u, side="right") - 1)
        seg = int(np.clip(seg, 0, len(x_knots) - 2))

        F_lo, F_hi = F[seg], F[seg + 1]
        x_lo, x_hi = x_knots[seg], x_knots[seg + 1]
        if F_hi - F_lo < 1e-30:
            return float(0.5 * (x_lo + x_hi))
        frac = (u - F_lo) / (F_hi - F_lo)
        return float(x_lo + frac * (x_hi - x_lo))

    def _spline_pdf(self, x: float, pi: dict, state: HMMState, center: float) -> float:
        """Плотность = производная кусочно-линейной CDF (кусочно-постоянная)."""
        lo, hi = pi["lo"], pi["hi"]
        x = float(np.clip(x, lo, hi))
        x_knots, F = self._cdf_knots(pi, state, center)
        masses = np.diff(F)
        n_seg = len(masses)

        seg = int(np.searchsorted(x_knots, x, side="right") - 1)
        seg = int(np.clip(seg, 0, n_seg - 1))

        width = x_knots[seg + 1] - x_knots[seg]
        if width < 1e-30:
            return 1.0 / max(pi["range"], 1e-8)
        return float(masses[seg] / width)

    def _integrate_spline(self, pi: dict, state: HMMState, center: float, n_grid: int = 200) -> float:
        """Численная проверка нормировки (∫f dx ≈ 1)."""
        lo, hi = pi["lo"], pi["hi"]
        xs = np.linspace(lo, hi, n_grid)
        pdf_vals = [self._spline_pdf(x, pi, state, center) for x in xs]
        return float(np.trapezoid(pdf_vals, xs))

    def _integrate_mixture(
        self, pi: dict, state: HMMState, center: float, n_grid: int = 200
    ) -> float:
        """Численная проверка нормировки смеси spline + narrow TN."""
        lo, hi = pi["lo"], pi["hi"]
        xs = np.linspace(lo, hi, n_grid)
        w_s = self._mix_weight(state)
        pdf_vals = []
        for x in xs:
            log_s = np.log(max(self._spline_pdf(x, pi, state, center), 1e-300))
            log_g = self._log_narrow_tn(center, x, pi)
            log_q = _logsumexp(np.array([
                np.log(w_s + 1e-300) + log_s,
                np.log(1.0 - w_s + 1e-300) + log_g,
            ]))
            pdf_vals.append(np.exp(log_q))
        return float(np.trapezoid(pdf_vals, xs))

    def _sample_continuous(self, x_j: float, pi: dict, state: HMMState) -> float:
        if not self._spline_ready():
            return super()._sample_continuous(x_j, pi, state)
        w_s = self._mix_weight(state)
        if np.random.rand() >= w_s:
            return self._sample_narrow_tn(x_j, pi)
        u = np.random.rand()
        x_knots, F = self._cdf_knots(pi, state, x_j)
        return self._invert_cdf(u, x_knots, F)

    def _log_q_continuous(
        self, x_from: float, x_to: float, pi: dict, state: HMMState
    ) -> float:
        if not self._spline_ready():
            return super()._log_q_continuous(x_from, x_to, pi, state)
        w_s = self._mix_weight(state)
        log_s = np.log(max(self._spline_pdf(x_to, pi, state, center=x_from), 1e-300))
        log_g = self._log_narrow_tn(x_from, x_to, pi)
        return float(_logsumexp(np.array([
            np.log(w_s + 1e-300) + log_s,
            np.log(1.0 - w_s + 1e-300) + log_g,
        ])))

    def _sample_integer(self, x_j: int, pi: dict, state: HMMState) -> int:
        if not self._spline_ready():
            return super()._sample_integer(x_j, pi, state)
        w_s = self._mix_weight(state)
        if np.random.rand() >= w_s:
            return self._sample_local_integer(x_j, pi)
        lo, hi = pi["lo"], pi["hi"]
        u = np.random.rand()
        x_knots, F = self._cdf_knots(pi, state, float(x_j))
        val = self._invert_cdf(u, x_knots, F)
        return int(np.clip(round(val), lo, hi))

    def _log_q_integer(self, x_from: int, x_to: int, pi: dict, state: HMMState) -> float:
        if not self._spline_ready():
            return super()._log_q_integer(x_from, x_to, pi, state)
        w_s = self._mix_weight(state)
        log_s = self._log_spline_integer(x_from, x_to, pi, state)
        log_g = self._log_local_integer(x_from, x_to, pi)
        return float(_logsumexp(np.array([
            np.log(w_s + 1e-300) + log_s,
            np.log(1.0 - w_s + 1e-300) + log_g,
        ])))


# ---------------------------------------------------------------------------
# Main algorithm class
# ---------------------------------------------------------------------------

class HMM_MCMC_TEST(HMM_MCMC):
    """H-MCMC-FMP с Baum-Welch (A-only) и spline-предложениями.

    Параметры BW и spline можно отключить флагами ``use_baum_welch`` /
    ``use_spline_proposal`` для A/B сравнения с базовым :class:`HMM_MCMC`.
    """

    def __init__(
        self,
        objective_func,
        budget: int,
        dict_to_optimize: dict,
        n_init: int = 16,
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
        use_baum_welch: bool = True,
        bw_refit_every: int = 5,
        bw_min_obs: int = 12,
        bw_n_em_iters: int = 3,
        bw_prior_strength: float = 25.0,
        bw_exploit_prior_scale: float = 3.0,
        bw_max_len: int = 64,
        use_spline_proposal: bool = True,
        spline_knots: int = 60,
        spline_floor: float = 0.05,
        spline_min_archive: int = 5,
        locality_sigma_fraction: float = 0.08,
        spline_mix_scale: float = 1.0,
        show_progress: bool = True,
        progress_desc: str | None = None,
        verbose_history: bool = True,
    ):
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
        )
        self.use_baum_welch = use_baum_welch
        self.bw_refit_every = bw_refit_every
        self.bw_min_obs = bw_min_obs
        self.bw_n_em_iters = bw_n_em_iters
        self.bw_prior_strength = bw_prior_strength
        self.bw_exploit_prior_scale = bw_exploit_prior_scale
        self.bw_max_len = bw_max_len

        self.use_spline_proposal = use_spline_proposal
        self.spline_knots = spline_knots
        self.spline_floor = spline_floor
        self.spline_min_archive = spline_min_archive
        self.locality_sigma_fraction = locality_sigma_fraction
        self.spline_mix_scale = spline_mix_scale
        self.show_progress = show_progress
        self.progress_desc = progress_desc
        self.verbose_history = verbose_history

    def _make_proposal_generator(self) -> FactorizedProposalGenerator:
        if self.use_spline_proposal:
            return SplineProposalGenerator(
                self.dict_to_optimize,
                sigma_fraction=self._sigma_fraction,
                temperature=self._temperature,
                wide_sigma_fraction=self._wide_sigma_fraction,
                kde_tau=self._kde_tau,
                spline_knots=self.spline_knots,
                spline_floor=self.spline_floor,
                spline_min_archive=self.spline_min_archive,
                locality_sigma_fraction=self.locality_sigma_fraction,
                spline_mix_scale=self.spline_mix_scale,
            )
        return FactorizedProposalGenerator(
            self.dict_to_optimize,
            sigma_fraction=self._sigma_fraction,
            temperature=self._temperature,
            wide_sigma_fraction=self._wide_sigma_fraction,
            kde_tau=self._kde_tau,
        )

    def _make_hmm_controller(self) -> HMMController:
        if self.use_baum_welch:
            return BaumWelchHMMController(
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
        return HMMController(
            window=self._hmm_window,
            obs_epsilon=self._hmm_obs_epsilon,
            lambda_noise=self._hmm_lambda_noise,
        )

    def main_loop(self):
        """Основной цикл с Baum-Welch HMM и spline-предложениями."""
        self._sobol_init = SobolInitializer(self.dict_to_optimize)
        self._proposal_gen = self._make_proposal_generator()

        init_configs = self._sobol_init.generate(self.n_init)
        pbar = tqdm(
            total=self.budget,
            desc=self.progress_desc or "H-MCMC-FMP-TEST",
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
            return min(self.data, key=lambda x: x[1])

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
            chain = MCMCChain(
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
            )
            self._chains.append(chain)

        self._orchestrator = GlobalOrchestrator(
            chains=self._chains,
            proposal_gen=self._proposal_gen,
            sobol_init=self._sobol_init,
            dict_to_optimize=self.dict_to_optimize,
            objective_func=self.objective_func,
            clone_noise=self._clone_noise,
        )

        step_counter = 0

        while len(self.data) < self.budget:
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

                self.history_table.append({
                    "Eval": len(self.data),
                    "Chain": chain.chain_id,
                    "State": chain.state.name,
                    "Loss": float(loss),
                    "Accepted": accepted,
                    "Rej_Streak": chain._rejection_streak,
                })

                pbar.update(1)

            step_counter += 1

            if step_counter % self.orchestrate_every == 0:
                if len(self.data) < self.budget:
                    new_evals = self._orchestrator.orchestrate(self.data, budget=self.budget)
                    for cfg, loss in new_evals:
                        if len(self.data) >= self.budget:
                            break
                        self.data.append((cfg, loss))

                        self.history_table.append({
                            "Eval": len(self.data),
                            "Chain": "ORCHESTRATOR",
                            "State": "RESET/CLONE",
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
                print("HMM MCMC TEST State History (last run):")
                with pd.option_context("display.max_rows", 200, "display.max_columns", None):
                    print(df)
                print("=" * 60 + "\n")
            except ImportError:
                pass

        return min(self.data, key=lambda x: x[1])


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_check_spline_density() -> None:
    """Проверка нормировки и положительности spline-плотности."""
    space = {
        "x0": {"type": "float", "values": [-500.0, 500.0]},
        "x1": {"type": "float", "values": [-500.0, 500.0]},
    }
    gen = SplineProposalGenerator(space, spline_min_archive=3)
    for i in range(10):
        cfg = {"x0": np.random.uniform(-400, 400), "x1": np.random.uniform(-400, 400)}
        gen.update_category_history(cfg, float(np.random.rand()))

    pi = gen._param_info[0]
    center = 0.0
    for state in HMMState:
        integral = gen._integrate_spline(pi, state, center, n_grid=400)
        assert 0.92 <= integral <= 1.08, f"spline integral={integral} for state={state.name}"
        xs = np.linspace(pi["lo"], pi["hi"], 50)
        pdfs = [gen._spline_pdf(x, pi, state, center) for x in xs]
        assert all(p > 0 for p in pdfs), f"non-positive pdf for state={state.name}"
    print("[self-check] spline density: OK (integral ~ 1, pdf > 0)")


def _self_check_mixture_density() -> None:
    """Проверка нормировки смеси spline + narrow TN."""
    space = {
        "x0": {"type": "float", "values": [-500.0, 500.0]},
        "x1": {"type": "float", "values": [-500.0, 500.0]},
    }
    gen = SplineProposalGenerator(space, spline_min_archive=3, spline_mix_scale=1.0)
    for i in range(10):
        cfg = {"x0": np.random.uniform(-400, 400), "x1": np.random.uniform(-400, 400)}
        gen.update_category_history(cfg, float(np.random.rand()))

    pi = gen._param_info[0]
    center = 0.0
    for state in HMMState:
        integral = gen._integrate_mixture(pi, state, center, n_grid=400)
        assert 0.92 <= integral <= 1.08, (
            f"mixture integral={integral} for state={state.name}"
        )
    print("[self-check] mixture density: OK (integral ~ 1)")


def _self_check_baum_welch() -> None:
    """Проверка что Baum-Welch сохраняет стохастичность строк A."""
    ctrl = BaumWelchHMMController(
        min_obs=8,
        refit_every=1,
        n_em_iters=5,
        prior_strength=25.0,
        exploit_prior_scale=3.0,
    )
    rng = np.random.default_rng(0)
    obs_seq = list(rng.normal(loc=-0.01, scale=0.08, size=30))
    ctrl._fit_transitions(obs_seq)
    row_sums = ctrl.A.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), f"A rows don't sum to 1: {row_sums}"
    assert np.all(ctrl.A >= 0), "negative transition probabilities"
    assert np.all(np.isfinite(ctrl.A)), "non-finite A entries"
    exploit_self = ctrl.A[int(HMMState.EXPLOIT), int(HMMState.EXPLOIT)]
    assert exploit_self >= 0.5, (
        f"EXPLOIT self-transition too low: {exploit_self:.3f} (expected >= 0.5)"
    )
    print(f"[self-check] Baum-Welch A:\n{ctrl.A}")
    print(f"[self-check] A[EXPLOIT,EXPLOIT]={exploit_self:.3f}")
    print("[self-check] Baum-Welch: OK (rows sum to 1, EXPLOIT anchored)")


def _self_check_schwefel_run() -> None:
    """Короткий прогон на Schwefel 2D."""
    from hpo_rl.backends.function import OptimizationBenchmarkBackend

    backend = OptimizationBenchmarkBackend(
        function_name="schwefel", dimensions=2, noise_std=0
    )
    objective = backend.evaluate
    space = backend.hp_space

    algo = HMM_MCMC_TEST(
        objective_func=objective,
        budget=40,
        dict_to_optimize=space,
        n_init=5,
        n_chains=1,
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
        use_baum_welch=True,
        use_spline_proposal=True,
        bw_refit_every=3,
        bw_min_obs=8,
    )
    best_cfg, best_loss = algo.main_loop()
    print(f"[self-check] Schwefel run: best_loss={best_loss:.4f}, cfg={best_cfg}")
    assert best_loss < 1e6, "Schwefel run returned unreasonable loss"
    print("[self-check] Schwefel run: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("HMM_MCMC_TEST self-checks")
    print("=" * 60)
    _self_check_spline_density()
    _self_check_mixture_density()
    _self_check_baum_welch()
    _self_check_schwefel_run()
    print("=" * 60)
    print("All self-checks passed.")
    print("=" * 60)
