"""H-MCMC-RAM: Baum-Welch HMM + diagonal RAM on archive-guided KDE mixture.

Extends :class:`HMM_MCMC_BW` with diagonal Robust Adaptive Metropolis (Vihola 2012)
that adapts the **narrow** component of the factorized EXPLOIT mixture (narrow TN +
archive KDE + wide TN). KDE archive jumps and base mixture Hastings are preserved.
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
from hpo_rl.baselines.HMM_MCMC_TEST import BaumWelchHMMController


# ---------------------------------------------------------------------------
# Diagonal Robust Adaptive Metropolis preconditioner
# ---------------------------------------------------------------------------

@dataclass
class DiagRAMConfig:
    target_accept: float = 0.234
    gamma: float = 0.7
    s_min_frac: float = 0.005
    s_max_frac: float = 0.5


class DiagRAMPreconditioner:
    """Per-chain diagonal scale vector ``s`` with Vihola-style RAM updates."""

    def __init__(
        self,
        dim: int,
        ranges: np.ndarray,
        config: DiagRAMConfig | None = None,
        initial_s: np.ndarray | None = None,
    ):
        self.dim = int(dim)
        self.ranges = np.asarray(ranges, dtype=np.float64)
        self.config = config or DiagRAMConfig()
        lo = self.config.s_min_frac * self.ranges
        hi = self.config.s_max_frac * self.ranges
        self._s_lo = np.maximum(lo, 1e-12)
        self._s_hi = np.maximum(hi, self._s_lo * 1.001)

        if initial_s is not None:
            self.s = self._clip_s(np.asarray(initial_s, dtype=np.float64))
        else:
            self.s = self._clip_s(0.1 * self.ranges)

        self.n_steps: int = 0

    def _clip_s(self, s: np.ndarray) -> np.ndarray:
        return np.clip(s, self._s_lo, self._s_hi)

    def init_from_std(self, std_vec: np.ndarray) -> None:
        """Set ``s`` from elite-archive marginal std (clipped)."""
        self.s = self._clip_s(np.maximum(np.asarray(std_vec, dtype=np.float64), self._s_lo))

    def eta(self) -> float:
        """Diminishing adaptation step size."""
        if self.n_steps <= 0:
            return 0.0
        d = max(self.dim, 1)
        return float(min(1.0, d * (self.n_steps ** (-self.config.gamma))))

    def update(self, alpha: float, z: np.ndarray) -> None:
        """Diagonal RAM update using acceptance and normalized proposal direction ``z``."""
        self.n_steps += 1
        eta_n = self.eta()
        if eta_n <= 0.0:
            return

        z = np.asarray(z, dtype=np.float64)
        z_norm_sq = float(np.dot(z, z))
        if z_norm_sq < 1e-30:
            return

        delta = 0.5 * eta_n * (float(alpha) - self.config.target_accept)
        log_s = np.log(self.s + 1e-30)
        log_s += delta * (z * z) / z_norm_sq
        self.s = self._clip_s(np.exp(log_s))


# ---------------------------------------------------------------------------
# Adaptive factorized proposal generator (RAM narrow-scale injection)
# ---------------------------------------------------------------------------

class AdaptiveFactorizedProposalGenerator(FactorizedProposalGenerator):
    """Factorized mixture; RAM injects per-dim narrow ``pi['sigma']`` before EXPLOIT steps."""

    def __init__(self, dict_to_optimize: dict, **kwargs):
        super().__init__(dict_to_optimize, **kwargs)
        self._continuous_indices: list[int] = [
            i for i, pi in enumerate(self._param_info) if pi["type"] in ("float", "int")
        ]
        self._ranges_cont: np.ndarray | None = None
        if self._continuous_indices:
            pis = [self._param_info[i] for i in self._continuous_indices]
            los = np.array([pi["lo"] for pi in pis], dtype=np.float64)
            his = np.array([pi["hi"] for pi in pis], dtype=np.float64)
            self._ranges_cont = his - los

    @property
    def continuous_dim(self) -> int:
        return len(self._continuous_indices)

    def config_to_vector(self, cfg: dict) -> np.ndarray:
        return np.array(
            [float(cfg[self._param_info[i]["name"]]) for i in self._continuous_indices],
            dtype=np.float64,
        )

    def apply_narrow_scales(self, s_vec: np.ndarray) -> None:
        """Write RAM-adapted scales into narrow Gaussian sigmas for continuous dims."""
        s_vec = np.maximum(np.asarray(s_vec, dtype=np.float64), 1e-12)
        for j, idx in enumerate(self._continuous_indices):
            self._param_info[idx]["sigma"] = float(s_vec[j])

    def default_initial_scales(self) -> np.ndarray:
        if self._ranges_cont is None:
            return np.array([], dtype=np.float64)
        return self.sigma_fraction * self._ranges_cont

    def elite_marginal_std(self, top_k: int = 50, min_std_frac: float = 0.01) -> np.ndarray:
        """Marginal std of top-k archive configs per continuous dim."""
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


# ---------------------------------------------------------------------------
# RAM-scaled KDE MCMC chain
# ---------------------------------------------------------------------------

class RamMCMCChain(MCMCChain):
    """MCMC chain with diagonal RAM on EXPLOIT narrow component of the KDE mixture."""

    def __init__(
        self,
        *args,
        precond: DiagRAMPreconditioner | None = None,
        use_ram: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.precond = precond
        self.use_ram = use_ram
        self._last_kernel: str = "factorized"

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

    def _factorized_step(
        self, objective_func, T_effective: float, is_burnin: bool
    ) -> tuple[list[tuple[dict, float]], bool]:
        self._last_kernel = "factorized"
        cat_only = np.random.rand() < self.p_cat_step
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

    def _ram_kde_step(
        self,
        objective_func,
        T_effective: float,
    ) -> tuple[list[tuple[dict, float]], bool]:
        gen = self.proposal_gen
        assert isinstance(gen, AdaptiveFactorizedProposalGenerator)
        assert self.precond is not None

        self._last_kernel = "ram_kde"
        gen.apply_narrow_scales(self.precond.s)

        x_before = copy.deepcopy(self.current_x)
        x_vec_before = gen.config_to_vector(x_before)

        cat_only = np.random.rand() < self.p_cat_step
        x_prime = gen.generate_proposal(
            self.current_x, HMMState.EXPLOIT, categorical_only=cat_only
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

        x_vec_prime = gen.config_to_vector(x_prime)
        s_safe = np.maximum(self.precond.s, 1e-12)
        z = (x_vec_prime - x_vec_before) / s_safe
        self.precond.update(alpha, z)

        delta_loss = loss_prime - self.current_loss_old_for_hmm
        O_t = float(delta_loss / (self.scale_factor + 1e-8))
        self._observations.append(O_t)

        if loss_prime < self.best_loss:
            self.best_x = copy.deepcopy(x_prime)
            self.best_loss = loss_prime

        return [(x_prime, loss_prime)], accepted

    def step(
        self,
        objective_func,
        progress: float = 0.0,
        is_burnin: bool = False,
    ) -> tuple[list[tuple[dict, float]], bool]:
        T_effective = self._prepare_step(progress, is_burnin)

        use_ram_now = (
            self.use_ram
            and not is_burnin
            and self.state == HMMState.EXPLOIT
            and self.precond is not None
            and isinstance(self.proposal_gen, AdaptiveFactorizedProposalGenerator)
            and self.proposal_gen.continuous_dim > 0
        )

        if use_ram_now:
            return self._ram_kde_step(objective_func, T_effective)
        return self._factorized_step(objective_func, T_effective, is_burnin)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class HMM_MCMC_RAM(HMM_MCMC_BW):
    """H-MCMC-FMP-BW with diagonal RAM on EXPLOIT narrow component of the KDE mixture."""

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
        rejection_streak: int = 10,
        use_baum_welch: bool = True,
        bw_refit_every: int = 5,
        bw_min_obs: int = 12,
        bw_n_em_iters: int = 3,
        bw_prior_strength: float = 25.0,
        bw_exploit_prior_scale: float = 3.0,
        bw_max_len: int = 64,
        use_ram: bool = True,
        ram_target_accept: float = 0.234,
        ram_gamma: float = 0.7,
        ram_s_min: float = 0.005,
        ram_s_max: float = 0.5,
        ram_precond_init: str = "archive",
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
        self.use_ram = use_ram
        self.ram_target_accept = ram_target_accept
        self.ram_gamma = ram_gamma
        self.ram_s_min = ram_s_min
        self.ram_s_max = ram_s_max
        self.ram_precond_init = ram_precond_init
        self._ram_config = DiagRAMConfig(
            target_accept=ram_target_accept,
            gamma=ram_gamma,
            s_min_frac=ram_s_min,
            s_max_frac=ram_s_max,
        )

    def _make_proposal_generator(self) -> AdaptiveFactorizedProposalGenerator:
        return AdaptiveFactorizedProposalGenerator(
            self.dict_to_optimize,
            sigma_fraction=self._sigma_fraction,
            temperature=self._temperature,
            wide_sigma_fraction=self._wide_sigma_fraction,
            kde_tau=self._kde_tau,
        )

    def _make_preconditioner(
        self, gen: AdaptiveFactorizedProposalGenerator
    ) -> DiagRAMPreconditioner | None:
        if gen.continuous_dim == 0 or gen._ranges_cont is None:
            return None

        if self.ram_precond_init == "archive" and len(gen._archive) >= 2:
            initial_s = gen.elite_marginal_std()
        else:
            initial_s = gen.default_initial_scales()

        return DiagRAMPreconditioner(
            dim=gen.continuous_dim,
            ranges=gen._ranges_cont,
            config=self._ram_config,
            initial_s=initial_s,
        )

    def main_loop(self):
        """Main loop with RAM-scaled KDE mixture and Baum-Welch HMM."""
        self._sobol_init = SobolInitializer(self.dict_to_optimize)
        self._proposal_gen = self._make_proposal_generator()

        init_configs = self._sobol_init.generate(self.n_init)
        pbar = tqdm(
            total=self.budget,
            desc=self.progress_desc or "H-MCMC-RAM",
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
            precond = self._make_preconditioner(self._proposal_gen)
            chain = RamMCMCChain(
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
                use_ram=self.use_ram,
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

                evals, accepted = chain.step(
                    self.objective_func,
                    progress=progress,
                    is_burnin=is_burnin,
                )

                for cfg, loss in evals:
                    if len(self.data) >= self.budget:
                        break
                    self.data.append((cfg, loss))
                    self._proposal_gen.update_category_history(cfg, loss)

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
                print("HMM MCMC RAM State History (last run):")
                with pd.option_context("display.max_rows", 200, "display.max_columns", None):
                    print(df)
                print("=" * 60 + "\n")
            except ImportError:
                pass

        return self._decode_best(min(self.data, key=lambda x: x[1]))


# ---------------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------------

def _self_check_precond_update() -> None:
    ranges = np.array([10.0, 1.0])
    cfg = DiagRAMConfig(target_accept=0.234, gamma=0.6)
    precond = DiagRAMPreconditioner(2, ranges, config=cfg, initial_s=0.5 * ranges)
    s0 = precond.s.copy()

    alphas = []
    for _ in range(400):
        z = np.random.randn(2)
        alpha = float(np.clip(np.random.beta(2, 5), 0.0, 1.0))
        alphas.append(alpha)
        precond.update(alpha, z)

    assert np.all(precond.s > 0), "scale must stay positive"
    assert np.all(precond.s >= precond._s_lo), "scale below lower clip"
    assert np.all(precond.s <= precond._s_hi), "scale above upper clip"
    assert not np.allclose(precond.s, s0), "scale should adapt over updates"
    print(f"[self-check] RAM precond mean alpha={float(np.mean(alphas)):.3f}")
    print("[self-check] RAM precond update: OK")


def _self_check_ram_kde_step() -> None:
    """EXPLOIT step uses ram_kde kernel, injects scales, proposals stay in bounds."""
    space = {
        "x0": {"type": "float", "values": [-5.0, 5.0]},
        "x1": {"type": "float", "values": [0.0, 1.0]},
    }
    gen = AdaptiveFactorizedProposalGenerator(space, sigma_fraction=0.1)
    for i in range(5):
        cfg = {"x0": float(i - 2), "x1": float(i) * 0.15}
        gen.update_category_history(cfg, float(i))

    ranges = gen._ranges_cont
    precond = DiagRAMPreconditioner(2, ranges, initial_s=0.2 * ranges)

    def loss_fn(cfg):
        v0 = cfg["x0"]
        v1 = cfg["x1"]
        return float(v0 * v0 + v1)

    chain = RamMCMCChain(
        chain_id=0,
        x0={"x0": 0.0, "x1": 0.5},
        loss0=0.25,
        proposal_gen=gen,
        hmm=BaumWelchHMMController(min_obs=10**9, refit_every=10**9),
        T_mcmc=1.0,
        scale_factor=1.0,
        precond=precond,
        use_ram=True,
    )
    chain.state = HMMState.EXPLOIT
    chain.hmm.force_state(HMMState.EXPLOIT)

    injected = precond.s.copy()
    evals, _ = chain.step(loss_fn, progress=0.5, is_burnin=False)
    assert chain._last_kernel == "ram_kde", f"expected ram_kde, got {chain._last_kernel}"
    assert len(evals) == 1, "ram_kde step should produce exactly one eval"

    cfg, _ = evals[0]
    assert -5.0 <= cfg["x0"] <= 5.0
    assert 0.0 <= cfg["x1"] <= 1.0

    idx0 = gen._continuous_indices[0]
    assert abs(gen._param_info[idx0]["sigma"] - injected[0]) < 1e-9
    print("[self-check] ram_kde step: OK")


def _self_check_budget_exactness() -> None:
    from hpo_rl.backends.function import OptimizationBenchmarkBackend

    backend = OptimizationBenchmarkBackend(
        function_name="schwefel", dimensions=2, noise_std=0
    )
    algo = HMM_MCMC_RAM(
        objective_func=backend.evaluate,
        budget=35,
        dict_to_optimize=backend.hp_space,
        n_init=5,
        n_chains=2,
        orchestrate_every=1000,
        burnin_fraction=0.0,
        use_baum_welch=False,
        use_ram=True,
        show_progress=False,
        verbose_history=False,
    )
    algo.main_loop()
    assert len(algo.data) == 35, f"expected 35 evals, got {len(algo.data)}"
    print("[self-check] budget exactness: OK")


def _self_check_ablation_parity() -> None:
    """With use_ram=False, only factorized kernel is used."""
    space = {"x": {"type": "float", "values": [-5.0, 5.0]}}

    def quad(cfg):
        return float(cfg["x"] ** 2)

    algo = HMM_MCMC_RAM(
        objective_func=quad,
        budget=20,
        dict_to_optimize=space,
        n_init=4,
        n_chains=1,
        orchestrate_every=1000,
        burnin_fraction=0.0,
        use_baum_welch=False,
        use_ram=False,
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
    print("HMM_MCMC_RAM self-checks")
    print("=" * 60)
    _self_check_precond_update()
    _self_check_ram_kde_step()
    _self_check_budget_exactness()
    _self_check_ablation_parity()
    print("=" * 60)
    print("All self-checks passed.")
    print("=" * 60)
