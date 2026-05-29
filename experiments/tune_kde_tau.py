import json
import numpy as np
from pathlib import Path
from hpo_rl.baselines.HMM_MCMC import HMM_MCMC
from hpo_rl.backends.function import OptimizationBenchmarkBackend

KDE_TAU_VALUES = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15]
N_SEEDS = 10
BUDGET = 200
DIMENSIONS = 10


def run_single(kde_tau: float, seed: int) -> float:
    np.random.seed(seed)

    backend = OptimizationBenchmarkBackend(
        function_name="schwefel", dimensions=DIMENSIONS, noise_std=0
    )
    dict_to_optimize = {
        f"x{i}": {"values": [-500, 500], "type": "float", "log": False}
        for i in range(DIMENSIONS)
    }

    alg = HMM_MCMC(
        objective_func=backend.evaluate,
        budget=BUDGET,
        dict_to_optimize=dict_to_optimize,
        n_init=5,
        n_chains=1,
        orchestrate_every=1000,
        T_mcmc=0.01,
        sigma_fraction=0.0055,
        wide_sigma_fraction=0.5,
        temperature=0.3,
        hmm_window=4,
        hmm_obs_epsilon=1e-8,
        hmm_lambda_noise=0.01,
        clone_noise=0.05,
        burnin_fraction=0.0,
        p_cat_step=0.0,
        anneal_T=True,
        kde_tau=kde_tau,
    )

    best_config, best_loss = alg.main_loop()
    return best_loss


if __name__ == "__main__":
    results = {}
    for tau in KDE_TAU_VALUES:
        print(f"\n{'='*60}")
        print(f"  kde_tau = {tau}  ({N_SEEDS} seeds, budget={BUDGET})")
        print(f"{'='*60}")

        seed_losses = []
        for s in range(N_SEEDS):
            best = run_single(tau, seed=s + 42)
            seed_losses.append(best)
            print(f"    seed {s}: best_loss = {best:.2f}")

        results[tau] = seed_losses
        mean_loss = np.mean(seed_losses)
        std_loss = np.std(seed_losses)
        print(f"  => kde_tau={tau}: mean={mean_loss:.2f} ± {std_loss:.2f}")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for tau in KDE_TAU_VALUES:
        losses = results[tau]
        print(f"  kde_tau={tau:5.2f}: mean={np.mean(losses):8.2f} ± {np.std(losses):6.2f}  min={np.min(losses):8.2f}")
    best_tau = min(results, key=lambda t: np.mean(results[t]))
    print(f"\n  Best kde_tau = {best_tau} (mean={np.mean(results[best_tau]):.2f})")
