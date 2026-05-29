"""
Animated GIF: CMA-ES vs SGA (Simple Genetic Algorithm) on the Rastrigin function (2D).

Saves: cmaes_vs_sga_rastrigin.gif
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Ellipse
from pathlib import Path

# ---------------------------------------------------------------------------
# Rastrigin function (2D)
# ---------------------------------------------------------------------------

def rastrigin(x: np.ndarray) -> float:
    n = len(x)
    return float(10 * n + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


# ---------------------------------------------------------------------------
# CMA-ES  –  records one population (array of shape [lam, dim]) per generation
# ---------------------------------------------------------------------------

class CMAESRecorder:
    """(mu/mu_w, lam)-CMA-ES with per-generation population recording."""

    def __init__(self, dim: int = 2, sigma0: float = 2.0,
                 popsize: int | None = None, budget: int = 2000,
                 bounds: tuple[float, float] = (-5.12, 5.12)):
        self.dim = dim
        self.sigma0 = sigma0
        self.budget = budget
        self.lb, self.ub = bounds

        lam = popsize if popsize else 4 + int(3 * np.log(dim))
        mu = lam // 2
        w_prime = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        self.weights = w_prime / w_prime.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()
        self.lam, self.mu = lam, mu

        # Step-size control
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        self.chiN = dim ** 0.5 * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Covariance matrix adaptation
        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) /
                       ((dim + 2) ** 2 + self.mueff))

    def run(self) -> tuple[list[np.ndarray], list[tuple[np.ndarray, float]]]:
        dim, lb, ub = self.dim, self.lb, self.ub

        xmean = np.random.uniform(lb, ub, dim)
        sigma = self.sigma0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        invsqrtC = np.eye(dim)
        eigeneval = 0

        populations: list[np.ndarray] = []
        best_history: list[tuple[np.ndarray, float]] = []
        cov_history: list[tuple[np.ndarray, np.ndarray, float]] = []
        best_score = float("inf")
        best_x = xmean.copy()
        evals = 0

        while evals < self.budget:
            arz = np.random.randn(self.lam, dim)
            arx = xmean + sigma * (arz @ (B * D).T)
            arx_clipped = np.clip(arx, lb, ub)

            fitness = np.array([rastrigin(x) for x in arx_clipped])
            evals += self.lam

            populations.append(arx_clipped.copy())

            idx = np.argsort(fitness)
            arx_sorted = arx[idx]
            arz_sorted = arz[idx]

            if fitness[idx[0]] < best_score:
                best_score = float(fitness[idx[0]])
                best_x = arx_clipped[idx[0]].copy()
            best_history.append((best_x.copy(), best_score))
            cov_history.append((xmean.copy(), C.copy(), sigma))

            # Mean update
            xold = xmean.copy()
            xmean = self.weights @ arx_sorted[: self.mu]

            # Path updates
            ps = ((1 - self.cs) * ps
                  + np.sqrt(self.cs * (2 - self.cs) * self.mueff)
                  * invsqrtC @ (xmean - xold) / sigma)
            hs = (np.linalg.norm(ps)
                  / np.sqrt(1 - (1 - self.cs) ** (2 * evals / self.lam))
                  / self.chiN < 1.4 + 2 / (dim + 1))
            pc = ((1 - self.cc) * pc
                  + hs * np.sqrt(self.cc * (2 - self.cc) * self.mueff)
                  * (xmean - xold) / sigma)

            # Covariance update
            artmp = (1 / sigma) * (arx_sorted[: self.mu] - xold)
            C = ((1 - self.c1 - self.cmu) * C
                 + self.c1 * (np.outer(pc, pc)
                               + (1 - hs) * self.cc * (2 - self.cc) * C)
                 + self.cmu * artmp.T @ np.diag(self.weights) @ artmp)

            # Step-size
            sigma *= np.exp((self.cs / self.ds)
                            * (np.linalg.norm(ps) / self.chiN - 1))

            # Eigendecomposition (lazy)
            if evals - eigeneval > self.lam / (self.c1 + self.cmu) / dim / 10:
                eigeneval = evals
                C = np.triu(C) + np.triu(C, 1).T
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
                invsqrtC = B @ np.diag(1.0 / D) @ B.T

        return populations, best_history, cov_history


# ---------------------------------------------------------------------------
# Simple Genetic Algorithm  –  records one population per generation
# ---------------------------------------------------------------------------

class SGARecorder:
    """Generational GA with tournament selection, uniform crossover, Gaussian mutation."""

    def __init__(self, dim: int = 2, popsize: int = 40, budget: int = 2000,
                 bounds: tuple[float, float] = (-5.12, 5.12),
                 mutation_prob: float = 0.15, crossover_prob: float = 0.8,
                 tournament_size: int = 3):
        self.dim = dim
        self.popsize = popsize
        self.budget = budget
        self.lb, self.ub = bounds
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size

    def _select(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        idx = np.random.choice(len(pop), self.tournament_size, replace=False)
        return pop[idx[np.argmin(fitness[idx])]].copy()

    def run(self) -> tuple[list[np.ndarray], list[tuple[np.ndarray, float]]]:
        lb, ub, dim = self.lb, self.ub, self.dim
        range_ = ub - lb

        pop = np.random.uniform(lb, ub, (self.popsize, dim))
        fitness = np.array([rastrigin(x) for x in pop])
        evals = self.popsize

        populations = [pop.copy()]
        best_idx = int(np.argmin(fitness))
        best_x = pop[best_idx].copy()
        best_score = float(fitness[best_idx])
        best_history: list[tuple[np.ndarray, float]] = [(best_x.copy(), best_score)]

        while evals < self.budget:
            new_pop = [pop[int(np.argmin(fitness))].copy()]  # elitism

            while len(new_pop) < self.popsize:
                p1 = self._select(pop, fitness)
                p2 = self._select(pop, fitness)

                # Uniform crossover
                if np.random.rand() < self.crossover_prob:
                    mask = np.random.rand(dim) < 0.5
                    child = np.where(mask, p1, p2)
                else:
                    child = p1.copy()

                # Gaussian mutation
                for i in range(dim):
                    if np.random.rand() < self.mutation_prob:
                        child[i] += np.random.randn() * 0.1 * range_
                        child[i] = np.clip(child[i], lb, ub)

                new_pop.append(child)

            pop = np.array(new_pop[: self.popsize])
            fitness = np.array([rastrigin(x) for x in pop])
            evals += self.popsize

            populations.append(pop.copy())
            bi = int(np.argmin(fitness))
            if fitness[bi] < best_score:
                best_score = float(fitness[bi])
                best_x = pop[bi].copy()
            best_history.append((best_x.copy(), best_score))

        return populations, best_history


# ---------------------------------------------------------------------------
# GIF generator
# ---------------------------------------------------------------------------

def build_gif(output: str = "cmaes_vs_sga_rastrigin.gif",
              budget: int = 2000,
              fps: int = 8,
              seed: int = 42) -> None:

    np.random.seed(seed)
    bounds = (-5.12, 5.12)

    # ---- Rastrigin landscape (background) ----------------------------------
    res = 400
    xs = np.linspace(*bounds, res)
    X, Y = np.meshgrid(xs, xs)
    Z = (10 * 2
         + (X ** 2 - 10 * np.cos(2 * np.pi * X))
         + (Y ** 2 - 10 * np.cos(2 * np.pi * Y)))

    # ---- Run optimizers ----------------------------------------------------
    print("Running CMA-ES …")
    cmaes = CMAESRecorder(dim=2, sigma0=2.0, budget=budget, bounds=bounds)
    cmaes_pops, cmaes_best, cmaes_cov = cmaes.run()

    # Precompute ellipse params (1σ and 2σ) for every generation
    def _ellipse_params(xmean, C, sigma, nsigma):
        eigvals, eigvecs = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(eigvals, 1e-20))
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        return xmean.copy(), angle, 2 * nsigma * sigma * D[1], 2 * nsigma * sigma * D[0]

    cmaes_ell_params = [
        (_ellipse_params(xm, C, s, 1.0), _ellipse_params(xm, C, s, 2.0))
        for xm, C, s in cmaes_cov
    ]

    print("Running SGA …")
    sga = SGARecorder(dim=2, popsize=40, budget=budget, bounds=bounds)
    sga_pops, sga_best = sga.run()

    n_frames = min(len(cmaes_pops), len(sga_pops))
    print(f"Rendering {n_frames} frames …")

    # ---- Build animation ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.set_facecolor("white")

    ALGO_COLORS = ["#c0392b", "#2471a3"]
    ALGO_TITLES = ["CMA-ES", "SGA (Simple Genetic Algorithm)"]

    # Pre-draw static background
    for ax, color, title in zip(axes, ALGO_COLORS, ALGO_TITLES):
        ax.contourf(X, Y, Z, levels=40, cmap="YlOrRd", alpha=0.75)
        ax.contour(X, Y, Z, levels=15, colors="black", alpha=0.18, linewidths=0.4)
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_xlabel("x₁", color="black")
        ax.set_ylabel("x₂", color="black")
        ax.tick_params(colors="black")
        for spine in ax.spines.values():
            spine.set_edgecolor("#aaa")
        ax.set_title(title, color=color, fontsize=12, fontweight="bold", pad=8)

    fig.tight_layout(pad=2.0)

    # Covariance ellipses for CMA-ES (1σ filled, 2σ dashed outline)
    cov_ell_fill = Ellipse((0, 0), 1, 1, angle=0,
                           fill=True, facecolor=ALGO_COLORS[0],
                           edgecolor="none", alpha=0.18, zorder=4)
    cov_ell_outer = Ellipse((0, 0), 1, 1, angle=0,
                            fill=False, edgecolor=ALGO_COLORS[0],
                            linewidth=1.8, linestyle="--", alpha=0.85, zorder=6)
    axes[0].add_patch(cov_ell_fill)
    axes[0].add_patch(cov_ell_outer)

    # Scatter artist containers (will be updated each frame)
    trail_artists: list[list] = [[], []]
    pop_scatters = [
        axes[0].scatter([], [], s=90, c=ALGO_COLORS[0], alpha=1.0,
                        edgecolors="black", linewidths=0.8, zorder=5),
        axes[1].scatter([], [], s=90, c=ALGO_COLORS[1], alpha=1.0,
                        edgecolors="black", linewidths=0.8, zorder=5),
    ]
    best_scatters = [
        axes[0].scatter([], [], s=420, c="gold", marker="*", zorder=9,
                        edgecolors="black", linewidths=1.2),
        axes[1].scatter([], [], s=420, c="gold", marker="*", zorder=9,
                        edgecolors="black", linewidths=1.2),
    ]
    # Global-optimum marker (static)
    for ax in axes:
        ax.scatter([0], [0], s=350, c="limegreen", marker="*", linewidths=1.5,
                   zorder=11, edgecolors="black", label="Global opt (0,0)")
        ax.legend(fontsize=8, loc="upper right",
                  facecolor="white", edgecolor="#aaa",
                  labelcolor="black")

    best_text = [
        axes[0].text(0.03, 0.97, "", transform=axes[0].transAxes,
                     fontsize=8, color="black", va="top",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8,
                               edgecolor="#aaa")),
        axes[1].text(0.03, 0.97, "", transform=axes[1].transAxes,
                     fontsize=8, color="black", va="top",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8,
                               edgecolor="#aaa")),
    ]

    TRAIL_LEN = 6  # how many past generations to show as ghost trail

    def update(frame: int):
        # Update covariance ellipses (CMA-ES only)
        (xm1, ang1, w1, h1), (xm2, ang2, w2, h2) = cmaes_ell_params[frame]
        cov_ell_fill.set_center(xm1)
        cov_ell_fill.width, cov_ell_fill.height, cov_ell_fill.angle = w1, h1, ang1
        cov_ell_outer.set_center(xm2)
        cov_ell_outer.width, cov_ell_outer.height, cov_ell_outer.angle = w2, h2, ang2

        for col, (pops, best_hist, ax, color, trail_list, ps, bs, bt) in enumerate(zip(
            [cmaes_pops, sga_pops],
            [cmaes_best, sga_best],
            axes,
            ALGO_COLORS,
            trail_artists,
            pop_scatters,
            best_scatters,
            best_text,
        )):
            # Remove old trail artists
            for art in trail_list:
                art.remove()
            trail_list.clear()

            # Draw ghost trail
            start = max(0, frame - TRAIL_LEN)
            for j in range(start, frame):
                alpha = 0.18 + 0.45 * (j - start) / TRAIL_LEN
                ghost = ax.scatter(
                    pops[j][:, 0], pops[j][:, 1],
                    s=40, c=color, alpha=alpha, zorder=3,
                    edgecolors="black", linewidths=0.4,
                )
                trail_list.append(ghost)

            # Current population
            pop = pops[frame]
            ps.set_offsets(pop)

            # Best so far
            bx, bsc = best_hist[frame]
            bs.set_offsets([[bx[0], bx[1]]])

            bt.set_text(f"Best f = {bsc:.4f}\n"
                        f"x = ({bx[0]:.3f}, {bx[1]:.3f})")

        return [cov_ell_fill, cov_ell_outer,
                *pop_scatters, *best_scatters, *best_text,
                *trail_artists[0], *trail_artists[1]]

    ani = animation.FuncAnimation(
        fig, update,
        frames=n_frames,
        interval=1000 // fps,
        blit=False,
    )

    print(f"Saving '{output}' …  (this may take a moment)")
    ani.save(output, writer="pillow", fps=fps, dpi=100)
    plt.close(fig)

    print(f"\nDone!  Saved → {output}")
    print(f"CMA-ES best result : {cmaes_best[-1][1]:.6f}")
    print(f"SGA    best result : {sga_best[-1][1]:.6f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _out_dir = Path(__file__).parent.parent / "logs" / "animate_cmaes_sga"
    _out_dir.mkdir(parents=True, exist_ok=True)
    build_gif(
        output=str(_out_dir / "cmaes_vs_sga_rastrigin.gif"),
        budget=2000,   # total function evaluations per algorithm
        fps=8,         # frames per second in the GIF
        seed=42,
    )
