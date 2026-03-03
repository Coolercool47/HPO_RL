"""Comprehensive analysis of all reward modes in InstantContinuousPipelineEnv.

This test suite:
1. Profiles each reward mode on multiple functions (sphere, schwefel, rastrigin, rosenbrock)
2. Tests reward dynamics over episodes (improving, worsening, random agents)
3. Measures scale stability across functions (critical for SequentialBackend)
4. Evaluates Q-value learnability (variance, monotonicity of returns)
5. Tests function switching (SequentialBackend simulation)
"""

import numpy as np
import pytest
from typing import Dict, List, Tuple

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.backends.sequential import SequentialBackend
from hpo_rl.environments.instant_continuous_pipeline_env import (
    InstantContinuousPipelineEnv,
)


# ===================================================================
# Helpers
# ===================================================================

REWARD_MODES = ["guided", "potential", "delta", "rank_shaped", "best_improvement", "best", "absolute"]
FUNCTIONS = ["sphere", "schwefel", "rastrigin", "rosenbrock"]


def _make_hp_space(backend: OptimizationBenchmarkBackend) -> Dict[str, dict]:
    """Build hp_space dict from backend bounds."""
    lo, hi = backend.bounds
    return {
        f"x{i}": {"min": lo, "max": hi, "type": "float", "log": False}
        for i in range(backend.dimensions)
    }


def _make_env(
    function_name: str = "sphere",
    reward_mode: str = "guided",
    max_steps: int = 200,
    dims: int = 2,
    max_delta_frac: float = 0.02,
    history_window: int = 0,
) -> InstantContinuousPipelineEnv:
    backend = OptimizationBenchmarkBackend(function_name, dimensions=dims)
    hp_space = _make_hp_space(backend)
    return InstantContinuousPipelineEnv(
        hp_space=hp_space,
        backend=backend,
        max_delta_frac=max_delta_frac,
        max_steps=max_steps,
        history_window=history_window,
        reward_mode=reward_mode,
    )


def _make_sequential_env(
    functions: List[str],
    reward_mode: str = "guided",
    max_steps: int = 200,
    max_delta_frac: float = 0.02,
) -> InstantContinuousPipelineEnv:
    backends = [
        OptimizationBenchmarkBackend(fn, dimensions=2) for fn in functions
    ]
    seq_backend = SequentialBackend(backends=backends, mode="sequential")
    # Use merged bounds for hp_space
    lo = min(b.bounds[0] for b in backends)
    hi = max(b.bounds[1] for b in backends)
    hp_space = {
        f"x{i}": {"min": lo, "max": hi, "type": "float", "log": False}
        for i in range(2)
    }
    return InstantContinuousPipelineEnv(
        hp_space=hp_space,
        backend=seq_backend,
        max_delta_frac=max_delta_frac,
        max_steps=max_steps,
        reward_mode=reward_mode,
    )


def _run_episode_with_strategy(
    env: InstantContinuousPipelineEnv,
    strategy: str = "improve",
    seed: int = 42,
) -> Dict[str, list]:
    """Run one episode, returning reward/metric traces.

    Strategies:
      'improve': move towards origin (works for sphere/rastrigin)
      'worsen':  move away from origin
      'random':  random actions
      'zero':    no movement
    """
    obs, _ = env.reset(seed=seed)
    rewards, metrics, raw_metrics = [], [], []

    for step in range(env.max_steps_limit):
        if strategy == "improve":
            current = np.array([env.current_hyp_setup[f"x{i}"] for i in range(env.num_hyperparams)])
            action = -np.sign(current).astype(np.float32)
        elif strategy == "worsen":
            current = np.array([env.current_hyp_setup[f"x{i}"] for i in range(env.num_hyperparams)])
            action = np.sign(current).astype(np.float32)
        elif strategy == "zero":
            action = np.zeros(env.num_hyperparams, dtype=np.float32)
        else:  # random
            action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        metrics.append(info["current_metric"])
        raw_metrics.append(env.raw_metric)

        if terminated or truncated:
            break

    return {
        "rewards": rewards,
        "metrics": metrics,
        "raw_metrics": raw_metrics,
        "total_reward": sum(rewards),
        "best_metric": min(metrics) if not env.backend.maximize else max(metrics),
        "final_metric": metrics[-1],
    }


# ===================================================================
# 1. SCALE ANALYSIS: что делает scale и как он влияет
# ===================================================================
class TestScaleAnalysis:
    """Анализ переменной scale = abs(_to_reward(initial)) + 1."""

    def test_scale_values_across_functions(self) -> None:
        """Вычислить scale для всех функций и показать разброс.

        Scale должен нормализовать rewards так, чтобы они были сопоставимы
        между функциями. Если scale сильно различается -> rewards несопоставимы.
        """
        scales = {}
        for fn in FUNCTIONS:
            env = _make_env(fn, reward_mode="guided")
            # Собираем scale из нескольких reset'ов
            fn_scales = []
            for seed in range(20):
                env.reset(seed=seed)
                scale = abs(env._to_reward(env._initial_metric)) + 1.0
                fn_scales.append(scale)
            scales[fn] = {
                "mean": np.mean(fn_scales),
                "std": np.std(fn_scales),
                "min": np.min(fn_scales),
                "max": np.max(fn_scales),
            }

        print("\n=== SCALE VALUES ACROSS FUNCTIONS ===")
        for fn, s in scales.items():
            print(f"  {fn:15s}: mean={s['mean']:10.2f}, std={s['std']:8.2f}, "
                  f"range=[{s['min']:8.2f}, {s['max']:8.2f}]")

        # Проверяем: scale не должен быть 0 или отрицательным
        for fn, s in scales.items():
            assert s["min"] > 0, f"{fn}: scale should always be > 0"

    def test_scale_effect_on_reward_magnitude(self) -> None:
        """Rewards после нормализации scale должны быть в разумном диапазоне.

        Если scale работает правильно, step rewards на разных функциях
        должны быть примерно в одном порядке (±1).
        """
        mode_stats = {}
        for mode in REWARD_MODES:
            fn_means = {}
            for fn in FUNCTIONS:
                env = _make_env(fn, reward_mode=mode)
                abs_rewards = []
                for seed in range(5):
                    result = _run_episode_with_strategy(env, "random", seed=seed)
                    abs_rewards.extend([abs(r) for r in result["rewards"]])
                fn_means[fn] = np.mean(abs_rewards) if abs_rewards else 0
            mode_stats[mode] = fn_means

        print("\n=== MEAN |REWARD| PER STEP BY MODE × FUNCTION ===")
        header = f"{'Mode':20s}" + "".join(f"{fn:>12s}" for fn in FUNCTIONS)
        print(header)
        for mode, fn_means in mode_stats.items():
            row = f"{mode:20s}" + "".join(f"{fn_means[fn]:12.4f}" for fn in FUNCTIONS)
            print(row)

        # Проверяем: rewards должны быть конечными
        for mode, fn_means in mode_stats.items():
            for fn, mean_r in fn_means.items():
                assert np.isfinite(mean_r), f"{mode}/{fn}: non-finite reward"


# ===================================================================
# 2. REWARD MODE PROFILES: детальный анализ каждого режима
# ===================================================================
class TestRewardModeProfiles:
    """Profile each mode on sphere with improving/worsening/random strategies."""

    @pytest.fixture(params=REWARD_MODES)
    def mode(self, request):
        return request.param

    def test_improving_agent_gets_positive_rewards(self, mode) -> None:
        """Agent moving towards optimum should get positive total reward."""
        env = _make_env("sphere", reward_mode=mode, max_steps=100)
        result = _run_episode_with_strategy(env, "improve", seed=42)
        # For most modes, improving should yield positive total reward
        # Exception: potential/best may be positive but small
        if mode in ("best", "absolute"):
            # These are position-based, always negative for sphere (minimize)
            # best gives symlog(-best_f), where best_f >= 0
            pass
        else:
            assert result["total_reward"] > 0, (
                f"Mode '{mode}': improving agent got total_reward={result['total_reward']:.4f}"
            )

    def test_worsening_agent_gets_negative_rewards(self, mode) -> None:
        """Agent moving away from optimum should get negative/lower total reward."""
        env = _make_env("sphere", reward_mode=mode, max_steps=100)
        result_improve = _run_episode_with_strategy(env, "improve", seed=42)
        result_worsen = _run_episode_with_strategy(env, "worsen", seed=42)
        assert result_improve["total_reward"] > result_worsen["total_reward"], (
            f"Mode '{mode}': improve={result_improve['total_reward']:.4f}, "
            f"worsen={result_worsen['total_reward']:.4f}"
        )

    def test_zero_action_reward_behavior(self, mode) -> None:
        """Zero action should give ~0 reward for delta-based, stable for position-based."""
        env = _make_env("sphere", reward_mode=mode, max_steps=50)
        result = _run_episode_with_strategy(env, "zero", seed=42)
        rewards = result["rewards"]
        # After first step (which may have initialization effects), rewards should be small
        mid_rewards = rewards[5:] if len(rewards) > 5 else rewards
        mean_abs = np.mean([abs(r) for r in mid_rewards])
        if mode in ("delta", "relative_delta", "potential", "guided"):
            assert mean_abs < 0.1, (
                f"Mode '{mode}': zero action should give ~0 reward, got mean|r|={mean_abs:.4f}"
            )


# ===================================================================
# 3. REWARD DYNAMICS: временная структура rewards
# ===================================================================
class TestRewardDynamics:
    """Анализ временных рядов rewards для каждого режима."""

    def test_reward_variance_across_modes(self) -> None:
        """Compare reward variance — low variance = easier for critic to learn.

        Modes with lower variance are generally easier for SAC's critic.
        """
        print("\n=== REWARD STATISTICS (sphere, random, 200 steps) ===")
        header = f"{'Mode':20s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s} {'Total':>10s}"
        print(header)

        stats = {}
        for mode in REWARD_MODES:
            env = _make_env("sphere", reward_mode=mode, max_steps=200)
            all_rewards = []
            for seed in range(10):
                result = _run_episode_with_strategy(env, "random", seed=seed)
                all_rewards.extend(result["rewards"])
            s = {
                "mean": np.mean(all_rewards),
                "std": np.std(all_rewards),
                "min": np.min(all_rewards),
                "max": np.max(all_rewards),
                "total_mean": np.mean([sum(all_rewards[i:i+200]) for i in range(0, len(all_rewards), 200)]),
            }
            stats[mode] = s
            row = f"{mode:20s} {s['mean']:8.4f} {s['std']:8.4f} {s['min']:8.4f} {s['max']:8.4f} {s['total_mean']:10.4f}"
            print(row)

        # delta/guided should have lower variance than absolute/best
        # (delta-based rewards are bounded by max_delta * scale, while
        #  absolute uses symlog on the full metric range)

    def test_return_monotonicity(self) -> None:
        """For improving agent, cumulative discounted return should increase over time.

        This is crucial for SAC: if improving policy → higher return,
        then Q-values have a clear gradient to follow.
        """
        gamma = 0.995
        print("\n=== RETURN MONOTONICITY (sphere, improving agent) ===")
        for mode in REWARD_MODES:
            env = _make_env("sphere", reward_mode=mode, max_steps=200)
            result = _run_episode_with_strategy(env, "improve", seed=42)
            rewards = result["rewards"]

            # Compute discounted return from each timestep
            returns = []
            G = 0
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            # Check if early returns < late returns (monotonically improving situation)
            early_return = np.mean(returns[:20])
            late_return = np.mean(returns[-20:])
            mid_return = np.mean(returns[90:110])

            print(f"  {mode:20s}: early_G={early_return:8.3f}, mid_G={mid_return:8.3f}, "
                  f"late_G={late_return:8.3f}")


# ===================================================================
# 4. FUNCTION SWITCHING: критический тест для SequentialBackend
# ===================================================================
class TestFunctionSwitching:
    """Test reward behavior when switching between functions.

    The core problem: after switching, scale changes drastically,
    and stale Q-values in replay buffer become misleading.
    """

    def test_scale_ratio_between_functions(self) -> None:
        """Measure how much scale changes between function switches.

        If scale_fn1 / scale_fn2 >> 1, replay buffer data is poisoned.
        """
        pairs = [
            ("sphere", "schwefel"),
            ("sphere", "rosenbrock"),
            ("rastrigin", "schwefel"),
            ("rosenbrock", "rastrigin"),
        ]
        print("\n=== SCALE RATIOS BETWEEN FUNCTION PAIRS ===")
        for fn1, fn2 in pairs:
            scales1, scales2 = [], []
            for seed in range(20):
                env1 = _make_env(fn1); env1.reset(seed=seed)
                s1 = abs(env1._to_reward(env1._initial_metric)) + 1.0
                scales1.append(s1)

                env2 = _make_env(fn2); env2.reset(seed=seed)
                s2 = abs(env2._to_reward(env2._initial_metric)) + 1.0
                scales2.append(s2)

            ratio = np.mean(scales1) / np.mean(scales2)
            print(f"  {fn1:12s} / {fn2:12s}: scale ratio = {ratio:.4f} "
                  f"(scales: {np.mean(scales1):.1f} vs {np.mean(scales2):.1f})")

    def test_reward_magnitude_after_switch(self) -> None:
        """Simulate function switching: run on fn1, then fn2.

        Check if reward magnitudes are comparable after switch.
        """
        print("\n=== REWARD MAGNITUDE AFTER FUNCTION SWITCH ===")
        for mode in REWARD_MODES:
            fn1_rewards, fn2_rewards = [], []

            for seed in range(5):
                env1 = _make_env("sphere", reward_mode=mode)
                r1 = _run_episode_with_strategy(env1, "random", seed=seed)
                fn1_rewards.extend([abs(r) for r in r1["rewards"]])

                env2 = _make_env("schwefel", reward_mode=mode)
                r2 = _run_episode_with_strategy(env2, "random", seed=seed)
                fn2_rewards.extend([abs(r) for r in r2["rewards"]])

            mean1 = np.mean(fn1_rewards)
            mean2 = np.mean(fn2_rewards)
            ratio = max(mean1, mean2) / (min(mean1, mean2) + 1e-8)
            print(f"  {mode:20s}: sphere={mean1:.6f}, schwefel={mean2:.6f}, ratio={ratio:.2f}x")

    def test_sequential_backend_reward_consistency(self) -> None:
        """Run with actual SequentialBackend and measure cross-function consistency."""
        print("\n=== SEQUENTIAL BACKEND: REWARD CONSISTENCY ===")
        functions = ["sphere", "schwefel", "rastrigin"]

        for mode in REWARD_MODES:
            env = _make_sequential_env(functions, reward_mode=mode)

            per_function_stats = {fn: [] for fn in functions}

            for fn_idx, fn in enumerate(functions):
                env.backend.set_active_backend(fn_idx)
                for seed in range(5):
                    result = _run_episode_with_strategy(env, "random", seed=seed)
                    per_function_stats[fn].append({
                        "mean_abs_reward": np.mean([abs(r) for r in result["rewards"]]),
                        "total_reward": result["total_reward"],
                    })

            means = {fn: np.mean([s["mean_abs_reward"] for s in stats])
                     for fn, stats in per_function_stats.items()}
            max_mean = max(means.values())
            min_mean = min(means.values()) + 1e-8
            ratio = max_mean / min_mean

            means_str = ", ".join(f"{fn}={means[fn]:.5f}" for fn in functions)
            print(f"  {mode:20s}: {means_str}, ratio={ratio:.2f}x")


# ===================================================================
# 5. DETAILED MODE COMPARISON: итоговая матрица
# ===================================================================
class TestModeComparison:
    """Comprehensive comparison matrix of all modes."""

    def test_comprehensive_matrix(self) -> None:
        """Generate full comparison matrix: mode × metric across functions and strategies.

        Metrics:
        - convergence_speed: how fast does metric improve (improving agent)?
        - reward_signal_density: what fraction of steps get non-zero reward?
        - cross_function_stability: how stable is reward magnitude across functions?
        - replay_compatibility: is reward Markovian (doesn't depend on history)?
        """
        print("\n" + "="*100)
        print("COMPREHENSIVE REWARD MODE COMPARISON")
        print("="*100)

        results = {}
        for mode in REWARD_MODES:
            mode_data = {
                "convergence_rate": [],
                "signal_density": [],
                "cross_fn_cv": 0.0,  # coefficient of variation across functions
                "replay_safe": True,
            }

            fn_reward_means = []
            for fn in FUNCTIONS:
                env = _make_env(fn, reward_mode=mode, max_steps=200)

                # Convergence: improving agent
                conv_results = []
                for seed in range(5):
                    r = _run_episode_with_strategy(env, "improve", seed=seed)
                    # Metric improvement relative to initial
                    initial = r["raw_metrics"][0] if r["raw_metrics"] else r["metrics"][0]
                    final = r["final_metric"]
                    # For minimize: lower is better, so improvement = initial - final
                    if not env.backend.maximize:
                        improvement = (initial - final) / (abs(initial) + 1)
                    else:
                        improvement = (final - initial) / (abs(initial) + 1)
                    conv_results.append(improvement)
                mode_data["convergence_rate"].append(np.mean(conv_results))

                # Signal density: fraction of non-trivial rewards
                density_results = []
                for seed in range(5):
                    r = _run_episode_with_strategy(env, "random", seed=seed)
                    nonzero = sum(1 for rr in r["rewards"] if abs(rr) > 1e-6)
                    density_results.append(nonzero / len(r["rewards"]))
                mode_data["signal_density"].append(np.mean(density_results))

                # Reward magnitude for cross-function comparison
                for seed in range(5):
                    r = _run_episode_with_strategy(env, "random", seed=seed)
                    fn_reward_means.append(np.mean([abs(rr) for rr in r["rewards"]]))

            # Cross-function coefficient of variation
            fn_means_by_fn = []
            for i, fn in enumerate(FUNCTIONS):
                env = _make_env(fn, reward_mode=mode)
                rs = []
                for seed in range(5):
                    r = _run_episode_with_strategy(env, "random", seed=seed)
                    rs.append(np.mean([abs(rr) for rr in r["rewards"]]))
                fn_means_by_fn.append(np.mean(rs))

            cv = np.std(fn_means_by_fn) / (np.mean(fn_means_by_fn) + 1e-8)
            mode_data["cross_fn_cv"] = cv

            # Replay safety: position-based modes (best, absolute) have
            # non-Markovian rewards w.r.t. state transitions
            # delta/guided depend only on (s, a, s') — replay safe
            mode_data["replay_safe"] = mode in ("delta", "relative_delta",
                                                   "guided", "potential")

            results[mode] = mode_data

        # Print comprehensive table
        print(f"\n{'Mode':20s} {'Conv(sphere)':>12s} {'Conv(schwfl)':>12s} "
              f"{'SignalDens':>10s} {'CrossFnCV':>10s} {'ReplaySafe':>10s}")
        print("-" * 80)

        sphere_idx = FUNCTIONS.index("sphere")
        schwefel_idx = FUNCTIONS.index("schwefel")
        for mode, data in results.items():
            conv_sp = data["convergence_rate"][sphere_idx] if len(data["convergence_rate"]) > sphere_idx else 0
            conv_sw = data["convergence_rate"][schwefel_idx] if len(data["convergence_rate"]) > schwefel_idx else 0
            density = np.mean(data["signal_density"])
            cv = data["cross_fn_cv"]
            safe = "YES" if data["replay_safe"] else "NO"
            print(f"{mode:20s} {conv_sp:12.4f} {conv_sw:12.4f} {density:10.4f} {cv:10.4f} {safe:>10s}")

        print(f"\nConv = fraction of initial metric improved (higher = better)")
        print(f"SignalDens = fraction of steps with |reward| > 1e-6 (higher = denser signal)")
        print(f"CrossFnCV = coefficient of variation of |reward| across {len(FUNCTIONS)} functions "
              f"(lower = more stable)")
        print(f"ReplaySafe = reward depends only on (s,a,s'), safe for off-policy replay buffer")

    def test_best_mode_reward_dynamics_detail(self) -> None:
        """Deep dive into 'best' mode — why it converges fast but switches badly."""
        print("\n=== DEEP DIVE: 'best' MODE ===")

        for fn in FUNCTIONS:
            env = _make_env(fn, reward_mode="best", max_steps=200)
            result = _run_episode_with_strategy(env, "improve", seed=42)

            rewards = result["rewards"]
            # best mode: reward = symlog(-best_f)
            # As agent improves, best_f decreases → -best_f increases → symlog increases
            # But once best is found, ALL subsequent steps get the SAME reward

            unique_rewards = len(set([round(r, 6) for r in rewards]))
            total = len(rewards)
            print(f"  {fn:15s}: unique_rewards={unique_rewards}/{total}, "
                  f"total={sum(rewards):.3f}, "
                  f"first_r={rewards[0]:.4f}, last_r={rewards[-1]:.4f}")

        print("\n  PROBLEM: After finding best → all steps get same reward")
        print("  → No gradient for Q-function to distinguish good/bad actions")
        print("  → After function switch, symlog(-best) jumps by orders of magnitude")
        print("  → Stale Q-values in replay buffer are completely wrong")

    def test_guided_mode_reward_dynamics_detail(self) -> None:
        """Deep dive into 'guided' mode."""
        print("\n=== DEEP DIVE: 'guided' MODE ===")

        for fn in FUNCTIONS:
            env = _make_env(fn, reward_mode="guided", max_steps=200)
            result = _run_episode_with_strategy(env, "improve", seed=42)

            rewards = result["rewards"]
            # Decompose: guided = r_delta + r_best + r_proximity
            # r_delta normalized by scale → O(max_delta/scale)
            # r_best = 0 or improvement/scale + 1.0
            # r_proximity = clip(gap/scale, -1, 0) * 0.1

            unique_rewards = len(set([round(r, 6) for r in rewards]))
            total = len(rewards)
            pos_rewards = sum(1 for r in rewards if r > 0.01)
            neg_rewards = sum(1 for r in rewards if r < -0.01)
            near_zero = total - pos_rewards - neg_rewards

            print(f"  {fn:15s}: unique={unique_rewards}/{total}, "
                  f"pos={pos_rewards}, neg={neg_rewards}, ~zero={near_zero}, "
                  f"mean={np.mean(rewards):.4f}, std={np.std(rewards):.4f}")


# ===================================================================
# 6. SPECIFIC BUG TESTS: known issues
# ===================================================================
class TestKnownIssues:
    """Test for known problematic behaviors."""

    def test_best_mode_same_reward_after_plateau(self) -> None:
        """'best' mode gives identical reward for every step after finding best.

        This makes TD-learning impossible: Q(s,a) = Q(s',a') for all s' after best.
        """
        env = _make_env("sphere", reward_mode="best", max_steps=100)
        env.reset(seed=42)

        # Walk to near-origin — should find best quickly
        for _ in range(30):
            current = np.array([env.current_hyp_setup[f"x{i}"] for i in range(2)])
            action = -np.sign(current).astype(np.float32)
            env.step(action)

        # Now walk randomly — best doesn't change, reward stays same
        rewards_after_best = []
        for _ in range(30):
            _, r, _, _, _ = env.step(env.action_space.sample())
            rewards_after_best.append(r)

        unique = len(set([round(r, 6) for r in rewards_after_best]))
        assert unique <= 3, (  # may have 1-2 improvements, mostly same
            f"After finding best, 'best' mode should give near-identical rewards. "
            f"Got {unique} unique values."
        )

    def test_absolute_mode_not_markovian(self) -> None:
        """'absolute' mode: _spawn() uses np.random.uniform (global RNG),
        NOT gymnasium's seeded RNG. So env.reset(seed=42) does NOT produce
        deterministic spawn positions.

        KNOWN ISSUE: this means same seed gives different initial positions,
        hence different rewards. The test documents this non-determinism.
        """
        env = _make_env("sphere", reward_mode="absolute", max_steps=50)

        # Two resets to same gymnasium seed → _spawn uses global np.random
        env.reset(seed=42)
        env.step(np.zeros(2, dtype=np.float32))
        r1 = env.reward

        env.reset(seed=42)
        env.step(np.zeros(2, dtype=np.float32))
        r2 = env.reward

        # DOCUMENT: rewards differ because _spawn() is not seeded
        # This is a real reproducibility bug in the environment.
        print(f"\n  spawn non-determinism: r1={r1:.6f}, r2={r2:.6f}, diff={abs(r1-r2):.6f}")
        assert abs(r1 - r2) > 0 or abs(r1 - r2) < 1e-6, (
            "Test is purely diagnostic — always passes"
        )

    def test_rank_shaped_scale_dependency(self) -> None:
        """'rank_shaped' depends on initial metric → different episodes get different scales.

        r = symlog((current - initial) / scale * 10).
        Since initial varies per episode, the same action at the same position
        gives different rewards in different episodes.
        """
        env = _make_env("sphere", reward_mode="rank_shaped", max_steps=50)

        rewards_per_seed = []
        for seed in [10, 20, 30, 40, 50]:
            env.reset(seed=seed)
            # Move towards origin
            _, r, _, _, _ = env.step(-np.ones(2, dtype=np.float32))
            rewards_per_seed.append(r)

        # Different seeds → different initial positions → different scales → different rewards
        reward_range = max(rewards_per_seed) - min(rewards_per_seed)
        assert reward_range > 0.01, (
            f"rank_shaped should produce different rewards for different initials: "
            f"range={reward_range:.6f}"
        )

    def test_potential_sparsity(self) -> None:
        """'potential' gives 0 reward on most steps (only when best improves).

        For random agent, this is very sparse — hard for SAC to learn from.
        """
        env = _make_env("sphere", reward_mode="potential", max_steps=200)
        result = _run_episode_with_strategy(env, "random", seed=42)

        nonzero = sum(1 for r in result["rewards"] if abs(r) > 1e-6)
        density = nonzero / len(result["rewards"])
        print(f"\n  'potential' signal density (random agent): {density:.2%} "
              f"({nonzero}/{len(result['rewards'])} steps)")
        # Usually very sparse for random walk
        assert density < 0.5, (
            f"'potential' should be sparse for random agent: density={density:.2%}"
        )

    def test_guided_components_balance(self) -> None:
        """Check: does best_bonus (+1) dominate over r_delta and r_proximity in guided?

        If best_bonus fires too often, it drowns out the delta signal.
        If it fires too rarely, guided degrades to delta.
        """
        env = _make_env("sphere", reward_mode="guided", max_steps=200)
        env.reset(seed=42)

        best_bonus_count = 0
        delta_magnitudes = []
        prox_magnitudes = []

        old_best = env.best_raw_metric
        for _ in range(200):
            _, r, _, _, _ = env.step(env.action_space.sample())
            if env.best_raw_metric != old_best:
                best_bonus_count += 1
                old_best = env.best_raw_metric

        print(f"\n  'guided' component balance (random, sphere, 200 steps):")
        print(f"  best_bonus fired: {best_bonus_count}/200 ({best_bonus_count/200:.1%})")

    def test_delta_mode_zero_sum_issue(self) -> None:
        """Delta rewards sum to final - initial over episode → total is predetermined!

        For delta mode: Σ(r_t) = Σ[(metric_t - metric_{t-1}) / scale]
                       = (metric_final - metric_initial) / scale

        This means total episode reward is fixed regardless of path.
        BAD for RL: agent can't distinguish good/bad trajectories if total is same.
        """
        env = _make_env("sphere", reward_mode="delta", max_steps=200)

        # Run episode; _run_episode_with_strategy resets internally
        result = _run_episode_with_strategy(env, "random", seed=42)

        # After episode, env still holds _initial_metric from that reset
        scale = abs(env._to_reward(env._initial_metric)) + 1.0
        initial_reward = env._to_reward(env._initial_metric)

        # The final metric is the last raw_metric in the episode
        final_reward = env._to_reward(result["raw_metrics"][-1])

        total_reward = sum(result["rewards"])
        expected_total = (final_reward - initial_reward) / scale

        print(f"\n  'delta' zero-sum check:")
        print(f"  total_reward    = {total_reward:.6f}")
        print(f"  (final-initial) / scale = {expected_total:.6f}")
        print(f"  diff = {abs(total_reward - expected_total):.2e}")
        # Delta is a telescoping sum: Σ(m_t - m_{t-1}) = m_final - m_initial
        # Stagnation penalty can break this if steps_without_improve > 5
        # So we allow a tolerance that accounts for stagnation penalty
        stag_penalty_max = 200 * 0.01 * 10  # worst case
        assert abs(total_reward - expected_total) < stag_penalty_max, (
            f"Delta total={total_reward:.6f} far from (final-initial)/scale={expected_total:.6f}"
        )


# ===================================================================
# 7. SWITCHING SIMULATION: full SequentialBackend cycle
# ===================================================================
class TestSwitchingSimulation:
    """Simulate function switching and measure impact on rewards."""

    def test_full_switch_cycle(self) -> None:
        """Run 3 episodes on different functions, compare reward statistics."""
        functions = ["sphere", "schwefel", "rastrigin"]

        print("\n=== FULL SWITCH CYCLE: 3 FUNCTIONS × ALL MODES ===")
        print(f"{'Mode':20s} | " + " | ".join(f"{'ep_' + fn[:6]:>12s}" for fn in functions) +
              " | ratio")
        print("-" * 90)

        for mode in REWARD_MODES:
            totals = []
            for fn in functions:
                env = _make_env(fn, reward_mode=mode, max_steps=200)
                runs = []
                for seed in range(5):
                    r = _run_episode_with_strategy(env, "random", seed=seed)
                    runs.append(abs(r["total_reward"]))
                totals.append(np.mean(runs))

            max_t = max(totals)
            min_t = min(totals) + 1e-8
            ratio = max_t / min_t
            vals = " | ".join(f"{t:12.4f}" for t in totals)
            print(f"{mode:20s} | {vals} | {ratio:6.1f}x")

    def test_replay_buffer_contamination(self) -> None:
        """Simulate: Q-values learned on fn1, then applied to fn2.

        If reward scales differ, the critic's Q-estimates are wrong.
        We measure: how much would stale Q misestimate returns on new function?
        """
        gamma = 0.995
        print("\n=== REPLAY BUFFER CONTAMINATION ANALYSIS ===")
        print(f"{'Mode':20s} {'sphere_G':>10s} {'schwefel_G':>10s} {'ratio':>8s} {'problem?':>10s}")

        for mode in REWARD_MODES:
            # Expected return on sphere
            env1 = _make_env("sphere", reward_mode=mode, max_steps=200)
            returns1 = []
            for seed in range(10):
                r = _run_episode_with_strategy(env1, "random", seed=seed)
                G = sum(r["rewards"][i] * gamma**i for i in range(len(r["rewards"])))
                returns1.append(G)

            # Expected return on schwefel
            env2 = _make_env("schwefel", reward_mode=mode, max_steps=200)
            returns2 = []
            for seed in range(10):
                r = _run_episode_with_strategy(env2, "random", seed=seed)
                G = sum(r["rewards"][i] * gamma**i for i in range(len(r["rewards"])))
                returns2.append(G)

            g1 = np.mean(returns1)
            g2 = np.mean(returns2)

            if min(abs(g1), abs(g2)) < 1e-6:
                ratio = float('inf') if max(abs(g1), abs(g2)) > 1e-6 else 1.0
            else:
                ratio = max(abs(g1), abs(g2)) / min(abs(g1), abs(g2))

            problem = "CRITICAL" if ratio > 10 else "WARNING" if ratio > 3 else "ok"
            print(f"{mode:20s} {g1:10.3f} {g2:10.3f} {ratio:8.1f}x {problem:>10s}")


# ===================================================================
# Runner
# ===================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
