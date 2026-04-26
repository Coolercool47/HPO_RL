# InstantContinuousPipelineEnv: issue description and fix plan (implemented)

## Summary

Training with PPO on `instant_continuous_pipeline` (e.g. `SequentialBackend` with Rastrigin / Rosenbrock / Schwefel) failed to show convergence: trajectories were erratic, often hitting search-space boundaries (“bang-bang” behaviour), and the policy did not learn a stable search strategy.

## Root causes

### 1. Double action squashing (critical)

Tianshou’s continuous policy already maps raw Gaussian samples to `[-1, 1]` via `Policy.map_action` when `action_bound_method="tanh"`. The environment applied a second nonlinearity, `action = sign(a) * |a|^2`, on top of that.

That composed mapping is flat near 0, strongly nonlinear elsewhere, and encourages saturation at the extremes—working against both learning and the stated goal of “smoother” control.

**Fix:** Use only `np.clip(…, -1, 1)` in `_take_action` and linear scaling `deltas = action * _max_delta`. Further tuning: change `max_delta_frac` in config, not the action law.

### 2. Seeding: `_spawn` used global `np.random` (critical)

`np.random.uniform` ignored Gymnasium’s per-env `self.np_random` set by `reset(seed=…)`. Parallel `DummyVectorEnv` workers no longer had independent, reproducible initial points.

**Fix:** Sample with `self.np_random.uniform(lo, hi)`.

### 3. Redundant / fragile reward initialisation (significant)

`reset` evaluated the backend once, set many fields, then `_compute_reward` evaluated the same config again. The `if self._initial_metric is None: … else: …` branch duplicated the same `evaluate` call (dead for behaviour, wasteful, risky with noisy metrics).

**Fix:** Single `self.backend.evaluate` per `_compute_reward` call. `reset` only calls `_spawn` and `_compute_reward`. On the first reward computation of the episode, set `_initial_metric` when it is still `None`.

### 4. `prev_raw_metric` and first evaluation (significant)

After removing the duplicate pre-eval in `reset`, delta-based modes need a well-defined “previous” metric for the first computation. If `prev_raw_metric` is `None`, it is set to the current `raw_metric` so the first internal delta is zero, matching the intended “no step yet” semantics.

### 5. Uninitialised episode state in `__init__` (significant)

State such as `reward`, `raw_metric`, `current_hyp_setup` was only set in `reset`, which can break any code that touches the env before the first `reset`.

**Fix:** Define sensible defaults in `__init__` (as before the regression).

### 6. History buffer at `reset` (moderate)

Filling the history with zeros made normalized coordinates look like the left bound of the box. The cyclic pipeline instead fills with the current representation.

**Fix:** After the first `_compute_reward` in `reset`, set each history row to `[ _current_param_vec(), reward ]` (tiled to the window size).

### 7. `step`: `self.reward` when not OOB (moderate)

After refactoring, the non-OOB branch no longer always aligned `self.reward` with the returned `reward` used in observations. **Fix:** always assign `self.reward = reward` after OOB handling.

## Reference files

- Implementation: `hpo_rl/environments/instant_continuous_pipeline_env.py`
- Policy action mapping: Tianshou `Policy.map_action` (tanh / clip) + Collector

## Out of scope (not changed here)

- `check.py` merged bounds for asymmetric benchmarks (e.g. `bukin_n6` single range vs per-axis bounds) — only matters if that function is used in `sequential` configs.

## Verification

- Smoke: `InstantContinuousPipelineEnv` + `OptimizationBenchmarkBackend`, `reset` and `step` run without error.
- Full check: `python run_experiment.py` with `config_continuous_ppo` and compare trajectories / learning curves to prior logs.

## Related configuration hints

- If actions still look too large or too small, adjust `max_delta_frac` in the env config instead of reintroducing env-side nonlinearities on `action`.
- For continuous PPO, very high `ent_coef` can keep Gaussian variance large; tune together with `action_bound_method` and `max_delta_frac`.
