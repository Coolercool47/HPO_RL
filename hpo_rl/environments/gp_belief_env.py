import gymnasium as gym
import numpy as np
import warnings
from typing import Dict, Any, Optional

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.exceptions import ConvergenceWarning

from hpo_rl.environments.instant_continuous_pipeline_env import InstantContinuousPipelineEnv


class GPBeliefContinuousPipelineEnv(InstantContinuousPipelineEnv):
    """
    Continuous-action HPO environment that adds Gaussian Process (GP)
    belief state features to the observation.

    The agent sees standard features plus:
    - GP predictive mean and std at the current point (normalized)
    - Gradient of GP mean and std at the current point (local information)
    """

    def __init__(
        self,
        hp_space: Dict[str, Any],
        backend,
        max_delta_frac: float = 0.1,
        max_steps: int = 200,
        obs_mode: str = "norm",
        history_window: int = 0,
        reward_mode: str = "bounded",
        oob_penalty: float = -10.0,
        terminate_on_oob: bool = False,
        oob_tolerance: int = 1,
        gp_update_freq: int = 5,
        gp_max_points: int = 50,
    ):
        self.points_x = []
        self.points_y = []
        self.gp = None
        self.gp_update_freq = gp_update_freq
        self.gp_max_points = gp_max_points
        self.epsilon = 1e-4
        self._gp_features_cache = None  # cached GP features between refits

        super().__init__(
            hp_space=hp_space,
            backend=backend,
            max_delta_frac=max_delta_frac,
            max_steps=max_steps,
            obs_mode=obs_mode,
            history_window=history_window,
            reward_mode=reward_mode,
            oob_penalty=oob_penalty,
            terminate_on_oob=terminate_on_oob,
            oob_tolerance=oob_tolerance,
        )

    # ------------------------------------------------------------------
    # Observation space (extend parent with GP features)
    # ------------------------------------------------------------------
    def _init_observation_space(self):
        super()._init_observation_space()

        # mean, std, grad_mean (per dim), grad_std (per dim)
        gp_features_dim = 1 + 1 + self.num_hyperparams + self.num_hyperparams

        if isinstance(self.observation_space, gym.spaces.Dict):
            low = self.observation_space.spaces['obs'].low
            high = self.observation_space.spaces['obs'].high
            new_low = np.concatenate([low, -np.inf * np.ones(gp_features_dim, dtype=np.float32)])
            new_high = np.concatenate([high, np.inf * np.ones(gp_features_dim, dtype=np.float32)])
            self.observation_space = gym.spaces.Dict({
                "obs": gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)
            })
            self.flat_obs_dim = int(new_low.shape[0])
        else:
            low = self.observation_space.low
            high = self.observation_space.high
            new_low = np.concatenate([low, -np.inf * np.ones(gp_features_dim, dtype=np.float32)])
            new_high = np.concatenate([high, np.inf * np.ones(gp_features_dim, dtype=np.float32)])
            self.observation_space = gym.spaces.Box(low=new_low, high=new_high, dtype=np.float32)
            self.flat_obs_dim = int(new_low.shape[0])

    # ------------------------------------------------------------------
    # GP data management (separated from _get_obs)
    # ------------------------------------------------------------------
    def _record_gp_point(self, x_norm: np.ndarray, y_reward: float):
        """Record a data point for GP. Avoids duplicates."""
        if len(self.points_x) == 0 or not np.allclose(self.points_x[-1], x_norm):
            self.points_x.append(x_norm.copy())
            self.points_y.append(y_reward)

    def _fit_gp(self):
        """Fit the GP model on collected data.

        Only refits every ``gp_update_freq`` points to amortize the O(n^3)
        cost of GP fitting.  Uses at most ``gp_max_points`` most recent
        points so the cost stays bounded (~50^3 = 125K ops max).
        """
        n_points = len(self.points_x)
        if n_points < 2:
            return

        need_refit = (self.gp is None) or (n_points % self.gp_update_freq == 0)
        if not need_refit:
            return

        # Use only the last gp_max_points to keep O(n^3) bounded
        if n_points > self.gp_max_points:
            X = np.array(self.points_x[-self.gp_max_points:])
            y = np.array(self.points_y[-self.gp_max_points:])
        else:
            X = np.array(self.points_x)
            y = np.array(self.points_y)

        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + \
                 WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed")
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=0,
            normalize_y=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            try:
                self.gp.fit(X, y)
            except Exception:
                self.gp = None

        # Invalidate feature cache on refit
        self._gp_features_cache = None

    def _compute_gp_features(self, current_x: np.ndarray) -> np.ndarray:
        """Compute GP belief features (pure - no side effects).

        Returns normalized GP mean, std, and their gradients.
        The key fix: normalize GP predictions by _y_train_mean/_y_train_std
        BEFORE applying tanh, so features aren't saturated for large-scale functions.
        """
        gp_mean_norm = 0.0
        gp_std_norm = 0.0
        gp_mean_grad = np.zeros(self.num_hyperparams, dtype=np.float32)
        gp_std_grad = np.zeros(self.num_hyperparams, dtype=np.float32)

        if self.gp is not None:
            x_test = current_x.reshape(1, -1)
            try:
                mean_val, std_val = self.gp.predict(x_test, return_std=True)

                # Normalize by GP's internal y-scale so tanh doesn't saturate.
                # normalize_y=True means GP stores _y_train_mean and _y_train_std.
                # predict() returns de-normalized values, so we re-normalize.
                y_std = max(float(self.gp._y_train_std), 1e-8)
                y_mean = float(self.gp._y_train_mean)
                gp_mean_norm = (float(mean_val[0]) - y_mean) / y_std
                gp_std_norm = float(std_val[0]) / y_std

                # Gradients via finite differences, clamped to [0, 1]
                for i in range(self.num_hyperparams):
                    x_plus = x_test.copy()
                    x_minus = x_test.copy()
                    x_plus[0, i] = min(x_test[0, i] + self.epsilon, 1.0)
                    x_minus[0, i] = max(x_test[0, i] - self.epsilon, 0.0)

                    denom = x_plus[0, i] - x_minus[0, i]
                    if denom < 1e-10:
                        continue

                    m_plus, s_plus = self.gp.predict(x_plus, return_std=True)
                    m_minus, s_minus = self.gp.predict(x_minus, return_std=True)

                    gp_mean_grad[i] = ((float(m_plus[0]) - float(m_minus[0])) / y_std) / denom
                    gp_std_grad[i] = ((float(s_plus[0]) - float(s_minus[0])) / y_std) / denom
            except Exception:
                pass

        return np.concatenate([
            [np.tanh(gp_mean_norm), np.tanh(gp_std_norm)],
            np.clip(gp_mean_grad, -5.0, 5.0),
            np.clip(gp_std_grad, -5.0, 5.0),
        ]).astype(np.float32)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.points_x = []
        self.points_y = []
        self.gp = None
        self._gp_features_cache = None
        obs, info = super().reset(seed=seed, options=options)

        # Warmup GP by sampling random points for initial landscape
        for _ in range(3):
            rand_vals = self._lo + np.random.rand(self.num_hyperparams) * self._range
            hyp_dict = {name: float(v) for name, v in zip(self.hp_names, rand_vals)}
            raw = self.backend.evaluate(hyp_dict)
            y_r = self._to_reward(raw)
            x_norm = (rand_vals - self._lo) / (self._range + 1e-8)
            self._record_gp_point(x_norm, y_r)

        # Record the initial spawn point too
        current_x = self._current_param_vec()
        self._record_gp_point(current_x, self._to_reward(self.raw_metric))
        self._fit_gp()

        # Refetch obs to include GP features
        obs = self._get_obs()
        return obs, info

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------
    def _compute_reward(self):
        if self.reward_mode != "auto_sigmoid":
            return super()._compute_reward()

        # auto_sigmoid: evaluate metric ourselves to avoid double evaluation
        self.raw_metric = self.backend.evaluate(self.current_hyp_setup)
        self.current_raw_metric = self.raw_metric
        y_curr = self._to_reward(self.raw_metric)

        ys = np.array(self.points_y + [y_curr])
        if len(ys) < 2:
            self.reward = 0.0
            return self.reward

        y_max = np.max(ys)
        y_min = np.min(ys)
        scale = max(y_max - y_min, 1e-6)
        center = (y_max + y_min) / 2.0

        # k chosen so sigmoid(y_max) ~ +1, sigmoid(y_min) ~ -1
        k = 8.0 / scale
        r = 2.0 / (1.0 + np.exp(-k * (y_curr - center))) - 1.0

        if self._is_improvement(self.raw_metric, self.best_raw_metric):
            r += 0.5
            self.best_raw_metric = self.raw_metric
            self.best_config_so_far = self.current_hyp_setup.copy()

        self.reward = float(r)
        return self.reward

    # ------------------------------------------------------------------
    # Step (owns the full lifecycle)
    # ------------------------------------------------------------------
    def step(self, action):
        # Save history BEFORE action (pre-state semantics, like parent)
        if self.history_window > 0:
            self._history_buf = np.roll(self._history_buf, -1, axis=0)
            entry = np.concatenate([self._current_param_vec(), [self.reward]])
            self._history_buf[-1] = entry

        old_vec = self._current_param_vec().copy()

        self._take_action(action)
        self.step_num_total += 1

        # Record the new point for GP AFTER action, BEFORE obs
        current_x = self._current_param_vec()
        reward = self._compute_reward()
        y_reward = self._to_reward(self.raw_metric)
        self._record_gp_point(current_x, y_reward)
        self._fit_gp()

        # OOB penalty
        if self._out_of_bounds:
            self._oob_consecutive_count += 1
            reward += self.oob_penalty * self._oob_violation_frac
            self.reward = reward
        else:
            self._oob_consecutive_count = 0

        observation = self._get_obs()
        info = self._get_info()
        info["out_of_bounds"] = self._out_of_bounds
        info["oob_consecutive_count"] = getattr(self, '_oob_consecutive_count', 0)
        info["oob_violation_frac"] = self._oob_violation_frac

        terminated = self._terminated_logic()
        truncated = self._truncated_logic()

        # Stagnation detection: terminate if agent doesn't move for several steps
        new_vec = self._current_param_vec()
        dist = np.linalg.norm(new_vec - old_vec)
        if self.step_num_total > 5 and dist < 1e-4:
            terminated = True
            info["stagnated"] = True

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observations (pure - no side effects)
    # ------------------------------------------------------------------
    def _get_obs(self):
        base_obs = super()._get_obs()
        current_x = self._current_param_vec()

        # Recompute GP features only when cache is invalidated (after refit)
        # or when no cache exists yet. Between refits, reuse cached features
        # to avoid 5x predict() calls per step.
        if self._gp_features_cache is None:
            self._gp_features_cache = self._compute_gp_features(current_x)

        if isinstance(base_obs, dict):
            return {"obs": np.concatenate([base_obs['obs'], self._gp_features_cache])}
        else:
            return np.concatenate([base_obs, self._gp_features_cache])

    # ------------------------------------------------------------------
    # Action: absolute positioning (not relative deltas)
    # ------------------------------------------------------------------
    def _take_action(self, action):
        """Agent outputs absolute coordinates in [-1, 1].

        Maps directly to [lo, hi], allowing the GP agent to 'teleport'
        to any point in the space in one step (Bayesian Optimization style).
        """
        action = np.asarray(action, dtype=np.float32).flatten()
        action_clipped = np.clip(action, -1.0, 1.0)

        # [-1, 1] -> [0, 1] -> [lo, hi]
        norm_vals = (action_clipped + 1.0) / 2.0
        abs_new_vals = self._lo + norm_vals * self._range

        # Absolute positioning can't go out of bounds
        self._out_of_bounds = False
        self._oob_violation_frac = 0.0

        # Round int parameters
        if np.any(self._is_int):
            abs_new_vals[self._is_int] = np.round(abs_new_vals[self._is_int])

        for i, hp_name in enumerate(self.hp_names):
            self.current_hyp_setup[hp_name] = float(abs_new_vals[i])
