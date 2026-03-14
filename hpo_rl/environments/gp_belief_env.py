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
    - GP predictive mean and std at the current point
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
        gp_update_freq: int = 1,
    ):
        self.points_x = []
        self.points_y = []
        self.gp = None
        self.gp_update_freq = gp_update_freq
        self.epsilon = 1e-4
        
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

    def _init_observation_space(self):
        super()._init_observation_space()
        
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

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        self.points_x = []
        self.points_y = []
        self.gp = None
        obs, info = super().reset(seed=seed, options=options)
        
        # Warmup GP + Reward Sigmoid by sampling bounds randomly
        # We sample points to give GP an initial landscape and establish scale
        for _ in range(3):
            # random absolute values
            rand_vals = self._lo + np.random.rand(self.num_hyperparams) * self._range
            hyp_dict = {name: float(v) for name, v in zip(self.hp_names, rand_vals)}
            raw = self.backend.evaluate(hyp_dict)
            y_r = self._to_reward(raw)
            x_norm = (rand_vals - self._lo) / (self._range + 1e-8)
            self.points_x.append(x_norm)
            self.points_y.append(y_r)
            
        # Refetch obs to include features from the newly updated GP
        obs = self._get_obs()
        return obs, info

    def _compute_reward(self):
        # Allow base class to evaluate metric and track best_raw_metric
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            super()._compute_reward()
        
        if self.reward_mode == "auto_sigmoid":
            # Current y (can be negative if minimizing, but _to_reward aligns it so bigger is better)
            y_curr = self._to_reward(self.raw_metric)
            
            # Combine history (including warmup) and current to find scale
            ys = np.array(self.points_y + [y_curr])
            if len(ys) < 2:
                self.reward = 0.0
                return self.reward
                
            y_max = np.max(ys)
            y_min = np.min(ys)
            
            # The gap defines the scale
            scale = max(y_max - y_min, 1e-6)
            center = (y_max + y_min) / 2.0
            
            # We want the max value to map closely to +1 (e.g. sigmoid(4) ~ 0.98)
            # So k * (y_max - center) = 4.0
            k = 8.0 / scale
            
            # Auto-adjustable sigmoid mapping values roughly between -1 and 1
            # Current max is +1, current min is -1. As new extrema are found, it readjusts
            r = 2.0 / (1.0 + np.exp(-k * (y_curr - center))) - 1.0
            
            # Optionally add a small discrete bonus for finding a NEW best point
            if self._is_improvement(self.raw_metric, self.best_raw_metric):
                r += 0.5 
                self.best_raw_metric = self.raw_metric
                self.best_config_so_far = self.current_hyp_setup.copy()

            self.reward = float(r)
        return self.reward

    def step(self, action):
        old_vec = self._current_param_vec().copy()
        obs, reward, terminated, truncated, info = super().step(action)
        new_vec = self._current_param_vec().copy()
        
        # Terminate if the agent refuses to move (stagnates in the same spot)
        # Allows to organically handle "found optimum and stopped" or penalize doing nothing
        dist = np.linalg.norm(new_vec - old_vec)
        if self.step_num_total > 1 and dist < 1e-3:
            terminated = True
            info["stagnated"] = True
            
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        base_obs = super()._get_obs()
        
        # We need the current point
        current_x = self._current_param_vec().copy()
        y_reward = self._to_reward(self.raw_metric)
        
        # We store point right after base _get_obs
        # base_obs is generated on initial reset() and also on step()
        
        if len(self.points_x) == 0 or not np.allclose(self.points_x[-1], current_x):
            self.points_x.append(current_x)
            self.points_y.append(y_reward)
            
        n_points = len(self.points_x)
        
        gp_mean = 0.0
        gp_std = 0.0
        gp_mean_grad = np.zeros(self.num_hyperparams, dtype=np.float32)
        gp_std_grad = np.zeros(self.num_hyperparams, dtype=np.float32)

        # Fit GP if enough points
        if n_points >= 2:
            if n_points % self.gp_update_freq == 0 or self.gp is None:
                # Use fixed variance (jitter) for deterministic target functions, 
                # restrict length scale to meaningful bounds for scaled features
                kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5) + \
                         WhiteKernel(noise_level=1e-4, noise_level_bounds="fixed")
                self.gp = GaussianProcessRegressor(
                    kernel=kernel, 
                    n_restarts_optimizer=0, 
                    normalize_y=True
                )
                X = np.array(self.points_x)
                y = np.array(self.points_y)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    try:
                        self.gp.fit(X, y)
                    except Exception:
                        pass
            
            if self.gp is not None:
                x_test = current_x.reshape(1, -1)
                try:
                    mean_val, std_val = self.gp.predict(x_test, return_std=True)
                    gp_mean = float(mean_val[0])
                    gp_std = float(std_val[0])
                    
                    # Compute gradients via finite differences
                    for i in range(self.num_hyperparams):
                        x_plus = x_test.copy()
                        x_minus = x_test.copy()
                        x_plus[0, i] += self.epsilon
                        x_minus[0, i] -= self.epsilon
                        
                        m_plus, s_plus = self.gp.predict(x_plus, return_std=True)
                        m_minus, s_minus = self.gp.predict(x_minus, return_std=True)
                        
                        gp_mean_grad[i] = (m_plus[0] - m_minus[0]) / (2 * self.epsilon)
                        gp_std_grad[i] = (s_plus[0] - s_minus[0]) / (2 * self.epsilon)
                except Exception:
                    pass

        # Package features
        # using tanh scaling similar to other features
        # (could use some dynamic scaling but GP naturally standardizes with normalize_y=True)
        gp_feats = np.concatenate([
            [np.tanh(gp_mean), np.tanh(gp_std)],
            np.clip(gp_mean_grad, -5.0, 5.0),
            np.clip(gp_std_grad, -5.0, 5.0)
        ]).astype(np.float32)
        
        if isinstance(base_obs, dict):
            orig_obs = base_obs['obs']
            final_obs = np.concatenate([orig_obs, gp_feats])
            return {"obs": final_obs}
        else:
            orig_obs = base_obs
            final_obs = np.concatenate([orig_obs, gp_feats])
            return final_obs

    def _take_action(self, action):
        """
        Переопределяем базовое поведение (относительных шагов).
        Теперь агент выдает абсолютные координаты в нормированном пространстве [-1, 1].
        Мы масштабируем их напрямую в [lo, hi]. Это позволяет GP-агенту "телепортироваться"
        в любую точку пространства за один шаг (что и есть суть Bayesian Optimization).
        """
        action = np.asarray(action, dtype=np.float32).flatten()
        action_clipped = np.clip(action, -1.0, 1.0)
        
        # Переводим [-1, 1] в [0, 1]
        norm_vals = (action_clipped + 1.0) / 2.0
        
        # Переводим в абсолютные значения гиперпараметров
        abs_new_vals = self._lo + norm_vals * self._range
        
        # Так как агент всегда генерирует action в [-1, 1],
        # жестких выходов за исходные границы (bounds) у нас физически не может быть.
        self._out_of_bounds = False
        self._oob_violation_frac = 0.0
        
        for i, hp_name in enumerate(self.hp_names):
            self.current_hyp_setup[hp_name] = abs_new_vals[i]
