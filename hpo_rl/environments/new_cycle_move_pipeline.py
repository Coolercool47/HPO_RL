import gymnasium as gym
import numpy as np
import itertools

from typing import Dict, Any, Optional, List
from hpo_rl.backends.base import EvaluationBackend
from hpo_rl.environments.base_env import BaseHPOEnv

class CyclicPipelineEnv(BaseHPOEnv):
    def __init__(self,
        hp_space: Dict[str, Any],
        backend: EvaluationBackend,
        step_sizes: List[int],
        num_bins: int = 20,
        max_steps: int = 200,
        # mode: bool = True,
        # sliding_window_size: int = 3
        ):

        super().__init__(hp_space, backend)

        self.num_bins = num_bins
        self.max_steps_limit = max_steps
        self.step_num_total = 0
        self.cur_step_num = 0
        # self.mode = mode
        # self.sliding_window_size = sliding_window_size
        self.step_sizes = step_sizes

        self.num_hyperparams = len(self.hp_names)
        # self.history_len = self.sliding_window_size * self.num_hyperparams
        # self.history = []

        self.cur_idx_dict = {} 

        self.current_hyp_setup = {}
        self.best_config_so_far = {}
        self.best_raw_metric = float('-inf') if self.backend.maximize else float('inf')
        self.current_raw_metric = float('-inf') if self.backend.maximize else float('inf')
        self.raw_metric = 0
        self.reward = 0

        self.hp_space_keys_iterator = itertools.cycle(self.hp_space_config)

        self.hp_lin_spaces = {} #инициализация np.linspace для гиперов 
        self._max_categorical_hyp_len = 0

        for hp_name, values in self.hp_space_config.items():
            if values["type"] == "float":
                self.hp_lin_spaces[hp_name] = np.linspace(values["values"][0], values["values"][1], num=self.num_bins, dtype=np.float32)
            elif values["type"] == "categorical":
                hyp_values_len = len(values["values"])
                if hyp_values_len > self._max_categorical_hyp_len:
                    self._max_categorical_hyp_len = hyp_values_len

        self._init_action_space()
        self._init_observation_space()

    def _init_action_space(self):
        self.step_sizes = [-i for i in self.step_sizes] + [0] + self.step_sizes
        self.step_sizes = sorted(self.step_sizes)

        num_actions = np.max(len(self.step_sizes), self._max_categorical_hyp_len)
        self.action_space = gym.spaces.Discrete(num_actions)

    def _init_observation_space(self):
        self.observation_space = gym.spaces.Dict(
            {
                "chosen_values": gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(self.num_hyperparams,), dtype=np.float32),
                "reward": gym.spaces.Box(low=float("-inf"), high=float("inf"), shape=(1,), dtype=np.float32),
                "optimized_param": gym.spaces.Box(low=0.0, high=1.0,shape=(self.num_hyperparams,),dtype=np.float32),
                "mask": gym.spaces.Box(low=0.0, high=1.0, shape=(self.action_space.n,), dtype=np.float32)
            })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed, options=options)

        self.current_hyp_setup = {}
        self.best_config_so_far = {}
        self.best_raw_metric = float('-inf') if self.backend.maximize else float('inf')
        self.current_raw_metric = float('-inf') if self.backend.maximize else float('inf')
        
        self.step_num_total = 0

        self.cur_step_num = 0

        # self.history = []

        self.hp_space_keys_iterator = itertools.cycle(self.hp_space_config)

        self._spawn()

        self._compute_reward()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        self._take_action(action)

        self.step_num_total += 1
        self.cur_step_num = (self.cur_step_num + 1) % self.num_hyperparams
        
        reward = self._compute_reward()
        observation = self._get_obs()
        info = self._get_info()
        terminated = self._terminated_logic()
        truncated = self._truncated_logic()

        return observation, reward, terminated, truncated, info

    def _calculate_mask(self):
        mask = np.ones(self.action_space.n, dtype=bool)

        return mask
    
    def _take_action(self, action):
        cur_hp_name = next(self.hp_space_keys_iterator)
        if self.current_hyp_setup[cur_hp_name]["type"] == "float":
            self.cur_idx_dict[cur_hp_name] = self.cur_idx_dict[cur_hp_name] + self.step_sizes[action]
            self.current_hyp_setup[cur_hp_name] = self.hp_lin_spaces[cur_hp_name][np.clip(self.cur_idx_dict[cur_hp_name], 0, len(self.hp_lin_spaces[cur_hp_name]))]

        elif self.current_hyp_setup[cur_hp_name]["type"] == "categorical":
            self.current_hyp_setup[cur_hp_name] = self.self.hp_space_config[cur_hp_name]["values"][action]
    
    def _spawn(self):
        for hp_name, values in self.hp_space_config.items():
            if values["type"] == "float":
                self.cur_idx_dict[hp_name] = np.random.randint(0, self.num_bins)
                self.current_hyp_setup[hp_name] = self.hp_lin_spaces[hp_name][self.cur_idx_dict[hp_name]]

            elif values["type"] == "categorical":
                self.current_hyp_setup[hp_name] = np.random.choice(values["values"])

    def _truncated_logic(self):
        return self.max_steps_limit > self.step_num_total

    def _terminated_logic(self):
        return False

    def _compute_reward(self):
        self.raw_metric = self.backend.evaluate(self.current_hyp_setup)
        if self.raw_metric > self.best_raw_metric:
            self.best_raw_metric = self.raw_metric
            self.best_config_so_far = self.current_hyp_setup
        self.reward = self.raw_metric
        return self.reward

    def _get_obs(self):
        observation = {}
        observation["reward"] = self.reward
        observation["chosen_values"] = [values for _, values in self.current_hyp_setup.values()]
        ohe = np.zeros(self.num_hyperparams)
        ohe[self.cur_step_num] = 1
        observation["optimized_param"] = ohe
        observation["mask"] = 
        return observation

    def _get_info(self) -> Dict[str, Any]:
        return {
            "best_config": self.best_config_so_far,
            "best_metric": self.best_raw_metric,
            "current_config": self.current_hyp_setup,
            "current_metric": self.current_raw_metric
        }

