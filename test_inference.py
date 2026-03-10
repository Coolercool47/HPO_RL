import sys, os
import ray, gymnasium as gym, torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from hpo_rl.environments.instant_continuous_pipeline_env import InstantContinuousPipelineEnv
from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.backends.sequential import SequentialBackend

def env_creator(config):
    backends = [OptimizationBenchmarkBackend(function_name='rastrigin', dimensions=2, maximize=False)]
    backend = SequentialBackend(backends=backends, mode='shuffle')
    child = backend.backends[0]
    lo, hi = child.bounds
    hp_space = {f'x{i}': {'type': 'float', 'values': [lo, hi]} for i in range(child.dimensions)}
    return InstantContinuousPipelineEnv(hp_space=hp_space, backend=backend, max_delta_frac=0.05, max_steps=200, history_window=3, reward_mode='absolute')

register_env('HPO-ContinuousPipeline-v0', env_creator)
ray.init(ignore_reinit_error=True, num_cpus=2, logging_level='ERROR')
ppo_config = PPOConfig().environment(env='HPO-ContinuousPipeline-v0').env_runners(num_env_runners=0).learners(num_learners=0)
algo = ppo_config.build_algo()

env = env_creator({})
obs, info = env.reset()

mod = algo.get_module('default_policy')
res = mod.forward_inference({'obs': torch.tensor(obs, dtype=torch.float32).unsqueeze(0)})
print('Keys:', list(res.keys()))

dist_class = algo.get_module('default_policy').get_exploration_action_dist_class()
print('dist_class:', dist_class)

algo.stop()
ray.shutdown()
