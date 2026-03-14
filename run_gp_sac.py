from hpo_rl.experiments.run_experiment import run_n_experiments
from hpo_rl.nets.base_net import BaseNet
from torch.optim import Adam
from tianshou.utils.net.continuous import ContinuousActorProbabilistic
from tianshou.utils.net.continuous import ContinuousCritic
import torch
import torch.nn as nn
from tianshou.utils.net.common import Net
from tianshou.algorithm.modelfree.sac import SACPolicy
import numpy as np


config_sac_gp = {
    "full_args": {
        "algorithm": {
            "name": "sac",
            "gamma": 0.95,
        },
        "optim": {
            "name": "TorchOptimizerFactory",
            "optim_class": Adam,
            "lr": 1e-3,
        },
        "net": {
            "actor": ContinuousActorProbabilistic,
            "critic": ContinuousCritic,
            "net": Net,
            "hidden_sizes": [128, 128],
        },
        "buffer":
            {
                "total_size": 100000,
                "buffer_num": 20,
                "stack_num": 1,
            },
        "trainer":
        {
            "max_epochs": 6,             
            "epoch_num_steps": 4000,
            "batch_size": 256,
            "collection_step_num_env_steps": 2000,
            "update_step_num_gradient_steps_per_sample": 1.0,
            "test_step_num_episodes": 20,
        },
        "policy": {
            "class": SACPolicy,
            "action_scaling": True,
        },
        "inference": {
            "n_episode": 2,
            "reset_before_collect": True,
        },
        "num_training_envs": 2,
        "num_test_envs": 2
    },
    "env": {
        "name": "gp_belief_pipeline",
        "max_delta_frac": 1.0,  # Больше не используется, так как у нас абсолютные шаги
        "history_window": 0,
        "obs_mode": "norm",
        "reward_mode": "auto_sigmoid", # Используем нашу новую самобалансирующуюся сигмоиду
        "max_steps": 200,
        "gp_update_freq": 1
    },
    "backend": {
        "name": "sequential",
        "mode": "shuffle",
        "backends": [
            {"name": "function", "function": "rastrigin", "dimensions": 2},
            {"name": "function", "function": "rosenbrock", "dimensions": 2},
            {"name": "function", "function": "schwefel", "dimensions": 2},
        ]
    }
}

if __name__ == "__main__":
    print("Testing GP Belief Agent with SAC and Sequential Backend...")
    run_n_experiments(config_sac_gp, n_experiments=1, inference_only=False)
