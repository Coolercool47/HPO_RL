from hpo_rl.experiments.run_experiment import run_experiment
from hpo_rl.experiments.run_experiment import run_n_experiments
from hpo_rl.models.simple_cnn import SimpleCNN
from hpo_rl.trainers.torch_trainer import TorchTrainer
from hpo_rl.data_processing.processors import pytorch_mnist_processor
from torch.optim import Adam
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.utils.net.discrete import DiscreteActor
from tianshou.utils.net.discrete import DiscreteCritic
import torch
from tianshou.utils.net.common import Net

# def stop_fn(score):
#     return False

if __name__ == "__main__":
    config_ppo = {
    "full_args": {
        "algorithm":
        {
            "name": "ppo",
            "gamma": 0.9,
            # "n_step_return_horizon": 3,
            # "target_update_freq": 320,
        },  
        "optim":
        {
            "name": "TorchOptimizerFactory",
            "optim_class": Adam,
            "lr": 1e-3,
        },
        "net":
        {
            "actor": DiscreteActor,
            "critic": DiscreteCritic, 
            "hidden_states": [64, 64],
            "net": Net
        },
        "trainer":
        {
            "max_epochs": 10,
            "epoch_num_steps": 1000,
            "batch_size": 64,
            "collection_step_num_env_steps": 10,
            "update_step_num_repetitions": 5,
            # "test_in_training": True,
            # "stop_fn": stop_fn
        },
        "policy":
        {
            "class": ProbabilisticActorPolicy,
            "dist_fn": torch.distributions.Categorical,
            "action_scaling": False,
            # "eps_training": 0.1,
            # "eps_inference": 0.05,
        },
        "inference": 
        {
            "n_episode": 1,
            "reset_before_collect": True,
        },
        "num_training_envs": 10,
        "num_test_envs": 10,
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        # "action_type": "continuous",
        "max_steps": 100,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25]
    },
    "backend": {
        "name": "function",
        "function": "rastrigin",
        "dimensions": 2
    }
    }

    config_dqn = {
    "full_args": {
        "algorithm":
        {
            "name": "dqn",
            "gamma": 0.9,
            # "n_step_return_horizon": 3,
            # "target_update_freq": 320,
        },
        "buffer":
        {
            "total_size": 10000,
            "buffer_num": 10,
        },  
        "optim":
        {
            "name": "TorchOptimizerFactory",
            "optim_class": Adam,
            "lr": 1e-3,
        },
        "net":
        {
            # "actor": DiscreteActor,
            # "critic": DiscreteCritic, 
            "hidden_states": [64, 64],
            "net": Net
        },
        "trainer":
        {
            "max_epochs": 10,
            "epoch_num_steps": 1000,
            "batch_size": 64,
            "collection_step_num_env_steps": 10,
            # "update_step_num_repetitions": 5,
            # "test_in_training": True,
            # "stop_fn": stop_fn
        },
        "policy":
        {
            "class": DiscreteQLearningPolicy,
            # "dist_fn": torch.distributions.Categorical,
            # "action_scaling": False,
            # "eps_training": 0.1,
            # "eps_inference": 0.05,
        },
        "inference": 
        {
            "n_episode": 1,
            "reset_before_collect": True,
        },
        "num_training_envs": 10,
        "num_test_envs": 10,
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        # "action_type": "continuous",
        "max_steps": 100,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25]
    },
    "backend": {
        "name": "function",
        "function": "rastrigin",
        "dimensions": 2
    }
    }

    config_reinforce = {
    "full_args": {
        "algorithm":
        {
            "name": "sac",
            "gamma": 0.9,
            # "n_step_return_horizon": 3,
            # "target_update_freq": 320,
        },
        "optim":
        {
            "name": "TorchOptimizerFactory",
            "optim_class": Adam,
            "lr": 1e-3,
        },
        "net":
        {
            "actor": DiscreteActor,
            "critic": DiscreteCritic, 
            "hidden_states": [64, 64],
            "net": Net
        },
        "trainer":
        {
            "max_epochs": 10,
            "epoch_num_steps": 1000,
            "batch_size": 64,
            "collection_step_num_env_steps": 10,
            # "update_step_num_repetitions": 5,
            # "test_in_training": True,
            # "stop_fn": stop_fn
        },
        "policy":
        {
            "class": ProbabilisticActorPolicy,
            "dist_fn": torch.distributions.Categorical,
            "action_scaling": False,
            # "eps_training": 0.1,
            # "eps_inference": 0.05,
        },
        "inference": 
        {
            "n_episode": 1,
            "reset_before_collect": True,
        },
        "num_training_envs": 10,
        "num_test_envs": 10,
    },
    "env": {
        "name": "cycle_move_pipeline",
        # "num_bins": 300,
        "action_type": "continuous",
        "max_steps": 100,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25]
    },
    "backend": {
        "name": "function",
        "function": "rastrigin",
        "dimensions": 2
    }
    }

    config_TPE = {
    "backend": {
        "name": "function",
        "function": "rastrigin",
        "dimensions": 2
    },
    "algorithm": {
        "name": "TPE",
        "N_init": 10,
        "N_s": 10,
        "budget": 100,
        "separation_value": 0.2
    }
    }

    config_real = {
        "algorithm": {
        "name": "PPO", 
        "verbose": 1,
        "gamma": 0.95,
        "learning_rate": 0.001,
        "total_timesteps": 4,
        "inference_timesteps": 4,
        "n_steps": 2,
        "batch_size": 2,
        "policy": "MultiInputPolicy"
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        "max_steps": 2,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25]
    },
    "backend": {
        "name": "real",
        "model": SimpleCNN,
        "trainer": TorchTrainer,
        "data_processor": pytorch_mnist_processor,
        "hp_space": {
            "n_params": {
                "refers_to": "model",
                "type": "int",
                "min": 1,
                "max": 512
            },
            "learning_rate":{
                "type": "float",
                "min": 0,
                "max": 0.1
            },
            "optimizer": {
                "refers_to": "train_loop",
                "type": "categorical",
                "values": ["SGD", "Adam"],
                "dependencies": ["learning_rate"]
            },
            "criterion": {
                "refers_to": "train_loop",
                "type": "categorical",
                "values": ["CrossEntropyLoss"]
            },
            "learning_rate": {
                "refers_to": "optimizer",
                "type": "float",
                "min": 0,
                "max": 1
            }
        }
    }
    }
    run_n_experiments(config_dqn, 3)