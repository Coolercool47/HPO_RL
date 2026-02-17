from hpo_rl.experiments.run_experiment import run_experiment
from hpo_rl.experiments.run_experiment import run_n_experiments
from hpo_rl.models.simple_cnn import SimpleCNN
from hpo_rl.trainers.torch_trainer import TorchTrainer
from hpo_rl.data_processing.processors import pytorch_mnist_processor

if __name__ == "__main__":
    config = {
    "algorithm": {
        "name": "SAC", 
        "verbose": 1,
        "gamma": 0.95,
        "learning_rate": 0.001,
        "total_timesteps": 50000,
        "inference_timesteps": 250,
        #"n_steps": 1000,
        "batch_size": 500,
        "policy": "MultiInputPolicy"
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        "max_steps": 250,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25],
        "action_type": "continuous"
    },
    "backend": {
        "name": "function",
        "function": "rastrigin",
        "dimensions": 10
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

    config_SimpleGA = {
        "backend": {
            "name": "function",
            "function": "rastrigin",
            "dimensions": 2
        },
        "algorithm": {
            "name": "SimpleGA",
            "N_pop": 20,
            "budget": 100,
            "mutation_prob": 0.1,
            "crossover_prob": 0.8,
            "tournament_size": 3,
            "elitism": True
        }
    }

    config_CMA_ES = {
        "backend": {
            "name": "function",
            "function": "rastrigin",
            "dimensions": 2
        },
        "algorithm": {
            "name": "CMA_ES",
            "N_pop": 10,
            "budget": 100,
            "initial_step_size": 0.5
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
    # Варианты запуска:
    # run_experiment(config_SimpleGA)
    # run_n_experiments(config_SimpleGA, 3)
    run_n_experiments(config, 1)
