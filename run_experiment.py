from hpo_rl.main_scripts.plot import plot
from hpo_rl.main_scripts.check import check
from hpo_rl.controller.controller import controller
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path
from datetime import datetime

from hpo_rl.models.simple_cnn import SimpleCNN
from hpo_rl.trainers.torch_trainer import TorchTrainer
from hpo_rl.data_processing.processors import pytorch_mnist_processor

# Пофиксить max/mix
# Сделать документацию
# Потыкать Real
# Составить Ipynb
# Доделать конфиги
# Инсталлятор/Деинсталлятор
# requirements.txt
# Readme.md
# Сделать картинки в формате Latex

def run_experiment(config):
    
    parsed_config = check(config)
    # print(config, parsed_config, sep="\n\n", end="\n\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = parsed_config.get("mode")

    algorithm_name = config.get("algorithm").get("name")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / algorithm_name / timestamp 
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = parsed_config.get("log_save_path", log_dir)

    expreiment_controller = controller(device=device, **parsed_config)
    if mode == "RL":
        expreiment_controller.train()
    best_result = expreiment_controller.inference()
    history = expreiment_controller.return_history()
    graphics = plot(history, best_result, save_path, expreiment_controller.backend)
    # graphics.plot_3d()
    graphics.plot_trajectory()


if __name__ == "__main__":
    config = {
    "algorithm": {
        "name": "PPO", 
        "verbose": 1,
        "gamma": 0.95,
        "learning_rate": 0.001,
        "total_timesteps": 100,
        "inference_timesteps": 100,
        "n_steps": 100,
        "batch_size": 50,
        "policy": "MultiInputPolicy"
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        "max_steps": 10,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25]
    },
    "backend": {
        "name": "function",
        "function": "sphere",
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
        "total_timesteps": 10000,
        "inference_timesteps": 100,
        "policy": "MultiInputPolicy"
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        "max_steps": 100,
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
                "values": [optim.SGD, optim.Adam],
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
    run_experiment(config)