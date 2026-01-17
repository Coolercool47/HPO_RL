from hpo_rl.main_scripts.plot import plot
from hpo_rl.main_scripts.check import check
from hpo_rl.controller.controller import controller
import torch
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
    print(parsed_config)
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
    graphics.plot_3d()
    graphics.plot_trajectory()

if __name__ == "__main__":
    config = {
    "algorithm": {
        "name": "RecurrentPPO", 
        "verbose": 1,
        "gamma": 0.95,
        "learning_rate": 0.001,
        "total_timesteps": 10000,
        "inference_timesteps": 100,
        "policy": "MultiInputLstmPolicy"
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        "max_steps": 100,
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
                "type": "int",
                "min": 1,
                "max": 512
            },
            "optimizer": {
                "type": "categorical",
                "values": ["SGD", "Adam"]
            },
            "criterion": {
                "type": "categorical",
                "values": ["CrossEntorpyLoss"]
            }
        }
    }
    }
    run_experiment(config_real)