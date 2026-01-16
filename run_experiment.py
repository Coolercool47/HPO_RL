from hpo_rl.main_scripts.plot import plot
from hpo_rl.main_scripts.check import check
from hpo_rl.controller.controller import controller
import torch
from pathlib import Path
from datetime import datetime

def run_experiment(config):
    parsed_config = check(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = parsed_config.get("mode")

    algorithm_name = config.get("algorithm").get("name")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / algorithm_name / timestamp 
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = parsed_config.get("log_save_path", log_dir)

    expreiment_controller = controller(device = device, **parsed_config)
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
        "name": "PPO", 
        "verbose": 1,
        "gamma": 0.95,
        "learning_rate": 0.001,
        "total_timesteps": 10000,
        "inference_timesteps": 50,
        "policy": "MultiInputPolicy"
    },
    "env": {
        "name": "cycle_move_pipeline",
        "num_bins": 300,
        "max_steps": 20,
        "reward_mode": "per_step",
        "step_sizes": [1, 5, 25]
    },
    "backend": {
        "name": "function",
        "function": "rastrigin",
        "dimensions": 2
    }
    }
    run_experiment(config)