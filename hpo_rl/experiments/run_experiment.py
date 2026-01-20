from hpo_rl.controller.plot import plot_and_save
from hpo_rl.controller.check import check
from hpo_rl.controller.controller import controller
import torch
from pathlib import Path
from datetime import datetime


def run_experiment(config):
    
    parsed_config = check(config)
    # print(config, parsed_config, sep="\n\n", end="\n\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = parsed_config.get("mode")

    algorithm_name = config.get("algorithm").get("name")
    backend_name = config.get("backend").get("name")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / algorithm_name / timestamp 
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = parsed_config.get("log_save_path", log_dir)

    expreiment_controller = controller(device=device, **parsed_config)
    if mode == "RL":
        expreiment_controller.train()
    best_result = expreiment_controller.inference()
    history = expreiment_controller.return_history()
    outputs = plot_and_save(history, best_result, save_path, expreiment_controller.backend)
    if backend_name == "function" and expreiment_controller.backend.dimensions == 2:
        outputs.plot_3d()
    outputs.plot_trajectory()
    outputs.save_history(as_latex=True)
    outputs.save_history(as_latex=False)