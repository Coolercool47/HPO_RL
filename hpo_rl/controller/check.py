import yaml
import os
import numpy as np

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from sb3_contrib import MaskablePPO, TRPO, RecurrentPPO

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.backends.real import RealTrainingBackend
from hpo_rl.backends.objective import ObjectiveBackend

from hpo_rl.baselines.BOHB import BOHB
from hpo_rl.baselines.TPE import TPE
from hpo_rl.baselines.hyperband import hyperband
from hpo_rl.baselines.SimpleGA import SimpleGA

from hpo_rl.models.simple_cnn import SimpleCNN

from hpo_rl.environments.cycle_move_pipeline import CyclicPipelineEnv

functions = {
    "rastrigin": {"min": -5.12, "max": 5.12},
    "sphere": {"min": -5, "max": 5},
    "rosenbrock": {"min": -1.0, "max": 1.0},
    "ackley": {"min": -32.768, "max": 32.768},
    "griewank": {"min": -600.0, "max": 600.0},
    "schwefel": {"min": -500.0, "max": 500.0},
    "levy": {"min": -10.0, "max": 10.0},
    "michalewicz": {"min": 0.0, "max": np.pi},
    "booth": {"min": -10.0, "max": 10.0},
    "beale": {"min": -4.5, "max": 4.5},
    "goldstein_price": {"min": -2.0, "max": 2.0},
    "shifted_sphere": {"min": -5.0, "max": 5.0},
    "shifted_rastrigin": {"min": -5.12, "max": 5.12}
}

ALGORITHMS_RL = {
    "A2C": A2C, "DQN": DQN, "PPO": PPO, "SAC": SAC, "TD3": TD3,
    "TRPO": TRPO, "MaskablePPO": MaskablePPO, "RecurrentPPO": RecurrentPPO,
}

ALGORITHMS_BASELINE = {
    "TPE": TPE,  "BOHB": BOHB, "hyperband": hyperband, "SimpleGA": SimpleGA
}

BACKENDS = {
    "function": OptimizationBenchmarkBackend, "real": RealTrainingBackend, "objective": ObjectiveBackend
}

ENVS = {
    "cycle_move_pipeline": CyclicPipelineEnv
}

MODELS = {
    "simle_cnn": SimpleCNN
}

def check(config):
    """Функция проверки конфигурации и задачи классов для последующей передачи в :class:`controller`.

    Args: 
        config: Необработанная конфигурация

    Поддерживаемые алгоритмы:
        - Обучение с подкреплением:
            - A2C
            - DQN
            - PPO
            - SAC
            - TD3
            - TRPO
            - MaskablePPO
            - RecurrentPPO
        - Классические
            - TPE
            - BOHB
            - Hyperband 
            - SimpleGA
    
    Поддерживаемые `backend`:
        - function
        - real
        - objective

    Поддерживемые среды:
        - cycle_move_pipeline

    Встренные модели для подбора гиперпараметров:
        - simle_cnn
    
    Returns:
        Конфигурацию для :class:`controller`
    
    """

    algorithm_name = config["algorithm"]["name"]
    alg_params = {}

    if algorithm_name in ALGORITHMS_RL:
        mode = "RL"
        algorithm_class = ALGORITHMS_RL[algorithm_name]
        for key, value in config["algorithm"].items():
            if key != "name":
                alg_params[key] = value

        env_name = config["env"]["name"]
        env_params = {}
        env_class = ENVS[env_name]
        for key, value in config["env"].items():
            if key != "name":
                env_params[key] = value

    elif algorithm_name in ALGORITHMS_BASELINE:
        mode = "baseline"
        algorithm_class = ALGORITHMS_BASELINE[algorithm_name]
        for key, value in config["algorithm"].items():
            if key != "name":
                alg_params[key] = value
    else:
        raise ValueError(f"Algorithm {algorithm_name} not supported")

    backend_name = config["backend"]["name"]
    backend_params = {}
    
    if backend_name == "function":
        backend_class = BACKENDS[backend_name]
        function_name = config["backend"]["function"]
        if function_name in functions:
            backend_params = {"function_name": function_name, "dimensions": config["backend"]["dimensions"]}
        else: 
            raise ValueError(f"Function {function_name} not supported")
            
        min_value = functions[function_name]["min"]
        max_value = functions[function_name]["max"]
        
        if mode == "RL":
            env_params["hp_space"] = {f"x{i}": {"min": min_value, "max": max_value, "type": "float", "log": False} for i in range(int(config["backend"]["dimensions"]))}
        elif mode == "baseline":
            alg_params["dict_to_optimize"] = {f"x{i}": {"values": (min_value, max_value), "type": "float", "log": False} for i in range(int(config["backend"]["dimensions"]))}
    
    elif backend_name == "real":
        backend_class = BACKENDS[backend_name]
        backend_config = config["backend"]
        backend_params = {
            "model": backend_config["model"], 
            "trainer": backend_config["trainer"], 
            "data_processor": backend_config["data_processor"], 
            "hp_space": backend_config["hp_space"]
        }
        if mode == "RL":
            env_params["hp_space"] = backend_config["hp_space"]
        elif mode == "baseline":
            alg_params["dict_to_optimize"] = backend_config["hp_space"]

    elif backend_name == "objective":
        backend_class = BACKENDS[backend_name]
        backend_config = config["backend"]
        backend_params = {
            "objective_function": backend_config["objective_function"], 
            "hp_space": backend_config["hp_space"]
        }
        if mode == "RL":
            env_params["hp_space"] = backend_config["hp_space"]
        elif mode == "baseline":
            alg_params["dict_to_optimize"] = backend_config["hp_space"]
    else:
        raise ValueError(f"Backend {backend_name} not supported")
    

    if mode == "RL":
        config_for_controller = {
            "mode": mode,
            "backend": {"class": backend_class, "params": backend_params},
            "algorithm": {"class": algorithm_class, "params": alg_params},
            "env": {"class": env_class, "params": env_params}
            }
    elif mode == "baseline":
        config_for_controller = {
            "mode": mode,
            "backend": {"class": backend_class, "params": backend_params},
            "algorithm": {"class": algorithm_class, "params": alg_params}
            }
            
    return config_for_controller