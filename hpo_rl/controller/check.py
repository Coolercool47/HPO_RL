import yaml
import os

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from sb3_contrib import MaskablePPO, TRPO, RecurrentPPO

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.backends.real import RealTrainingBackend
from hpo_rl.backends.objective import ObjectiveBackend

from hpo_rl.baselines.BOHB import BOHB
from hpo_rl.baselines.TPE import TPE
from hpo_rl.baselines.hyperband import hyperband

from hpo_rl.models.simple_cnn import SimpleCNN

from hpo_rl.environments.cycle_move_pipeline import CyclicPipelineEnv

ALGORITHMS = {
    "A2C": A2C, "DQN": DQN, "PPO": PPO, "SAC": SAC, "TD3": TD3,
    "TRPO": TRPO, "MaskablePPO": MaskablePPO, "RecurrentPPO": RecurrentPPO,
    "TPE": TPE,  "BOHB": BOHB, "hyperband": hyperband
}

BACKENDS = {
    "function": OptimizationBenchmarkBackend, "real": RealTrainingBackend, "objective": ObjectiveBackend
}

BASELINES = {
    "TPE": TPE, "BOHB": BOHB, "hyperband": hyperband
}

ENVS = {
    "cycle_move_pipeline": CyclicPipelineEnv
}

MODELS = {
    "simle_cnn": SimpleCNN
}

def check(config):
    current_dir = os.path.dirname(os.path.abspath(__file__))

    config_path_alg = os.path.join(current_dir, '..', '..', 'configs', 'alg.yaml')
    config_path_alg = os.path.normpath(config_path_alg)
    with open(config_path_alg, 'r', encoding='utf-8') as a:
        alg_config = yaml.safe_load(a)

    config_path_functions = os.path.join(current_dir, '..', '..', 'configs', 'functions.yaml')
    config_path_functions = os.path.normpath(config_path_functions)
    with open(config_path_functions, 'r', encoding='utf-8') as f:
        functions_config = yaml.safe_load(f)

    algorithm_name = config.get("algorithm").get("name")
    alg_params = {}

    if algorithm_name in alg_config.get("algorithms").get("RL"):
        mode = "RL"
        algorithm_class = ALGORITHMS.get(algorithm_name)
        for key, value in config.get("algorithm").items():
            if key != "name":
                alg_params[key] = value

        env_name = config.get("env").get("name")
        env_params = {}
        env_class = ENVS.get(env_name)
        for key, value in config.get("env").items():
            if key != "name":
                print(key, value)
                env_params[key] = value

    elif algorithm_name in alg_config.get("algorithms").get("baselines"):
        mode = "baseline"
        algorithm_class = ALGORITHMS.get(algorithm_name)
        for key, value in config.get("algorithm").items():
            if key != "name":
                alg_params[key] = value
    else:
        raise

    backend_name = config.get("backend").get("name")
    backend_params = {}
    
    if backend_name == "function":
        backend_class = BACKENDS.get(backend_name)
        function_name = config.get("backend").get("function")
        if function_name in functions_config.get("functions"):
            backend_params = {"function_name": function_name, "dimensions": config.get("backend").get("dimensions")} #Сделать в yaml файле проверку на dim
        else: 
            raise
        min_value = functions_config.get("functions").get(function_name).get("min")
        max_value = functions_config.get("functions").get(function_name).get("max")
        if mode == "RL": #Определение гиперов должно быть не тут
            env_params["hp_space"] = {f"x{i}": {"min": min_value, "max": max_value, "type": "float", "log": False} for i in range(int(config.get("backend").get("dimensions")))}
        elif mode == "baseline":
            alg_params["dict_to_optimize"] = {f"x{i}": {"values": (min_value, max_value), "type": "float", "log": False} for i in range(int(config.get("backend").get("dimensions")))}
    
    elif backend_name == "real":
        backend_class = BACKENDS.get(backend_name)
        backend_config = config.get("backend")
        # TODO сделать проверки
        backend_params = {"model": backend_config.get("model"), "trainer": backend_config.get("trainer"), "data_processor": backend_config.get("data_processor"), "hp_space": backend_config.get("hp_space")}
        if mode == "RL":
            env_params["hp_space"] = backend_config.get("hp_space")
        elif mode == "baseline":
            alg_params["dict_to_optimize"] = backend_config.get("hp_space")

    elif backend_name == "objective":
        backend_class = BACKENDS.get(backend_name)
        backend_config = config.get("backend")
        backend_params = {"num_epochs": backend_config.get("num_epochs"), "objective_function": backend_config.get("objective_function"), "hp_space":backend_config.get("hp_space")}
        if mode == "RL":
            env_params["hp_space"] = backend_config.get("hp_space")
        elif mode == "baseline":
            alg_params["dict_to_optimize"] = backend_config.get("hp_space")
    else:
        raise
    

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