import yaml
import os
import numpy as np
import tianshou as ts
import datetime

import tianshou.algorithm.optim as opt

from tianshou.algorithm.modelfree.a2c import A2C
from tianshou.algorithm.modelfree.bdqn import BDQNPolicy, BDQN
from tianshou.algorithm.modelfree.c51 import C51Policy, C51
from tianshou.algorithm.modelfree.ddpg import DDPG, ContinuousDeterministicPolicy
from tianshou.algorithm.modelfree.discrete_sac import DiscreteSAC, DiscreteSACPolicy
from tianshou.algorithm.modelfree.dqn import DQN, DiscreteQLearningPolicy
from tianshou.algorithm.modelfree.fqf import FQF,FQFPolicy
from tianshou.algorithm.modelfree.iqn import IQN,IQNPolicy
from tianshou.algorithm.modelfree.npg import NPG
from tianshou.algorithm.modelfree.ppo import PPO
from tianshou.algorithm.modelfree.qrdqn import QRDQN, QRDQNPolicy
from tianshou.algorithm.modelfree.rainbow import RainbowDQN
from tianshou.algorithm.modelfree.redq import REDQ, REDQPolicy
from tianshou.algorithm.modelfree.reinforce import Reinforce, ProbabilisticActorPolicy, DiscreteActorPolicy
from tianshou.algorithm.modelfree.sac import SAC, SACPolicy
from tianshou.algorithm.modelfree.td3 import TD3
from tianshou.algorithm.modelfree.trpo import TRPO
from hpo_rl.alg.recurrent_ppo import ChunkedRNNPPO
from hpo_rl.alg.recurrent_dqn import ChunkedRNNDQN
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import WandbLogger
from tianshou.utils import TensorboardLogger

from tianshou.trainer import OffPolicyTrainerParams
from tianshou.trainer import OnPolicyTrainerParams

from tianshou.data import VectorReplayBuffer

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.backends.real import RealTrainingBackend
from hpo_rl.backends.objective import ObjectiveBackend

from hpo_rl.baselines.BOHB import BOHB
from hpo_rl.baselines.TPE import TPE
from hpo_rl.baselines.hyperband import hyperband
from hpo_rl.baselines.SimpleGA import SimpleGA
from hpo_rl.baselines.CMA_ES import CMA_ES

from hpo_rl.models.simple_cnn import SimpleCNN

from hpo_rl.environments.cycle_move_pipeline import CyclicPipelineEnv
from hpo_rl.environments.new_cycle_move_pipeline import CyclicPipelineEnvNew


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
    "onpolicy": {
        "a2c": A2C,
        "npg": NPG,
        "ppo": PPO,
        "recurrent_ppo": ChunkedRNNPPO,
        "reinforce": Reinforce,
        "trpo": TRPO,
    },
    "offpolicy": {
        "bdqn": BDQN,
        "c51": C51,
        "ddpg": DDPG,
        "discrete_sac": DiscreteSAC,
        "dqn": DQN,
        "recurrent_dqn": ChunkedRNNDQN,
        "fqf": FQF,
        "iqn": IQN,
        "qrdqn": QRDQN,
        "rainbow": RainbowDQN,
        "redq": REDQ,
        "sac": SAC,
        "td3": TD3,
    }
}

OPTIMIZERS = {
    "TorchOptimizerFactory": opt.TorchOptimizerFactory,
    "AdamOptimizerFactory": opt.AdamOptimizerFactory,
    "RMSpropOptimizerFactory": opt.RMSpropOptimizerFactory
}

ALGORITHMS_BASELINE = {
    "TPE": TPE,  "BOHB": BOHB, "hyperband": hyperband, "SimpleGA": SimpleGA, "CMA_ES": CMA_ES
}

BACKENDS = {
    "function": OptimizationBenchmarkBackend, "real": RealTrainingBackend, "objective": ObjectiveBackend
}

ENVS = {
    "cycle_move_pipeline": CyclicPipelineEnv,
    "new_cycle_move_pipeline": CyclicPipelineEnvNew
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
            - CMA_ES
    
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

    algorithm_name = config["full_args"]["algorithm"]["name"]
    alg_params = {}

    if algorithm_name in ALGORITHMS_RL["offpolicy"] or algorithm_name in ALGORITHMS_RL["onpolicy"]:
        mode = "RL"

        env_name = config["env"]["name"]
        env_params = {}
        env_class = ENVS[env_name]
        for key, value in config["env"].items():
            if key != "name":
                env_params[key] = value
        # env = env_class(**env_params)
        

        if algorithm_name in ALGORITHMS_RL["offpolicy"]:
            algorithm_class = ALGORITHMS_RL["offpolicy"][algorithm_name]
        else: 
            algorithm_class = ALGORITHMS_RL["onpolicy"][algorithm_name]
        for key, value in config["full_args"]["algorithm"].items():
            if key != "name":
                alg_params[key] = value
        if config["full_args"]["net"].get("critic"):
            alg_params["critic"] = config["full_args"]["net"]["critic"]

        optim_name = config["full_args"]["optim"]["name"]
        optim_class = OPTIMIZERS[optim_name]
        optim_params = {}
        for key, value in config["full_args"]["optim"].items():
            if key != "name":
                optim_params[key] = value
        alg_params["optim"] = optim_class(**optim_params)
        
        if algorithm_name in ALGORITHMS_RL["offpolicy"]:
            trainer_class = OffPolicyTrainerParams
        else:
            trainer_class = OnPolicyTrainerParams
        
        trainer_params = {}
        for key, value in config["full_args"]["trainer"].items():
            trainer_params[key] = value

        if "test_step_num_episodes" not in trainer_params and config["full_args"].get("num_test_envs", {}):
            trainer_params["test_step_num_episodes"] = config["full_args"]["num_test_envs"]


        if config["full_args"].get("buffer"):
            buffer_params = {}
            for key, value in config["full_args"]["buffer"].items():
                buffer_params[key] = value
            buffer = VectorReplayBuffer(**buffer_params)

        training_collector_params = {}
        if config["full_args"].get("training_collector_kwargs"):
            for key, value in config["full_args"]["training_collector_kwargs"].items():
                    training_collector_params[key] = value
            if config["full_args"].get("buffer"):
                training_collector_params["buffer"] = buffer

        test_collector_params = {}
        if config["full_args"].get("test_collector_kwargs"):
            for key, value in config["full_args"]["test_collector_kwargs"].items():
                    test_collector_params[key] = value

        policy_params = {}
        policy_class = config["full_args"]["policy"]["class"]
        for key, value in config["full_args"]["policy"].items():
                if key != "class":
                    policy_params[key] = value
        if config["full_args"]["net"].get("actor"):
            policy_params["actor"] = config["full_args"]["net"]["actor"]
        

        inference_params = {}
        if config["full_args"].get("inference"):
            for key, value in config["full_args"]["inference"].items():
                    inference_params[key] = value

        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # logger = TensorboardLogger(SummaryWriter(f"log/{algorithm_name}/{timestamp}"), update_interval=1, training_interval=1)
        logger = WandbLogger(update_interval=1, training_interval=1, save_interval = 1, entity = "hpo_rl", project="HPO_RL_stats_and_models", name=f"log/{algorithm_name}/{timestamp}")
        logger.load(SummaryWriter(f"log/{algorithm_name}/{timestamp}"))

        save = os.path.join(f"log/{algorithm_name}/{timestamp}", "best_policy.pth")

        if config["full_args"]["net"].get("net"):
            net = config["full_args"]["net"]["net"]
        net_params = {}
        for key, value in config["full_args"]["net"].items():
            if key != "net" and key != "critic" and key != "actor":
                net_params[key] = value
        # print(net_params)
        # hidden_states = config["full_args"]["net"]["hidden_states"]

    elif algorithm_name in ALGORITHMS_BASELINE:
        mode = "baseline"
        algorithm_class = ALGORITHMS_BASELINE[algorithm_name]
        for key, value in config["full_args"]["algorithm"].items():
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
            "net": net,
            "policy": {"class": policy_class, "params": policy_params},
            "trainer": {"class": trainer_class, "params": trainer_params},
            "logger": logger,
            "backend": {"class": backend_class, "params": backend_params},
            "algorithm": {"class": algorithm_class, "params": alg_params},
            "env": {"class": env_class, "params": env_params},
            "training_collector_kwargs": training_collector_params,
            "test_collector_kwargs": test_collector_params,
            "inference_kwargs": inference_params,
            "n_training_envs": config["full_args"]["num_training_envs"],
            "n_inference_envs": config["full_args"]["num_test_envs"],
            "alg_name": algorithm_name,
            "net_params": net_params,
            "save": save
            }
        
    elif mode == "baseline":
        config_for_controller = {
            "mode": mode,
            "backend": {"class": backend_class, "params": backend_params},
            "algorithm": {"class": algorithm_class, "params": alg_params}
            }
            
    return config_for_controller