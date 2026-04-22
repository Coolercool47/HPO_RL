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

from tianshou.utils.net.discrete import IntrinsicCuriosityModule

from tianshou.data import VectorReplayBuffer

from hpo_rl.backends.function import OptimizationBenchmarkBackend
from hpo_rl.backends.objective import ObjectiveBackend
from hpo_rl.backends.sequential import SequentialBackend

from hpo_rl.baselines.BOHB import BOHB
from hpo_rl.baselines.TPE import TPE
from hpo_rl.baselines.hyperband import hyperband
from hpo_rl.baselines.SimpleGA import SimpleGA
from hpo_rl.baselines.CMA_ES import CMA_ES
from hpo_rl.baselines.HMM_MCMC import HMM_MCMC

from hpo_rl.environments.new_cycle_move_pipeline import CyclicPipelineEnvNew
from hpo_rl.environments.instant_continuous_pipeline_env import InstantContinuousPipelineEnv
from hpo_rl.environments.gp_belief_env import GPBeliefContinuousPipelineEnv


functions = {
    "rastrigin": {"values": [-5.12, 5.12], "type": "float", "log": False},
    "sphere": {"values": [-5, 5], "type": "float", "log": False},
    "rosenbrock": {"values": [-1.0, 1.0], "type": "float", "log": False},
    "ackley": {"values": [-32.768, 32.768], "type": "float", "log": False},
    "griewank": {"values": [-600.0, 600.0], "type": "float", "log": False},
    "schwefel": {"values": [-500.0, 500.0], "type": "float", "log": False},
    "levy": {"values": [-10.0, 10.0], "type": "float", "log": False},
    "michalewicz": {"values": [0.0, np.pi], "type": "float", "log": False},
    "booth": {"values": [-10.0, 10.0], "type": "float", "log": False},
    "beale": {"values": [-4.5, 4.5], "type": "float", "log": False},
    "goldstein_price": {"values": [-2.0, 2.0], "type": "float", "log": False},
    "shifted_sphere": {"values": [-5.0, 5.0], "type": "float", "log": False},
    "shifted_rastrigin": {"values": [-5.12, 5.12], "type": "float", "log": False},
    "bukin_n6": {"values": [-15.0, 3.0], "type": "float", "log": False},
    "cross_in_tray": {"values": [-10.0, 10.0], "type": "float", "log": False},
    "drop_wave": {"values": [-5.12, 5.12], "type": "float", "log": False},
    "eggholder": {"values": [-512.0, 512.0], "type": "float", "log": False},
    "holder_table": {"values": [-10.0, 10.0], "type": "float", "log": False},
    "schaffer_n2": {"values": [-100.0, 100.0], "type": "float", "log": False},
    "schaffer_n4": {"values": [-100.0, 100.0], "type": "float", "log": False},
    "shubert": {"values": [-10.0, 10.0], "type": "float", "log": False},
    "dejong_n5": {"values": [-65.536, 65.536], "type": "float", "log": False},
    "easom": {"values": [-100.0, 100.0], "type": "float", "log": False},
    "levy_n13": {"values": [-10.0, 10.0], "type": "float", "log": False},
    "langermann": {"values": [0.0, 10.0], "type": "float", "log": False},
    "styblinski_tang": {"values":[-5.0, 5.0], "type": "float", "log": False}
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
    "TPE": TPE,  "BOHB": BOHB, "hyperband": hyperband, "SimpleGA": SimpleGA, "CMA_ES": CMA_ES, "HMM_MCMC": HMM_MCMC
}

BACKENDS = {
    "function": OptimizationBenchmarkBackend, "objective": ObjectiveBackend, "sequential": SequentialBackend
}

ENVS = {
    "new_cycle_move_pipeline": CyclicPipelineEnvNew,
    "instant_continuous_pipeline": InstantContinuousPipelineEnv,
    "gp_belief_pipeline": GPBeliefContinuousPipelineEnv,
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

        # Путь для загрузки чекпоинта (опционально)
        load = config["full_args"].get("load_checkpoint", None)

        # --- ICM ---
        icm_raw = config["full_args"].get("icm")
        if icm_raw is not None:
            if "feature_net" not in icm_raw:
                raise ValueError(
                    "ICM config requires 'feature_net' — готовый инстанс nn.Module. "
                    "Пример: 'feature_net': MLP(input_dim=state_dim, output_dim=64, hidden_sizes=[64])"
                )
            icm_feature_net = icm_raw["feature_net"]
            icm_feature_dim = icm_raw.get("feature_dim", 64)
            icm_hidden = icm_raw.get("hidden_sizes", [64])

            icm_optim_name = icm_raw.get("optim", {}).get("name", "AdamOptimizerFactory")
            icm_optim_class = OPTIMIZERS.get(icm_optim_name, opt.AdamOptimizerFactory)
            icm_optim_params = {k: v for k, v in icm_raw.get("optim", {}).items() if k != "name"}
            if not icm_optim_params:
                icm_optim_params = {"lr": 1e-3}
            icm_optim = icm_optim_class(**icm_optim_params)

            for required_key in ("lr_scale", "reward_scale", "forward_loss_weight"):
                if required_key not in icm_raw:
                    raise ValueError(
                        f"ICM config requires '{required_key}'. "
                        "Задайте lr_scale, reward_scale и forward_loss_weight явно."
                    )

            alg_params["icm"] = {
                "feature_net": icm_feature_net,
                "feature_dim": icm_feature_dim,
                "hidden_sizes": icm_hidden,
                "optim": icm_optim,
                "lr_scale": icm_raw["lr_scale"],
                "reward_scale": icm_raw["reward_scale"],
                "forward_loss_weight": icm_raw["forward_loss_weight"],
            }

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
            if "noise_std" in config["backend"]:
                backend_params["noise_std"] = config["backend"]["noise_std"]
        else: 
            raise ValueError(f"Function {function_name} not supported")
            
        min_value = functions[function_name]["values"][0]
        max_value = functions[function_name]["values"][1]
        
        if mode == "RL":
            env_params["hp_space"] = {f"x{i}": {"values": [min_value, max_value], "type": "float", "log": False} for i in range(int(config["backend"]["dimensions"]))}
        elif mode == "baseline":
            alg_params["dict_to_optimize"] = {f"x{i}": {"values": [min_value, max_value], "type": "float", "log": False} for i in range(int(config["backend"]["dimensions"]))}
    
    elif backend_name == "sequential":
        backend_config = config["backend"]
        seq_mode = backend_config.get("mode", "random")
        backend_list = backend_config["backends"]  # список описаний бэкендов
        
        child_backends = []
        all_hp_spaces = []
        
        for entry in backend_list:
            entry_type = entry["name"]
            
            if entry_type == "function":
                fn = entry["function"]
                dims = entry["dimensions"]
                noise = entry.get("noise_std", 0.0)
                if fn not in functions:
                    raise ValueError(f"Function {fn} not supported")
                child_backends.append(
                    OptimizationBenchmarkBackend(function_name=fn, dimensions=dims, noise_std=noise)
                )
                min_v, max_v = functions[fn]["values"][0], functions[fn]["values"][1]
                all_hp_spaces.append(
                    {f"x{i}": {"values": [min_v, max_v], "type": "float", "log": False}
                     for i in range(int(dims))}
                )
                
            elif entry_type == "objective":
                child_backends.append(
                    ObjectiveBackend(
                        objective_function=entry["objective_function"],
                        hp_space=entry["hp_space"],
                    )
                )
                all_hp_spaces.append(entry["hp_space"])
                
            else:
                raise ValueError(f"Backend type '{entry_type}' not supported inside sequential")
        
        backend_class = SequentialBackend
        
        # hp_space: объединение по всем дочерним бэкендам
        # Берём ключи из первого, расширяем диапазоны по всем
        merged_hp_space = {}
        for hp_space in all_hp_spaces:
            for key, val in hp_space.items():
                if key not in merged_hp_space:
                    merged_hp_space[key] = dict(val)
                else:
                    existing = merged_hp_space[key]
                    if "values" in val and "values" in existing and getattr(val, "get", lambda x: "float")("type") in ("float", "int"):
                        existing["values"][0] = min(existing["values"][0], val["values"][0])
                        existing["values"][1] = max(existing["values"][1], val["values"][1])
                    elif "min" in val and "min" in existing:
                        existing["min"] = min(existing["min"], val["min"])
                    if "max" in val and "max" in existing:
                        existing["max"] = max(existing["max"], val["max"])
                    if "values" in val and "values" in existing:
                        # Для tuple (min, max)
                        if isinstance(val["values"], tuple) and isinstance(existing["values"], tuple):
                            existing["values"] = (
                                min(existing["values"][0], val["values"][0]),
                                max(existing["values"][1], val["values"][1]),
                            )

        backend_params = {
            "backends": child_backends,
            "mode": seq_mode,
        }

        if mode == "RL":
            env_params["hp_space"] = merged_hp_space
        elif mode == "baseline":
            alg_params["dict_to_optimize"] = merged_hp_space
    
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
            "save": save,
            "load": load
            }
        
    elif mode == "baseline":
        config_for_controller = {
            "mode": mode,
            "backend": {"class": backend_class, "params": backend_params},
            "algorithm": {"class": algorithm_class, "params": alg_params}
            }
            
    return config_for_controller