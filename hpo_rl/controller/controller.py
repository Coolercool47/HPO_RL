from tqdm.auto import tqdm
import gymnasium as gym
import tianshou as ts
from tianshou.data import CollectStats
from tianshou.utils.space_info import SpaceInfo
import gymnasium
from gymnasium.spaces import flatdim
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import unflatten
import torch
import pandas as pd
import os
import wandb
import time

class RainbowNetWrapper(torch.nn.Module):
    def __init__(self, model, action_num, num_atoms):
        super().__init__()
        self.model = model
        self.action_num = action_num
        self.num_atoms = num_atoms

    def forward(self, obs, state=None, info={}):
        # 1. Get flat logits from the standard Net [Batch, Action * Atoms]
        logits, hidden = self.model(obs, state=state, info=info)
        # 2. Reshape to [Batch, Action, Atoms] -> [64, 7, 51]
        logits = logits.view(-1, self.action_num, self.num_atoms)
        return logits, hidden
    
class controller():
    """
    Класс, скрепляющий конфигурацию с `backend` и `algorithm`.

    Args:
        device: device для подсчета RL алгоритмов
        mode: режим работы "RL" или "baseline"
        backend: получает класс `backend` и конфигурацию для него
        algorithm: алгоритм выбора гиперпараметров 
        env: среда для RL алгоритма
        save: путь сохранения RL модели
        load: путь загрузки RL модели

    Attributes:
        device: device для подсчета RL алгоритмов
        mode: режим работы "RL" или "baseline"
        backend: класс `backend`
        algorithm: алгоритм выбора гиперпараметров 
        env: среда для RL алгоритма
        save_loc: путь сохранения RL модели
        
    Note:
        `device`, `env`, `save_loc`, `load_loc` задаются только для случая `mode` = "RL"
    """
    def __init__(self, 
                 mode, backend, algorithm, alg_name=None, policy=None, 
                 trainer=None, logger=None, net=None, net_params = {},
                 training_collector_kwargs={}, test_collector_kwargs={}, 
                 inference_kwargs={}, n_training_envs=1, n_inference_envs=1,
                 env=None, save=None, load=None):
        
        self.mode = mode
        
        # Инициализация Backend
        backend_class = backend.get("class")
        self.backend = backend_class(**(backend.get("params")))

        # Группировка алгоритмов по парадигмам
        ON_POLICY_AC = ["ppo", "a2c", "trpo", "npg"]
        OFF_POLICY_TWIN_AC = ["sac", "td3"] # 1 Actor + 2 Critics
        OFF_POLICY_SINGLE_AC = ["ddpg", "discrete_sac"] # 1 Actor + 1 Critic
        VALUE_BASED = ["dqn", "rainbow", "c51", "qrdqn", "iqn", "fqf"] 
        PURE_POLICY = ["reinforce"]

        if self.mode == "RL":
            self.save_loc = save
            self.save_bool = save is not None
            self.inference_kwargs = inference_kwargs

            self.last_periodic_save = time.time()
            SAVE_INTERVAL_SECONDS = 30 * 60 
            
            # 1. Среды
            env_class = env.get("class")
            env_params = env.get("params")
            self.env = FlattenObservation(env_class(backend=self.backend, **env_params))

            def make_env():
                return FlattenObservation(env_class(backend=self.backend, **env_params))

            training_envs = ts.env.DummyVectorEnv([make_env for _ in range(n_training_envs)])
            test_envs = ts.env.DummyVectorEnv([make_env for _ in range(n_inference_envs)])

            # 2. Определение размерностей
            state_shape = flatdim(self.env.observation_space)
            if isinstance(self.env.action_space, gym.spaces.Discrete):
                action_shape = self.env.action_space.n
            else:
                action_shape = self.env.action_space.shape

            # 3. Извлечение классов
            NetClass = net
            PolicyClass = policy["class"]
            PolicyParams = policy["params"]
            AlgoClass = algorithm["class"]
            AlgoParams = algorithm["params"]
            
            ActorClass = PolicyParams.pop("actor", None)
            CriticClass = AlgoParams.pop("critic", None)

            # --- ФАБРИКА СБОРОК ---

            # On-Policy Actor-Critic (PPO, A2C...)
            if alg_name in ON_POLICY_AC or alg_name in OFF_POLICY_SINGLE_AC:
                net_a = NetClass(state_shape=state_shape, action_shape = action_shape, **net_params)
                net_c = NetClass(state_shape=state_shape, action_shape = action_shape, **net_params)
                
                actor = ActorClass(preprocess_net=net_a, action_shape=action_shape)
                critic = CriticClass(preprocess_net=net_c)
                
                _policy = PolicyClass(actor=actor, action_space=self.env.action_space, **PolicyParams)
                self.algo = AlgoClass(policy=_policy, critic=critic, **AlgoParams)

            # Off-Policy Twin Actor-Critic (SAC, TD3...)
            elif alg_name in OFF_POLICY_TWIN_AC:
                net_a = NetClass(state_shape=state_shape, **net_params)
                actor = ActorClass(preprocess_net=net_a, action_shape=action_shape)
                
                critics = []
                optim = AlgoParams.pop("optim")
                for _ in range(2):
                    net_c = NetClass(state_shape=state_shape, action_shape=action_shape,concat=True, **net_params)
                    critics.append({"critic":CriticClass(preprocess_net=net_c), "optim": optim})

                _policy = PolicyClass(actor=actor, action_space=self.env.action_space, **PolicyParams)
                self.algo = AlgoClass(policy=_policy, policy_optim =optim, critic=critics[0]["critic"], critic_optim = critics[0]["optim"], critic2=critics[1]["critic"], critic2_optim = critics[1]["optim"], **AlgoParams)
            # Value-Based (DQN...)
            elif alg_name in VALUE_BASED:
                num_atoms = PolicyParams.get("num_atoms", 51)
    
                if alg_name in ["rainbow", "c51"]:
                    # Initialize standard net with total flat size (7 * 51 = 357)
                    base_net = NetClass(
                        state_shape=state_shape, 
                        action_shape=action_shape * num_atoms, 
                        **net_params
                    )
                    # Wrap it to reshape output to [Batch, 7, 51]
                    q_net = RainbowNetWrapper(base_net, action_shape, num_atoms)
                else:
                    q_net = NetClass(
                        state_shape=state_shape, 
                        action_shape=action_shape, 
                        **net_params
                    )
                _policy = PolicyClass(model=q_net, action_space=self.env.action_space, **PolicyParams)
                self.algo = AlgoClass(policy=_policy, **AlgoParams)
            # Pure Policy (REINFORCE)
            elif alg_name in PURE_POLICY:
                net_a = NetClass(state_shape=state_shape, **net_params)
                actor = ActorClass(preprocess_net=net_a, action_shape=action_shape)
                _policy = PolicyClass(actor=actor, action_space=self.env.action_space, **PolicyParams)
                self.algo = AlgoClass(policy=_policy, **AlgoParams)

            # 4. Инициализация Коллекторов и Трейнера
            training_collector = ts.data.Collector[CollectStats](
                self.algo, training_envs, **training_collector_kwargs
            )
            test_collector = ts.data.Collector[CollectStats](
                self.algo, test_envs, **test_collector_kwargs
            )

            def save_best_fn(policy):
                if self.save_loc:
                    dir_name = os.path.dirname(self.save_loc)
                    if dir_name:
                        os.makedirs(dir_name, exist_ok=True)
                    
                    torch.save(policy.state_dict(), self.save_loc)
                    print(f"Model saved locally to: {self.save_loc}")
                    if wandb.run is not None:
                        # artifact = wandb.Artifact(name=f"{alg_name}_model", type="model")
                        # artifact.add_file(self.save_loc)
                        # wandb.log_artifact(artifact)
                        wandb.save(self.save_loc)
                    return self.save_loc
                return None
            
            def periodic_train_hook(epoch, env_step):
                current_time = time.time()
                
                if current_time - self.last_periodic_save >= SAVE_INTERVAL_SECONDS:
                    if self.save_loc:
                        self.last_periodic_save = current_time
                        
                        periodic_path = self.save_loc.replace("best_policy.pth", "periodic_backup.pth")
                        os.makedirs(os.path.dirname(periodic_path), exist_ok=True)
                        
                        policy_to_save = self.algo.policy if hasattr(self.algo, 'policy') else self.algo
                        torch.save(policy_to_save.state_dict(), periodic_path)
                        
                        print(f"[30-Min Dump] Saved to {periodic_path}")
                        
                        if wandb.run is not None:
                            wandb.save(periodic_path)
            
            self.trainer_initialized = trainer["class"](
                training_collector=training_collector,
                test_collector=test_collector,
                logger=logger,
                training_fn=periodic_train_hook, 
                save_best_fn = save_best_fn,
                **trainer["params"]
            )

        elif self.mode == "baseline":
            algorithm_class = algorithm.get("class")
            if self.backend.maximize:
                objective_func = lambda *args, **kwargs: -self.backend.evaluate(*args, **kwargs)
            else:
                objective_func = self.backend.evaluate
            self.algorithm = algorithm_class(objective_func=objective_func, **(algorithm.get("params")))

    def train(self):
        if self.mode == "RL":
            result = self.algo.run_training(self.trainer_initialized)
            
            # if self.save_bool:
            #     import torch
            #     p = self.algo.policy if hasattr(self.algo, "policy") else self.algo
            #     torch.save(p.state_dict(), self.save_loc)
            
            print(f"Finished training in {result.timing.total_time:.2f} seconds")

    def inference(self):
        if self.mode == "RL":
            collector = ts.data.Collector[CollectStats](self.algo, self.env, exploration_noise=True)
            collector.reset_buffer()
            result = collector.collect(**self.inference_kwargs)     
            
            n_steps = result.n_collected_steps 
            buffer = collector.buffer
            self.history = []
            for i in range(n_steps):
                step_info = buffer.info[i]
                # print(step_info)
                current_config = step_info["current_config"]
                current_metric = step_info["current_metric"]
                # if hasattr(current_config, "to_dict"):
                #     current_config = current_config.to_dict()
                # step_rew = buffer.rew[i]
                self.history.append((current_config, current_metric))
        
        elif self.mode == "baseline":
            self.algorithm.main_loop()
            if self.backend.maximize:
                self.history = [(cfg, -score) for cfg, score in self.algorithm.data]
            else:
                self.history = list(self.algorithm.data)
        return min(self.history, key=lambda x: x[-1]) if not self.backend.maximize else max(self.history, key=lambda x: x[-1])
    
    def return_history(self):
        return self.history
    