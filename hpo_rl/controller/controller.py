from tqdm.auto import tqdm
import numpy as np
import gymnasium as gym
import tianshou as ts
from tianshou.data import CollectStats
from tianshou.utils.space_info import SpaceInfo
from tianshou.utils.net.common import Net
import gymnasium
from gymnasium.spaces import flatdim
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import unflatten
import torch
import pandas as pd
import os
import wandb
import time

from tianshou.algorithm.modelbased.icm import ICMOnPolicyWrapper, ICMOffPolicyWrapper
from tianshou.utils.net.discrete import IntrinsicCuriosityModule
from hpo_rl.alg.recurrent_icm import RecurrentICMOnPolicyWrapper

from hpo_rl.backends.sequential import SequentialBackend
from hpo_rl.nets.icm_feature_net import ICMFeatureNet
from hpo_rl.nets.rainbow_net import RainbowNetWrapper

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
        ON_POLICY_AC = ["ppo", "a2c", "trpo", "npg", "recurrent_ppo"]
        OFF_POLICY_TWIN_AC = ["sac", "td3"] # 1 Actor + 2 Critics
        OFF_POLICY_SINGLE_AC = ["ddpg", "discrete_sac"] # 1 Actor + 1 Critic
        VALUE_BASED = ["dqn", "recurrent_dqn", "rainbow", "c51", "qrdqn", "iqn", "fqf"] 
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
            self.env = FlattenObservation(env_class(backend=self.backend, **env_params)) if net == Net else env_class(backend=self.backend, **env_params)

            def make_env():
                # Each training/test env gets its own SequentialBackend instance so
                # that backend.next_backend() calls in env.reset() are independent
                # (no mid-episode contamination from other parallel envs).
                # Child backends (OptimizationBenchmarkBackend) are shared since
                # they are stateless / cache-safe.
                if isinstance(self.backend, SequentialBackend):
                    per_env_backend = SequentialBackend(
                        backends=self.backend.backends,
                        mode=self.backend.mode,
                    )
                else:
                    per_env_backend = self.backend
                return FlattenObservation(env_class(backend=per_env_backend, **env_params)) if net == Net else env_class(backend=per_env_backend, **env_params)

            training_envs = ts.env.DummyVectorEnv([make_env for _ in range(n_training_envs)])
            test_envs = ts.env.DummyVectorEnv([make_env for _ in range(n_inference_envs)])

            # 2. Определение размерностей
            if net == Net:
                state_shape = flatdim(self.env.observation_space)
            elif isinstance(self.env.observation_space, gym.spaces.Dict) and "obs" in self.env.observation_space.spaces:
                state_shape = flatdim(self.env.observation_space["obs"])
            else:
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
            actor_kwargs = PolicyParams.pop("actor_kwargs", {})
            CriticClass = AlgoParams.pop("critic", None)
            icm_config = AlgoParams.pop("icm", None)

            # --- ФАБРИКА СБОРОК ---

            # On-Policy Actor-Critic (PPO, A2C...)
            if alg_name in ON_POLICY_AC or alg_name in OFF_POLICY_SINGLE_AC:
                net_a = NetClass(state_shape=state_shape, action_shape=0, **net_params)
                net_c = NetClass(state_shape=state_shape, action_shape=0, **net_params)
                
                # #region agent log
                import json as _json, time as _time, os as _os
                _log_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(__file__))), "debug-e46846.log")
                with open(_log_path, "a") as _f: _f.write(_json.dumps({"sessionId":"e46846","hypothesisId":"H1_bottleneck","location":"controller.py:net_creation","message":"Net output_dim after creation","data":{"net_a_output_dim": net_a.output_dim, "net_c_output_dim": net_c.output_dim, "action_shape": str(action_shape), "state_shape": state_shape, "NetClass": NetClass.__name__},"timestamp":int(_time.time()*1000)}) + "\n")
                # #endregion
                
                actor = ActorClass(preprocess_net=net_a, action_shape=action_shape, **actor_kwargs)
                
                # #region agent log
                _actor_info = {"actor_class": ActorClass.__name__, "actor_kwargs": str(actor_kwargs)}
                if hasattr(actor, 'sigma_param'):
                    _actor_info["has_sigma_param"] = True
                if hasattr(actor, 'sigma'):
                    _actor_info["sigma_type"] = str(type(actor.sigma))
                with open(_log_path, "a") as _f: _f.write(_json.dumps({"sessionId":"e46846","hypothesisId":"H12_sigma","location":"controller.py:actor_created","message":"Actor configuration","data":_actor_info,"timestamp":int(_time.time()*1000)}) + "\n")
                # #endregion
                critic = CriticClass(preprocess_net=net_c)
                
                _policy = PolicyClass(actor=actor, action_space=self.env.action_space, **PolicyParams)
                self.algo = AlgoClass(policy=_policy, critic=critic, **AlgoParams)

            # Off-Policy Twin Actor-Critic (SAC, TD3...)
            elif alg_name in OFF_POLICY_TWIN_AC:
                net_a = NetClass(state_shape=state_shape, **net_params)
                actor = ActorClass(preprocess_net=net_a, action_shape=action_shape, **actor_kwargs)
                
                critics = []
                optim = AlgoParams.pop("optim")
                for _ in range(2):
                    net_c = NetClass(state_shape=state_shape, action_shape=action_shape, concat=True, **net_params)
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
            else:
                supported = ON_POLICY_AC + OFF_POLICY_TWIN_AC + OFF_POLICY_SINGLE_AC + VALUE_BASED + PURE_POLICY
                raise ValueError(f"Algorithm '{alg_name}' is not supported. Supported: {supported}")

            # 3.5. ICM обёртка (Intrinsic Curiosity Module)
            if icm_config is not None:
                icm_model_cls = icm_config.get("model_class", IntrinsicCuriosityModule)
                feature_net = ICMFeatureNet(icm_config["feature_net"])
                icm_feature_dim = icm_config["feature_dim"]

                icm_model = icm_model_cls(
                    feature_net=feature_net,
                    feature_dim=icm_config["feature_dim"],
                    action_dim=int(action_shape) if np.isscalar(action_shape) else int(action_shape[0]),
                    hidden_sizes=icm_config.get("hidden_sizes", ()),
                )
                icm_optim = icm_config["optim"]
                icm_lr_scale = icm_config["lr_scale"]
                icm_reward_scale = icm_config["reward_scale"]
                icm_forward_loss_weight = icm_config["forward_loss_weight"]

                RECURRENT_ALGOS = ["recurrent_ppo", "recurrent_dqn"]

                if alg_name in ON_POLICY_AC + PURE_POLICY:
                    wrapper_cls = RecurrentICMOnPolicyWrapper if alg_name in RECURRENT_ALGOS else ICMOnPolicyWrapper
                    self.algo = wrapper_cls(
                        wrapped_algorithm=self.algo,
                        model=icm_model,
                        optim=icm_optim,
                        lr_scale=icm_lr_scale,
                        reward_scale=icm_reward_scale,
                        forward_loss_weight=icm_forward_loss_weight,
                    )
                elif alg_name in OFF_POLICY_TWIN_AC + OFF_POLICY_SINGLE_AC + VALUE_BASED:
                    self.algo = ICMOffPolicyWrapper(
                        wrapped_algorithm=self.algo,
                        model=icm_model,
                        optim=icm_optim,
                        lr_scale=icm_lr_scale,
                        reward_scale=icm_reward_scale,
                        forward_loss_weight=icm_forward_loss_weight,
                    )
                print(f"ICM wrapper applied to '{alg_name}' (reward_scale={icm_reward_scale}, lr_scale={icm_lr_scale})")

            # 4. Загрузка чекпоинта (если указан load)
            self.load_loc = load
            if self.load_loc:
                # Нормализация пути: замена / на os.sep, устранение escaped-символов
                # (например "\f" → form feed вместо "\final...")
                normalized = os.path.normpath(self.load_loc)
                if not os.path.isfile(normalized) and os.path.isfile(self.load_loc):
                    normalized = self.load_loc  # fallback: оригинальный путь работает
                self.load_loc = normalized

            self._checkpoint_loaded = False
            if self.load_loc and os.path.isfile(self.load_loc):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                state_dict = torch.load(self.load_loc, map_location=device, weights_only=False)
                
                if "_optimizers" in state_dict:
                    # Полный чекпоинт algo.state_dict() — сети + оптимизаторы
                    self.algo.load_state_dict(state_dict)
                    print(f"Loaded full checkpoint (networks + optimizers) from: {self.load_loc}")
                else:
                    # Только policy.state_dict() — только веса актора
                    self.algo.policy.load_state_dict(state_dict)
                    print(f"Loaded policy weights (actor only) from: {self.load_loc}")
                    print("  Warning: optimizer state not restored, training continues with fresh optimizer")
                self._checkpoint_loaded = True
            elif self.load_loc:
                print(f"WARNING: checkpoint NOT found at: {self.load_loc}")
                print(f"  repr: {repr(self.load_loc)}")
                print(f"  Hint: если путь содержит backslash, используйте r\"...\" или '/'")
                print(f"  Модель будет инициализирована случайными весами!")

            # 5. Инициализация Коллекторов и Трейнера
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
                # Periodic checkpoint save (every 30 min)
                current_time = time.time()
                
                if current_time - self.last_periodic_save >= SAVE_INTERVAL_SECONDS:
                    if self.save_loc:
                        self.last_periodic_save = current_time
                        
                        periodic_path = self.save_loc.replace("best_policy.pth", "periodic_backup.pth")
                        os.makedirs(os.path.dirname(periodic_path), exist_ok=True)
                        
                        torch.save(self.algo.state_dict(), periodic_path)
                        
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
            
            # Сохранение финальной модели (полный чекпоинт: сети + оптимизаторы)
            if self.save_loc:
                final_path = self.save_loc.replace("best_policy.pth", "final_policy.pth")
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                torch.save(self.algo.state_dict(), final_path)
                print(f"Final model saved to: {final_path}")
                if wandb.run is not None:
                    wandb.save(final_path)
            
            print(f"Finished training in {result.timing.total_time:.2f} seconds")

    def inference(self):
        if self.mode == "RL":
            # During inference, the caller (run_n_experiments) explicitly sets the active backend
            # via set_active_backend() before each inference call. That call automatically locks
            # the backend to prevent next_backend() from switching during env.reset().
            # After inference, we unlock for subsequent training.
            collector = ts.data.Collector[CollectStats](self.algo, self.env, exploration_noise=False)
            collector.reset_buffer()
            result = collector.collect(**self.inference_kwargs)

            # Unlock backend for subsequent training (if it was locked)
            if isinstance(self.backend, SequentialBackend):
                self.backend.unlock()
            
            n_steps = result.n_collected_steps 
            buffer = collector.buffer
            self.history = []
            self.rewards = []
            for i in range(n_steps):
                step_info = buffer.info[i]
                current_config = step_info["current_config"]
                current_metric = step_info["current_metric"]
                step_rew = float(buffer.rew[i])
                self.history.append((current_config, current_metric))
                self.rewards.append(step_rew)
        
        elif self.mode == "baseline":
            self.algorithm.main_loop()
            if self.backend.maximize:
                self.history = [(cfg, -score) for cfg, score in self.algorithm.data]
            else:
                self.history = list(self.algorithm.data)
        return min(self.history, key=lambda x: x[-1]) if not self.backend.maximize else max(self.history, key=lambda x: x[-1])
    
    def return_history(self):
        return self.history

    def return_rewards(self):
        return getattr(self, 'rewards', [])
    