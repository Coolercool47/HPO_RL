"""Контроллер экспериментов HPO: связывает backend, алгоритм (RL или baseline) и среду."""

from tqdm.auto import tqdm
import inspect
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
        """Инициализирует контроллер эксперимента.

        Args:
            mode: ``"RL"`` или ``"baseline"``
            backend: dict с ключами ``class`` и ``params`` для бэкенда
            algorithm: dict с ключами ``class`` и ``params`` для алгоритма
            alg_name: имя RL-алгоритма (``"ppo"``, ``"recurrent_dqn"`` и т.д.)
            policy: конфигурация политики (``class``, ``params``)
            trainer: конфигурация тренера Tianshou
            logger: логгер (WandbLogger / TensorboardLogger)
            net: класс нейросети
            net_params: параметры нейросети
            training_collector_kwargs: аргументы training collector
            test_collector_kwargs: аргументы test collector
            inference_kwargs: аргументы collector при инференсе
            n_training_envs: число параллельных сред обучения
            n_inference_envs: число параллельных сред тестирования
            env: конфигурация среды (``class``, ``params``)
            save: путь сохранения лучшей политики
            load: путь загрузки чекпоинта
        """
        
        self.mode = mode
        
        backend_class = backend.get("class")
        self.backend = backend_class(**(backend.get("params")))

        ON_POLICY_AC = ["ppo", "a2c", "trpo", "npg", "recurrent_ppo"]
        OFF_POLICY_TWIN_AC = ["sac", "td3"] 
        OFF_POLICY_SINGLE_AC = ["ddpg", "discrete_sac"]
        VALUE_BASED = ["dqn", "recurrent_dqn", "rainbow", "c51", "qrdqn", "iqn", "fqf"] 
        PURE_POLICY = ["reinforce"]

        if self.mode == "RL":
            self.save_loc = save
            self.save_bool = save is not None
            self.inference_kwargs = inference_kwargs

            self.last_periodic_save = time.time()
            SAVE_INTERVAL_SECONDS = 30 * 60 
            
            env_class = env.get("class")
            env_params = env.get("params")
            self.env = FlattenObservation(env_class(backend=self.backend, **env_params)) if net == Net else env_class(backend=self.backend, **env_params)

            def make_env():
                """Фабрика среды: отдельный бэкенд на каждый воркер в vector env."""
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

            NetClass = net
            PolicyClass = policy["class"]
            PolicyParams = policy["params"]
            AlgoClass = algorithm["class"]
            AlgoParams = algorithm["params"]

            net_params = dict(net_params)
            _net_init_params = inspect.signature(NetClass.__init__).parameters
            if "device" in _net_init_params:
                if net_params.get("device") is None:
                    net_params["device"] = str(
                        torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    )
            else:
                net_params.pop("device", None)
            
            ActorClass = PolicyParams.pop("actor", None)
            actor_kwargs = PolicyParams.pop("actor_kwargs", {})
            CriticClass = AlgoParams.pop("critic", None)
            icm_config = AlgoParams.pop("icm", None)

            if alg_name in ON_POLICY_AC or alg_name in OFF_POLICY_SINGLE_AC:
                net_a = NetClass(state_shape=state_shape, action_shape=0, **net_params)
                net_c = NetClass(state_shape=state_shape, action_shape=0, **net_params)
                actor = ActorClass(preprocess_net=net_a, action_shape=action_shape, **actor_kwargs)
                critic = CriticClass(preprocess_net=net_c)
                
                _policy = PolicyClass(actor=actor, action_space=self.env.action_space, **PolicyParams)
                self.algo = AlgoClass(policy=_policy, critic=critic, **AlgoParams)

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
            elif alg_name in VALUE_BASED:
                num_atoms = PolicyParams.get("num_atoms", 51)
    
                if alg_name in ["rainbow", "c51"]:
                    base_net = NetClass(
                        state_shape=state_shape, 
                        action_shape=action_shape * num_atoms, 
                        **net_params
                    )
                    q_net = RainbowNetWrapper(base_net, action_shape, num_atoms)
                else:
                    q_net = NetClass(
                        state_shape=state_shape, 
                        action_shape=action_shape, 
                        **net_params
                    )
                _policy = PolicyClass(model=q_net, action_space=self.env.action_space, **PolicyParams)
                self.algo = AlgoClass(policy=_policy, **AlgoParams)
            elif alg_name in PURE_POLICY:
                net_a = NetClass(state_shape=state_shape, **net_params)
                actor = ActorClass(preprocess_net=net_a, action_shape=action_shape)
                _policy = PolicyClass(actor=actor, action_space=self.env.action_space, **PolicyParams)
                self.algo = AlgoClass(policy=_policy, **AlgoParams)
            else:
                supported = ON_POLICY_AC + OFF_POLICY_TWIN_AC + OFF_POLICY_SINGLE_AC + VALUE_BASED + PURE_POLICY
                raise ValueError(f"Algorithm '{alg_name}' is not supported. Supported: {supported}")

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

            self.load_loc = load
            if self.load_loc:
                normalized = os.path.normpath(self.load_loc)
                if not os.path.isfile(normalized) and os.path.isfile(self.load_loc):
                    normalized = self.load_loc  
                self.load_loc = normalized

            self._checkpoint_loaded = False
            if self.load_loc and os.path.isfile(self.load_loc):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                state_dict = torch.load(self.load_loc, map_location=device, weights_only=False)
                
                if "_optimizers" in state_dict:
                    self.algo.load_state_dict(state_dict)
                    print(f"Loaded full checkpoint (networks + optimizers) from: {self.load_loc}")
                else:
                    self.algo.policy.load_state_dict(state_dict)
                    print(f"Loaded policy weights (actor only) from: {self.load_loc}")
                    print("  Warning: optimizer state not restored, training continues with fresh optimizer")
                self._checkpoint_loaded = True
            elif self.load_loc:
                print(f"WARNING: checkpoint NOT found at: {self.load_loc}")
                print(f"  repr: {repr(self.load_loc)}")
                print(f"  Hint: если путь содержит backslash, используйте r\"...\" или '/'")
                print(f"  Модель будет инициализирована случайными весами!")

            training_collector = ts.data.Collector[CollectStats](
                self.algo, training_envs, **training_collector_kwargs
            )
            test_collector = ts.data.Collector[CollectStats](
                self.algo, test_envs, **test_collector_kwargs
            )

            def save_best_fn(policy):
                """Сохраняет лучшую политику на диск и в W&B при улучшении метрики."""
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
                """Периодически сохраняет полный чекпоинт алгоритма (раз в 30 минут)."""
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
        """Запускает обучение RL-алгоритма (только ``mode="RL"``).

        По завершении сохраняет финальный чекпоинт в ``final_policy.pth``.
        """
        if self.mode == "RL":
            result = self.algo.run_training(self.trainer_initialized)
            
            if self.save_loc:
                final_path = self.save_loc.replace("best_policy.pth", "final_policy.pth")
                os.makedirs(os.path.dirname(final_path), exist_ok=True)
                torch.save(self.algo.state_dict(), final_path)
                print(f"Final model saved to: {final_path}")
                if wandb.run is not None:
                    wandb.save(final_path)
            
            print(f"Finished training in {result.timing.total_time:.2f} seconds")

    def inference(self):
        """Выполняет инференс: сбор траектории RL или ``main_loop`` baseline.

        Заполняет ``history`` (и ``rewards`` для RL).

        Returns:
            tuple: лучшая пара (конфигурация, метрика) по направлению оптимизации backend.
        """
        if self.mode == "RL":
            collector = ts.data.Collector[CollectStats](self.algo, self.env, exploration_noise=False)
            collector.reset_buffer()
            result = collector.collect(**self.inference_kwargs)

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
        """Возвращает историю последнего инференса.

        Returns:
            list: список пар (конфигурация, метрика).
        """
        return self.history

    def return_rewards(self):
        """Возвращает пошаговые награды RL (для baseline — пустой список).

        Returns:
            list[float]: награды за шаги эпизода.
        """
        return getattr(self, 'rewards', [])
    