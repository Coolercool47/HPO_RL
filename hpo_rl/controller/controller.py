from tqdm.auto import tqdm
import gymnasium as gym
import tianshou as ts
from tianshou.data import CollectStats
from tianshou.utils.space_info import SpaceInfo
import gymnasium
from gymnasium.spaces import flatdim
from gymnasium.wrappers import FlattenObservation
from gymnasium.spaces.utils import unflatten

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
                 mode, 
                 backend, 
                 algorithm,
                 alg_name = None,
                 policy = None, 
                 trainer = None,
                 logger = None,
                 net = None,
                 hidden_states = None,
                 training_collector_kwargs= {}, 
                 test_collector_kwargs = {}, 
                 inference_kwargs = {},
                 n_training_envs = 1,
                 n_inference_envs = 1,
                 device = None, 
                 env = None, 
                 save = None, 
                 load = None):
        """Инициализация класса controller

        Args:
            device: device для подсчета RL алгоритмов
            mode: режим работы "RL" или "baseline"
            backend: получает класс `backend` и конфигурацию для него
            algorithm: алгоритм выбора гиперпараметров 
            env: среда для RL алгоритма
            save: путь сохранения RL модели
            load: путь загрузки RL модели

        """
        # mode: "baseline"/"RL"
        # backend: {class: backend_style_class, params: function_or_real_params}
        # algorithm: {name:name, class: algorithm_style_class, params: alg_params}
        # env: {class: env_style_class, params: env_params}
        # save: save_location
        # load: load_location

        self.mode = mode
        self.device = device
        backend_class = backend.get("class")
        self.backend = backend_class(**(backend.get("params")))
        if self.mode == "RL":

            self.save_loc = save
            if self.save_loc != None:
                self.save_bool  = True
            else: 
                self.save_bool = False 
        
            load_loc = load
            if load_loc != None:
                load_bool = True
            else:
                load_bool = False
            if not load_bool:
                if alg_name in ["ppo"]:
                    env_class = env.get("class")
                    self.env = env_class(backend=self.backend, **(env.get("params")))

                    self.env = FlattenObservation(self.env)

                    training_envs = ts.env.DummyVectorEnv([lambda: FlattenObservation(env_class(backend=self.backend, **(env.get("params")))) for _ in range(n_training_envs)])
                    test_envs = ts.env.DummyVectorEnv([lambda: FlattenObservation(env_class(backend=self.backend, **(env.get("params")))) for _ in range(n_inference_envs)])

                    self.inference_kwargs = inference_kwargs
                    
                    state_shape = flatdim(self.env.observation_space)
                    if isinstance(self.env.action_space, gym.spaces.Discrete):
                        action_shape = self.env.action_space.n
                    elif isinstance(self.env.action_space, gym.spaces.Box):
                        action_shape = self.env.action_space.shape

                    print(state_shape,action_shape)

                    net_c = net(state_shape=state_shape,action_shape=action_shape, hidden_sizes=hidden_states)
                    net_a = net(state_shape=state_shape,action_shape=action_shape, hidden_sizes=hidden_states)

                    actor_class = policy["params"].pop("actor")(preprocess_net=net_a, action_shape=action_shape)
                    policy_class = policy["class"]
                    policy_initialized = policy_class(actor = actor_class, action_space = self.env.action_space, **policy["params"])

                    critic_class = algorithm["params"].pop("critic")(preprocess_net=net_c)
                    algorithm_class = algorithm["class"]
                    self.algorithm_initialized = algorithm_class(critic = critic_class, policy = policy_initialized, **algorithm["params"])
                    
                    training_collector = ts.data.Collector[CollectStats](
                        self.algorithm_initialized,
                        training_envs,
                        **training_collector_kwargs
                    )
                    test_collector = ts.data.Collector[CollectStats](
                        self.algorithm_initialized,
                        test_envs,
                        **test_collector_kwargs
                    )

                    trainer_class = trainer["class"]
                    self.trainer_initialized = trainer_class(
                            training_collector=training_collector,
                            test_collector=test_collector,
                            logger=logger,
                            **trainer["params"])
                else:
                    algorithm_class = algorithm.get("class")
                    self.algorithm = algorithm_class.load(load_loc, device=device)

        elif self.mode == "baseline":
            algorithm_class = algorithm.get("class")
            self.algorithm = algorithm_class(objective_func=lambda *args, **kwargs: -self.backend.evaluate(*args, **kwargs), **(algorithm.get("params"))) #интегрировать backend в baseline'ы  

    def train(self):
        """Запускает обучение модели
        
        note:
            Работает только для `mode` == "RL"

        """
        if self.mode == "RL":
            result = self.algorithm_initialized.run_training(self.trainer_initialized)
            if self.save_bool:
                self.algorithm_initialized.save(self.save_loc)
            print(f"Finished training in {result.timing.total_time} seconds")
    
    def inference(self):
        """Запускает инференс модели
        
        Returns:
            Лучшие параметры модели
        """

        # Добавить сохранение лучшей модели
    
        if self.mode == "RL":
            collector = ts.data.Collector[CollectStats](self.algorithm_initialized, self.env, exploration_noise=True)
            collector.reset_buffer()
            result = collector.collect(**self.inference_kwargs)     
            n_steps = result.n_collected_steps         
            buffer = collector.buffer
            self.history = []
            for i in range(n_steps):
                step_info = buffer.info[i]
                current_config = step_info["current_config"]
                step_rew = buffer.rew[i]
                self.history.append((current_config, step_rew))
            # print(self.history)
        elif self.mode == "baseline":
            self.algorithm.main_loop()
            self.history = [(i[0], -i[1]) for i in self.algorithm.data]
        return min(self.history, key=lambda x: x[-1]) if not self.backend.maximize else max(self.history, key=lambda x: x[-1])

    def return_history(self):
        """Возвращает историю гиперпараметров работы инференса
        
        Returns:
            История гиперпараметров 
        """
        return self.history