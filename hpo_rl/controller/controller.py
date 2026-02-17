from tqdm.auto import tqdm

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
    def __init__(self, mode, backend, algorithm, device = None, env = None, save = None, load = None):
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
        # algorithm: {class: algorithm_style_class, params: alg_params}
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
                env_class = env.get("class")
                self.env = env_class(backend=self.backend, **(env.get("params")))

                self.total_timesteps = algorithm.get("params").pop("total_timesteps")
                self.inference_timesteps = algorithm.get("params").pop("inference_timesteps")

                algorithm_class = algorithm.get("class")
                self.algorithm = algorithm_class(device=self.device, env=self.env, **(algorithm.get("params")))
            else:
                algorithm_class = algorithm.get("class")
                self.algorithm = algorithm_class.load(load_loc, device=device)

        elif self.mode == "baseline":
            algorithm_class = algorithm.get("class")
            if self.backend.maximize:
                objective_func = lambda *args, **kwargs: -self.backend.evaluate(*args, **kwargs)
            else:
                objective_func = self.backend.evaluate
            self.algorithm = algorithm_class(objective_func=objective_func, **(algorithm.get("params")))

    def train(self):
        """Запускает обучение модели
        
        note:
            Работает только для `mode` == "RL"

        """
        if self.mode == "RL":
            self.algorithm.learn(total_timesteps=self.total_timesteps, progress_bar = True)
            if self.save_bool:
                self.algorithm.save(self.save_loc)
    
    def inference(self):
        """Запускает инференс модели
        
        Returns:
            Лучшие параметры модели
        """

        # Добавить сохранение лучшей модели
    
        if self.mode == "RL":
            
            self.history = []
            env = self.algorithm.env
            obs = env.reset()
            inference_bar = tqdm(total=int(self.inference_timesteps if self.inference_timesteps <= self.env.max_steps_limit else self.env.max_steps_limit),desc="Inference", position=0, leave=True)
            for _ in range(self.inference_timesteps):
                action, _states = self.algorithm.predict(obs, deterministic=True)
                obs, _rewards, done, infos = env.step(action)
                info = infos[0]
                config = info.get("current_config")
                metric = info.get("current_metric")
                self.history.append([config, metric])
                inference_bar.update(1)
                if done:
                    inference_bar.close()
                    print("DONE")
                    break
            if not done:
                inference_bar.close()
                print("Did not finish inference episode")
                

        elif self.mode == "baseline":
            self.algorithm.main_loop()
            if self.backend.maximize:
                self.history = [(cfg, -score) for cfg, score in self.algorithm.data]
            else:
                self.history = list(self.algorithm.data)
        return min(self.history, key=lambda x: x[-1]) if not self.backend.maximize else max(self.history, key=lambda x: x[-1])

    def return_history(self):
        """Возвращает историю гиперпараметров работы инференса
        
        Returns:
            История гиперпараметров 
        """
        return self.history