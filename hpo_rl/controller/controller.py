from hpo_rl.controller.parallelization import parallelization
import numpy as np

class controller():
    def __init__(self, device, mode, backend, algorithm, env = None, save = None, load = None):
        """ 
        mode: "baseline"/"RL"
        backend: {class: backend_style_class, params: function_or_real_params}
        algorithm: {class: algorithm_style_class, params: alg_params}
        env: {class: env_style_class, params: env_params}
        save: save_location
        load: load_location
        """
        self.mode = mode
        self.parallelization = parallelization(self.mode)
        self.device = device

        backend_class = backend.get("class")
        print(backend.get("params"))
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
                self.algorithm = algorithm_class(device = self.device, env = self.env, **(algorithm.get("params")))
            else:
                algorithm_class = algorithm.get("class")
                self.algorithm = algorithm_class.load(load_loc, device = device)

        elif self.mode == "baseline":
            algorithm_class = algorithm.get("class")
            self.algorithm = algorithm_class(objective_func = self.backend.evaluate, **(algorithm.get("params"))) #интегрировать backend в baseline'ы  

    def train(self):
        if self.mode == "RL":
            # print(type(self.total_timesteps))
            self.algorithm.learn(total_timesteps = self.total_timesteps)
            if self.save_bool:
                self.algorithm.save(self.save_loc)
    
    def inference(self):
        # Добавить сохранение лучшей модели
        if self.mode == "RL":
            self.history = []
            # print(self.history)
            env = self.algorithm.env
            obs = env.reset()
            for _ in range(self.inference_timesteps):
                action, _states = self.algorithm.predict(obs, deterministic=True)
                obs, _rewards, done, infos = env.step(action)
                info = infos[0]
                config = info.get("current_config")
                metric = info.get("current_metric")
                if done:
                    print("DONE")
                    break
                self.history.append([config, metric])

        elif self.mode == "baseline":
            self.algorithm.main_loop()
            self.history = self.algorithm.data
        # print(self.history)
        return min(self.history, key=lambda x: x[-1])

    def return_history(self):
        return self.history