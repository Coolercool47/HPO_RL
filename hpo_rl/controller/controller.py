from parallelization import parallelization

class controller():
    def __init__(self, mode, backend, algorithm, env = None, save = None, load = None):
        """ 
        mode: "baseline"/"RL"
        backend: {class: backend_style_class, params: function_or_real_params}
        algorithm: {class: algorithm_style_class, params: alg_params}
        env: {class: env_style_class, params: env_params}
        save: {save_bool: True_or_False, save_loc: save_location}
        load: {load_bool: True_or_False, load_loc: laod_location}
        """
        self.mode = mode
        self.parallelization = parallelization(self.mode)

        backend_class = backend.get("class")
        self.backend = backend_class(**(backend.get("params"))) 

        if self.mode == "RL":
            env_class = env.get("class")
            self.env = env_class(backend = self.backend, **(env.get("params")))

            self.total_timesteps = algorithm.get("params").pop("total_timesteps")
            self.inference_timesteps = algorithm.get("params").pop("inference_timesteps")

            algorithm_class = algorithm.get("class")
            self.algorithm = algorithm_class(self.env, **(algorithm.get("params")))

            self.save_bool = save.get("save_bool")
            self.save_loc = save.get("save_loc")

            self.load_bool = load.get("load_bool")
            self.load_loc = load.get("load_loc")

        elif self.mode == "baseline":
            algorithm_class = algorithm.get("class")
            self.algorithm = algorithm_class(objective_func = self.backend.evaluate, **(algorithm.get("params"))) #интегрировать backend в baseline'ы
          

    def train(self):
        if self.mode == "RL":
            self.algorithm.train(total_timesteps = self.total_timesteps)
            if self.save_bool:
                self.algorithm.save(self.save_loc)
    
    def inference(self):
        if self.mode == "RL":
            self.history = []

            env = self.algorithm.env
            obs = env.reset()
            for _ in range(self.inference_timesteps):
                action, _states = self.algorithm.predict(obs)
                obs, _rewards, _dones, infos = env.step(action)
                info = infos[0]
                config = info.get("current_config")
                metric = info.get("current_metric")

                self.history.append([config[:], metric])

        elif self.mode == "baseline":
            self.algorithm.main_loop()
            self.history = self.algorithm.data

    def return_history(self):
        return self.history