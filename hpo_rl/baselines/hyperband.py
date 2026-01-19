import numpy as np
from tqdm.auto import tqdm

class hyperband:
    def __init__(self, R, nu, objective_func, dict_to_optimize):
        self.R = R
        self.nu = nu
        self.objective_func = objective_func
        self.dict_to_optimize = dict_to_optimize
        self.s_max = int(np.floor(np.log(self.R)/np.log(self.nu)))
        self.B = (self.s_max+1)*R
        self.N_min = len(dict_to_optimize)+1

    def main_loop(self):
        parent_bar = tqdm(total=self.s_max, position=0)
        for s in range(self.s_max, -1, -1):
            n = int(np.ceil(self.B*self.nu**s/(self.R*(s+1))))
            r = self.R/(self.nu**s)
            
            T = self.get_config(n)
            
            for i in range(0, s + 1):
                n_i = int(np.floor(n/self.nu**(i)))
                r_i = r*self.nu**i
                
                L = []
                for t in T:
                    loss = self.objective_func(t)
                    L.append(loss)
                
                if i < s:
                    params_with_loss = list(zip(T, L))
                    k = int(np.floor(n_i / self.nu))
                    T = self.top_k(params_with_loss, max(1, k))
            parent_bar.update(1)
        parent_bar.close()
        best_idx = np.argmin([self.objective_func(t, self.R, self.dict_to_optimize) for t in T])
        return T[best_idx]
     
    def get_config(self, n):
        configs = []
        for _ in range(n):
            config = {} 
            
            for param_name, param_info in self.dict_to_optimize.items():
                if param_info["type"] == "float":
                    config[param_name] = np.random.uniform(
                        param_info["min"], param_info["max"]
                    )
                elif param_info["type"] == "categorical":
                    config[param_name] = np.random.choice(param_info["values"])
            
            configs.append(config)

        return configs
    
    def top_k(self, params_with_loss, k):
        params_with_loss = sorted(params_with_loss, key=lambda x: x[1])
        return [x[0] for x in params_with_loss[:k]]