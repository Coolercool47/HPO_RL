import numpy as np
import copy
from tqdm.auto import tqdm

class CMA_ES:
    """Класс, реализурующий алгоритм (mu/mu_w, lambda)-CMA-ES.

        Статья: `The CMA Evolution Strategy: A Tutorial <https://arxiv.org/pdf/1604.00772>`_

        Args:
            objective_func: целевая функция (минимизация)
            N_pop: размер популяции (lambda); если None — 4 + 3*ln(N)
            budget: количество вызовов целевой функции
            dict_to_optimize: конфигурация параметров (float, int, categorical)
            initial_step_size: начальное стандартное отклонение (sigma)

        Attributes:
            objective_func: целевая функция
            budget: бюджет вызовов
            dict_to_optimize: пространство гиперпараметров
            data: история (config, score)
            population: текущая популяция

        Пример::

            def objective_function(params):
                return sum(v ** 2 for v in params.values())

            dict_config = {"x0": {"type": "float", "values": [0.0, 1.0]}}

            cma = CMA_ES(objective_func=objective_function, N_pop=20, budget=200,
                         dict_to_optimize=dict_config)
            best_config, best_score = cma.main_loop()
        """

    def __init__(self, objective_func, N_pop, budget, dict_to_optimize, initial_step_size=0.5):
        """Инициализирует CMA-ES.

        Args:
            objective_func: целевая функция (минимизация)
            N_pop: размер популяции; если None — вычисляется автоматически
            budget: количество вызовов целевой функции
            dict_to_optimize: конфигурация допустимых гиперпараметров
            initial_step_size: начальное стандартное отклонение (sigma)
        """
        self.objective_func = objective_func
        self.budget = budget
        self.dict_to_optimize = dict_to_optimize
        self.initial_step_size = initial_step_size
        
        self.data = []
        self.population = []
        
        self.param_names = []
        self.bounds = [] 
        self.types = []  
        self.cat_maps = [] 
        
        for key, info in self.dict_to_optimize.items():
            self.param_names.append(key)
            self.types.append(info['type'])
            
            if info['type'] == 'float':
                self.bounds.append(info['values'])
                self.cat_maps.append(None)
            elif info['type'] == 'int':
                self.bounds.append(info['values'])  
                self.cat_maps.append(None)
            elif info['type'] == 'categorical':
                values = info['values']
                self.bounds.append([0, len(values) - 1])
                self.cat_maps.append(values)
            else:
                raise ValueError(f"Unknown type: {info['type']}")
                
        self.bounds = np.array(self.bounds, dtype=float)
        self.N = len(self.param_names)
        self.gen = 0

        if N_pop is None:
            self.lambd = 4 + int(3 * np.log(self.N))
        else:
            self.lambd = N_pop
            
        self.mu = self.lambd // 2
        
        weights_raw = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights_raw / np.sum(weights_raw) 
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)    

        self.cc = (4 + self.mu_eff / self.N) / (self.N + 4 + 2 * self.mu_eff / self.N)
        self.c_sigma = (self.mu_eff + 2) / (self.N + self.mu_eff + 5)
        self.c1 = 2 / ((self.N + 1.3)**2 + self.mu_eff)
        self.c_mu = min(1 - self.c1, 
                        2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.N + 2)**2 + self.mu_eff))
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1)/(self.N + 1)) - 1) + self.c_sigma

        self.chiN = np.sqrt(self.N) * (1 - 1/(4*self.N) + 1/(21 * self.N**2))

    def reset(self):
        """Сброс состояния перед запуском."""
        self.data = []
        self.population = []
        self.initialize()

    def initialize(self):
        """Инициализация внутренних переменных состояния CMA-ES (m, sigma, C, paths)."""
        print(f"Initializing CMA-ES with dimension N={self.N}, lambda={self.lambd}, mu={self.mu}...")
        
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        self.xmean = lower + (upper - lower) * 0.5
        
        domain_range = np.mean(upper - lower)
        self.sigma = self.initial_step_size * domain_range

        self.pc = np.zeros(self.N)
        self.ps = np.zeros(self.N)
        
        self.B = np.eye(self.N)
        self.D = np.ones(self.N)
        self.C = np.eye(self.N) 
        
        self.gen = 0

    def _vector_to_config(self, vector):
        """Преобразует вещественный вектор CMA-ES в словарь параметров (с округлением)."""
        vector_clipped = np.clip(vector, self.bounds[:, 0], self.bounds[:, 1])
        
        config = {}
        for i, val in enumerate(vector_clipped):
            name = self.param_names[i]
            p_type = self.types[i]
            
            if p_type == 'float':
                config[name] = float(val)
            elif p_type == 'int':
                config[name] = int(np.round(val))
            elif p_type == 'categorical':
                idx = int(np.round(val))
                idx = max(0, min(len(self.cat_maps[i]) - 1, idx))
                config[name] = self.cat_maps[i][idx]
        return config

    def main_loop(self):
        """Исполняет основной цикл CMA-ES.

        Returns:
            кортеж (наилучшая конфигурация, наилучшая оценка)
        """
        if not self.population and self.gen == 0:
            self.initialize()

        pbar = tqdm(total=self.budget, position=0)
        pbar.update(len(self.data)) 

        while len(self.data) < self.budget:
            self.gen += 1
            
            offspring_params = [] 
            configs = []       
            penalties = []       
            
            alpha = 1.0
            
            for k in range(self.lambd):
                z_k = np.random.randn(self.N)
                y_k = self.B @ (self.D * z_k)
                x_k = self.xmean + self.sigma * y_k
                
                x_repaired = np.clip(x_k, self.bounds[:, 0], self.bounds[:, 1])
                penalty = alpha * np.sum((x_k - x_repaired)**2)
                
                offspring_params.append((x_k, y_k, z_k))
                configs.append(self._vector_to_config(x_repaired))
                penalties.append(penalty)

            fitness_values = []
            for i, config in enumerate(configs):
                if len(self.data) >= self.budget:
                    break
                score = self.objective_func(config)
                self.data.append((config, score))
                
                fitness = score + penalties[i]
                fitness_values.append(fitness)
                pbar.update(1)
            
            if len(self.data) >= self.budget:
                break

            sorted_indices = np.argsort(fitness_values)
            
            best_indices = sorted_indices[:self.mu]
            
            y_w = np.zeros(self.N)
            for i, idx in enumerate(best_indices):
                _, y_k, _ = offspring_params[idx]
                y_w += self.weights[i] * y_k
            
            self.xmean = self.xmean + self.sigma * y_w

            inv_D = 1.0 / self.D
            C_inv_half_y_w = self.B @ (inv_D * (self.B.T @ y_w))
            
            self.ps = (1 - self.c_sigma) * self.ps + \
                      np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * C_inv_half_y_w
            
            norm_ps = np.linalg.norm(self.ps)
            
            self.sigma = self.sigma * np.exp((self.c_sigma / self.d_sigma) * (norm_ps / self.chiN - 1))

            hs_cond = (norm_ps / np.sqrt(1 - (1 - self.c_sigma)**(2*self.gen))) < (1.4 + 2/(self.N+1)) * self.chiN
            h_sigma = 1.0 if hs_cond else 0.0
            
            d_hs = (1 - h_sigma) * self.cc * (2 - self.cc)
            
            self.pc = (1 - self.cc) * self.pc + \
                      h_sigma * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y_w
            
            rank_mu_update = np.zeros((self.N, self.N))
            for i, idx in enumerate(best_indices):
                _, y_k, _ = offspring_params[idx]
                rank_mu_update += self.weights[i] * np.outer(y_k, y_k)

            self.C = (1 + self.c1 * d_hs - self.c1 - self.c_mu) * self.C + \
                     self.c1 * np.outer(self.pc, self.pc) + \
                     self.c_mu * rank_mu_update
                     
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            
            vals, vecs = np.linalg.eigh(self.C)
            
            vals = np.maximum(vals, 1e-14)
            
            self.D = np.sqrt(vals)
            self.B = vecs

        pbar.close()
        best_overall = min(self.data, key=lambda x: x[1])
        return best_overall