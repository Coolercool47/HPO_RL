import numpy as np
from scipy.stats import norm
from .TPE import TPE
from tqdm.auto import tqdm

class BOHB:
    """Класс, реализурующий алгоритм BOHB.
        
        Статья:  `BOHB: Robust and Efficient Hyperparameter Optimization at Scale <https://arxiv.org/pdf/1807.01774>`_

        Args:
            R: максимальное количество ресурсов, выделяемое под единственную конфигурацию
            nu: контролирует пропорцию отбрасываемых конгфигураций
            objective_func: целевая функция, возвращающая оценку
            dict_to_optimize: конфигурация допустимых гиперпараметров
            min_points_in_model: минимальное количество точек для запуска алгоритма :class:`TPE`
            top_n_percent: перцентиль данных, попадающих в "хорошую" выборку :class:`TPE`
            num_samples: сколько сделать сэмплирований в логике :class:`TPE`
            random_fraction: вероятность запуска случайного поиска, вместо :class:`TPE`

        Attributes:
            R: ресурсы под конфигурацию
            nu: пропорция отбрасываемых конгфигураций
            objective_func: целевая функция
            dict_to_optimize: конфигурация допустимых гиперпараметров
            min_points_in_model: минимальное количество точек для :class:`TPE`
            top_n_percent: перцентиль
            num_samples: количество сэмплов для :class:`TPE`
            random_fraction: вероятность случайного поиска
            s_max: количество итераций для каждого из бюджетов
            B: бюджет, контролирующий выделение ресурсов для модели
            data: датасет с посчитанными оценками

        Пример::
           
            def objective_function(params): 
                score = ...
                return score

            dict_config = {
                "x0": {type: float, min: 0.0, max:1.0} , 
                "x1": {type: categorical, values: ["a", "b"]}
            }

            bohb = BOHB(R=9, nu=3, objective_func=objective_function, dict_to_optimize=dict_config)
            best_config = bohb.main_loop()
            
        """
    def __init__(self, R, nu, objective_func, dict_to_optimize, 
                 min_points_in_model=None, 
                 top_n_percent=0.15, 
                 num_samples=64, 
                 random_fraction=0.3):
        """
        Инициализирует BOHB

        Args:
            R: максимальное количество ресурсов, выделяемое под единственную конфигурацию
            nu: контролирует пропорцию отбрасываемых конгфигураций
            objective_func: целевая функция, возвращающая оценку, согласно которой будут отбрасываться значения
            dict_to_optimize: конфигурация допустимых гиперпараметров
            min_points_in_model: минимальное количество точек для запуска алгоритма :class:`TPE`
            top_n_percent: перцентиль данных, попадающих в "хорошую" выборку :class:`TPE`
            num_samples: сколько сделать сэмплирований в логике :class:`TPE`
            random_fraction: вероятность запуска случайного поиска, вместо :class:`TPE`
        
        """
        self.R = R
        self.nu = nu
        self.objective_func = objective_func
        self.dict_to_optimize = dict_to_optimize
        
        self.s_max = int(np.floor(np.log(self.R)/np.log(self.nu)))
        self.B = (self.s_max+1)*R
        
        self.data = [] 
        
        dim = len(dict_to_optimize)
        self.min_points_in_model = min_points_in_model if min_points_in_model else dim + 1
        
        self.top_n_percent = top_n_percent
        self.num_samples = num_samples
        self.random_fraction = random_fraction

    def reset(self):
        """Сбрасывает состояние алгоритма для повторного запуска (run_n_experiments)."""
        self.data = []

    def main_loop(self):
        """Исполняет логику алгоритма BOHB с учетом введленных параметров

        Returns: 
            наилучшая найденная конфигурация гиперпараметров
        """
        best_overall_config = None
        best_overall_loss = np.inf

        parent_bar = tqdm(total=self.s_max, desc="Overall", position=0)

        for s in range(self.s_max, -1, -1):
            n = int(np.ceil(self.B * self.nu**s / (self.R * (s+1))))
            r = self.R / (self.nu**s)
            
            T = self.get_config(n)

            for i in range(0, s + 1):
                n_i = int(np.floor(n / self.nu**(i)))
                r_i = r * self.nu**i
                
                L = []
                for t in T:
                    loss = self.objective_func(t)
                    L.append(loss)
                    
                    self.data.append((t, loss, r_i))
                    
                    if loss < best_overall_loss:
                        best_overall_loss = loss
                        best_overall_config = t

                if i < s:
                    params_with_loss = list(zip(T, L))
                    T = self.top_k(params_with_loss, int(np.floor(n_i / self.nu)))
            parent_bar.update(1)
        parent_bar.close()
        return best_overall_config

    def get_config(self, n):
        """
        Функция возвращающее определенное количество конфигураций гиперпараметров

        Args: 
            n: количество возвращемых конфигураций гиперпароаметров

        Returns: 
            Конфигурации гиперпараметров

        """
        
        configs = []
        for _ in range(n):
            if np.random.rand() < self.random_fraction:
                configs.append(self.sample_random_config())
                continue

            budget_groups = {}
            for row in self.data:
                b = row[2]
                if b not in budget_groups: budget_groups[b] = []
                budget_groups[b].append(row)
            
            sorted_budgets = sorted(budget_groups.keys(), reverse=True)
            valid_budget_data = None
            
            for b in sorted_budgets:
                if len(budget_groups[b]) >= self.min_points_in_model + 2:
                    valid_budget_data = budget_groups[b]
                    break
            
            if valid_budget_data is None:
                configs.append(self.sample_random_config())
            else:
                tpe_data = [(row[0], row[1]) for row in valid_budget_data]
                
                gamma_f = lambda x: self.top_n_percent

                tpe_sampler = TPE(
                    objective_func=None,  
                    N_init=0,             
                    N_s=self.num_samples,
                    budget=0,             
                    dict_to_optimize=self.dict_to_optimize,
                    gamma_func=gamma_f
                )
                
                tpe_sampler.data = tpe_data
                
                configs.append(tpe_sampler.suggest())
        
        return configs

    def sample_random_config(self):
        """Функция возращающая единственную конфигурацию гиперпараметров из равномерного распределения

        Returns: 
            конфигурация гиперпараметров

        """
        config = {}
        for param_name, info in self.dict_to_optimize.items():
            if info["type"] == "float":
                config[param_name] = np.random.uniform(info["min"], info["max"])
            elif info["type"] == "categorical":
                config[param_name] = np.random.choice(info["values"])
        return config

    def top_k(self, params_with_loss, k):
        """Функция возвращающая k лучших значений гиперпараметров

        Args: 
            params_with_loss: выболрка гиперпараметров
            k: количество лучших возвращаемых значений

        Returns: 
            k лучших значений гиперпараметров
        
        """
        params_with_loss = sorted(params_with_loss, key=lambda x: x[1])
        return [x[0] for x in params_with_loss[:k]]