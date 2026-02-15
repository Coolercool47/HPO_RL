import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm

class TPE:
    """Класс, реализурующий алгоритм BOHB.
        
        Статьи:  
            `Algorithms for Hyper-Parameter Optimization <https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf>`_
            `Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for Better Empirical Performance <https://arxiv.org/pdf/2304.11127>`_

        Args:
            objective_func: целевая функция, возвращающая оценку
            N_init: количество изначальных конфигураций
            N_s: количество кандидатов для оптимизации функции получения
            budget: количество итераций работы алгоритма, отвечающее за количество вызовов целевой функции
            dict_to_optimize: конфигурация допустимых гиперпараметров
            separation_value: перцентиль данных, попадающих в "хорошую" и "плохую" выборку 

        Attributes:
            objective_func: целевая функция
            N_init: количество изначальных конфигураций
            N_s: количество кандидатов для функции получения
            budget: количество итераций алгоритма
            dict_to_optimize: конфигурация допустимых гиперпараметров
            gamma_func: функция, задающая перцентиль данных, которые попадают "хорошую" и "плохую" выборку
            data: датасет с посчитанными оценками

        Пример::
           
            def objective_function(params): 
                score = ...
                return score

            dict_config = {
                "x0": {type: float, min: 0.0, max:1.0} , 
                "x1": {type: categorical, values: ["a", "b"]}
            }

            tpe = TPE()
            best_config = tpe.main_loop(objective_func = objective_function, N_init = 3, N_s = 3, budget = 200, dict_to_optimize = dict_config, separation_value = 0.3)
            
        """
    def __init__(self, objective_func, N_init, N_s, budget, dict_to_optimize, separation_value):
        """
        Инициализирует TPE

        Args:
            objective_func: целевая функция, возвращающая оценку
            N_init: количество изначальных конфигураций
            N_s: количество кандидатов для оптимизации функции получения
            budget: количество итераций работы алгоритма, отвечающее за количество вызовов целевой функции
            dict_to_optimize: конфигурация допустимых гиперпараметров
            separation_value: перцентиль данных, попадающих в "хорошую" и "плохую" выборку 
        
        """
        self.objective_func = objective_func
        self.N_init = N_init
        self.N_s = N_s  
        self.budget = budget
        self.dict_to_optimize = dict_to_optimize
        self.gamma_func = lambda x: separation_value
        self.data = []

    def initialize(self):
        """Семплирует `N_init` раз гиперпараметры, согласно равномерному распределению"""
        print(f"Initializing with {self.N_init} random samples...")
        for _ in range(self.N_init):
            setup = {}
            # print(self.dict_to_optimize.items())
            for param_name, info in self.dict_to_optimize.items():
                if info["type"] == "float":
                    value = np.random.uniform(info["values"][0], info["values"][1])
                elif info["type"] == "categorical" or info["type"] == "int":
                    value = np.random.choice(info["values"])
                setup[param_name] = value
            score = self.objective_func(setup)
            self.data.append((setup, score))

    def main_loop(self):
        """Исполняет логику алгоритма TPE с учетом введленных параметров

        Returns: 
            наилучшая найденная конфигурация гиперпараметров

        """
        if len(self.data) < self.N_init:
            self.initialize()

        param_names = list(self.dict_to_optimize.keys())

        parent_bar = tqdm(total=self.budget, position=0)

        for _ in range(self.budget):
            n = len(self.data)
            gamma = self.gamma_func(n)
            
            D_l, D_g = self.D_split(n, gamma)
            
            weights_l = self.count_weights_uniform(len(D_l))
            weights_g = self.count_weights_uniform(len(D_g))
            
            b_l = self.count_bandwidths(D_l)
            b_g = self.count_bandwidths(D_g)

            candidates = []
            for _ in range(self.N_s):
                cand = {}
                for param in param_names:
                    cand[param] = self.sample_from_kde(
                        param, D_l, b_l[param], weights_l
                    )
                candidates.append(cand)

            best_sample = None
            best_score = -np.inf
            
            for cand in candidates:
                log_l = 0
                log_g = 0
                
                for param in param_names:
                    p_l = self.evaluate_kde(cand[param], param, D_l, weights_l, b_l[param])
                    p_g = self.evaluate_kde(cand[param], param, D_g, weights_g, b_g[param])
                    log_l += np.log(p_l)
                    log_g += np.log(p_g)
                
                ratio = log_l - log_g
                
                if ratio > best_score:
                    best_score = ratio
                    best_sample = cand

            result = self.objective_func(best_sample)
            self.data.append((best_sample, result))

            parent_bar.update(1)
            
            # if len(self.data) % 10 == 0:
            #     print(f"Iter {len(self.data)}/{self.budget}: Best Score = {min(d[1] for d in self.data)}")
        parent_bar.close()
        best_overall = min(self.data, key=lambda x: x[1])
        return best_overall

    def D_split(self, n, gamma):
        """Разделение датасета на "плохой" и "хороший"
        
        Args: 
            n: количество данных в self.data
            gamma: условие роазделения данных

        Returns: 
            Возвращает "плохой" и "хороший" датасет

        """
        sorted_data = sorted(self.data, key=lambda x: x[1])
        split_idx = int(np.ceil(n * gamma))
        split_idx = max(1, min(n - 1, split_idx))
        D_l = sorted_data[:split_idx]
        D_g = sorted_data[split_idx:]
        return D_l, D_g

    def count_weights_uniform(self, N_group):
        """Подсчет весов
        
        Args: 
            N_group: количество данных в датасете

        Returns: 
            1 / количество данных в датасете

        """
        if N_group == 0: return []
        return [1.0 / N_group]

    def count_bandwidths(self, D_group):
        """Подсчет ширины окна согласно правилу Скотта и magic_clipping
        
        Args: 
            D_group: количество данных в датасете

        Returns: 
            массив с шириной окна

        """
        bandwidths = {}
        param_names = list(self.dict_to_optimize.keys())
        N = len(D_group)

        magic_exponent_alpha = 2.0 

        for param in param_names:
            info = self.dict_to_optimize[param]
            values = np.array([pt[0][param] for pt in D_group])

            if info["type"] == "float":
                L, R = info["values"]
                domain_range = R - L
                
                if N <= 1:
                    sigma = domain_range * 0.2
                else:
                    sigma = np.std(values)
                
                b_scott = 1.06 * sigma * (max(N, 1) ** (-0.2))

                b_magic = domain_range /  (N ** (magic_exponent_alpha))
                
                min_bandwidth_factor = 0.05 
                b_min = max(domain_range * min_bandwidth_factor, b_magic)

                b_final = max(b_scott,b_min)
                bandwidths[param] = b_final
            
            elif info["type"] == "categorical" or info["type"] == "int":
                bandwidths[param] = 0.2

        return bandwidths

    def evaluate_kde(self, x, param_name, D_group, weights, bandwidth):
        """Подсчет ядер. Для категориальных данных используется ядро Этчисона-Эйткена, для нерерывных - усеченное нормальное распределение
        
        Args: 
            x: значение, для которого ведется подсчет ядра
            param_name: название праматра, для которого составляется ядро
            D_group: датасет, по которому строится ядро
            weights: веса при ядрах и априорной верятности
            bandwidth: ширина окна ядер

        Returns: 
            правдоподобие каждого ядра

        """
        info = self.dict_to_optimize[param_name]
        w_prior = weights[0]
        w_obs = weights[0]
        
        if info["type"] == "categorical" or info["type"] == "int":
            num_categories = len(info["values"])
            prior_prob = 1.0 / num_categories
            probability = w_prior * prior_prob
            
            if len(D_group) > 0:
                h = bandwidth
                obs_counts = {val: 0 for val in info["values"]}
                for pt in D_group:
                    obs_counts[pt[0][param_name]] += 1
                
                likelihood_sum = 0
                for cat in info["values"]:
                    count = obs_counts[cat]
                    if count == 0: continue
                    
                    if x == cat:
                        prob_contribution = (1.0 - h)
                    else:
                        prob_contribution = h / (num_categories - 1)
                    
                    likelihood_sum += (count / len(D_group)) * prob_contribution
                
                probability += w_obs * likelihood_sum
                
            return probability

        elif info["type"] == "float":
            domain_size = info["values"][1] - info["values"][0]
            prior_prob = 1.0 / domain_size
            probability = w_prior * prior_prob

            observations = [pt[0][param_name] for pt in D_group]
            
            if len(observations) > 0:
                obs_array = np.array(observations)
                pdf_vals = norm.pdf(x, loc=obs_array, scale=bandwidth)
                
                L, R = info["values"]
                Z = norm.cdf(R, loc=obs_array, scale=bandwidth) - norm.cdf(L, loc=obs_array, scale=bandwidth)
                
                k_vals = pdf_vals / Z
                probability += w_obs * np.mean(k_vals)

            return probability

    def sample_from_kde(self, param_name, D_group, bandwidth, weights):
        """Семплирование из ядер. Для категориальных данных используется равномерное распределение, для нерерывных - усеченное нормальное распределение
        
        Args: 
            param_name: название праматра, для которого составляется ядро
            D_group: датасет, по которому строится ядро
            bandwidth: ширина окна ядер
            weights: веса при ядрах и априорной верятности

        Returns: 
            сэмпл данных

        """
        info = self.dict_to_optimize[param_name]
        
        if info["type"] == "categorical" or info["type"]=="int":
            w_prior = weights[0]
            if np.random.rand() < w_prior or len(D_group) == 0:
                return np.random.choice(info["values"])
            
            observations = [pt[0][param_name] for pt in D_group]
            center_val = np.random.choice(observations)
            
            h = bandwidth
            if np.random.rand() < (1.0 - h):
                return center_val
            else:
                others = [v for v in info["values"] if v != center_val]
                return np.random.choice(others)

        elif info["type"] == "float":
            L, R = info["values"]
            w_prior = 0.1
            if np.random.rand() < w_prior or len(D_group) == 0:
                while True:
                    sample =  np.random.normal((L+R)/2, (R-L)**2)
                    if L <= sample <= R:
                        return sample   
            
            observations = [pt[0][param_name] for pt in D_group]
            center = np.random.choice(observations)
            
            while True:
                sample = np.random.normal(center, bandwidth)
                if L <= sample <= R:
                    return sample   
        
    def suggest(self):
        """Выполняет `main_loop`, только без подсчета целевой функции
        
        Args: 
            param_name: название праматра, для которого составляется ядро
            D_group: датасет, по которому строится ядро
            bandwidth: ширина окна ядер
            weights: веса при ядрах и априорной верятности

        Returns: 
             наилучшая найденная конфигурация гиперпараметров

        """
        param_names = list(self.dict_to_optimize.keys())
        n = len(self.data)
        
        gamma = self.gamma_func(n)
        D_l, D_g = self.D_split(n, gamma)
        
        weights_l = self.count_weights_uniform(len(D_l))
        weights_g = self.count_weights_uniform(len(D_g))
        
        b_l = self.count_bandwidths(D_l)
        b_g = self.count_bandwidths(D_g)

        candidates = []
        for _ in range(self.N_s):
            cand = {}
            for param in param_names:
                cand[param] = self.sample_from_kde(
                    param, D_l, b_l[param]
                )
            candidates.append(cand)

        best_sample = None
        best_score = -np.inf
        
        for cand in candidates:
            log_l = 0
            log_g = 0
            
            for param in param_names:
                p_l = self.evaluate_kde(cand[param], param, D_l, weights_l, b_l[param])
                
                p_g = self.evaluate_kde(cand[param], param, D_g, weights_g, b_g[param])
                
                log_l += np.log(p_l)
                log_g += np.log(p_g)
            
            ratio = log_l - log_g
            
            if ratio > best_score:
                best_score = ratio
                best_sample = cand

        return best_sample