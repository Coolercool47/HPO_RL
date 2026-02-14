import numpy as np
import copy
from tqdm.auto import tqdm

class CMA_ES:
    """Класс, реализурующий алгоритм (mu/mu_w, lambda)-CMA-ES.
    
    Алгоритм основан на статье "The CMA Evolution Strategy: A Tutorial".
    Адаптирует ковариационную матрицу распределения для эффективного поиска в непрерывном пространстве.
    
    Args:
        objective_func: целевая функция (минимизация)
        N_pop: размер популяции (lambda). Если None, вычисляется по формуле 4 + 3*ln(N).
        budget: количество вызовов целевой функции.
        dict_to_optimize: конфигурация параметров (float, int, categorical).
        initial_step_size: начальное стандартное отклонение (sigma).
    """

    def __init__(self, objective_func, N_pop, budget, dict_to_optimize, initial_step_size=0.5):
        self.objective_func = objective_func
        self.budget = budget
        self.dict_to_optimize = dict_to_optimize
        self.initial_step_size = initial_step_size
        
        # История и текущее состояние
        self.data = []
        self.population = []
        
        # 1. Преобразование пространства параметров в векторную форму
        self.param_names = []
        self.bounds = [] # [[low, high], ...]
        self.types = []  # 'float', 'int', 'categorical'
        self.cat_maps = [] # Для categorical храним списки значений
        
        for key, info in self.dict_to_optimize.items():
            self.param_names.append(key)
            self.types.append(info['type'])
            
            if info['type'] == 'float':
                self.bounds.append(info['values'])
                self.cat_maps.append(None)
            elif info['type'] == 'int':
                self.bounds.append(info['values']) # [min, max]
                self.cat_maps.append(None)
            elif info['type'] == 'categorical':
                # Категориальные мапим на индексы [0, len-1]
                values = info['values']
                self.bounds.append([0, len(values) - 1])
                self.cat_maps.append(values)
            else:
                raise ValueError(f"Unknown type: {info['type']}")
                
        self.bounds = np.array(self.bounds, dtype=float)
        self.N = len(self.param_names) # Размерность задачи (N)
        self.gen = 0

        # 2. Настройка параметров CMA-ES (на основе Table 1 из статьи)
        # Если N_pop не задан пользователем, берем дефолтный: 4 + 3*ln(N)
        if N_pop is None:
            self.lambd = 4 + int(3 * np.log(self.N))
        else:
            self.lambd = N_pop
            
        # Количество родителей (mu) - обычно половина популяции
        self.mu = self.lambd // 2
        
        # Веса для рекомбинации (weights): w_i propto ln(mu+1/2) - ln(i)
        weights_raw = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights_raw / np.sum(weights_raw) # Нормализация, сумма = 1
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)    # Variance effective selection mass

        # Параметры адаптации (Step-size control & Covariance adaptation)
        # Equation 56
        self.cc = (4 + self.mu_eff / self.N) / (self.N + 4 + 2 * self.mu_eff / self.N)
        # Equation 55
        self.c_sigma = (self.mu_eff + 2) / (self.N + self.mu_eff + 5)
        # Equation 57
        self.c1 = 2 / ((self.N + 1.3)**2 + self.mu_eff)
        # Equation 58 (alpha_cov = 2)
        self.c_mu = min(1 - self.c1, 
                        2 * (self.mu_eff - 2 + 1/self.mu_eff) / ((self.N + 2)**2 + self.mu_eff))
        # Equation 55 (damping)
        self.d_sigma = 1 + 2 * max(0, np.sqrt((self.mu_eff - 1)/(self.N + 1)) - 1) + self.c_sigma

        # Ожидание ||N(0,I)|| (Approximation)
        self.chiN = np.sqrt(self.N) * (1 - 1/(4*self.N) + 1/(21 * self.N**2))

    def reset(self):
        """Сброс состояния перед запуском."""
        self.data = []
        self.population = []
        self.initialize()

    def initialize(self):
        """Инициализация внутренних переменных состояния CMA-ES (m, sigma, C, paths)."""
        print(f"Initializing CMA-ES with dimension N={self.N}, lambda={self.lambd}, mu={self.mu}...")
        
        # Начальная точка (mean) - центр диапазона
        lower = self.bounds[:, 0]
        upper = self.bounds[:, 1]
        self.xmean = lower + (upper - lower) * 0.5
        
        # Начальный шаг (sigma)
        # Можно масштабировать, но для простоты берем скаляр * средний разброс
        domain_range = np.mean(upper - lower)
        self.sigma = self.initial_step_size * domain_range

        # Пути эволюции (Evolution paths)
        self.pc = np.zeros(self.N)
        self.ps = np.zeros(self.N)
        
        # Ковариационная матрица и её разложение
        self.B = np.eye(self.N)
        self.D = np.ones(self.N)
        self.C = np.eye(self.N) # B * D^2 * B.T
        
        self.gen = 0

    def _vector_to_config(self, vector):
        """Преобразует вещественный вектор CMA-ES в словарь параметров (с округлением)."""
        # Сначала ограничиваем значения границами (Box constraints via clipping)
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
                # Защита от выхода за границы индекса (хоть мы и делали clip)
                idx = max(0, min(len(self.cat_maps[i]) - 1, idx))
                config[name] = self.cat_maps[i][idx]
        return config

    def main_loop(self):
        """Основной цикл CMA-ES."""
        if not self.population and self.gen == 0:
            self.initialize()

        # Если перезапуск (data не пуста), нужно синхронизироваться, 
        # но CMA-ES сложнее восстановить из истории, чем SimpleGA.
        # Поэтому предполагаем, что reset() вызывается перед новым запуском.
        
        # Прогресс бар
        pbar = tqdm(total=self.budget, position=0)
        pbar.update(len(self.data)) # Если уже есть данные

        while len(self.data) < self.budget:
            self.gen += 1
            
            # 1. Sampling (Eq. 38-40)
            # Генерируем lambda потомков
            offspring_params = [] # список векторов z, y, x
            configs = []          # список словарей для objective_func
            
            for k in range(self.lambd):
                # z ~ N(0, I)
                z_k = np.random.randn(self.N)
                # y ~ N(0, C)  -> y = B * D * z
                y_k = self.B @ (self.D * z_k)
                # x ~ N(m, sigma^2 C) -> x = m + sigma * y
                x_k = self.xmean + self.sigma * y_k
                
                offspring_params.append((x_k, y_k, z_k))
                configs.append(self._vector_to_config(x_k))

            # 2. Evaluation
            fitness_values = []
            for config in configs:
                if len(self.data) >= self.budget:
                    break
                score = self.objective_func(config)
                self.data.append((config, score))
                fitness_values.append(score)
                pbar.update(1)
            
            if len(self.data) >= self.budget:
                break

            # 3. Selection and Recombination (Eq. 41-42)
            # Сортируем потомков по фитнесу (минимизация)
            sorted_indices = np.argsort(fitness_values)
            
            # Выбираем топ mu
            best_indices = sorted_indices[:self.mu]
            
            # Среднее взвешенное векторов смещения выбранных потомков
            # <y>_w = sum(w_i * y_i:lambda)
            y_w = np.zeros(self.N)
            for i, idx in enumerate(best_indices):
                _, y_k, _ = offspring_params[idx]
                y_w += self.weights[i] * y_k
            
            # Обновление среднего значения распределения
            # m <-- m + cm * sigma * <y>_w (здесь cm=1 по умолчанию)
            self.xmean = self.xmean + self.sigma * y_w

            # 4. Step-size control (Eq. 43-44)
            # C^(-1/2) = B * D^(-1) * B.T
            # Но нам нужно умножить C^(-1/2) на y_w.
            # y_w уже в координатах пространства. Перевод обратно: z_w = B.T * y_w / D (упрощенно)
            # Точнее: C^(-1/2) * y_w = B * D^(-1) * B.T * (B * D * z_w_recombined) = B * z_w_recombined
            # Для эффективности считаем через сохраненные параметры:
            # Для выбранных шагов восстановим z составляющие: z = D^-1 * B^T * y
            
            inv_D = 1.0 / self.D
            # C^-1/2 * y_w
            C_inv_half_y_w = self.B @ (inv_D * (self.B.T @ y_w))
            
            # Обновление пути эволюции для sigma (ps)
            self.ps = (1 - self.c_sigma) * self.ps + \
                      np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff) * C_inv_half_y_w
            
            norm_ps = np.linalg.norm(self.ps)
            
            # Обновление sigma
            self.sigma = self.sigma * np.exp((self.c_sigma / self.d_sigma) * (norm_ps / self.chiN - 1))

            # 5. Covariance matrix adaptation (Eq. 45-47)
            # Heaviside function for h_sigma (stall update if ps is too large)
            hs_cond = (norm_ps / np.sqrt(1 - (1 - self.c_sigma)**(2*self.gen))) < (1.4 + 2/(self.N+1)) * self.chiN
            h_sigma = 1.0 if hs_cond else 0.0
            
            d_hs = (1 - h_sigma) * self.cc * (2 - self.cc) # поправка для ранга-1, если h=0
            
            # Обновление пути эволюции для C (pc)
            self.pc = (1 - self.cc) * self.pc + \
                      h_sigma * np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * y_w
            
            # Rank-1 update component: pc * pc.T
            # Rank-mu update component: sum(w_i * y_i * y_i.T)
            
            rank_mu_update = np.zeros((self.N, self.N))
            for i, idx in enumerate(best_indices):
                _, y_k, _ = offspring_params[idx]
                # outer product y_k * y_k.T (но y_k масштабирован sigma, в формуле (47) y_i:lambda = (x-m)/sigma)
                # В коде выше y_k = (x-m)/sigma. Все верно.
                # Но для обновления C используется (y_k / sigma)? Нет.
                # В статье: y_i:lambda = (x - m_old) / sigma_old. Это наше y_k.
                rank_mu_update += self.weights[i] * np.outer(y_k, y_k)

            # Обновление матрицы C
            self.C = (1 + self.c1 * d_hs - self.c1 - self.c_mu) * self.C + \
                     self.c1 * np.outer(self.pc, self.pc) + \
                     self.c_mu * rank_mu_update
                     
            # 6. Eigendecomposition (для следующего шага)
            # C симметрична. Для стабильности делаем enforce symmetry
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            
            # Собственные числа и вектора
            vals, vecs = np.linalg.eigh(self.C)
            
            # Numerical stability: собственные числа должны быть положительными
            vals = np.maximum(vals, 1e-14)
            
            self.D = np.sqrt(vals)
            self.B = vecs
            # C = B * diag(D^2) * B.T

        pbar.close()
        # Возвращаем лучший результат из всей истории
        best_overall = min(self.data, key=lambda x: x[1])
        return best_overall