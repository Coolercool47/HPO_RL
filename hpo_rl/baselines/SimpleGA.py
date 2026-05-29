import numpy as np
import copy
from tqdm.auto import tqdm

class SimpleGA:
    """Класс, реализурующий простой генетический алгоритм (GA) для HPO.
        
        Работает с непрерывными и категориальными параметрами.
        Использует турнирную селекцию, равномерное скрещивание и мутацию.

        Args:
            objective_func: целевая функция, возвращающая оценку (минимизация)
            N_pop: размер популяции
            budget: количество итераций работы алгоритма (вызовов целевой функции)
            dict_to_optimize: конфигурация допустимых гиперпараметров
            mutation_prob: вероятность мутации отдельного гена
            crossover_prob: вероятность скрещивания двух родителей
            tournament_size: размер турнира для селекции
            elitism: сохранять ли лучшую особь в новое поколение

        Attributes:
            data: история всех оценок (список кортежей (config, score))
            population: текущая популяция (список config)

        Пример::

            def objective_function(params):
                return sum(v ** 2 for v in params.values())

            dict_config = {"x0": {"type": "float", "values": [0.0, 1.0]}}

            ga = SimpleGA(objective_func=objective_function, N_pop=20, budget=200,
                          dict_to_optimize=dict_config)
            best_config, best_score = ga.main_loop()
    """

    def __init__(self, objective_func, N_pop, budget, dict_to_optimize,
                 mutation_prob=0.1, crossover_prob=0.8, tournament_size=3, elitism=True):
        """Инициализирует генетический алгоритм.

        Args:
            objective_func: целевая функция (минимизация)
            N_pop: размер популяции
            budget: количество вызовов целевой функции
            dict_to_optimize: конфигурация допустимых гиперпараметров
            mutation_prob: вероятность мутации гена
            crossover_prob: вероятность скрещивания
            tournament_size: размер турнира
            elitism: сохранять ли лучшую особь
        """
        self.objective_func = objective_func
        self.N_pop = N_pop
        self.budget = budget
        self.dict_to_optimize = dict_to_optimize
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.data = [] 
        self.population = []

    def reset(self):
        """Сбрасывает состояние алгоритма для повторного запуска (run_n_experiments)."""
        self.data = []
        self.population = []

    def initialize(self):
        """Создает начальную популяцию случайным образом"""
        print(f"Initializing population with {self.N_pop} individuals...")
        self.population = []
        for _ in range(self.N_pop):
            individual = self._random_individual()
            self.population.append(individual)

    def _random_individual(self):
        """Генерирует одну случайную конфигурацию.

        Returns:
            словарь конфигурации гиперпараметров
        """
        setup = {}
        for param_name, info in self.dict_to_optimize.items():
            p_type = info["type"]
            values = info["values"]

            if p_type == "float":
                L, R = values
                value = np.random.uniform(L, R)

            elif p_type == "int":
                L, R = values
                value = np.random.randint(L, R + 1)

            elif p_type == "categorical":
                value = np.random.choice(values)

            else:
                raise ValueError(f"Unknown parameter type '{p_type}' for '{param_name}'")

            setup[param_name] = value
        return setup

    def main_loop(self):
        """Исполняет логику генетического алгоритма.
        
        Returns:
            кортеж (наилучшая конфигурация, наилучшая оценка)
        """
        if not self.population:
            self.initialize()

        scores = []
        parent_bar = tqdm(total=self.budget, position=0)
        
        if len(self.data) == 0:
            for ind in self.population:
                score = self.objective_func(ind)
                self.data.append((ind, score))
                scores.append(score)
                parent_bar.update(1)
                if len(self.data) >= self.budget:
                    break
        else:
            scores = [x[1] for x in self.data[-len(self.population):]]

        while len(self.data) < self.budget:
            new_population = []
            
            if self.elitism:
                best_idx = np.argmin(scores)
                new_population.append(copy.deepcopy(self.population[best_idx]))

            while len(new_population) < self.N_pop:
                parent1 = self.tournament_selection(scores)
                parent2 = self.tournament_selection(scores)

                if np.random.rand() < self.crossover_prob:
                    child = self.crossover(parent1, parent2)
                else:
                    child = copy.deepcopy(parent1)

                child = self.mutate(child)
                new_population.append(child)

            self.population = new_population
            scores = []

            for ind in self.population:
                if len(self.data) >= self.budget:
                    break
                
                score = self.objective_func(ind)
                self.data.append((ind, score))
                scores.append(score)
                parent_bar.update(1)

        parent_bar.close()
        best_overall = min(self.data, key=lambda x: x[1])
        return best_overall

    def tournament_selection(self, scores):
        """Выбирает родителя методом турнира.

        Returns:
            конфигурация выбранного родителя
        """
        indices = np.random.choice(len(self.population), size=self.tournament_size, replace=False)
        best_idx = indices[0]
        best_score = scores[best_idx]
        
        for idx in indices[1:]:
            if scores[idx] < best_score:
                best_score = scores[idx]
                best_idx = idx
        return self.population[best_idx]

    def crossover(self, p1, p2):
        """Равномерное скрещивание."""
        child = {}
        for key in self.dict_to_optimize.keys():
            if np.random.rand() < 0.5:
                child[key] = p1[key]
            else:
                child[key] = p2[key]
        return child

    def mutate(self, individual):
        """Мутация особи.

        Returns:
            мутированная копия особи
        """
        mutated_ind = copy.deepcopy(individual)
        for key, info in self.dict_to_optimize.items():
            if np.random.rand() < self.mutation_prob:
                p_type = info["type"]
                values = info["values"]

                if p_type == "float":
                    L, R = values
                    domain_range = R - L
                    sigma = domain_range * 0.1
                    delta = np.random.normal(0, sigma)
                    val = mutated_ind[key] + delta
                    mutated_ind[key] = np.clip(val, L, R)

                elif p_type == "int":
                    L, R = values
                    domain_range = max(1, R - L)
                    sigma = max(1.0, domain_range * 0.1)
                    delta = int(np.round(np.random.normal(0, sigma)))
                    val = int(mutated_ind[key]) + delta
                    mutated_ind[key] = int(np.clip(val, L, R))

                elif p_type == "categorical":
                    vals = values
                    if len(vals) > 1:
                        current = mutated_ind[key]
                        choices = [v for v in vals if v != current]
                        mutated_ind[key] = np.random.choice(choices)
                    else:
                        mutated_ind[key] = vals[0]

                else:
                    raise ValueError(f"Unknown parameter type '{p_type}' for '{key}'")

        return mutated_ind