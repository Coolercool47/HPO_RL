"""
H-MCMC-FMP: Hierarchical MCMC with Factorized Mixture Proposals.

Адаптивный роевой MCMC с Марковским управлением для оптимизации
гиперпараметров глубоких нейросетей в гетерогенном пространстве
(непрерывные, целочисленные и категориальные параметры).

Архитектура:
    - SobolInitializer: квазислучайная инициализация через TMS-сеть (Соболь).
    - HMMController: локальный HMM с алгоритмом Витерби для управления состоянием цепи.
    - FactorizedProposalGenerator: факторизованное смесевое предложение q(X'|X, S_hmm).
    - MCMCChain: одна цепь Метрополиса — Гастингса с HMM-управлением.
    - GlobalOrchestrator: макро-уровневый оркестратор (pruning, cloning, reseeding).
    - HMM_MCMC: фасад, совместимый с интерфейсом baselines (main_loop / reset / data).
"""

import numpy as np
import math
import copy
from enum import IntEnum
from tqdm.auto import tqdm
from scipy.special import logsumexp


# ---------------------------------------------------------------------------
#  Скрытые состояния HMM
# ---------------------------------------------------------------------------
class HMMState(IntEnum):
    EXPLOIT = 0  # Loss стабильно падает → микро-шаги
    EXPLORE = 1  # Loss стагнирует или шумит → большие прыжки
    TRAPPED = 2  # Loss не меняется, дисперсия нулевая → ждём Оркестратора


# ---------------------------------------------------------------------------
#  SobolInitializer
# ---------------------------------------------------------------------------
class SobolInitializer:
    """Квазислучайная инициализация стартовых точек через последовательность Соболя.

    Если scipy доступна — используется `scipy.stats.qmc.Sobol`, иначе
    фоллбэк на стратифицированный случайный семплинг (Latin Hypercube).
    """

    def __init__(self, dict_to_optimize: dict):
        self.dict_to_optimize = dict_to_optimize
        self.param_names = list(dict_to_optimize.keys())
        self.dim = len(self.param_names)

    def generate(self, n_points: int, seed: int = 42) -> list[dict]:
        """Возвращает список из n_points конфигураций в [0,1]^D → mapped."""
        unit_points = self._sobol_unit_cube(n_points, seed)
        configs = []
        for i in range(n_points):
            config = self._unit_to_config(unit_points[i])
            configs.append(config)
        return configs

    # ----- private -----
    def _sobol_unit_cube(self, n: int, seed: int) -> np.ndarray:
        """Генерирует n точек в [0,1]^D."""
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=self.dim, scramble=True, seed=seed)
            # Sobol требует 2^m точек; берём ближайшую степень двойки ≥ n
            m = int(np.ceil(np.log2(max(n, 2))))
            raw = sampler.random_base2(m)   # shape (2^m, dim)
            return raw[:n]
        except ImportError:
            # Фоллбэк: Latin Hypercube (стратифицированный)
            rng = np.random.default_rng(seed)
            result = np.zeros((n, self.dim))
            for j in range(self.dim):
                perm = rng.permutation(n)
                for i in range(n):
                    result[i, j] = (perm[i] + rng.random()) / n
            return result

    def _unit_to_config(self, u: np.ndarray) -> dict:
        """Отображает вектор из [0,1]^D в словарь гиперпараметров."""
        config = {}
        for j, name in enumerate(self.param_names):
            info = self.dict_to_optimize[name]
            p_type = info["type"]
            values = info["values"]

            if p_type == "float":
                lo, hi = values[0], values[1]
                config[name] = lo + (hi - lo) * u[j]
            elif p_type == "int":
                lo, hi = int(values[0]), int(values[1])
                config[name] = int(np.round(lo + (hi - lo) * u[j]))
            elif p_type == "categorical":
                idx = int(u[j] * len(values))
                idx = min(idx, len(values) - 1)
                config[name] = values[idx]
            else:
                raise ValueError(f"Unknown type: {p_type}")
        return config


# ---------------------------------------------------------------------------
#  HMMController (Витерби + экспертная инициализация)
# ---------------------------------------------------------------------------
class HMMController:
    """Локальный HMM-контроллер для одной MCMC-цепи.

    Скрытые состояния: EXPLOIT, EXPLORE, TRAPPED.

    Эмиссионная модель — непрерывная:
        O_t = log(Loss_t / (Loss_{t-1} + ε))

        P(O_t | S_t) = (1 - λ) · N(O_t | μ_S, σ_S²) + λ · c_noise

    Экспертные параметры:
        EXPLOIT:  μ = -0.01, σ = 0.15  (плавное улучшение)
        EXPLORE:  μ =  0.00, σ = 0.30  (высокая дисперсия)
        TRAPPED:  μ =  0.00, σ = 0.005 (почти нулевая дисперсия)

    Матрица переходов задаётся экспертно (без Баума — Велча).
    Текущее состояние определяется алгоритмом Витерби по последним W наблюдениям.
    """

    N_STATES = 3

    def __init__(self, window: int = 8, obs_epsilon: float = 1e-8,
                 lambda_noise: float = 0.01):
        """
        Args:
            window: длина окна наблюдений для Витерби.
            obs_epsilon: ε в знаменателе log(Loss_t / (Loss_{t-1} + ε)).
            lambda_noise: вес робастного фонового шума в эмиссии.
        """
        self.window = window
        self.obs_epsilon = obs_epsilon
        self.lambda_noise = lambda_noise

        # --- Экспертные матрицы ---
        # Начальные вероятности π
        self.pi = np.array([0.7, 0.2, 0.1])

        # Матрица переходов A[i, j] = P(S_t = j | S_{t-1} = i)
        self.A = np.array([
            [0.95,    0.05,    0.00],   # из EXPLOIT: остаемся долго, собирая локальный минимум
            [0.15,    0.85,    0.00],   # из EXPLORE: даем цепи время (85%) на исследование
            [0.60,    0.20,    0.20],   # из TRAPPED: возвращаемся в EXPLOIT(60%) или EXPLORE(20%) после сброса
        ])

        # --- Непрерывная эмиссионная модель ---
        # Параметры гауссовых компонент для каждого состояния [EXPLOIT, EXPLORE, TRAPPED]
        self.emission_mu = np.array([-0.01, 0.00, 0.00])
        self.emission_sigma = np.array([0.15, 0.30, 0.005])

        # Фоновый шум: равномерная плотность на широком интервале [-10, 10]
        self.c_noise = 1.0 / 20.0  # = 0.05

        # Текущее состояние (до первого наблюдения)
        self.state = HMMState.EXPLORE

    def reset(self):
        self.state = HMMState.EXPLORE

    def force_state(self, new_state: HMMState):
        """Принудительно устанавливает скрытое состояние (например, из-за внешнего предохранителя)."""
        self.state = new_state

    def observe(self, obs_history: list[float]) -> HMMState:
        """Принимает историю наблюдений O_t, прогоняет Витерби.

        Args:
            obs_history: список O_0, O_1, ..., O_t за последние шаги цепи.

        Returns:
            Текущее скрытое состояние HMM.
        """
        if len(obs_history) < 2:
            return self.state

        # Берём только последние W наблюдений
        obs_seq = obs_history[-self.window:]

        self.state = HMMState(self._viterbi(obs_seq))
        return self.state

    def _log_emission(self, obs: float, state: int) -> float:
        """Логарифм эмиссионной плотности P(O_t | S_t) для непрерывного наблюдения.

        P(O | S) = (1 - λ) · N(O | μ_S, σ_S²) + λ · c_noise
        """
        mu = self.emission_mu[state]
        sigma = self.emission_sigma[state]

        # Гауссова компонента
        z = (obs - mu) / sigma
        log_gauss = -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
        gauss = np.exp(log_gauss)

        # Смесь с робастным фоновым шумом
        density = (1.0 - self.lambda_noise) * gauss + self.lambda_noise * self.c_noise
        return np.log(density + 1e-30)

    def _viterbi(self, obs_seq: list[float]) -> int:
        """Алгоритм Витерби для непрерывных наблюдений.

        Возвращает наиболее вероятное последнее состояние.
        Все вычисления в логарифмической шкале для предотвращения underflow.
        """
        T = len(obs_seq)
        if T == 0:
            return int(self.state)

        log_pi = np.log(self.pi + 1e-30)
        log_A = np.log(self.A + 1e-30)

        # δ[t, s] — лог-вероятность наиболее вероятного пути к состоянию s в момент t
        delta = np.full((T, self.N_STATES), -np.inf)
        psi = np.zeros((T, self.N_STATES), dtype=int)

        # Инициализация: δ_0(s) = log π(s) + log P(O_0 | s)
        for s in range(self.N_STATES):
            delta[0, s] = log_pi[s] + self._log_emission(obs_seq[0], s)

        # Рекурсия
        for t in range(1, T):
            for j in range(self.N_STATES):
                scores = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = scores[psi[t, j]] + self._log_emission(obs_seq[t], j)

        # Backtracking (нужно только последнее состояние)
        path = np.zeros(T, dtype=int)
        path[T - 1] = int(np.argmax(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return int(path[-1])


# ---------------------------------------------------------------------------
#  FactorizedProposalGenerator
# ---------------------------------------------------------------------------
class FactorizedProposalGenerator:
    """Генератор факторизованных смесевых предложений.

    q(X' | X, S_hmm) = ∏_j q_j(x'_j | x_j, S_hmm)

    Для каждого параметра j определяется одномерная смесь распределений,
    веса которой зависят от текущего состояния HMM (EXPLOIT / EXPLORE / TRAPPED).

    Непрерывные (float):
        q_j = w_local · N(x'_j | x_j, σ²) + w_global · U(x'_j | lo, hi)
        где σ = sigma_fraction · (hi - lo).

    Целочисленные (int):
        q_j = w_local · U_disc(x'_j | x_j-1, x_j+1) + w_global · U_disc(x'_j | lo, hi)

    Категориальные (categorical):
        q_j = w_stay · I(x'_j = x_j) + w_uniform · (1/C) + w_history · P_boltz(x'_j)
        P_boltz(c) = exp(-mean_loss(c) / τ) / Z,  Z = Σ_c' exp(-mean_loss(c') / τ)
    """

    # --- Веса смесей для каждого состояния HMM ---
    # Формат для float: (w_local, w_wide, w_uniform)
    # Формат для int: (w_local, w_global)
    # Формат для categorical: (w_stay, w_uniform, w_history)
    FLOAT_WEIGHTS = {
        HMMState.EXPLOIT: (1.00, 0.00, 0.00),  # Строгий локальный поиск
        HMMState.EXPLORE: (0.60, 0.40, 0.00),  # Умеренное исследование: 60% шанс остаться в локальной зоне, 40% широкий прыжок
        HMMState.TRAPPED: (0.00, 0.50, 0.50),  # Панический сброс: 50% супер-широкий гаусс, 50% равномерное
    }
    INT_WEIGHTS = {
        HMMState.EXPLOIT: (1.00, 0.00),
        HMMState.EXPLORE: (0.40, 0.60),
        HMMState.TRAPPED: (0.00, 1.00),
    }
    CATEGORICAL_WEIGHTS = {
        HMMState.EXPLOIT: (0.95, 0.00, 0.05), # uniform=0.0 для EXPLOIT
        HMMState.EXPLORE: (0.05, 0.25, 0.70),
        HMMState.TRAPPED: (0.00, 0.25, 0.75),
    }

    # Малая константа для числовой стабильности (log(0) → log(eps))
    _EPS = 1e-30

    def __init__(self, dict_to_optimize: dict,
                 sigma_fraction: float = 0.10,
                 temperature: float = 1.0):
        """
        Args:
            dict_to_optimize: пространство гиперпараметров.
                Формат: {name: {"type": "float"|"int"|"categorical",
                                "values": [lo, hi] | [c1, c2, ...]}}.
            sigma_fraction: σ для гауссова локального шага,
                выраженная как доля диапазона (hi - lo).
            temperature: τ для Больцмановского softmax по истории категорий.
        """
        self.dict_to_optimize = dict_to_optimize
        self.param_names: list[str] = list(dict_to_optimize.keys())
        self.sigma_fraction = sigma_fraction
        self.temperature = temperature
        self.dim = len(dict_to_optimize)


        # ---- Предвычисление параметров по каждому измерению ----
        self._param_info: list[dict] = []
        for name in self.param_names:
            info = dict_to_optimize[name]
            p_type = info["type"]
            values = info["values"]

            rec: dict = {"name": name, "type": p_type, "values": values}
            if p_type == "float":
                rec["lo"] = float(values[0])
                rec["hi"] = float(values[1])
                rec["range"] = rec["hi"] - rec["lo"]
                # σ для гауссова шага
                rec["sigma"] = self.sigma_fraction * rec["range"]
            elif p_type == "int":
                rec["lo"] = int(values[0])
                rec["hi"] = int(values[1])
                rec["range"] = rec["hi"] - rec["lo"]
            elif p_type == "categorical":
                rec["n_categories"] = len(values)
                # Индекс значения → позиция (для быстрого поиска)
                rec["val_to_idx"] = {v: i for i, v in enumerate(values)}
            rec_type = rec  # alias
            self._param_info.append(rec)

        # ---- История потерь по категориям (для Boltzmann softmax) ----
        # category_history[param_name][category_value] → list[float] (losses)
        self._category_history: dict[str, dict] = {}
        for pi in self._param_info:
            if pi["type"] == "categorical":
                self._category_history[pi["name"]] = {
                    v: [] for v in pi["values"]
                }

    # ------------------------------------------------------------------
    #  Публичный API
    # ------------------------------------------------------------------

    def generate_proposal(self, current_x: dict, state: HMMState) -> dict:
        """Генерирует кандидата X' ~ q(· | X, S_hmm).

        Для каждого параметра j независимо семплируем x'_j из одномерной смеси.

        Args:
            current_x: текущая конфигурация X.
            state: текущее состояние HMM.

        Returns:
            Новая конфигурация X'.
        """
        x_new: dict = {}

        for pi in self._param_info:
            name = pi["name"]
            p_type = pi["type"]
            x_j = current_x[name]

            if p_type == "float":
                x_new[name] = self._sample_continuous(x_j, pi, state)
            elif p_type == "int":
                x_new[name] = self._sample_integer(x_j, pi, state)
            elif p_type == "categorical":
                x_new[name] = self._sample_categorical(x_j, pi, state)

        return x_new

    def log_proposal_density(self, x_from: dict, x_to: dict,
                             state: HMMState) -> float:
        """Вычисляет log q(X_to | X_from, S_hmm).

        log q(X' | X, S) = Σ_j log q_j(x'_j | x_j, S)

        Каждый log q_j считается через logsumexp для числовой стабильности:
            log q_j = logsumexp(log(w_k) + log(p_k))  по компонентам k смеси.

        Args:
            x_from: конфигурация X (центр предложения).
            x_to: конфигурация X' (точка, в которой считаем плотность).
            state: текущее состояние HMM.

        Returns:
            log q(X_to | X_from, S_hmm).
        """
        log_q_total = 0.0

        for pi in self._param_info:
            name = pi["name"]
            p_type = pi["type"]
            val_from = x_from[name]
            val_to = x_to[name]

            if p_type == "float":
                log_q_total += self._log_q_continuous(val_from, val_to, pi, state)
            elif p_type == "int":
                log_q_total += self._log_q_integer(val_from, val_to, pi, state)
            elif p_type == "categorical":
                log_q_total += self._log_q_categorical(val_from, val_to, pi, state)

        return log_q_total

    def update_category_history(self, config: dict, loss: float) -> None:
        """Обновляет историю потерь для категориальных параметров.

        Вызывается после каждой оценки целевой функции (и при init, и при step).

        Args:
            config: оценённая конфигурация.
            loss: значение целевой функции.
        """
        for pi in self._param_info:
            if pi["type"] == "categorical":
                name = pi["name"]
                val = config[name]
                self._category_history[name][val].append(loss)

    # ------------------------------------------------------------------
    #  Семплирование (private)
    # ------------------------------------------------------------------

    def _sample_continuous(self, x_j: float, pi: dict,
                           state: HMMState) -> float:
        """Семплирует x'_j для непрерывного параметра."""
        w_local, w_wide, w_uniform = self.FLOAT_WEIGHTS[state]
        lo, hi = pi["lo"], pi["hi"]

        r = np.random.rand()
        if r < w_local:
            # Узкий Гауссов шаг (EXPLOIT)
            x_new = np.random.normal(x_j, pi["sigma"])
        elif r < w_local + w_wide:
            # Широкий Гауссов шаг (EXPLORE): разброс масштабируется от размерности пространства
            sigma_wide = pi["range"] * (0.35 / np.sqrt(self.dim))
            x_new = np.random.normal(x_j, sigma_wide)
        else:
            # Равномерный сброс (TRAPPED)
            return float(np.random.uniform(lo, hi))

        # Отражение (bouncing) от границ для Гауссовых прыжков
        while x_new < lo or x_new > hi:
            if x_new < lo:
                x_new = lo + (lo - x_new)
            elif x_new > hi:
                x_new = hi - (x_new - hi)
        return float(x_new)

    def _sample_integer(self, x_j: int, pi: dict,
                        state: HMMState) -> int:
        """Семплирует x'_j для целочисленного параметра.

        q_j = w_local · U_disc(x_j-1, x_j+1) + w_global · U_disc(lo, hi)

        1. С вероятностью w_local — выбираем из {x_j-1, x_j, x_j+1} ∩ [lo, hi].
        2. С вероятностью w_global — равномерный из {lo, ..., hi}.
        """
        w_local, w_global = self.INT_WEIGHTS[state]
        lo, hi = pi["lo"], pi["hi"]

        if np.random.rand() < w_local:
            # Локальный дискретный шаг: {x_j-1, x_j, x_j+1} ∩ [lo, hi]
            candidates = [v for v in [x_j - 1, x_j, x_j + 1]
                          if lo <= v <= hi]
            x_new = int(np.random.choice(candidates))
        else:
            # Глобальный равномерный дискретный
            x_new = int(np.random.randint(lo, hi + 1))

        return x_new

    def _sample_categorical(self, x_j, pi: dict,
                            state: HMMState):
        """Семплирует x'_j для категориального параметра.

        q_j = w_stay · I(x'_j = x_j) + w_uniform · (1/C) + w_history · P_boltz(x'_j)

        Шаги:
            1. Вычислить P_boltz для всех категорий.
            2. Построить итоговое PMF: p(c) = w_stay·I(c=x_j) + w_uniform/C + w_history·P_boltz(c).
            3. Нормализовать (на случай числовых ошибок) и семплировать.
        """
        w_stay, w_uniform, w_history = self.CATEGORICAL_WEIGHTS[state]
        categories = pi["values"]
        C = pi["n_categories"]
        name = pi["name"]

        # Больцмановские вероятности
        p_boltz = self._boltzmann_probs(name, categories)

        # Построение итогового PMF
        pmf = np.zeros(C)
        for i, cat in enumerate(categories):
            # Компонента «stay»: w_stay если cat == x_j, иначе 0
            stay = w_stay if cat == x_j else 0.0
            # Компонента «uniform»: w_uniform / C
            uniform = w_uniform / C
            # Компонента «history»: w_history · P_boltz(cat)
            history = w_history * p_boltz[i]
            pmf[i] = stay + uniform + history

        # Нормализация (защита от числовых ошибок)
        pmf_sum = pmf.sum()
        if pmf_sum > 0:
            pmf /= pmf_sum
        else:
            pmf = np.ones(C) / C

        # Семплирование
        idx = np.random.choice(C, p=pmf)
        return categories[idx]

    # ------------------------------------------------------------------
    #  Логарифм плотности предложения (private)
    # ------------------------------------------------------------------

    def _log_q_continuous(self, x_from: float, x_to: float,
                          pi: dict, state: HMMState) -> float:
        """log q_j для непрерывного параметра через logsumexp."""
        w_local, w_wide, w_uniform = self.FLOAT_WEIGHTS[state]
        lo, hi = pi["lo"], pi["hi"]

        log_components = []

        # log N(x_to | x_from, sigma^2)
        if w_local > 0:
            sigma = pi["sigma"]
            z = (x_to - x_from) / (sigma + self._EPS)
            log_g = -0.5 * z * z - np.log(sigma + self._EPS) - 0.5 * np.log(2.0 * np.pi)
            log_components.append(np.log(w_local + self._EPS) + log_g)

        # log N(x_to | x_from, sigma_wide^2)
        if w_wide > 0:
            sigma_wide = pi["range"] * (0.35 / np.sqrt(self.dim))
            z = (x_to - x_from) / (sigma_wide + self._EPS)
            log_g_wide = -0.5 * z * z - np.log(sigma_wide + self._EPS) - 0.5 * np.log(2.0 * np.pi)
            log_components.append(np.log(w_wide + self._EPS) + log_g_wide)

        # log U(x_to | lo, hi)
        if w_uniform > 0:
            log_uniform_val = -np.log(pi["range"] + self._EPS)
            log_components.append(np.log(w_uniform + self._EPS) + log_uniform_val)

        return float(logsumexp(np.array(log_components)))

    def _log_q_integer(self, x_from: int, x_to: int,
                       pi: dict, state: HMMState) -> float:
        """log q_j для целочисленного параметра через logsumexp.

        q_j(x' | x, S) = w_local · U_disc(x' | {x-1,x,x+1}∩[lo,hi])
                        + w_global · U_disc(x' | {lo,...,hi})

        log q_j = logsumexp([log w_local + log p_local,
                             log w_global + log p_global])

        p_local = 1/|{x-1,x,x+1}∩[lo,hi]|   если x' ∈ {x-1,x,x+1}∩[lo,hi], иначе 0.
        p_global = 1/(hi - lo + 1).
        """
        w_local, w_global = self.INT_WEIGHTS[state]
        lo, hi = pi["lo"], pi["hi"]

        # Локальная компонента: {x_from-1, x_from, x_from+1} ∩ [lo, hi]
        local_candidates = [v for v in [x_from - 1, x_from, x_from + 1]
                            if lo <= v <= hi]
        n_local = len(local_candidates)

        # Глобальная компонента: {lo, ..., hi}
        n_global = hi - lo + 1

        log_components = []

        # Локальная: x_to должен попадать в local_candidates
        if x_to in local_candidates:
            log_p_local = -np.log(n_local)
            log_components.append(np.log(w_local + self._EPS) + log_p_local)
        else:
            # x_to вне локальной окрестности → вклад локальной компоненты = 0
            log_components.append(-np.inf)

        # Глобальная: x_to ∈ {lo, ..., hi}  (всегда true для валидных конфигураций)
        log_p_global = -np.log(n_global)
        log_components.append(np.log(w_global + self._EPS) + log_p_global)

        return float(logsumexp(np.array(log_components)))

    def _log_q_categorical(self, x_from, x_to,
                           pi: dict, state: HMMState) -> float:
        """log q_j для категориального параметра через logsumexp.

        q_j(x' | x, S) = w_stay · I(x'=x) + w_uniform · (1/C) + w_history · P_boltz(x')

        log q_j = logsumexp([log(w_stay · I(x'=x)),
                             log(w_uniform / C),
                             log(w_history · P_boltz(x'))])
        """
        w_stay, w_uniform, w_history = self.CATEGORICAL_WEIGHTS[state]
        categories = pi["values"]
        C = pi["n_categories"]
        name = pi["name"]

        # Больцмановские вероятности
        p_boltz = self._boltzmann_probs(name, categories)
        idx_to = pi["val_to_idx"][x_to]

        log_components = []

        # Компонента «stay»: w_stay · I(x_to == x_from)
        if x_to == x_from:
            log_components.append(np.log(w_stay + self._EPS))
        else:
            # I(x_to == x_from) = 0 → вклад = 0
            log_components.append(-np.inf)

        # Компонента «uniform»: w_uniform / C
        log_components.append(np.log(w_uniform / C + self._EPS))

        # Компонента «history»: w_history · P_boltz(x_to)
        log_components.append(np.log(w_history * p_boltz[idx_to] + self._EPS))

        return float(logsumexp(np.array(log_components)))

    # ------------------------------------------------------------------
    #  Вспомогательные методы (private)
    # ------------------------------------------------------------------

    def _boltzmann_probs(self, param_name: str,
                         categories: list) -> np.ndarray:
        """Вычисляет Больцмановские вероятности P_boltz(c) для категорий.

        P_boltz(c) = exp(-mean_loss(c) / τ) / Z
        Z = Σ_c' exp(-mean_loss(c') / τ)

        Если для категории c нет наблюдений, используется eps-сглаживание:
        mean_loss(c) подставляется как максимальный mean_loss среди
        наблюдённых + 1 (штраф за неизвестность).

        Вычисления в log-пространстве через logsumexp.

        Args:
            param_name: имя категориального параметра.
            categories: список допустимых значений.

        Returns:
            np.ndarray формы (C,) — нормализованные вероятности.
        """
        C = len(categories)
        history = self._category_history[param_name]

        # Вычислить средние потери для каждой категории
        mean_losses = np.zeros(C)
        observed = np.zeros(C, dtype=bool)

        for i, cat in enumerate(categories):
            losses = history[cat]
            if len(losses) > 0:
                mean_losses[i] = np.mean(losses)
                observed[i] = True

        # Если нет наблюдений ни у одной категории → равномерное
        if not np.any(observed):
            return np.ones(C) / C

        # Штраф для ненаблюдённых (сдвиг на 1 единицу температуры, чтобы сохранить инвариантность)
        tau = self.temperature + self._EPS
        penalty = np.max(mean_losses[observed]) + tau
        mean_losses[~observed] = penalty

        # log-softmax: log P_boltz(c) = -mean_loss(c)/τ - logsumexp(-mean_loss/τ)
        neg_scaled = -mean_losses / tau   # shape (C,)

        # logsumexp для нормализации
        log_Z = logsumexp(neg_scaled)
        log_probs = neg_scaled - log_Z
        probs = np.exp(log_probs)

        # Защита от вырожденных случаев (все probs ≈ 0 из-за переполнения)
        prob_sum = probs.sum()
        if prob_sum < self._EPS:
            return np.ones(C) / C
        probs /= prob_sum

        return probs


# ---------------------------------------------------------------------------
#  MCMCChain
# ---------------------------------------------------------------------------
import numpy as np
import copy

class MCMCChain:
    """Одна цепь Метрополиса — Гастингса с HMM-управлением.

    Критерий MH:
        α = min(1, exp(log_likelihood_ratio + log_hastings_ratio))
    """

    def __init__(self, chain_id: int, x0: dict, loss0: float,
                 proposal_gen: FactorizedProposalGenerator,
                 hmm: HMMController,
                 T_mcmc: float = 1.0, T_min: float | None = None,
                 scale_factor: float = 1.0):
        self.chain_id = chain_id
        self.current_x = copy.deepcopy(x0)
        self.current_loss = loss0
        self.current_loss_old_for_hmm = loss0
        self.best_x = copy.deepcopy(x0)
        self.best_loss = loss0
        self.T_mcmc_init = T_mcmc
        self.T_mcmc = T_mcmc
        self.T_min = T_min if T_min is not None else T_mcmc * 0.01
        self.scale_factor = scale_factor

        self.proposal_gen = proposal_gen
        self.hmm = hmm
        self.state = HMMState.EXPLORE

        # Храним готовые наблюдения O_t для Витерби, а не сырой Loss
        self._observations: list[float] =[]
        self._rejection_streak: int = 0
        self._stagnation_limit: int = 10

    def reset_at(self, x0: dict, loss0: float):
        """Перезапускает цепь в новой точке (для pruning/cloning)."""
        self.current_x = copy.deepcopy(x0)
        self.current_loss = loss0
        self.best_x = copy.deepcopy(x0)
        self.best_loss = loss0
        self.hmm.reset()
        self.state = HMMState.EXPLORE
        
        # Очищаем наблюдения. Пока не накопится история, цепь будет в EXPLORE
        self._observations =[]
        self._rejection_streak = 0

    def step(self, objective_func, progress: float = 0.0, is_burnin: bool = False) -> tuple[dict, float]:
        """Один шаг MH с температурным затуханием."""
        
        # 0. Линейное затухание температуры
        if is_burnin:
            self.T_mcmc = self.T_mcmc_init
        else:
            self.T_mcmc = self.T_min + (self.T_mcmc_init - self.T_min) * (1.0 - progress)

        # 1. Обновляем состояние HMM (если есть достаточная история)
        # Если истории мало, принудительно остаемся в текущем (обычно EXPLORE)
        if len(self._observations) >= 2:
            self.state = self.hmm.observe(self._observations[-self.hmm.window:])
        else:
            self.state = self.hmm.state

        # Failsafe: Переопределяем состояние при высоком rejection streak
        # В период burn-in даем цепи свободу и не применяем строгие ограничения
        if not is_burnin and self._rejection_streak >= self._stagnation_limit:
            if self.state != HMMState.TRAPPED:
                self.state = HMMState.TRAPPED
                self.hmm.force_state(HMMState.TRAPPED)  # Синхронизируем HMM с внешним вмешательством
            
            # Агрессивный "отжиг" для мгновенного выхода из TRAPPED:
            # С множителем 4.0 вероятность пробития барьера возрастает в тысячи раз за 2-3 шага
            agitation_level = self._rejection_streak - self._stagnation_limit + 1
            self.T_mcmc = self.T_mcmc * (4.0 ** agitation_level)

        # 2. Генерируем кандидата
        x_prime = self.proposal_gen.generate_proposal(self.current_x, self.state)

        # 3. Оцениваем кандидата
        loss_prime = objective_func(x_prime)
        self.current_loss_old_for_hmm = self.current_loss

        # 5. Вычисляем α (критерий MH)
        alpha = self._acceptance_probability(x_prime, loss_prime)

        # 6. Принятие / отклонение
        accepted = False
        if np.random.rand() < alpha:
            self.current_x = copy.deepcopy(x_prime)
            self.current_loss = loss_prime
            self._rejection_streak = 0
            accepted = True
        else:
            self._rejection_streak += 1

        # 4. Формируем наблюдение O_t ДЛЯ HMM на основе кандидата!
        # Вычисляем разницу, нормированную на глобальный масштаб (scale_factor).
        # Это защищает от проблемы отрицательных значений Loss, где log_ratio ломается:
        delta_loss = loss_prime - self.current_loss_old_for_hmm
        O_t = float(delta_loss / (self.scale_factor + 1e-8))
        
        # Ограничиваем O_t, чтобы Viterbi не сошел с ума от гигантских выбросов:
        # Успешный прыжок (сильно меньше 0) "обрезаем" до -0.05, чтобы он читался как чистый EXPLOIT
        # Худшие прыжки "обрезаем" до 0.8, чтобы они попадали в дисперсию EXPLORE 
        O_t = max(min(O_t, 0.8), -0.05)

        self._observations.append(O_t)

        # Ограничиваем размер хранимой истории для экономии памяти

        # 7. Обновляем лучший результат
        if loss_prime < self.best_loss:
            self.best_x = copy.deepcopy(x_prime)
            self.best_loss = loss_prime

        return x_prime, loss_prime, accepted

    def _acceptance_probability(self, x_prime: dict, loss_prime: float) -> float:
        """Вычисляет вероятность принятия α по формуле MH."""
        
        # log likelihood ratio
        log_likelihood_ratio = -(loss_prime - self.current_loss) / (self.T_mcmc + 1e-30)

        # log Hastings ratio
        log_q_reverse = self.proposal_gen.log_proposal_density(
            x_from=x_prime, x_to=self.current_x, state=self.state
        )
        log_q_forward = self.proposal_gen.log_proposal_density(
            x_from=self.current_x, x_to=x_prime, state=self.state
        )
        log_hastings_ratio = log_q_reverse - log_q_forward

        log_alpha = log_likelihood_ratio + log_hastings_ratio
        
        # np.exp(min(..., 0.0)) безопасно ограничивает вероятность сверху единицей
        return float(np.exp(min(log_alpha, 0.0)))


# ---------------------------------------------------------------------------
#  GlobalOrchestrator
# ---------------------------------------------------------------------------
class GlobalOrchestrator:
    """Макро-уровневый оркестратор: pruning, cloning, reseeding.

    Каждые E шагов:
        1. Сбор глобальной статистики категорий.
        2. Pruning & Cloning: убить TRAPPED-цепи, клонировать лучшую.
        3. Reseeding: при глобальном коллапсе перезапустить часть цепей.
    """

    def __init__(self, chains: list[MCMCChain],
                 proposal_gen: FactorizedProposalGenerator,
                 sobol_init: SobolInitializer,
                 dict_to_optimize: dict,
                 objective_func,
                 clone_noise: float = 0.05,
                 collapse_threshold: float = 0.01,
                 reseed_fraction: float = 0.3):
        """
        Args:
            clone_noise: σ шума при клонировании (как доля диапазона параметра).
            collapse_threshold: порог для детекции «коллапса роя»
                (все цепи слишком близко друг к другу по loss).
            reseed_fraction: доля цепей, перезапускаемых при reseeding.
        """
        self.chains = chains
        self.proposal_gen = proposal_gen
        self.sobol_init = sobol_init
        self.dict_to_optimize = dict_to_optimize
        self.objective_func = objective_func
        self.clone_noise = clone_noise
        self.collapse_threshold = collapse_threshold
        self.reseed_fraction = reseed_fraction

    def orchestrate(self, data: list, budget: int = 1) -> list[tuple[dict, float]]:
        """Выполняет один раунд оркестрации.

        Args:
            data: текущий список (config, loss).
            budget: общий бюджет (для вычисления progress).

        Returns:
            Список (config, loss) от пересозданных цепей (для data).
        """
        new_evals = []
        progress = len(data) / max(budget, 1)

        # 1. Pruning & Cloning — только цепи, которые прожили
        #    достаточно шагов после последнего reset (иначе HMM не успевает
        #    прогреться и сразу ставит TRAPPED → бесконечный цикл).
        min_age = max(self.chains[0].hmm.window, 10) if self.chains else 10
        trapped_ids = [
            i for i, c in enumerate(self.chains)
            if c.state == HMMState.TRAPPED and len(c._observations) >= min_age
        ]
        if trapped_ids:
            exploit_chains = [i for i, c in enumerate(self.chains)
                              if c.state == HMMState.EXPLOIT]
            if exploit_chains:
                donor_idx = min(exploit_chains,
                                key=lambda i: self.chains[i].best_loss)
            else:
                donor_idx = min(range(len(self.chains)),
                                key=lambda i: self.chains[i].best_loss)

            donor = self.chains[donor_idx]
            for idx in trapped_ids:
                new_x = self._add_noise(donor.best_x)
                new_loss = self.objective_func(new_x)
                new_evals.append((new_x, new_loss))
                self.chains[idx].reset_at(new_x, new_loss)
                self.proposal_gen.update_category_history(new_x, new_loss)

        # 2. Reseeding при глобальном коллапсе
        #    Отключаем во второй половине оптимизации: к этому моменту
        #    сходимость цепей — ожидаемое поведение, а не коллапс.
        if progress < 0.5 and self._detect_collapse():
            n_reseed = max(1, int(len(self.chains) * self.reseed_fraction))
            sorted_chains = sorted(range(len(self.chains)),
                                   key=lambda i: self.chains[i].best_loss)
            worst_ids = sorted_chains[-n_reseed:]
            new_configs = self.sobol_init.generate(
                n_reseed, seed=np.random.randint(0, 2**31)
            )
            for i, idx in enumerate(worst_ids):
                new_loss = self.objective_func(new_configs[i])
                new_evals.append((new_configs[i], new_loss))
                self.chains[idx].reset_at(new_configs[i], new_loss)
                self.proposal_gen.update_category_history(new_configs[i], new_loss)

        return new_evals

    def _add_noise(self, config: dict) -> dict:
        """Добавляет гауссовский шум к конфигурации (для клонирования)."""
        noisy = {}
        for name, info in self.dict_to_optimize.items():
            p_type = info["type"]
            values = info["values"]
            val = config[name]

            if p_type == "float":
                lo, hi = values[0], values[1]
                sigma = (hi - lo) * self.clone_noise
                noisy[name] = float(np.clip(val + np.random.normal(0, sigma), lo, hi))
            elif p_type == "int":
                lo, hi = int(values[0]), int(values[1])
                step = np.random.choice([-1, 0, 1])
                noisy[name] = int(np.clip(int(val) + step, lo, hi))
            elif p_type == "categorical":
                # С малой вероятностью мутируем
                if np.random.rand() < 0.2 and len(values) > 1:
                    choices = [v for v in values if v != val]
                    noisy[name] = np.random.choice(choices)
                else:
                    noisy[name] = val
            else:
                noisy[name] = val
        return noisy

    def _detect_collapse(self) -> bool:
        """Детектирует глобальный коллапс: все best_loss слишком близки."""
        losses = [c.best_loss for c in self.chains]
        if len(losses) < 2:
            return False
        spread = np.std(losses) / (np.abs(np.mean(losses)) + 1e-30)
        return spread < self.collapse_threshold


# ---------------------------------------------------------------------------
#  HMM_MCMC — фасад, совместимый с интерфейсом baselines
# ---------------------------------------------------------------------------
class HMM_MCMC:
    """H-MCMC-FMP: Hierarchical MCMC with Factorized Mixture Proposals.

    Совместим с интерфейсом baselines:
        - __init__(objective_func, ..., dict_to_optimize, ...)
        - self.data = [(config, score), ...]
        - reset()
        - main_loop() → (best_config, best_score)

    Args:
        objective_func: целевая функция (минимизация).
        budget: общее количество вызовов целевой функции.
        dict_to_optimize: пространство параметров.
        n_init: количество начальных точек Соболя (Фаза I).
        n_chains: количество параллельных MCMC-цепей (K).
        orchestrate_every: частота запуска Оркестратора (E шагов).
        T_mcmc: температура цепей MH.
        sigma_fraction: σ для локального гауссова шага (доля диапазона).
        temperature: τ для Softmax Больцмана по категориям.
        hmm_window: длина окна наблюдений для Витерби.
        hmm_obs_epsilon: ε в знаменателе log-ratio наблюдений HMM.
        hmm_lambda_noise: вес робастного фонового шума в эмиссии HMM.
        clone_noise: σ шума при клонировании (доля диапазона).
    """

    def __init__(self, objective_func, budget: int, dict_to_optimize: dict,
                 n_init: int = 16, n_chains: int = 4,
                 orchestrate_every: int = 5, T_mcmc: float = 1.0,
                 T_min: float | None = None, burnin_fraction: float = 0.10,
                 sigma_fraction: float = 0.10, temperature: float = 1.0,
                 hmm_window: int = 8, hmm_obs_epsilon: float = 1e-8,
                 hmm_lambda_noise: float = 0.01, clone_noise: float = 0.05):
        self.objective_func = objective_func
        self.budget = budget
        self.dict_to_optimize = dict_to_optimize
        self.n_init = n_init
        self.n_chains = n_chains
        self.orchestrate_every = orchestrate_every
        self.T_mcmc = T_mcmc
        self.T_min = T_min
        self.burnin_fraction = burnin_fraction

        # Параметры компонентов
        self._sigma_fraction = sigma_fraction
        self._temperature = temperature
        self._hmm_window = hmm_window
        self._hmm_obs_epsilon = hmm_obs_epsilon
        self._hmm_lambda_noise = hmm_lambda_noise
        self._clone_noise = clone_noise

        # Состояние
        self.data: list[tuple[dict, float]] = []
        self._chains: list[MCMCChain] = []
        self._proposal_gen: FactorizedProposalGenerator | None = None
        self._sobol_init: SobolInitializer | None = None
        self._orchestrator: GlobalOrchestrator | None = None
        self.history_table: list[dict] = []

    def reset(self):
        """Сброс для повторного запуска (run_n_experiments)."""
        self.data = []
        self._chains = []
        self._proposal_gen = None
        self._sobol_init = None
        self._orchestrator = None
        self.history_table = []

    def main_loop(self):
        """Основной цикл H-MCMC-FMP.

        Фаза I:  Инициализация Соболя → оценка → отбор K лучших → старт цепей.
        Фаза II: Параллельный MCMC (каждая цепь: HMM → proposal → MH).
        Фаза III: Каждые E шагов — оркестрация (pruning, cloning, reseeding).

        Returns:
            (best_config, best_loss)
        """
        # ========== Фаза I: Инициализация ==========
        self._sobol_init = SobolInitializer(self.dict_to_optimize)
        self._proposal_gen = FactorizedProposalGenerator(
            self.dict_to_optimize,
            sigma_fraction=self._sigma_fraction,
            temperature=self._temperature,
        )

        # Генерация и оценка начальных точек
        init_configs = self._sobol_init.generate(self.n_init)
        pbar = tqdm(total=self.budget, desc="H-MCMC-FMP")

        init_scores = []
        for cfg in init_configs:
            if len(self.data) >= self.budget:
                break
            score = self.objective_func(cfg)
            self.data.append((cfg, score))
            init_scores.append((cfg, score))
            self._proposal_gen.update_category_history(cfg, score)
            pbar.update(1)

        if len(self.data) >= self.budget:
            pbar.close()
            return min(self.data, key=lambda x: x[1])

        # Вычисляем амплитуду ландшафта для адаптивной температуры
        losses = [score for _, score in init_scores]
        loss_std = float(np.std(losses))
        # Если все точки имеют одинаковый loss, не масштабируем
        scale_factor = loss_std if loss_std > 1e-5 else 1.0
        
        scaled_T_mcmc = self.T_mcmc * scale_factor
        scaled_T_min = self.T_min * scale_factor if self.T_min is not None else None

        # Масштабируем температуру для расчета распределения Больцмана категориальных параметров
        self._proposal_gen.temperature = self._temperature * scale_factor

        # Отбор K лучших как стартовых позиций цепей
        init_scores.sort(key=lambda x: x[1])
        k = min(self.n_chains, len(init_scores))
        best_inits = init_scores[:k]

        # Создаём MCMC-цепи
        self._chains = []
        for i, (cfg, score) in enumerate(best_inits):
            hmm = HMMController(
                window=self._hmm_window,
                obs_epsilon=self._hmm_obs_epsilon,
                lambda_noise=self._hmm_lambda_noise,
            )
            chain = MCMCChain(
                chain_id=i,
                x0=cfg,
                loss0=score,
                proposal_gen=self._proposal_gen,
                hmm=hmm,
                T_mcmc=scaled_T_mcmc,
                T_min=scaled_T_min,
                scale_factor=scale_factor,
            )
            self._chains.append(chain)

        # Создаём Оркестратор
        self._orchestrator = GlobalOrchestrator(
            chains=self._chains,
            proposal_gen=self._proposal_gen,
            sobol_init=self._sobol_init,
            dict_to_optimize=self.dict_to_optimize,
            objective_func=self.objective_func,
            clone_noise=self._clone_noise,
        )

        # ========== Фаза II + III: Основной цикл ==========
        step_counter = 0

        while len(self.data) < self.budget:
            # Round-robin по цепям
            for chain in self._chains:
                if len(self.data) >= self.budget:
                    break

                progress = len(self.data) / self.budget
                
                cfg, loss, accepted = chain.step(self.objective_func, progress=progress)
                self.data.append((cfg, loss))
                self._proposal_gen.update_category_history(cfg, loss)
                
                # Сохраняем состояние ПОСЛЕ шага, чтобы зафиксировать случай, 
                # когда цепь экстренно перешла в TRAPPED внутри метода step()
                current_state_name = chain.state.name
                
                # Логируем шаг
                self.history_table.append({
                    "Eval": len(self.data),
                    "Chain": chain.chain_id,
                    "State": current_state_name,  # Состояние, в котором генерировалась точка
                    "Loss": float(loss),
                    "Accepted": accepted,
                    "Rej_Streak": chain._rejection_streak
                })
                
                pbar.update(1)

            step_counter += 1

            # Фаза III: Оркестрация каждые E шагов
            if step_counter % self.orchestrate_every == 0:
                if len(self.data) < self.budget:
                    new_evals = self._orchestrator.orchestrate(self.data, budget=self.budget)
                    for cfg, loss in new_evals:
                        if len(self.data) >= self.budget:
                            break
                        self.data.append((cfg, loss))
                        
                        # Оркестратор может менять стейты, логируем рестарт как специальное событие
                        self.history_table.append({
                            "Eval": len(self.data),
                            "Chain": "ORCHESTRATOR",
                            "State": "RESET/CLONE",
                            "Loss": float(loss),
                            "Accepted": True,
                            "Rej_Streak": 0
                        })
                        pbar.update(1)

        pbar.close()
        
        # Вывод таблицы истории
        try:
            import pandas as pd
            df = pd.DataFrame(self.history_table)
            print("\n" + "="*60)
            print("HMM MCMC State History (last run):")
            # Печатаем полностью, если строк до 200, иначе обрезаем
            with pd.option_context('display.max_rows', 200, 'display.max_columns', None):
                print(df)
            print("="*60 + "\n")
            df.to_csv("hmm_mcmc_history.csv", index=False)
        except ImportError:
            pass

        return min(self.data, key=lambda x: x[1])
