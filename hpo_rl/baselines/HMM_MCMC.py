"""H-MCMC-FMP: иерархический MCMC с факторизованными смесевыми предложениями.

Компоненты: :class:`SobolInitializer`, :class:`HMMController`,
:class:`FactorizedProposalGenerator`, :class:`MCMCChain`, :class:`GlobalOrchestrator`.
"""

import numpy as np
import math
import copy
import os
from enum import IntEnum
from tqdm.auto import tqdm
from scipy.stats.qmc import Sobol


def _logsumexp(a):
    """Чистый numpy logsumexp"""
    arr = np.asarray(a, dtype=np.float64)
    a_max = arr.max()
    if a_max == -np.inf:
        return -np.inf
    return float(a_max + np.log(np.sum(np.exp(arr - a_max))))


def decode_config(cfg: dict, dict_to_optimize: dict) -> dict:
    """Декодирует внутренние log10-координаты в реальные значения гиперпараметров."""
    out = dict(cfg)
    for name, info in dict_to_optimize.items():
        if info.get("type") == "float" and info.get("log"):
            out[name] = float(10.0 ** float(cfg[name]))
    return out


def _float_param_bounds(info: dict) -> tuple[float, float, float, float, bool]:
    """Возвращает (lo, hi, real_lo, real_hi, is_log) для float-параметра."""
    real_lo, real_hi = float(info["values"][0]), float(info["values"][1])
    if info.get("log"):
        return float(np.log10(real_lo)), float(np.log10(real_hi)), real_lo, real_hi, True
    return real_lo, real_hi, real_lo, real_hi, False

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)
_SQRT2 = math.sqrt(2.0)


def _std_cdf(x: float) -> float:
    """Φ(x) — CDF стандартной нормали через math.erf."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _std_ppf(p: float) -> float:
    """Φ⁻¹(p) — квантильная функция стандартной нормали."""
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -_rational_approx(math.sqrt(-2.0 * math.log(p)))
    else:
        return _rational_approx(math.sqrt(-2.0 * math.log(1.0 - p)))


def _rational_approx(t: float) -> float:
    """Рациональная аппроксимация в формуле обратной Φ⁻¹ стандартной нормали."""
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)


def _tn_rvs(a: float, b: float, loc: float, scale: float) -> float:
    """Семплирование из усечённой нормали TN(loc, scale², [lo, hi])."""
    cdf_a = _std_cdf(a)
    cdf_b = _std_cdf(b)
    u = np.random.uniform(cdf_a + 1e-15, cdf_b - 1e-15)
    return float(_std_ppf(u) * scale + loc)


def _tn_logpdf(x: float, a: float, b: float,
               loc: float, scale: float) -> float:
    """log-плотность усечённой нормали."""
    z = (x - loc) / scale
    if z < a - 1e-10 or z > b + 1e-10:
        return -np.inf
    log_phi = -0.5 * z * z - _LOG_SQRT_2PI
    cdf_diff = _std_cdf(b) - _std_cdf(a)
    if cdf_diff < 1e-30:
        return -np.inf
    return float(log_phi - math.log(scale) - math.log(cdf_diff))

class HMMState(IntEnum):
    """Скрытые состояния локального HMM: EXPLOIT, EXPLORE, TRAPPED."""

    EXPLOIT = 0
    EXPLORE = 1
    TRAPPED = 2

class SobolInitializer:
    """Квазислучайная инициализация стартовых точек через последовательность Соболя.

    Если scipy доступна — используется `scipy.stats.qmc.Sobol`, иначе
    фоллбэк на стратифицированный случайный семплинг (латинский гиперкуб).
    """

    def __init__(self, dict_to_optimize: dict):
        """Инициализирует генератор точек Соболя.

        Args:
            dict_to_optimize: пространство гиперпараметров
        """
        self.dict_to_optimize = dict_to_optimize
        self.param_names = list(dict_to_optimize.keys())
        self.dim = len(self.param_names)

    def generate(self, n_points: int, seed: int = 42) -> list[dict]:
        """Возвращает список из n_points конфигураций: [0,1]^D → пространство гиперпараметров."""
        unit_points = self._sobol_unit_cube(n_points, seed)
        configs = []
        for i in range(n_points):
            config = self._unit_to_config(unit_points[i])
            configs.append(config)
        return configs

    def _sobol_unit_cube(self, n: int, seed: int) -> np.ndarray:
        """Генерирует n точек в [0,1]^D."""
        sampler = Sobol(d=self.dim, scramble=True, seed=seed)
        m = int(np.ceil(np.log2(max(n, 2))))
        raw = sampler.random_base2(m)  
        return raw[:n]

    def _unit_to_config(self, u: np.ndarray) -> dict:
        """Отображает вектор из [0,1]^D в словарь гиперпараметров."""
        config = {}
        for j, name in enumerate(self.param_names):
            info = self.dict_to_optimize[name]
            p_type = info["type"]
            values = info["values"]

            if p_type == "float":
                lo, hi, _, _, _ = _float_param_bounds(info)
                config[name] = lo + (hi - lo) * u[j]
            elif p_type == "int":
                lo, hi = int(values[0]), int(values[1])
                config[name] = int(np.round(lo + (hi - lo) * u[j]))
            elif p_type == "categorical":
                idx = int(u[j] * len(values))
                idx = min(idx, len(values) - 1)
                config[name] = values[idx]
            else:
                raise ValueError(f"Неизвестный тип: {p_type}")
        return config

class HMMController:
    """Локальный HMM-контроллер одной MCMC-цепи.

    Состояния: EXPLOIT, EXPLORE, TRAPPED. Эмиссия — смесь гауссиан по log-ratio loss.
    Текущее состояние определяется алгоритмом Витерби по окну наблюдений.
    """

    N_STATES = 3

    def __init__(self, window: int = 8, obs_epsilon: float = 1e-8,
                 lambda_noise: float = 0.01):
        """Инициализирует локальный HMM-контроллер.

        Args:
            window: длина окна наблюдений для Витерби.
            obs_epsilon: ε в знаменателе log(Loss_t / (Loss_{t-1} + ε)).
            lambda_noise: вес робастного фонового шума в эмиссии.
        """
        self.window = window
        self.obs_epsilon = obs_epsilon
        self.lambda_noise = lambda_noise
        
        self.pi = np.array([1.0, 0.0, 0.0])

        self.A = np.array([
            [0.60,    0.40,    0.00],   
            [0.50,    0.50,    0.00],  
            [0.60,    0.20,    0.20], 
        ])

        self.emission_mu    = np.array([-0.01,  0.10,  0.015])
        self.emission_sigma = np.array([ 0.06,  0.25,  0.020])

        self.c_noise = 1.0 / 20.0 

        self.state = HMMState.EXPLOIT

    def reset(self):
        """Сбрасывает состояние HMM в EXPLOIT."""
        self.state = HMMState.EXPLOIT

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

        obs_seq = obs_history[-self.window:]

        self.state = HMMState(self._viterbi(obs_seq))
        return self.state

    def _log_emission(self, obs: float, state: int) -> float:
        """Логарифм эмиссионной плотности P(O_t | S_t) для непрерывного наблюдения.

        P(O | S) = (1 - λ) · N(O | μ_S, σ_S²) + λ · c_noise
        """
        mu = self.emission_mu[state]
        sigma = self.emission_sigma[state]

        z = (obs - mu) / sigma
        log_gauss = -0.5 * z * z - np.log(sigma) - 0.5 * np.log(2.0 * np.pi)
        gauss = np.exp(log_gauss)

        density = (1.0 - self.lambda_noise) * gauss + self.lambda_noise * self.c_noise
        return np.log(density + 1e-30)

    def _viterbi(self, obs_seq: list[float]) -> int:
        """Алгоритм Витерби для непрерывных наблюдений.

        Возвращает наиболее вероятное последнее состояние.
        Все вычисления в логарифмической шкале для предотвращения исчезновения вероятностей.
        """
        T = len(obs_seq)
        if T == 0:
            return int(self.state)

        log_pi = np.log(self.pi + 1e-30)
        log_A = np.log(self.A + 1e-30)

        delta = np.full((T, self.N_STATES), -np.inf)
        psi = np.zeros((T, self.N_STATES), dtype=int)

        for s in range(self.N_STATES):
            delta[0, s] = log_pi[s] + self._log_emission(obs_seq[0], s)

        for t in range(1, T):
            for j in range(self.N_STATES):
                scores = delta[t - 1] + log_A[:, j]
                psi[t, j] = int(np.argmax(scores))
                delta[t, j] = scores[psi[t, j]] + self._log_emission(obs_seq[t], j)

        path = np.zeros(T, dtype=int)
        path[T - 1] = int(np.argmax(delta[T - 1]))
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return int(path[-1])

class FactorizedProposalGenerator:
    """Генератор факторизованных смесевых предложений q(X' | X, S_hmm).

    Для float/int — смесь узкого гаусса, KDE по архиву и широкого шага.
    Для categorical — смесь «остаться», равномерного и исторического распределений.
    """

    FLOAT_WEIGHTS = {
        HMMState.EXPLOIT: (0.90, 0.10, 0.00),  
        HMMState.EXPLORE: (0.00, 0.80, 0.20), 
        HMMState.TRAPPED: (0.00, 0.50, 0.50),  
    }
    INT_WEIGHTS = {
        HMMState.EXPLOIT: (0.90, 0.10, 0.00),
        HMMState.EXPLORE: (0.00, 0.80, 0.20),
        HMMState.TRAPPED: (0.00, 0.50, 0.50),
    }
    CATEGORICAL_WEIGHTS = {
        HMMState.EXPLOIT: (0.30, 0.00, 0.70),  
        HMMState.EXPLORE: (0.10, 0.40, 0.50), 
        HMMState.TRAPPED: (0.00, 0.80, 0.20),  
    }

    _EPS = 1e-30

    def __init__(self, dict_to_optimize: dict,
                 sigma_fraction: float = 0.10,
                 temperature: float = 0.60,
                 wide_sigma_fraction: float = 0.40,
                 kde_tau: float = 0.05):
        """Инициализирует генератор факторизованных смесевых предложений.

        Args:
            dict_to_optimize: пространство гиперпараметров.
            sigma_fraction: σ для узкого гауссова шага (доля диапазона).
            wide_sigma_fraction: σ для широкого гауссова шага (доля диапазона).
            temperature: τ для Boltzmann softmax по категориям.
            kde_tau: τ для Boltzmann-взвешивания Archive-KDE (после min-max нормализации).
        """
        self.dict_to_optimize = dict_to_optimize
        self.param_names: list[str] = list(dict_to_optimize.keys())
        self.sigma_fraction = sigma_fraction
        self.temperature = temperature
        self.dim = len(dict_to_optimize)
        self.wide_sigma_fraction = wide_sigma_fraction
        self._archive: list[tuple[dict, float]] = []
        self._archive_max = 200  
        self._kde_tau = kde_tau
        self._param_info: list[dict] = []
        for name in self.param_names:
            info = dict_to_optimize[name]
            p_type = info["type"]
            values = info["values"]

            rec: dict = {"name": name, "type": p_type, "values": values}
            if p_type == "float":
                lo, hi, real_lo, real_hi, is_log = _float_param_bounds(info)
                rec["log"] = is_log
                rec["real_lo"] = real_lo
                rec["real_hi"] = real_hi
                rec["lo"] = lo
                rec["hi"] = hi
                rec["range"] = rec["hi"] - rec["lo"]
                rec["sigma"] = self.sigma_fraction * rec["range"]
                rec["sigma_wide"] = self.wide_sigma_fraction * rec["range"]
            elif p_type == "int":
                rec["lo"] = int(values[0])
                rec["hi"] = int(values[1])
                rec["range"] = rec["hi"] - rec["lo"]
            elif p_type == "categorical":
                rec["n_categories"] = len(values)
                rec["val_to_idx"] = {v: i for i, v in enumerate(values)}
            self._param_info.append(rec)

        self._category_history: dict[str, dict] = {}
        for pi in self._param_info:
            if pi["type"] == "categorical":
                self._category_history[pi["name"]] = {
                    v: [] for v in pi["values"]
                }

    def generate_proposal(self, current_x: dict, state: HMMState,
                          categorical_only: bool = False) -> dict:
        """Генерирует кандидата X' ~ q(· | X, S_hmm).

        Для каждого параметра j независимо семплируем x'_j из одномерной смеси.

        Args:
            current_x: текущая конфигурация X.
            state: текущее состояние HMM.
            categorical_only: если True, изменяем ТОЛЬКО категориальные
                параметры, оставляя float/int без изменений.

        Returns:
            Новая конфигурация X'.
        """
        x_new: dict = {}
        explore_dims: set | None = None
        if state == HMMState.EXPLORE:
            fi_indices = [i for i, pi in enumerate(self._param_info)
                          if pi["type"] in ("float", "int")]
            if fi_indices:
                n_mod = max(1, len(fi_indices) // 5)  
                explore_dims = set(np.random.choice(
                    fi_indices, size=min(n_mod, len(fi_indices)), replace=False))

        for i, pi in enumerate(self._param_info):
            name = pi["name"]
            p_type = pi["type"]
            x_j = current_x[name]

            if p_type == "categorical":
                x_new[name] = self._sample_categorical(x_j, pi, state)
            elif categorical_only:
                x_new[name] = x_j
            elif explore_dims is not None and i not in explore_dims:
                x_new[name] = x_j
            elif p_type == "float":
                x_new[name] = self._sample_continuous(x_j, pi, state)
            elif p_type == "int":
                x_new[name] = self._sample_integer(x_j, pi, state)

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
        """Обновляет историю потерь для категориальных параметров и KDE-архив.

        Args:
            config: оценённая конфигурация.
            loss: значение целевой функции.
        """
        for pi in self._param_info:
            if pi["type"] == "categorical":
                name = pi["name"]
                val = config[name]
                self._category_history[name][val].append(loss)

        self._archive.append((config, loss))
        self._archive.sort(key=lambda x: x[1])
        if len(self._archive) > self._archive_max:
            self._archive = self._archive[:self._archive_max]

    def _archive_boltzmann_weights(self) -> np.ndarray:
        """Boltzmann-веса для архивных членов (min-max нормализация → softmax)."""
        n = len(self._archive)
        if n == 0:
            return np.array([])
        losses = np.array([entry[1] for entry in self._archive])
        l_min, l_max = float(losses.min()), float(losses.max())
        if l_max - l_min < 1e-10:
            return np.ones(n) / n
        normalized = (losses - l_min) / (l_max - l_min) 
        log_w = -normalized / self._kde_tau
        log_w -= _logsumexp(log_w)
        return np.exp(log_w)

    def _kde_bandwidth(self, pi: dict) -> float:
        """Ширина полосы KDE как доля диапазона параметра (3%)."""
        return 0.03 * pi["range"]

    def _sample_continuous(self, x_j: float, pi: dict,
                           state: HMMState) -> float:
        """Семплирует x'_j для непрерывного параметра (3-компонентная смесь).

        Компоненты:
            1. narrow:  TN(x' | current_x, σ²)
            2. KDE:     TN(x' | archive_center_i, h²), center выбран по Boltzmann-весам
            3. wide:    TN(x' | current_x, σ_wide²)
        """
        w_narrow, w_kde, w_wide = self.FLOAT_WEIGHTS[state]
        lo, hi = pi["lo"], pi["hi"]
        name = pi["name"]

        r = np.random.rand()
        if r < w_narrow:
            sigma = pi["sigma"]
            a, b = (lo - x_j) / sigma, (hi - x_j) / sigma
            return _tn_rvs(a, b, loc=x_j, scale=sigma)
        elif r < w_narrow + w_kde and len(self._archive) >= 3:
            weights = self._archive_boltzmann_weights()
            idx = int(np.random.choice(len(self._archive), p=weights))
            center = float(self._archive[idx][0][name])
            h = self._kde_bandwidth(pi)
            a, b = (lo - center) / h, (hi - center) / h
            return _tn_rvs(a, b, loc=center, scale=h)
        else:
            sigma_wide = pi["sigma_wide"]
            a, b = (lo - x_j) / sigma_wide, (hi - x_j) / sigma_wide
            return _tn_rvs(a, b, loc=x_j, scale=sigma_wide)

    def _sample_integer(self, x_j: int, pi: dict,
                        state: HMMState) -> int:
        """Семплирует x'_j для целочисленного параметра.

        1. w_local:  U_disc(x_j-1, x_j+1)
        2. w_kde:    KDE из архива (round)
        3. w_global: U_disc(lo, hi)
        """
        w_local, w_kde, w_global = self.INT_WEIGHTS[state]
        lo, hi = pi["lo"], pi["hi"]
        name = pi["name"]

        r = np.random.rand()
        if r < w_local:
            candidates = [v for v in [x_j - 1, x_j, x_j + 1] if lo <= v <= hi]
            return int(np.random.choice(candidates))
        elif r < w_local + w_kde and len(self._archive) >= 3:
            weights = self._archive_boltzmann_weights()
            idx = int(np.random.choice(len(self._archive), p=weights))
            center = float(self._archive[idx][0][name])
            h = max(self._kde_bandwidth(pi), 1.0)
            val = _tn_rvs((lo - 0.5 - center) / h, (hi + 0.5 - center) / h,
                          loc=center, scale=h)
            return int(np.clip(round(val), lo, hi))
        else:
            return int(np.random.randint(lo, hi + 1))

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

        p_boltz = self._boltzmann_probs(name, categories)

        pmf = np.zeros(C)
        for i, cat in enumerate(categories):
            stay = w_stay if cat == x_j else 0.0
            uniform = w_uniform / C
            history = w_history * p_boltz[i]
            pmf[i] = stay + uniform + history

        pmf_sum = pmf.sum()
        if pmf_sum > 0:
            pmf /= pmf_sum
        else:
            pmf = np.ones(C) / C

        idx = np.random.choice(C, p=pmf)
        return categories[idx]

    def _log_q_continuous(self, x_from: float, x_to: float,
                          pi: dict, state: HMMState) -> float:
        """log q_j для непрерывного параметра (3-комп. смесь через logsumexp).

        q_j = w_narrow·TN(x'|x,σ²) + w_kde·Σ_i boltz_w_i·TN(x'|center_i, h²) + w_wide·TN(x'|x,σ_wide²)
        """
        w_narrow, w_kde, w_wide = self.FLOAT_WEIGHTS[state]
        lo, hi = pi["lo"], pi["hi"]
        name = pi["name"]

        log_components = []

        if w_narrow > 0:
            sigma = pi["sigma"]
            a, b = (lo - x_from) / sigma, (hi - x_from) / sigma
            log_g = _tn_logpdf(x_to, a, b, loc=x_from, scale=sigma)
            log_components.append(np.log(w_narrow + self._EPS) + log_g)

        if w_kde > 0 and len(self._archive) >= 3:
            bw = self._archive_boltzmann_weights()
            h = self._kde_bandwidth(pi)
            n_use = min(len(self._archive), 50)  
            kde_logpdfs = []
            for idx in range(n_use):
                center = float(self._archive[idx][0][name])
                a, b = (lo - center) / h, (hi - center) / h
                log_tn = _tn_logpdf(x_to, a, b, loc=center, scale=h)
                kde_logpdfs.append(np.log(bw[idx] + self._EPS) + log_tn)
            log_kde = _logsumexp(np.array(kde_logpdfs))
            log_components.append(np.log(w_kde + self._EPS) + log_kde)

        if w_wide > 0:
            sigma_wide = pi["sigma_wide"]
            a, b = (lo - x_from) / sigma_wide, (hi - x_from) / sigma_wide
            log_g_wide = _tn_logpdf(x_to, a, b, loc=x_from, scale=sigma_wide)
            log_components.append(np.log(w_wide + self._EPS) + log_g_wide)

        return float(_logsumexp(np.array(log_components)))

    def _log_q_integer(self, x_from: int, x_to: int,
                       pi: dict, state: HMMState) -> float:
        """log q_j для целочисленного параметра (3-комп. смесь через logsumexp).

        q_j = w_local·U(x'|x±1) + w_kde·Σ_i boltz_w_i·P_int(x'|center_i,h) + w_global·U(x'|lo..hi)
        """
        w_local, w_kde, w_global = self.INT_WEIGHTS[state]
        lo, hi = pi["lo"], pi["hi"]
        name = pi["name"]

        local_candidates = [v for v in [x_from - 1, x_from, x_from + 1] if lo <= v <= hi]
        n_local = len(local_candidates)
        n_global = hi - lo + 1

        log_components = []

        if x_to in local_candidates:
            log_components.append(np.log(w_local + self._EPS) - np.log(n_local))
        else:
            log_components.append(-np.inf)

        if w_kde > 0 and len(self._archive) >= 3:
            bw = self._archive_boltzmann_weights()
            h = max(self._kde_bandwidth(pi), 1.0)
            n_use = min(len(self._archive), 50)
            p_kde = 0.0
            for idx in range(n_use):
                center = float(self._archive[idx][0][name])
                p_val = _std_cdf((x_to + 0.5 - center) / h) - _std_cdf((x_to - 0.5 - center) / h)
                p_kde += bw[idx] * max(p_val, 0.0)
            if p_kde > 1e-30:
                log_components.append(np.log(w_kde + self._EPS) + np.log(p_kde))
            else:
                log_components.append(-np.inf)

        log_components.append(np.log(w_global + self._EPS) - np.log(n_global))

        return float(_logsumexp(np.array(log_components)))

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

        p_boltz = self._boltzmann_probs(name, categories)
        idx_to = pi["val_to_idx"][x_to]

        log_components = []
        if x_to == x_from:
            log_components.append(np.log(w_stay + self._EPS))
        else:
            log_components.append(-np.inf)
        log_components.append(np.log(w_uniform / C + self._EPS))
        log_components.append(np.log(w_history * p_boltz[idx_to] + self._EPS))

        return float(_logsumexp(np.array(log_components)))

    def _boltzmann_probs(self, param_name: str,
                         categories: list) -> np.ndarray:
        """Boltzmann-вероятности категорий по p10-квантилю loss.

        Args:
            param_name: имя категориального параметра
            categories: допустимые значения

        Returns:
            массив вероятностей формы (C,)
        """
        C = len(categories)
        history = self._category_history[param_name]

        scores = np.zeros(C)
        observed = np.zeros(C, dtype=bool)

        for i, cat in enumerate(categories):
            losses = history[cat]
            n = len(losses)
            if n >= 4:
                scores[i] = np.percentile(losses, 10)
                observed[i] = True
            elif n >= 1:
                scores[i] = min(losses)
                observed[i] = True

        if not np.any(observed):
            return np.ones(C) / C

        best_score = np.min(scores[observed])
        scores[~observed] = best_score - 0.3 * (abs(best_score) + self._EPS)

        s_min = scores.min()
        s_max = scores.max()
        s_range = s_max - s_min

        if s_range < self._EPS:
            return np.ones(C) / C

        normalized = (scores - s_min) / s_range  

        tau = self.temperature + self._EPS
        neg_scaled = -normalized / tau

        log_Z = _logsumexp(neg_scaled)
        log_probs = neg_scaled - log_Z
        probs = np.exp(log_probs)

        prob_sum = probs.sum()
        if prob_sum < self._EPS:
            return np.ones(C) / C
        probs /= prob_sum

        return probs

class MCMCChain:
    """Одна цепь Метрополиса — Гастингса с HMM-управлением.

    Критерий MH:
        α = min(1, exp(log_likelihood_ratio + log_hastings_ratio))
    """

    def __init__(self, chain_id: int, x0: dict, loss0: float,
                 proposal_gen: FactorizedProposalGenerator,
                 hmm: HMMController,
                 T_mcmc: float = 1.0, T_min: float | None = None,
                 scale_factor: float = 1.0,
                 p_cat_step: float = 0.30,
                 anneal_T: bool = True):
        """Инициализирует одну MCMC-цепь.

        Args:
            chain_id: идентификатор цепи
            x0: начальная конфигурация
            loss0: начальный loss
            proposal_gen: генератор предложений
            hmm: локальный HMM-контроллер
            T_mcmc: температура MH
            T_min: минимальная температура при отжиге
            scale_factor: масштаб нормализации loss
            p_cat_step: вероятность шага только по категориальным параметрам
            anneal_T: включить температурное затухание
        """
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
        self.p_cat_step = p_cat_step
        self.anneal_T = anneal_T

        self.proposal_gen = proposal_gen
        self.hmm = hmm
        self.state = HMMState.EXPLORE

        self._observations: list[float] =[]
        self._rejection_streak: int = 0
        self._stagnation_limit: int = 10

    def reset_at(self, x0: dict, loss0: float):
        """Перезапускает цепь в новой точке (для обрезки и клонирования)."""
        self.current_x = copy.deepcopy(x0)
        self.current_loss = loss0
        self.best_x = copy.deepcopy(x0)
        self.best_loss = loss0
        self.hmm.reset()
        self.state = HMMState.EXPLORE
        
        self._observations =[]
        self._rejection_streak = 0

    def step(self, objective_func, progress: float = 0.0, is_burnin: bool = False) -> tuple[dict, float]:
        """Один шаг MH с температурным затуханием."""
        
        if is_burnin:
            self.T_mcmc = self.T_mcmc_init
        elif self.anneal_T:
            self.T_mcmc = self.T_min + (self.T_mcmc_init - self.T_min) * (1.0 - progress)
        else:
            self.T_mcmc = self.T_mcmc_init

        if len(self._observations) >= 2:
            self.state = self.hmm.observe(self._observations[-self.hmm.window:])
        else:
            self.state = self.hmm.state

        if not is_burnin and self._rejection_streak >= self._stagnation_limit:
            if self.state != HMMState.TRAPPED:
                self.state = HMMState.TRAPPED
                self.hmm.force_state(HMMState.TRAPPED)
            n_over = self._rejection_streak - self._stagnation_limit + 1
            boost_factor = 1.0 + min(n_over * 3.0, 50.0)
            self.T_mcmc = self.T_mcmc_init * boost_factor

        cat_only = np.random.rand() < self.p_cat_step
        x_prime = self.proposal_gen.generate_proposal(
            self.current_x, self.state, categorical_only=cat_only
        )

        loss_prime = objective_func(x_prime)
        self.current_loss_old_for_hmm = self.current_loss

        if self.state == HMMState.EXPLORE:
            alpha = 1.0 if loss_prime <= self.current_loss else 0.0
        else:
            alpha = self._acceptance_probability(x_prime, loss_prime, self.T_mcmc)

        accepted = False
        if np.random.rand() < alpha:
            self.current_x = copy.deepcopy(x_prime)
            self.current_loss = loss_prime
            self._rejection_streak = 0
            accepted = True
        else:
            self._rejection_streak += 1

        delta_loss = loss_prime - self.current_loss_old_for_hmm
        O_t = float(delta_loss / (self.scale_factor + 1e-8))

        self._observations.append(O_t)

        if loss_prime < self.best_loss:
            self.best_x = copy.deepcopy(x_prime)
            self.best_loss = loss_prime

        return x_prime, loss_prime, accepted

    def _acceptance_probability(self, x_prime: dict, loss_prime: float,
                                 T_effective: float) -> float:
        """Вычисляет вероятность принятия α по формуле MH.

        Дельта лосса нормализуется на scale_factor → T_mcmc задаётся в безразмерных
        единицах (доля наблюдаемого разброса) и инвариантен к масштабу функции.
        """
        delta_normalized = (loss_prime - self.current_loss) / (self.scale_factor + 1e-8)
        log_likelihood_ratio = -delta_normalized / (T_effective + 1e-100)

        log_q_reverse = self.proposal_gen.log_proposal_density(
            x_from=x_prime, x_to=self.current_x, state=self.state
        )
        log_q_forward = self.proposal_gen.log_proposal_density(
            x_from=self.current_x, x_to=x_prime, state=self.state
        )
        log_hastings_ratio = log_q_reverse - log_q_forward

        log_alpha = log_likelihood_ratio + log_hastings_ratio
        
        return float(np.exp(min(log_alpha, 0.0)))

class GlobalOrchestrator:
    """Макро-уровневый оркестратор: обрезка, клонирование, перезапуск.

    Каждые E шагов:
        1. Сбор глобальной статистики категорий.
        2. Обрезка и клонирование: убить TRAPPED-цепи, клонировать лучшую.
        3. Перезапуск: при глобальном коллапсе перезапустить часть цепей.
    """

    def __init__(self, chains: list[MCMCChain],
                 proposal_gen: FactorizedProposalGenerator,
                 sobol_init: SobolInitializer,
                 dict_to_optimize: dict,
                 objective_func,
                 clone_noise: float = 0.05,
                 collapse_threshold: float = 0.01,
                 reseed_fraction: float = 0.3):
        """Инициализирует глобальный оркестратор цепей.

        Args:
            clone_noise: σ шума при клонировании (как доля диапазона параметра).
            collapse_threshold: порог для детекции «коллапса роя»
                (все цепи слишком близко друг к другу по loss).
            reseed_fraction: доля цепей, перезапускаемых при перезапуске роя.
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
                if np.random.rand() < 0.2 and len(values) > 1:
                    choices = [v for v in values if v != val]
                    noisy[name] = np.random.choice(choices)
                else:
                    noisy[name] = val
            else:
                noisy[name] = val
        return noisy

    def _detect_collapse(self) -> bool:
        """Детектирует глобальный коллапс: все лучшие loss слишком близки."""
        losses = [c.best_loss for c in self.chains]
        if len(losses) < 2:
            return False
        spread = np.std(losses) / (np.abs(np.mean(losses)) + 1e-30)
        return spread < self.collapse_threshold

class HMM_MCMC:
    """H-MCMC-FMP: иерархический MCMC для оптимизации гиперпараметров.

        Args:
            objective_func: целевая функция (минимизация)
            budget: количество вызовов целевой функции
            dict_to_optimize: пространство параметров
            n_init: число стартовых точек Соболя
            n_chains: число параллельных MCMC-цепей
            orchestrate_every: частота оркестрации (обрезка/клонирование)
            T_mcmc: температура MH
            sigma_fraction: σ локального шага (доля диапазона)
            temperature: τ для Boltzmann по категориям

        Attributes:
            data: история (config, score)
            history_table: таблица метрик по шагам

        Пример::

            hmm = HMM_MCMC(objective_func=f, budget=200, dict_to_optimize=space)
            best_config, best_score = hmm.main_loop()
    """

    def __init__(self, objective_func, budget: int, dict_to_optimize: dict,
                 n_init: int = 16, n_chains: int = 4,
                 orchestrate_every: int = 5, T_mcmc: float = 1.0,
                 T_min: float | None = None, anneal_T: bool = True,
                 burnin_fraction: float = 0.10,
                 sigma_fraction: float = 0.10, temperature: float = 0.60,
                 hmm_window: int = 8, hmm_obs_epsilon: float = 1e-8,
                 hmm_lambda_noise: float = 0.01, clone_noise: float = 0.05,
                 wide_sigma_fraction: float = 0.40,
                 p_cat_step: float = 0.30,
                 kde_tau: float = 0.05):
        """Инициализирует H-MCMC-FMP.

        Args:
            objective_func: целевая функция (минимизация)
            budget: бюджет вызовов целевой функции
            dict_to_optimize: пространство гиперпараметров
            n_init: число точек инициализации Соболя
            n_chains: число параллельных цепей
            orchestrate_every: период оркестрации
            T_mcmc: начальная температура MH
            T_min: минимальная температура
            anneal_T: включить температурный отжиг
            burnin_fraction: доля разогрева (burn-in)
            sigma_fraction: σ узкого шага
            temperature: τ для категориальных параметров
            hmm_window: окно наблюдений Витерби
            hmm_obs_epsilon: ε в log-ratio наблюдений
            hmm_lambda_noise: вес шума в эмиссии HMM
            clone_noise: σ шума при клонировании
            wide_sigma_fraction: σ широкого шага
            p_cat_step: вероятность категориального шага
            kde_tau: τ для Archive-KDE
        """
        self.dict_to_optimize = dict_to_optimize
        self._log_param_names = {
            name
            for name, info in dict_to_optimize.items()
            if info.get("type") == "float" and info.get("log")
        }
        self._raw_objective_func = objective_func
        self.objective_func = self._wrap_objective(objective_func)
        self.budget = budget
        self.n_init = n_init
        self.n_chains = n_chains
        self.orchestrate_every = orchestrate_every
        self.T_mcmc = T_mcmc
        self.T_min = T_min
        self.anneal_T = anneal_T
        self.burnin_fraction = burnin_fraction
        self._wide_sigma_fraction = wide_sigma_fraction
        self._p_cat_step = p_cat_step
        self._kde_tau = kde_tau

        self._sigma_fraction = sigma_fraction
        self._temperature = temperature
        self._hmm_window = hmm_window
        self._hmm_obs_epsilon = hmm_obs_epsilon
        self._hmm_lambda_noise = hmm_lambda_noise
        self._clone_noise = clone_noise

        self.data: list[tuple[dict, float]] = []
        self._chains: list[MCMCChain] = []
        self._proposal_gen: FactorizedProposalGenerator | None = None
        self._sobol_init: SobolInitializer | None = None
        self._orchestrator: GlobalOrchestrator | None = None
        self.history_table: list[dict] = []

    def decode_config(self, cfg: dict) -> dict:
        """Декодирует внутренний конфиг (log10 θ) в пользовательские значения."""
        return decode_config(cfg, self.dict_to_optimize)

    def _wrap_objective(self, objective_func):
        if not self._log_param_names:
            return objective_func

        def wrapped(cfg: dict) -> float:
            return objective_func(self.decode_config(cfg))

        return wrapped

    def _decode_best(self, result: tuple[dict, float]) -> tuple[dict, float]:
        cfg, loss = result
        return self.decode_config(cfg), loss

    def reset(self):
        """Сброс для повторного запуска (run_n_experiments)."""
        self.data = []
        self._chains = []
        self._proposal_gen = None
        self._sobol_init = None
        self._orchestrator = None
        self.history_table = []

    @staticmethod
    def _compute_scale(losses: list[float]) -> float:
        """Робастный масштаб нормализации loss (q90−q10 или max−min)."""
        n = len(losses)
        if n == 0:
            return 1.0
        if n == 1:
            return max(abs(losses[0]), 1.0)
        arr = np.array(losses)
        if n >= 10:
            q90, q10 = np.percentile(arr, [90, 10])
            scale = q90 - q10
        else:
            scale = float(arr.max() - arr.min())
        return max(scale, 1e-5)

    def main_loop(self):
        """Основной цикл H-MCMC-FMP.

        Фаза I:  Инициализация Соболя → оценка → отбор K лучших → старт цепей.
        Фаза II: Параллельный MCMC (каждая цепь: HMM → proposal → MH).
        Фаза III: Каждые E шагов — оркестрация (обрезка, клонирование, перезапуск).

        Returns:
            (best_config, best_loss)
        """
        self._sobol_init = SobolInitializer(self.dict_to_optimize)
        self._proposal_gen = FactorizedProposalGenerator(
            self.dict_to_optimize,
            sigma_fraction=self._sigma_fraction,
            temperature=self._temperature,
            wide_sigma_fraction=self._wide_sigma_fraction,
            kde_tau=self._kde_tau,
        )

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
            return self._decode_best(min(self.data, key=lambda x: x[1]))

        losses = [score for _, score in init_scores]
        scale_factor = self._compute_scale(losses)

        scaled_T_mcmc = self.T_mcmc
        scaled_T_min = self.T_min if self.T_min is not None else None

        self._proposal_gen.temperature = self._temperature

        init_scores.sort(key=lambda x: x[1])
        k = min(self.n_chains, len(init_scores))
        best_inits = init_scores[:k]

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
                p_cat_step=self._p_cat_step,
                anneal_T=self.anneal_T,
            )
            self._chains.append(chain)

        self._orchestrator = GlobalOrchestrator(
            chains=self._chains,
            proposal_gen=self._proposal_gen,
            sobol_init=self._sobol_init,
            dict_to_optimize=self.dict_to_optimize,
            objective_func=self.objective_func,
            clone_noise=self._clone_noise,
        )

        step_counter = 0

        while len(self.data) < self.budget:
            for chain in self._chains:
                if len(self.data) >= self.budget:
                    break

                progress = len(self.data) / self.budget
                is_burnin = progress < self.burnin_fraction
                cfg, loss, accepted = chain.step(self.objective_func, progress=progress, is_burnin=is_burnin)
                self.data.append((cfg, loss))
                self._proposal_gen.update_category_history(cfg, loss)
                
                current_state_name = chain.state.name
                
                self.history_table.append({
                    "Eval": len(self.data),
                    "Chain": chain.chain_id,
                    "State": current_state_name,  
                    "Loss": float(loss),
                    "Accepted": accepted,
                    "Rej_Streak": chain._rejection_streak
                })
                
                pbar.update(1)

            step_counter += 1

            if step_counter % self.orchestrate_every == 0:
                if len(self.data) < self.budget:
                    new_evals = self._orchestrator.orchestrate(self.data, budget=self.budget)
                    for cfg, loss in new_evals:
                        if len(self.data) >= self.budget:
                            break
                        self.data.append((cfg, loss))
                        
                        self.history_table.append({
                            "Eval": len(self.data),
                            "Chain": "ORCHESTRATOR",
                            "State": "RESET/CLONE",
                            "Loss": float(loss),
                            "Accepted": True,
                            "Rej_Streak": 0
                        })
                        pbar.update(1)

                if len(self.data) / self.budget < 0.5:
                    all_losses = [s for _, s in self.data]
                    scale_factor = self._compute_scale(all_losses)
                    for chain in self._chains:
                        chain.scale_factor = scale_factor

        pbar.close()
        try:
            import pandas as pd
            df = pd.DataFrame(self.history_table)
            print("\n" + "="*60)
            print("HMM MCMC State History (last run):")
            with pd.option_context('display.max_rows', 200, 'display.max_columns', None):
                print(df)
            print("="*60 + "\n")
        except ImportError:
            pass

        return self._decode_best(min(self.data, key=lambda x: x[1]))
