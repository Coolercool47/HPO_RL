from typing import Dict, Any, Optional, List
import math
import gymnasium as gym
import numpy as np

from hpo_rl.backends.base import EvaluationBackend
from hpo_rl.environments.base_env import BaseHPOEnv


class CyclicPipelineEnv(BaseHPOEnv):
    def __init__(
        self,
        hp_space: Dict[str, Any],
        backend: EvaluationBackend,
        num_bins: int = 20,
        max_steps: int = 200,
        sparse_reward: bool = False,
        step_sizes: Optional[List[int]] = None,
        step_size_penalty_coef: float = 0.0,  # Коэффициент штрафа за размер шага
        reward_mode: str = "per_step",  # "per_step" - "Награда" после каждого шага, "per_cycle" - "Награда" после выбора всех параметров
        action_type: str = "discrete",  # "discrete" - Дискретное пространство действий, "continuous" - Непрерывное пространство действий
        max_step_bins: Optional[int] = None,  # Максимальный шаг в бинах для Нерперывного пространства действий (None = без ограничений)
        adaptive_step_penalty: bool = False,  # Если True, штраф за размер шага увеличивается в течении эпизода
        use_history: bool = True,  # Если False, предает минимальую observation для RecurrentPPO
        history_cycles: int = 3  # Сколько циклов хранить в истории
    ):
        super().__init__(hp_space, backend)

        self.num_bins = num_bins
        self.max_steps_limit = max_steps
        self.sparse_reward = sparse_reward
        self.step_size_penalty_coef = step_size_penalty_coef
        self.reward_mode = reward_mode
        self.action_type = action_type
        self.max_step_bins = max_step_bins
        self.adaptive_step_penalty = adaptive_step_penalty
        self.use_history = use_history 
        self.history_cycles = history_cycles
        
        if reward_mode not in ["per_step", "per_cycle"]:
            raise ValueError(f"reward_mode must be 'per_step' or 'per_cycle', got {reward_mode}")
        if action_type not in ["discrete", "continuous"]:
            raise ValueError(f"action_type must be 'discrete' or 'continuous', got {action_type}")
        
        # Если max_step_bins не указан, используем 10% от num_bins по умолчанию
        if self.max_step_bins is None and action_type == "continuous":
            self.max_step_bins = max(1, num_bins // 10)
        
        self.action_grid = np.linspace(0.0, 1.0, num=num_bins, dtype=np.float32)
        self.num_hyperparams = len(self.hp_names)
        
        # Вычисляем длину скользящего окна - это гарантирует, что история содержит целое число циклов
        self.history_len = self.history_cycles * self.num_hyperparams

        self.param_types = []
        for name in self.hp_names:
            param_info = self.hp_space_config[name]
            if 'discrete' in param_info:
                self.param_types.append('discrete')
            elif 'continuous' in param_info:
                self.param_types.append('continuous')

        # ACTION SPACE
        if action_type == "discrete":
            # Действие определяет размер шага относительно текущей позиции
            # Формат: 0..N-1 = шаги назад, N = остаться, N+1..2N = шаги вперед
            if step_sizes is None:
                # По умолчанию используем 4 размера шагов
                self.step_sizes = [
                    max(1, num_bins // 5),
                    max(1, num_bins // 10),
                    max(1, num_bins // 20),
                    1
                ]
            else:
                self.step_sizes = sorted(step_sizes, reverse=True)
            
            self.num_actions = 2 * len(self.step_sizes) + 1  # шаги назад + шаги вперед + остаться
            self.action_space = gym.spaces.Discrete(self.num_actions)
        elif action_type == "continuous":
            # Нормализованное значение [0, 1] для текущего параметра
            self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            self.step_sizes = []

        # OBSERVATION SPACE
        # Для RecurrentPPO (use_history=False): минимальный observation + prev_reward + prev_action для RL² адаптации
        # Для Feedforward (use_history=True): скользящее окно (reward_history + action_history)
        obs_spaces = {
            "active_param": gym.spaces.Box(low=0, high=1, shape=(self.num_hyperparams,), dtype=np.float32),
            "chosen_values": gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_hyperparams,), dtype=np.float32),
        }
        
        if self.use_history:
            # Для Feedforward: скользящее окно для компенсации отсутствия памяти
            obs_spaces["prev_values"] = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_hyperparams,), dtype=np.float32)
            
            # История rewards: sign(reward) * log(1 + |reward|) / log(1 + max_reward)
            # Нормализовано в [-1, 1] с сохранением масштаба
            obs_spaces["reward_history"] = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.history_len,), dtype=np.float32)
            
            # История действий: для discrete — signed_step (направление и размер), для continuous — значение
            # Нормализовано в [-1, 1]: отрицательные = шаг назад, положительные = шаг вперёд, 0 = stay
            obs_spaces["action_history"] = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.history_len,), dtype=np.float32)
        else:
            # Для LSTM (RL² подход): prev_reward + prev_action, LSTM сам построит историю
            obs_spaces["prev_reward"] = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            obs_spaces["prev_action"] = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.observation_space = gym.spaces.Dict(obs_spaces)

        self.cursor_idx = 0 # Индекс текущего оптимизируемого гиперпараметра
        self.steps_total = 0
        self.current_indices = np.zeros(self.num_hyperparams, dtype=np.int32)
        
        # История предыдущих значений параметров из предыдущего цикла
        self.prev_cycle_indices = np.zeros(self.num_hyperparams, dtype=np.int32)

        self.current_metric = 0.0
        self.prev_reward = 0.0  # "Награда" за предыдущий шаг (для LSTM)
        self.prev_action = 0.0  # Действие за предыдущий шаг (для LSTM)
        
        # Буферы скользящего окна
        self.reward_history_buffer = np.zeros(self.history_len, dtype=np.float32)
        self.action_history_buffer = np.zeros(self.history_len, dtype=np.float32)
        
        # Параметрд для отслеживания статистики прогресса
        self.best_metric_so_far = -float('inf')
        self.best_config_so_far = {}
        self.final_config_options = {}
        self.steps_without_improvement = 0
        
        # Для режима per_cycle: накапливаем метрики в течение цикла
        self.cycle_metrics = []
        self.cycle_start_metric = 0.0
        self.cycle_start_best_metric = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.cursor_idx = 0
        self.steps_total = 0
        self.prev_reward = 0.0
        self.prev_action = 0.0
        self.reward_history_buffer.fill(0.0)
        self.action_history_buffer.fill(0.0)

        self.current_indices = self.np_random.integers(0, self.num_bins, size=self.num_hyperparams)
        self.prev_cycle_indices = self.current_indices.copy()

        self._update_config_from_indices()
        config = self._assemble_config(self.final_config_options)
        self.current_metric = self.backend.evaluate(config)

        self.best_metric_so_far = self.current_metric
        self.best_config_so_far = config.copy()
        
        self.steps_without_improvement = 0
        self.cycle_metrics = []
        self.cycle_start_metric = self.current_metric
        self.cycle_start_best_metric = self.best_metric_so_far  # Сохраняем лучшую метрику для per_cycle

        return self._get_obs(), self._get_info()

    def step(self, action):

        # Обрабатываем действие в зависимости от типа пространства действий
        param_idx = self.cursor_idx
        old_idx = self.current_indices[param_idx]
        
        if self.action_type == "discrete":
            action = int(action)
            num_step_sizes = len(self.step_sizes)
            step_size = 0
            if action == num_step_sizes: # Не изменять текущий индекс гиперпараметра
                delta_idx = 0
                step_size = 0

            elif action < num_step_sizes:  # Уменьшить индекс гиперпараметра
                step_size = self.step_sizes[action]
                delta_idx = -step_size

            else:  # Увеличить индекс гиперпараметра
                step_idx = action - num_step_sizes - 1
                step_size = self.step_sizes[step_idx]
                delta_idx = step_size
            
            new_idx = np.clip(old_idx + delta_idx, 0, self.num_bins - 1)
        elif self.action_type == "continuous":
            # Непрерывное действие нормализованное в [0, 1]
            if isinstance(action, np.ndarray):
                normalized_value = float(np.clip(action[0], 0.0, 1.0))
            else:
                normalized_value = float(np.clip(action, 0.0, 1.0))
            
            # Преобразуем нормализованное значение в целевой индекс
            target_idx = int(np.clip(normalized_value * (self.num_bins - 1), 0, self.num_bins - 1))
            
            # Ограничиваем максимальный шаг
            if self.max_step_bins is not None:
                max_step = self.max_step_bins
                delta = target_idx - old_idx
                if abs(delta) > max_step:
                    # Ограничиваем шаг до max_step_bins
                    new_idx = old_idx + np.sign(delta) * max_step
                    new_idx = int(np.clip(new_idx, 0, self.num_bins - 1))
                else:
                    new_idx = target_idx
            else:
                new_idx = target_idx
            
            step_size = abs(new_idx - old_idx)

        if new_idx != old_idx or self.action_type == "continuous":
            self.current_indices[param_idx] = new_idx
        
        self._update_config_from_indices()
        config = self._assemble_config(self.final_config_options)
        
        # Вычисляем reward в зависимости от режима
        reward = 0.0
        
        if new_idx != old_idx or self.action_type == "continuous":
            # Оцениваем новую конфигурацию
            new_metric = self.backend.evaluate(config)
            
            if self.reward_mode == "per_step":
                # Награда - это просто разница метрик.
                
                diff = new_metric - self.current_metric
                reward = np.sign(diff) * np.log(np.abs(diff) + 1.0)

                # Обновляем статистику
                if new_metric > self.best_metric_so_far:
                    self.best_metric_so_far = new_metric
                    self.best_config_so_far = config.copy()
                    self.steps_without_improvement = 0
                else:
                    self.steps_without_improvement += 1

                self.current_metric = new_metric
            elif self.reward_mode == "per_cycle":
                # Сохраняем метрику для цикла, reward будет выдан в конце цикла
                self.cycle_metrics.append(new_metric)
                self.current_metric = new_metric
                
                # Обновляем лучшее значение
                if new_metric > self.best_metric_so_far:
                    self.best_metric_so_far = new_metric
                    self.best_config_so_far = config.copy()
                    self.steps_without_improvement = 0
                else:
                    self.steps_without_improvement += 1
                
                reward = 0.0
        else:
            # Нет изменения позиции - небольшой штраф за бездействие
            reward = -0.05

        # Нормализуем действие в [-1, 1]
        if self.action_type == "discrete":

            max_step = max(self.step_sizes) if self.step_sizes else 1
            normalized_action = float(delta_idx) / max_step
        elif self.action_type == "continuous": 

            actual_delta = new_idx - old_idx
            max_possible_delta = self.max_step_bins if self.max_step_bins else (self.num_bins - 1)
            normalized_action = float(actual_delta) / max(max_possible_delta, 1)
        
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        
        self.steps_total += 1
        self.cursor_idx = (self.cursor_idx + 1) % self.num_hyperparams
        
        # Проверяем, завершен ли цикл
        cycle_completed = False
        if self.cursor_idx == 0 and self.steps_total > 0:
            cycle_completed = True
            # Сохраняем текущие значения как предыдущие для следующего цикла
            self.prev_cycle_indices = self.current_indices.copy()
            
            if self.reward_mode == "per_cycle":
                # Не требует априорного знания об оптимуме
                if len(self.cycle_metrics) > 0:
                    cycle_best_metric = max(self.cycle_metrics)
                    diff_from_best = cycle_best_metric - self.cycle_start_best_metric
                    
                    # Рассчет награды пропорционально улучшению
                    if diff_from_best > 0:
                        scale_factor = 100.0
                        reward = math.log(1 + diff_from_best * scale_factor) + 1.0
                    else:
                        reward = -0.05
                else:
                    # Штраф за отсутствие улучшения глобального результата
                    reward = -0.1
                
                # Сбрасываем для следующего цикла
                self.cycle_start_metric = self.current_metric
                self.cycle_start_best_metric = self.best_metric_so_far  # Обновляем для следующего цикла
                self.cycle_metrics = []

        self.reward_history_buffer = np.roll(self.reward_history_buffer, -1)
        normalized_reward = reward / (1.0 + abs(reward))
        self.reward_history_buffer[-1] = normalized_reward
        
        self.action_history_buffer = np.roll(self.action_history_buffer, -1)
        self.action_history_buffer[-1] = normalized_action
        
        # Сохраняем для LSTM варианта (RL²)
        self.prev_reward = reward
        self.prev_action = normalized_action

        terminated = False
        truncated = self.steps_total >= self.max_steps_limit

        info = self._get_info()
        info['step_size'] = step_size
        info['steps_without_improvement'] = self.steps_without_improvement
        info['cycle_completed'] = cycle_completed

        return self._get_obs(), reward, terminated, truncated, info

    def _update_config_from_indices(self):
        for i, name in enumerate(self.hp_names):
            idx = self.current_indices[i]
            norm_val = self.action_grid[idx]
            param_info = self.hp_space_config[name]

            val = None
            if 'discrete' in param_info:
                opts = param_info['discrete']['choices']
                c_idx = int(np.clip(math.floor(norm_val * len(opts)), 0, len(opts)-1))
                val = opts[c_idx]
            elif 'continuous' in param_info:
                c_info = param_info['continuous']
                l, h = c_info['range']
                if c_info.get('log', False):
                    val = np.exp(np.log(l) + norm_val * (np.log(h) - np.log(l)))
                else:
                    val = l + norm_val * (h - l)
                if c_info.get('type') == 'int':
                    val = int(round(val))

            self.final_config_options[name] = val

    def _get_obs(self) -> Dict[str, np.ndarray]:
        norm_values = self.current_indices.astype(np.float32) / (self.num_bins - 1)

        one_hot = np.zeros(self.num_hyperparams, dtype=np.float32)
        one_hot[self.cursor_idx] = 1.0

        obs = {
            "active_param": one_hot,
            "chosen_values": norm_values,
        }
        
        if self.use_history:

            # Для Feedforward: скользящее окно (rewards + actions)
            prev_norm_values = self.prev_cycle_indices.astype(np.float32) / (self.num_bins - 1)
            obs["prev_values"] = prev_norm_values
            # История rewards (уже нормализованы через softsign в буфере)
            obs["reward_history"] = self.reward_history_buffer.copy()
            # История действий: signed step в [-1, 1]
            obs["action_history"] = self.action_history_buffer.copy()
        else:
            
            # Для LSTM (RL² подход): prev_reward + prev_action для адаптации внутри эпизода
            # Используем softsign для нормализации reward
            normalized_prev_reward = self.prev_reward / (1.0 + abs(self.prev_reward))
            obs["prev_reward"] = np.array([normalized_prev_reward], dtype=np.float32)
            obs["prev_action"] = np.array([self.prev_action], dtype=np.float32)
        
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "best_config": self.best_config_so_far,
            "best_metric": self.best_metric_so_far,
            "current_config": self.final_config_options.copy(),
            "current_metric": self.current_metric
        }