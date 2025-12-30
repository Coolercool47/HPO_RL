from typing import Dict, Any, Optional, List
import math
import gymnasium as gym
import numpy as np

from hpo_rl.environments.cycle_move_pipeline import CyclicPipelineEnv
from hpo_rl.backends.base import EvaluationBackend

class PPOPipelineEnv(CyclicPipelineEnv):
    def __init__(
        self,
        hp_space: Dict[str, Any],
        backend: EvaluationBackend,
        num_bins: int = 20,
        max_steps: int = 200,
        step_sizes: Optional[List[int]] = None,
        reward_mode: str = "per_step",  # "per_step" - "Награда" после каждого шага, "per_cycle" - "Награда" после выбора всех параметров
        action_type: str = "discrete",  # "discrete" - Дискретное пространство действий, "continuous" - Непрерывное пространство действий
        max_step_bins: Optional[int] = None,  # Максимальный шаг в бинах для Нерперывного пространства действий (None = без ограничений)
        history_cycles: int = 3  # Сколько ЦИКЛОВ хранить в истории (умножается на num_hyperparams)
    ):
        super().__init__(
            hp_space=hp_space, 
            backend=backend, 
            num_bins=num_bins, 
            max_steps=max_steps, 
            step_sizes=step_sizes, 
            reward_mode=reward_mode, 
            action_type=action_type, 
            max_step_bins=max_step_bins, 
            use_history=False, # Для RecurrentPPO обычно false, так как LSTM сам строит историю
            history_cycles=history_cycles
        )

        # OBSERVATION SPACE
        # Для RecurrentPPO (use_history=False): минимальный observation + prev_reward + prev_action для RL² адаптации
        # Для Feedforward (use_history=True): скользящее окно (reward_history + action_history)
        obs_spaces = {
            "active_param": gym.spaces.Box(low=0, high=1, shape=(self.num_hyperparams,), dtype=np.float32),
            "chosen_values": gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_hyperparams,), dtype=np.float32),
        }
        
        # Для LSTM (RL² подход): prev_reward + prev_action, LSTM сам построит историю
        obs_spaces["prev_reward"] = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_spaces["prev_action"] = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        self.observation_space = gym.spaces.Dict(obs_spaces)

        self.cursor_idx = 0
        self.steps_total = 0
        self.current_indices = np.zeros(self.num_hyperparams, dtype=np.int32)
        
        # История предыдущих значений параметров (из предыдущего цикла)
        self.prev_cycle_indices = np.zeros(self.num_hyperparams, dtype=np.int32)

        self.current_metric = 0.0
        self.prev_reward = 0.0  # Reward за предыдущий шаг (для LSTM)
        self.prev_action = 0.0  # Действие за предыдущий шаг (нормализованное, для LSTM)
        self.best_metric_so_far = -float('inf')
        self.best_config_so_far = {}
        self.final_config_options = {}
        
        # Для отслеживания прогресса (используется только для статистики)
        self.steps_without_improvement = 0  # Счетчик шагов без улучшения
        
        # Для режима per_cycle: накапливаем метрики в течение цикла
        self.cycle_metrics = []  # Метрики за текущий цикл
        self.cycle_start_metric = 0.0  # Метрика в начале цикла
        self.cycle_start_best_metric = 0.0  # Лучшая метрика в начале цикла (для сравнения в конце)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Обрабатываем действие в зависимости от типа
        param_idx = self.cursor_idx
        old_idx = self.current_indices[param_idx]
        
        if self.action_type == "discrete":
            # Дискретные шаги
            action = int(action)
            num_step_sizes = len(self.step_sizes)
            step_size = 0
            if action == num_step_sizes:  # остаться на месте
                delta_idx = 0
                step_size = 0
            elif action < num_step_sizes:  # шаг назад (уменьшить)
                step_size = self.step_sizes[action]
                delta_idx = -step_size
            else:  # шаг вперед (увеличить)
                step_idx = action - num_step_sizes - 1
                step_size = self.step_sizes[step_idx]
                delta_idx = step_size
            
            new_idx = np.clip(old_idx + delta_idx, 0, self.num_bins - 1)
        else:  # continuous
            # Непрерывное действие: нормализованное значение [0, 1]
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
            
            step_size = abs(new_idx - old_idx)  # Для статистики

        # Обновляем индекс если он изменился (для continuous всегда обновляем)
        if new_idx != old_idx or self.action_type == "continuous":
            self.current_indices[param_idx] = new_idx
        
        # Всегда обновляем конфигурацию для info
        self._update_config_from_indices()
        config = self._assemble_config(self.final_config_options)
        
        # Вычисляем reward в зависимости от режима
        reward = 0.0
        
        if new_idx != old_idx or self.action_type == "continuous":
            # Оцениваем новую конфигурацию
            new_metric = self.backend.evaluate(config)
            
            if self.reward_mode == "per_step":
                # ПРОСТАЯ система наград:
                # Награда - это просто разница метрик (относительное улучшение).
                # Без нормализации tanh и логарифмических бонусов, чтобы сохранить масштаб улучшений (regret).
                
                diff = new_metric - self.current_metric
                reward = np.sign(diff) * np.log(np.abs(diff) + 1.0)
                
                
                # Обновляем статистику (но не даем за это доп. награду)
                if new_metric > self.best_metric_so_far:
                    self.best_metric_so_far = new_metric
                    self.best_config_so_far = config.copy()
                    self.steps_without_improvement = 0
                else:
                    self.steps_without_improvement += 1

                self.current_metric = new_metric
            else:  # per_cycle
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
                
                reward = 0.0  # Нет reward до конца цикла
        else:
            # Нет изменения позиции - небольшой штраф за бездействие
            reward = -0.05

        # Нормализуем действие в [-1, 1] с семантикой: отрицательное = назад, положительное = вперёд
        if self.action_type == "discrete":
            # Для дискретных: delta / max_step_size
            max_step = max(self.step_sizes) if self.step_sizes else 1
            normalized_action = float(delta_idx) / max_step  # [-1, 1]
        else:
            # Для continuous: delta / max_possible_step
            actual_delta = new_idx - old_idx
            # Делим на max_step_bins (если задан) или num_bins-1
            max_possible_delta = self.max_step_bins if self.max_step_bins else (self.num_bins - 1)
            normalized_action = float(actual_delta) / max(max_possible_delta, 1)  # [-1, 1]
        
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        
        self.steps_total += 1
        self.cursor_idx = (self.cursor_idx + 1) % self.num_hyperparams
        
        # Проверяем, завершен ли цикл (все параметры выбраны)
        cycle_completed = False
        if self.cursor_idx == 0 and self.steps_total > 0:
            cycle_completed = True
            # Сохраняем текущие значения как предыдущие для следующего цикла
            self.prev_cycle_indices = self.current_indices.copy()
            
            if self.reward_mode == "per_cycle":
                # Выдаем reward на основе улучшения относительно ЛУЧШЕГО найденного значения
                # Не требует априорного знания об оптимуме
                if len(self.cycle_metrics) > 0:
                    cycle_best_metric = max(self.cycle_metrics)
                    
                    # Улучшение относительно лучшего значения НА НАЧАЛО цикла
                    # (best_metric_so_far уже обновлён внутри цикла, поэтому используем сохранённое значение)
                    diff_from_best = cycle_best_metric - self.cycle_start_best_metric
                    
                    if diff_from_best > 0:
                        # Новый глобальный рекорд! Награда пропорциональна улучшению
                        scale_factor = 100.0
                        reward = math.log(1 + diff_from_best * scale_factor) + 1.0
                    else:
                        # Цикл не улучшил глобальный результат
                        reward = -0.05  # Небольшой штраф за цикл без прогресса
                else:
                    # Агент не изменил ни один параметр за цикл - штраф
                    reward = -0.1
                
                # Сбрасываем для следующего цикла
                self.cycle_start_metric = self.current_metric
                self.cycle_start_best_metric = self.best_metric_so_far  # Обновляем для следующего цикла
                self.cycle_metrics = []

        # Обновляем скользящее окно ПОСЛЕ вычисления финального reward (важно для per_cycle!)
        self.reward_history_buffer = np.roll(self.reward_history_buffer, -1)
        # Используем softsign вместо tanh: r / (1 + |r|)
        # Лучше сохраняет различия между большими значениями:
        # softsign(1)=0.5, softsign(2)=0.67, softsign(5)=0.83, softsign(10)=0.91
        normalized_reward = reward / (1.0 + abs(reward))  # softsign, диапазон (-1, 1)
        self.reward_history_buffer[-1] = normalized_reward
        
        self.action_history_buffer = np.roll(self.action_history_buffer, -1)
        self.action_history_buffer[-1] = normalized_action
        
        # Сохраняем для LSTM варианта (RL²)
        self.prev_reward = reward  # Сырое значение, нормализуем в _get_obs
        self.prev_action = normalized_action

        terminated = False
        truncated = self.steps_total >= self.max_steps_limit

        info = self._get_info()
        # Добавляем информацию для диагностики
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