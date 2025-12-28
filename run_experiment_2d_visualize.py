import yaml
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, List, Tuple, Optional

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO, TRPO, RecurrentPPO

from hpo_rl.core.factory import build_backend, build_env
from hpo_rl.core.register import initialize_framework

initialize_framework()

AGENT_REGISTRY = {
    "A2C": A2C,
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "TRPO": TRPO,
    "MaskablePPO": MaskablePPO,
    "RecurrentPPO": RecurrentPPO,
}


def generate_benchmark_hp_space(backend) -> Dict[str, Any]:
    hp_space = {}
    if hasattr(backend, 'dimensions') and hasattr(backend, 'bounds'):
        dims = backend.dimensions
        bounds = backend.bounds
        for i in range(dims):
            hp_space[f"x{i}"] = {
                "continuous": {"range": [float(bounds[0]), float(bounds[1])]}
            }
    return hp_space


def collect_trajectory(agent, env, num_episodes: int = 1, eval_seed: Optional[int] = None) -> List[Tuple[float, float, float]]:
    """
    Собирает траекторию поиска: (x0, x1, metric)
    Собирает все точки, которые посещает агент (current_config), а не только лучшие
    
    Args:
        eval_seed: Seed для инициализации стартовой точки (None = случайный)
    """
    trajectory = []
    step_sizes_used = []  # Для статистики
    vec_env = agent.get_env()
    
    for episode in range(num_episodes):
        # Устанавливаем seed для стартовой точки если указан
        if eval_seed is not None:
            vec_env.seed(eval_seed + episode)
        
        # VecEnv.reset() возвращает только наблюдения, не кортеж
        obs = vec_env.reset()
        done = False
        step_count = 0
        
        # Получаем начальную точку из первого шага
        # (так как reset() не возвращает info в VecEnv)
        action, _states = agent.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        
        # Обрабатываем info из первого шага
        if isinstance(infos, (list, tuple)) and len(infos) > 0:
            info = infos[0]
        else:
            info = infos if isinstance(infos, dict) else {}
        
        # Получаем начальную конфигурацию
        current_config = info.get("current_config", info.get("best_config", {}))
        current_metric = info.get("current_metric", info.get("best_metric", -float('inf')))
        
        if len(current_config) >= 2:
            x0 = current_config.get("x0", 0.0)
            x1 = current_config.get("x1", 0.0)
            trajectory.append((x0, x1, current_metric))
        
        done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
        
        while not done:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
            
            # Обрабатываем info
            if isinstance(infos, (list, tuple)) and len(infos) > 0:
                info = infos[0]
            else:
                info = infos if isinstance(infos, dict) else {}
            
            # Получаем текущую конфигурацию (все точки, которые посещает агент)
            current_config = info.get("current_config", info.get("best_config", {}))
            current_metric = info.get("current_metric", info.get("best_metric", -float('inf')))
            
            # Собираем статистику по размерам шагов
            step_size = info.get("step_size", 0)
            if step_size > 0:
                step_sizes_used.append(step_size)
            
            if len(current_config) >= 2:
                x0 = current_config.get("x0", 0.0)
                x1 = current_config.get("x1", 0.0)
                trajectory.append((x0, x1, current_metric))
            
            step_count += 1
            if step_count > 1000:  # Защита от бесконечного цикла
                break
    
    # Выводим статистику по размерам шагов
    if step_sizes_used:
        from collections import Counter
        step_size_counts = Counter(step_sizes_used)
        print(f"\n=== Статистика размеров шагов ===")
        print(f"Всего шагов: {len(step_sizes_used)}")
        print(f"Средний размер шага: {np.mean(step_sizes_used):.2f}")
        print(f"Медианный размер шага: {np.median(step_sizes_used):.2f}")
        print(f"Распределение шагов:")
        for size, count in sorted(step_size_counts.items(), reverse=True):
            percentage = (count / len(step_sizes_used)) * 100
            print(f"  Размер {size:2d}: {count:4d} раз ({percentage:5.1f}%)")
    
    return trajectory


def visualize_2d_trajectory(backend, trajectory: List[Tuple[float, float, float]], 
                           save_path: str = "trajectory_2d.png"):
    """
    Визуализирует траекторию поиска на 2D плоскости с контурными линиями функции
    """
    if len(trajectory) == 0:
        print("Траектория пуста, нечего визуализировать")
        return
    
    # Извлекаем координаты
    x0_vals = [t[0] for t in trajectory]
    x1_vals = [t[1] for t in trajectory]
    # metrics содержит reward, нужно конвертировать в реальные значения функции
    rewards = [t[2] for t in trajectory]
    if backend.maximize:
        metrics = rewards  # reward = value
    else:
        metrics = [-r for r in rewards]  # reward = -value, значит value = -reward
    
    # Создаем сетку для контурных линий
    bounds = backend.bounds
    x0_range = np.linspace(bounds[0], bounds[1], 100)
    x1_range = np.linspace(bounds[0], bounds[1], 100)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    # Вычисляем значения функции на сетке (реальные значения, не reward)
    # Нужно получить реальное значение функции, а не reward
    Z = np.zeros_like(X0)
    for i in range(X0.shape[0]):
        for j in range(X0.shape[1]):
            config = {"x0": X0[i, j], "x1": X1[i, j]}
            # Получаем reward из backend
            reward = backend.evaluate(config)
            # Конвертируем обратно в значение функции
            if backend.maximize:
                # Если maximize=True, то reward = value (без инверсии)
                Z[i, j] = reward
            else:
                # Если maximize=False, то reward = -value, значит value = -reward
                Z[i, j] = -reward
    
    # Создаем фигуру с двумя подграфиками
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Подграфик 1: Контурные линии + траектория
    contour = ax1.contour(X0, X1, Z, levels=20, cmap='viridis', alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.contourf(X0, X1, Z, levels=20, cmap='viridis', alpha=0.3)
    
    # Рисуем траекторию
    ax1.plot(x0_vals, x1_vals, 'r-', linewidth=2, alpha=0.7, label='Траектория')
    ax1.scatter(x0_vals[0], x1_vals[0], c='green', s=100, marker='o', 
                label='Начало', zorder=5, edgecolors='black', linewidths=2)
    ax1.scatter(x0_vals[-1], x1_vals[-1], c='red', s=100, marker='*', 
                label='Конец', zorder=5, edgecolors='black', linewidths=2)
    
    # Отмечаем лучшую точку (для минимизации - минимальное значение)
    best_idx = np.argmin(metrics)
    ax1.scatter(x0_vals[best_idx], x1_vals[best_idx], c='yellow', s=150, 
                marker='X', label='Лучшая точка', zorder=5, edgecolors='black', linewidths=2)
    
    ax1.set_xlabel('x0', fontsize=12)
    ax1.set_ylabel('x1', fontsize=12)
    ax1.set_title('Траектория поиска на контурной карте функции (минимизация)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Подграфик 2: 3D поверхность + траектория
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X0, X1, Z, cmap='viridis', alpha=0.6, 
                           linewidth=0, antialiased=True)
    
    # Рисуем траекторию в 3D
    ax2.plot(x0_vals, x1_vals, metrics, 'r-', linewidth=2, alpha=0.8, label='Траектория')
    ax2.scatter(x0_vals[0], x1_vals[0], metrics[0], c='green', s=100, 
                marker='o', label='Начало', edgecolors='black', linewidths=2)
    ax2.scatter(x0_vals[-1], x1_vals[-1], metrics[-1], c='red', s=100, 
                marker='*', label='Конец', edgecolors='black', linewidths=2)
    ax2.scatter(x0_vals[best_idx], x1_vals[best_idx], metrics[best_idx], 
                c='yellow', s=150, marker='X', label='Лучшая точка', 
                edgecolors='black', linewidths=2)
    
    ax2.set_xlabel('x0', fontsize=12)
    ax2.set_ylabel('x1', fontsize=12)
    ax2.set_zlabel('Значение функции (минимизация)', fontsize=12)
    ax2.set_title('3D визуализация траектории (минимизация)', fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"График сохранен в {save_path}")
    plt.close()  # Закрываем фигуру, чтобы освободить память


def run_experiment_with_visualization(config: Dict[str, Any], pretrained_model_path: str = None, 
                                     transfer_learning: bool = False, fine_tune_steps: int = 0,
                                     exploration_boost: float = 1.5, eval_seed: Optional[int] = None):
    agent_cfg = config.get('agent')
    agent_name = agent_cfg['name']
    
    log_dir_root = os.path.join("logs", agent_name)
    print(f"Логи будут сохранены в: {log_dir_root}")
    os.makedirs(log_dir_root, exist_ok=True)
    
    print("\n=== Настройка эксперимента ===")
    backend_cfg = config.get('backend')
    backend = build_backend(backend_cfg)
    
    # Проверяем, что задача 2D
    if backend.dimensions != 2:
        print(f"ВНИМАНИЕ: Задача {backend.dimensions}D, визуализация предназначена для 2D")
    
    hp_space = config.get('hp_space', {})
    if not hp_space:
        hp_space = generate_benchmark_hp_space(backend)
    
    env_cfg = config.get('environment')
    
    # При transfer learning синхронизируем параметры среды с исходной моделью
    if transfer_learning and pretrained_model_path:
        if 'params' not in env_cfg:
            env_cfg['params'] = {}
        
        # Определяем reward_mode из имени файла модели
        if 'per_cycle' in pretrained_model_path.lower():
            if env_cfg['params'].get('reward_mode') != 'per_cycle':
                env_cfg['params']['reward_mode'] = 'per_cycle'
                print(f"✓ Автоматически установлен reward_mode='per_cycle' для совместимости с обученной моделью")
        elif env_cfg['params'].get('reward_mode') is None:
            # Если reward_mode не указан, используем per_step по умолчанию
            env_cfg['params']['reward_mode'] = 'per_step'
            print(f"✓ Используется reward_mode='per_step' (по умолчанию)")
        
        # Синхронизируем параметры штрафа, если они не указаны
        # Используем значения из стандартного конфига для PPO
        if env_cfg['params'].get('step_size_penalty_coef') is None:
            env_cfg['params']['step_size_penalty_coef'] = 0.15
            print(f"✓ Автоматически установлен step_size_penalty_coef=0.15")
        if env_cfg['params'].get('adaptive_step_penalty') is None:
            env_cfg['params']['adaptive_step_penalty'] = True
            print(f"✓ Автоматически включен adaptive_step_penalty=True")
    
    # Для TD3 и SAC автоматически переключаем на continuous действия, если не указано явно
    if agent_name in ["TD3", "SAC"] and env_cfg.get('params', {}).get('action_type') != "continuous":
        if 'params' not in env_cfg:
            env_cfg['params'] = {}
        env_cfg['params']['action_type'] = "continuous"
        if 'max_step_bins' not in env_cfg['params']:
            env_cfg['params']['max_step_bins'] = 30  # 10% от 300 бинов
        print(f"ВНИМАНИЕ: {agent_name} требует continuous действия. Автоматически установлено action_type='continuous'")
    
    # Проверяем, нужно ли создавать параллельные среды
    n_envs = agent_cfg.get('params', {}).get('n_envs', 1)
    if n_envs > 1:
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        print(f"Создание {n_envs} параллельных сред...")
        
        def make_env():
            env = build_env(env_cfg, backend=backend, hp_space=hp_space)
            return env
        
        # Используем DummyVecEnv для простоты (SubprocVecEnv может быть быстрее, но сложнее)
        env = DummyVecEnv([make_env for _ in range(n_envs)])
        print(f"✓ Создано {n_envs} параллельных сред")
    else:
        env = build_env(env_cfg, backend=backend, hp_space=hp_space)
    
    # Создаем уникальный путь для логов на основе функции и режима
    function_name = backend_cfg.get('params', {}).get('function_name', 'unknown')
    if transfer_learning:
        if fine_tune_steps > 0:
            log_suffix = f"{function_name}_transfer_finetune{fine_tune_steps}"
        else:
            log_suffix = f"{function_name}_transfer"
    else:
        log_suffix = function_name
    
    log_path = os.path.join(log_dir_root, log_suffix)
    # Monitor wrapper работает по-разному для VecEnv и обычных сред
    if n_envs > 1:
        # Для VecEnv Monitor уже встроен или можно использовать VecMonitor
        from stable_baselines3.common.vec_env import VecMonitor
        env = VecMonitor(env, filename=log_path, info_keywords=("best_metric", "best_config", "current_metric", "current_config"))
    else:
        env = Monitor(env, filename=log_path, info_keywords=("best_metric", "best_config", "current_metric", "current_config"))
    
    print("\n=== Создание/Загрузка агента ===")
    agent_class = AGENT_REGISTRY[agent_name]
    
    # Для TD3 и SAC автоматически используем MultiInputPolicy, если не указано явно
    if agent_name in ["TD3", "SAC"] and agent_cfg.get('policy') != "MultiInputPolicy":
        agent_cfg['policy'] = "MultiInputPolicy"
        print(f"ВНИМАНИЕ: {agent_name} требует MultiInputPolicy для dict observation space. Автоматически установлено policy='MultiInputPolicy'")
    
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Загрузка предобученной модели из: {pretrained_model_path}")
        print(f"Функция для обучения: {backend_cfg.get('params', {}).get('function_name', 'unknown')}")
        agent = agent_class.load(pretrained_model_path, env=env)
        print("✓ Модель успешно загружена")
        
        if transfer_learning:
            print(f"\n=== Режим переноса знаний ===")
            print(f"Тестирование на новой функции: {backend_cfg.get('params', {}).get('function_name', 'unknown')}")
            
            # Увеличиваем exploration для предотвращения переобучения
            # Для PPO нужно получить текущий ent_coef и установить новый через learn()
            original_ent_coef = None
            new_ent_coef = None
            
            # Пытаемся получить текущий ent_coef
            if hasattr(agent, 'ent_coef'):
                if isinstance(agent.ent_coef, float):
                    original_ent_coef = agent.ent_coef
                elif hasattr(agent.ent_coef, 'value'):
                    original_ent_coef = agent.ent_coef.value()
                else:
                    # Пробуем получить из параметров
                    try:
                        original_ent_coef = float(agent.ent_coef)
                    except:
                        original_ent_coef = 0.01  # Значение по умолчанию
            elif hasattr(agent, 'policy') and hasattr(agent.policy, 'ent_coef'):
                if isinstance(agent.policy.ent_coef, float):
                    original_ent_coef = agent.policy.ent_coef
                elif hasattr(agent.policy.ent_coef, 'value'):
                    original_ent_coef = agent.policy.ent_coef.value()
                else:
                    try:
                        original_ent_coef = float(agent.policy.ent_coef)
                    except:
                        original_ent_coef = 0.01
            else:
                # Если ent_coef не найден, используем значение по умолчанию для PPO
                original_ent_coef = 0.01
            
            # Если ent_coef равен 0 или очень мал, устанавливаем базовое значение
            if original_ent_coef is None or original_ent_coef < 1e-6:
                original_ent_coef = 0.01
                print(f"⚠️  ent_coef не найден или равен 0, используем базовое значение: {original_ent_coef:.4f}")
            
            new_ent_coef = original_ent_coef * exploration_boost
            print(f"Увеличена энтропия для exploration: {original_ent_coef:.4f} -> {new_ent_coef:.4f}")
            
            # Увеличиваем learning rate для более быстрой адаптации
            original_lr = None
            if hasattr(agent, 'learning_rate'):
                if isinstance(agent.learning_rate, float):
                    original_lr = agent.learning_rate
                elif hasattr(agent.learning_rate, 'value'):
                    original_lr = agent.learning_rate.value()
                else:
                    try:
                        original_lr = float(agent.learning_rate)
                    except:
                        original_lr = 0.0003  # Значение по умолчанию для PPO
            elif hasattr(agent, 'lr_schedule'):
                # Для PPO learning rate может быть в lr_schedule
                if hasattr(agent.lr_schedule, 'initial_value'):
                    original_lr = agent.lr_schedule.initial_value
                else:
                    original_lr = 0.0003
            else:
                original_lr = 0.0003
            
            new_lr = original_lr * 2.0  # Удваиваем learning rate для адаптации
            print(f"Увеличен learning rate для адаптации: {original_lr:.6f} -> {new_lr:.6f}")
            
            if fine_tune_steps > 0:
                print(f"Дообучение на {fine_tune_steps} шагах с повышенным exploration...")
                
                # Устанавливаем ent_coef перед learn() для PPO
                # Для PPO ent_coef может быть schedule или float
                if hasattr(agent, 'ent_coef'):
                    try:
                        # Если это schedule, устанавливаем новое значение
                        if hasattr(agent.ent_coef, '__call__'):
                            # Это schedule, создаем новую функцию
                            agent.ent_coef = lambda _: new_ent_coef
                        else:
                            agent.ent_coef = new_ent_coef
                    except Exception as e:
                        print(f"⚠️  Не удалось установить ent_coef напрямую: {e}")
                
                # Устанавливаем learning_rate
                if hasattr(agent, 'policy') and hasattr(agent.policy, 'optimizer'):
                    for param_group in agent.policy.optimizer.param_groups:
                        param_group['lr'] = new_lr
                
                # Для PPO можно также обновить через lr_schedule
                if hasattr(agent, 'lr_schedule'):
                    if hasattr(agent.lr_schedule, 'initial_value'):
                        agent.lr_schedule.initial_value = new_lr
                
                agent.learn(
                    total_timesteps=fine_tune_steps, 
                    reset_num_timesteps=False,
                    progress_bar=False
                )
                print("✓ Дообучение завершено")
            else:
                print("Тестирование без дообучения (zero-shot transfer)")
                print("⚠️  Рекомендуется использовать --fine-tune-steps для лучшей адаптации")
    else:
        if pretrained_model_path:
            print(f"ВНИМАНИЕ: Файл модели не найден: {pretrained_model_path}")
            print("Создание новой модели...")
        # Определяем устройство для обучения
        import torch
        device = agent_cfg.get('params', {}).get('device', None)
        if device is None:
            # Автоматически определяем доступность GPU
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"✓ GPU доступен: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("⚠️  GPU недоступен, используется CPU")
        else:
            if device.startswith('cuda') and not torch.cuda.is_available():
                print(f"⚠️  ВНИМАНИЕ: Указан {device}, но GPU недоступен. Используется CPU")
                device = 'cpu'
            else:
                print(f"Используется устройство: {device}")
        
        # Создаем агента с указанным устройством
        agent_params = agent_cfg.get('params', {}).copy()
        agent_params['device'] = device  # Переопределяем device
        # Убираем n_envs из параметров агента, так как env уже создан с нужным количеством сред
        agent_params.pop('n_envs', None)
        
        agent = agent_class(
            policy=agent_cfg['policy'],
            env=env,
            **agent_params
        )
        
        train_steps = config.get('training', {}).get('total_timesteps', 5000)
        print(f"\n=== Обучение ({train_steps} шагов) ===")
        agent.learn(total_timesteps=train_steps)
        
        # Сохраняем модель, если указан путь
        save_path = config.get('save_path')
        if save_path:
            # Если save_path не содержит расширения, добавляем .zip
            if not save_path.endswith('.zip'):
                save_path = save_path + '.zip'
            model_path = os.path.join(log_dir_root, save_path)
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            agent.save(model_path)
            print(f"✓ Модель сохранена в: {model_path}")
    
    print("\n=== Сбор траектории ===")
    if eval_seed is not None:
        print(f"Используется eval_seed={eval_seed} для стартовой точки")
    trajectory = collect_trajectory(agent, env, num_episodes=1, eval_seed=eval_seed)
    print(f"Собрано {len(trajectory)} точек траектории")
    
    if len(trajectory) > 0:
        # Для минимизации ищем минимальное значение функции (не reward)
        rewards = [t[2] for t in trajectory]
        if backend.maximize:
            metrics = rewards
            best_metric = max(metrics)
        else:
            metrics = [-r for r in rewards]  # Конвертируем reward обратно в значение функции
            best_metric = min(metrics)
        
        print(f"Лучшая метрика (значение функции): {best_metric:.6f}")
        if hasattr(backend, 'global_optimum_value'):
            print(f"Глобальный оптимум: {backend.global_optimum_value:.6f}")
            print(f"Отклонение от оптимума: {abs(best_metric - backend.global_optimum_value):.6f}")
        
        print("\n=== Визуализация ===")
        # Создаем уникальное имя файла на основе функции, режима и seed
        function_name = backend_cfg.get('params', {}).get('function_name', 'unknown')
        seed_suffix = f"_seed{eval_seed}" if eval_seed is not None else ""
        if pretrained_model_path:
            if fine_tune_steps > 0:
                save_path = os.path.join(log_dir_root, f"trajectory_{function_name}_transfer_finetune{fine_tune_steps}{seed_suffix}.png")
            else:
                save_path = os.path.join(log_dir_root, f"trajectory_{function_name}_transfer{seed_suffix}.png")
        else:
            save_path = os.path.join(log_dir_root, f"trajectory_{function_name}{seed_suffix}.png")
        visualize_2d_trajectory(backend, trajectory, save_path=save_path)
    else:
        print("Не удалось собрать траекторию")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Визуализация 2D оптимизации с поддержкой transfer learning"
    )
    parser.add_argument("--config", type=str, default="configs/function_2d_test.yaml",
                       help="Путь к конфигурационному файлу")
    parser.add_argument("--pretrained-model", type=str, default=None,
                       help="Путь к предобученной модели для transfer learning")
    parser.add_argument("--transfer-learning", action="store_true",
                       help="Режим переноса знаний (тестирование на новой функции)")
    parser.add_argument("--fine-tune-steps", type=int, default=0,
                       help="Количество шагов для дообучения (0 = zero-shot transfer)")
    parser.add_argument("--exploration-boost", type=float, default=1.5,
                       help="Множитель для увеличения exploration при transfer learning (по умолчанию 1.5)")
    parser.add_argument("--agent", type=str, default=None,
                       help="Имя алгоритма для использования (PPO, TD3, SAC и т.д.). Переопределяет значение из конфига")
    parser.add_argument("--eval-seed", type=int, default=None,
                       help="Seed для выбора стартовой точки при инференсе (None = случайный)")
    
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        
        # Переопределяем алгоритм, если указан в аргументах
        if args.agent:
            if args.agent not in AGENT_REGISTRY:
                print(f"ОШИБКА: Неизвестный алгоритм '{args.agent}'")
                print(f"Доступные алгоритмы: {', '.join(AGENT_REGISTRY.keys())}")
                exit(1)
            if 'agent' not in main_config:
                main_config['agent'] = {}
            main_config['agent']['name'] = args.agent
            print(f"Используется алгоритм: {args.agent}")
        
        if args.pretrained_model and not args.transfer_learning:
            print("ВНИМАНИЕ: Указан --pretrained-model, но не указан --transfer-learning")
            print("Используется режим переноса знаний по умолчанию")
            args.transfer_learning = True
        
        run_experiment_with_visualization(
            main_config, 
            pretrained_model_path=args.pretrained_model,
            transfer_learning=args.transfer_learning,
            fine_tune_steps=args.fine_tune_steps,
            exploration_boost=args.exploration_boost,
            eval_seed=args.eval_seed
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)

