import yaml
from typing import Dict, Any
import argparse
from hpo_rl.core.factory import build_backend, build_env

from stable_baselines3 import A2C, DQN, PPO, SAC
from sb3_contrib import MaskablePPO
from hpo_rl.core.register import initialize_framework
initialize_framework()

# Реестр для внешних компонентов (агентов)
AGENT_REGISTRY = {
    "A2C": A2C,
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
    "MaskablePPO": MaskablePPO,
}


def run_experiment(config: Dict[str, Any]):
    """
    Простой линейный оркестратор для запуска одного эксперимента.

    Args:
        config (Dict[str, Any]): Словарь с полной конфигурацией эксперимента.
    """
    print(yaml.dump(config, default_flow_style=False))

    # Сборка Бэкенда
    print("1. Сборка вычислительного бэкенда...")
    backend_cfg = config.get('backend')
    if not backend_cfg:
        raise ValueError("Секция 'backend' отсутствует в конфигурационном файле.")
    backend = build_backend(backend_cfg)
    print(f"Бэкенд '{backend_cfg['name']}' успешно создан.")

    # Сборка RL-Среды
    print("Сборка RL-среды")
    env_cfg = config.get('environment')
    if not env_cfg:
        raise ValueError("Секция 'environment' отсутствует в конфигурационном файле.")

    # Передаем весь блок environment и уже созданный бэкенд в фабрику
    env = build_env(env_cfg, backend=backend)
    print(f"Среда '{env_cfg['name']}' успешно создана.")

    # Сборка RL-Агента
    print("Сборка RL-агента")
    agent_cfg = config.get('agent')
    if not agent_cfg:
        raise ValueError("Секция 'agent' отсутствует в конфигурационном файле.")

    agent_name = agent_cfg['name']
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Агент '{agent_name}' не зарегистрирован в AGENT_REGISTRY.")

    agent_class = AGENT_REGISTRY[agent_name]

    agent = agent_class(
        policy=agent_cfg['policy'],
        env=env,
        **(agent_cfg.get('params', {}))
    )
    print(f"Агент '{agent_name}' с политикой '{agent_cfg['policy']}' успешно создан.")

    # Запуск обучения
    training_cfg = config.get('training', {})
    total_timesteps = training_cfg.get('total_timesteps', 1000)

    print(f"Запуск обучения на {total_timesteps} шагов...")
    agent.learn(total_timesteps=total_timesteps)
    print("Обучение завершено!")

    # Сохранение модели (опционально)
    save_path = config.get('save_path')
    if save_path:
        print(f"Сохранение обученного агента в {save_path}...")
        agent.save(save_path)
        print("Агент успешно сохранен.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск HPO эксперимента.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dummy_test.yaml",
        help="Путь к YAML-файлу конфигурации эксперимента."
    )
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            main_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ОШИБКА: Файл конфигурации не найден по пути: {args.config}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"ОШИБКА: Не удалось прочитать YAML файл: {e}")
        exit(1)

    run_experiment(main_config)
