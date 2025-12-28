import yaml
import argparse
import copy
import traceback
import os
from typing import Dict, Any

from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.monitor import Monitor  # <--- ВАЖНО
from sb3_contrib import MaskablePPO, TRPO, RecurrentPPO

from hpo_rl.core.factory import build_backend, build_env
from hpo_rl.core.register import initialize_framework

initialize_framework()

AGENT_REGISTRY = {
    "A2C": A2C,
    "DQN": DQN,
    "PPO": PPO,
    "SAC": SAC,
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


def evaluate_agent_performance(agent, env_config_name: str, num_episodes: int = 1, patience: int = 20):
    print(f"\nОценка агента: {env_config_name} (Patience={patience})")
    vec_env = agent.get_env()
    obs = vec_env.reset()
    best_metric_global = -float('inf')

    for i in range(num_episodes):
        terminated = False
        step_counter = 0
        episode_best_metric = -float('inf')
        steps_without_improvement = 0

        while not terminated:
            action, _states = agent.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            terminated = dones[0]
            info = infos[0]

            current_best_in_env = info.get("best_metric", -float('inf'))

            if current_best_in_env > episode_best_metric + 1e-6:
                episode_best_metric = current_best_in_env
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            if current_best_in_env > best_metric_global:
                best_metric_global = current_best_in_env

            step_counter += 1

            if steps_without_improvement >= patience:
                print(f"Early Stopping: Нет улучшений {patience} шагов.")
                break

        print(f"Эпизод {i+1} завершен за {step_counter} шагов. Лучшее: {episode_best_metric:.4f}")

    print(f"Максимальная награда (метрика): {best_metric_global:.6f}")
    return best_metric_global


def run_experiment(config: Dict[str, Any]):
    agent_cfg = config.get('agent')
    agent_name = agent_cfg['name']

    log_dir_root = os.path.join("logs", agent_name)
    print(f"Логи будут сохранены в: {log_dir_root}")
    os.makedirs(log_dir_root, exist_ok=True)

    print("\nTask A Setup")
    backend_cfg = config.get('backend')
    backend = build_backend(backend_cfg)

    hp_space = config.get('hp_space', {})
    if not hp_space:
        hp_space = generate_benchmark_hp_space(backend)

    env_cfg = config.get('environment')
    env = build_env(env_cfg, backend=backend, hp_space=hp_space)

    log_path_a = os.path.join(log_dir_root, "task_a_rastrigin")
    env = Monitor(env, filename=log_path_a, info_keywords=("best_metric",))

    print("\nCreating Agent")
    agent_class = AGENT_REGISTRY[agent_name]
    agent = agent_class(
        policy=agent_cfg['policy'],
        env=env,
        device='cuda:0',
        **(agent_cfg.get('params', {}))
    )

    train_steps = config.get('training', {}).get('total_timesteps', 5000)
    print(f"Training Task A ({train_steps} steps)")
    agent.learn(total_timesteps=train_steps)
    # agent.save(os.path.join(log_dir_root, "model_task_a"))  # Сохраним модельку на всякий

    evaluate_agent_performance(agent, "Task A Eval", patience=20)

    print("Transfer to Task B")

    backend_cfg_eval = copy.deepcopy(backend_cfg)
    backend_cfg_eval["params"]["function_name"] = "sphere"
    backend_eval = build_backend(backend_cfg_eval)
    hp_space_eval = generate_benchmark_hp_space(backend_eval)

    env_eval = build_env(env_cfg, backend=backend_eval, hp_space=hp_space_eval)

    log_path_b = os.path.join(log_dir_root, "task_b_sphere_finetune")
    env_eval = Monitor(env_eval, filename=log_path_b, info_keywords=("best_metric",))

    agent.set_env(env_eval)

    try:
        new_lr = 0.0005
        for param_group in agent.policy.optimizer.param_groups:
            param_group["lr"] = new_lr
        print(f"LR reduced to {new_lr}")
    except:
        print("Warning: Could not update LR manually (maybe scheduler used).")

    print("Zero-Shot Eval")
    evaluate_agent_performance(agent, "Task B Zero-Shot", patience=50)

    tune_steps = 1000
    print(f"Few-Shot Finetuning ({tune_steps} steps)")
    agent.learn(total_timesteps=tune_steps)

    print("Final Eval")
    evaluate_agent_performance(agent, "Task B Final", patience=100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/new_experiment.yaml")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            main_config = yaml.safe_load(f)
        run_experiment(main_config)
    except Exception as e:
        traceback.print_exc()
        exit(1)
