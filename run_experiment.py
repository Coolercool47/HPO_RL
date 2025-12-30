import yaml
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from stable_baselines3 import A2C, DQN, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO, TRPO, RecurrentPPO

from hpo_rl.core.factory import build_backend, build_env
from hpo_rl.core.register import initialize_framework

initialize_framework()

AGENT_REGISTRY = {
    "A2C": A2C, "DQN": DQN, "PPO": PPO, "SAC": SAC, "TD3": TD3,
    "TRPO": TRPO, "MaskablePPO": MaskablePPO, "RecurrentPPO": RecurrentPPO,
}

RECURRENT_ALGORITHMS = {"RecurrentPPO"}


def generate_benchmark_hp_space(backend) -> Dict[str, Any]:
    hp_space = {}
    if hasattr(backend, 'dimensions') and hasattr(backend, 'bounds'):
        for i in range(backend.dimensions):
            hp_space[f"x{i}"] = {
                "continuous": {"range": [float(backend.bounds[0]), float(backend.bounds[1])]}
            }
    return hp_space


def _extract_point(info: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
    """Извлекает (x0, x1, reward) из info."""
    config = info.get("current_config", {})
    reward = info.get("current_metric", -float('inf'))
    if config and "x0" in config and "x1" in config:
        return (config["x0"], config["x1"], reward)
    return None


def collect_trajectory(agent, env, agent_name: str, num_episodes: int = 1,
                       eval_seed: Optional[int] = None) -> Tuple[List[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    """Собирает траекторию поиска, выбирает эпизод с максимальной наградой."""
    all_results = []
    is_recurrent = agent_name in RECURRENT_ALGORITHMS
    vec_env = agent.get_env()

    for episode in range(num_episodes):
        if eval_seed is not None:
            vec_env.seed(eval_seed + episode)

        obs = vec_env.reset()
        trajectory = []
        best_reward = -float('inf')

        try:
            infos = vec_env.env_method('get_info')
            start_info = infos[0] if infos else {}
        except Exception:
            start_info = {}

        start_point = _extract_point(start_info)
        if start_point:
            trajectory.append(start_point)
            best_reward = start_point[2]

        done = False
        step = 0
        states = None
        final_point = None

        while not done:
            predict_kwargs = {"deterministic": True}
            if is_recurrent:
                predict_kwargs["state"] = states
                predict_kwargs["episode_start"] = np.array([step == 0] * vec_env.num_envs)

            action, states = agent.predict(obs, **predict_kwargs)
            obs, _, dones, infos = vec_env.step(action)
            done, info = dones[0], infos[0]

            point = _extract_point(info)
            if point:
                trajectory.append(point)
                best_reward = max(best_reward, point[2])
                if done:
                    final_point = point

            step += 1
            if step > 1000:
                break

        if final_point is None and trajectory:
            final_point = trajectory[-1]

        all_results.append({
            "trajectory": trajectory,
            "final_point": final_point,
            "best_reward": best_reward
        })

    best_idx = np.argmax([r["best_reward"] for r in all_results])
    best = all_results[best_idx]
    print(f"\nBest episode reward: {best['best_reward']:.6f} (episode {best_idx}/{num_episodes})")

    return best["trajectory"], best["final_point"]


def visualize_2d_trajectory(backend, trajectory: List[Tuple[float, float, float]],
                           eval_mode: str = "best",
                           final_point: Optional[Tuple[float, float, float]] = None,
                           save_path: str = "trajectory_2d.png"):
    """Визуализирует траекторию в 2D и 3D."""
    if not trajectory:
        print("Empty trajectory, skipping visualization")
        return

    x0_vals = [t[0] for t in trajectory]
    x1_vals = [t[1] for t in trajectory]
    rewards = [t[2] for t in trajectory]
    metrics = rewards if backend.maximize else [-r for r in rewards]

    # Сетка для contour/surface
    bounds = backend.bounds
    grid = np.linspace(bounds[0], bounds[1], 100)
    X0, X1 = np.meshgrid(grid, grid)

    Z = np.zeros_like(X0)
    for i in range(X0.shape[0]):
        for j in range(X0.shape[1]):
            val = backend.evaluate({"x0": X0[i, j], "x1": X1[i, j]})
            Z[i, j] = val if backend.maximize else -val

    fig, (ax1, _) = plt.subplots(1, 2, figsize=(16, 6))

    # 2D контур
    contour = ax1.contour(X0, X1, Z, levels=20, cmap='viridis', alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.contourf(X0, X1, Z, levels=20, cmap='viridis', alpha=0.3)

    ax1.plot(x0_vals, x1_vals, 'r-', linewidth=2, alpha=0.7, label='Trajectory')
    ax1.scatter(x0_vals[0], x1_vals[0], c='green', s=100, marker='o',
                label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax1.scatter(x0_vals[-1], x1_vals[-1], c='red', s=100, marker='*',
                label='End', zorder=5, edgecolors='black', linewidths=2)

    # Выделяем результирующую точку
    if eval_mode == "final" and final_point:
        fx, fy, fr = final_point
        fval = fr if backend.maximize else -fr
        ax1.scatter(fx, fy, c='yellow', s=150, marker='X',
                    label='Final (result)', zorder=5, edgecolors='black', linewidths=2)
        marker_coords, marker_value = (fx, fy), fval
    else:
        best_idx = np.argmax(metrics) if backend.maximize else np.argmin(metrics)
        ax1.scatter(x0_vals[best_idx], x1_vals[best_idx], c='yellow', s=150,
                    marker='X', label='Best', zorder=5, edgecolors='black', linewidths=2)
        marker_coords = (x0_vals[best_idx], x1_vals[best_idx])
        marker_value = metrics[best_idx]

    opt_type = "max" if backend.maximize else "min"
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_title(f'Trajectory on contour ({opt_type})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 3D поверхность
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X0, X1, Z, cmap='viridis', alpha=0.6, linewidth=0, antialiased=True)

    ax2.plot(x0_vals, x1_vals, metrics, 'r-', linewidth=2, alpha=0.8, label='Trajectory')
    ax2.scatter(x0_vals[0], x1_vals[0], metrics[0], c='green', s=100,
                marker='o', label='Start', edgecolors='black', linewidths=2)
    ax2.scatter(x0_vals[-1], x1_vals[-1], metrics[-1], c='red', s=100,
                marker='*', label='End', edgecolors='black', linewidths=2)
    ax2.scatter(*marker_coords, marker_value, c='yellow', s=150, marker='X',
                label='Best/Final', edgecolors='black', linewidths=2)

    ax2.set_xlabel('x0')
    ax2.set_ylabel('x1')
    ax2.set_zlabel(f'Value ({opt_type})')
    ax2.set_title(f'3D trajectory ({opt_type})')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def _get_attr(obj, attr, default=None):
    """Безопасно получает атрибут, пробуя разные способы."""
    val = getattr(obj, attr, None)
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return val
    if hasattr(val, 'value'):
        return val.value()
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _get_ent_coef(agent) -> float:
    val = _get_attr(agent, 'ent_coef')
    if val is not None:
        return val
    if hasattr(agent, 'policy'):
        val = _get_attr(agent.policy, 'ent_coef')
        if val is not None:
            return val
    return 0.01


def _get_learning_rate(agent) -> float:
    val = _get_attr(agent, 'learning_rate')
    if val is not None:
        return val
    if hasattr(agent, 'lr_schedule') and hasattr(agent.lr_schedule, 'initial_value'):
        return agent.lr_schedule.initial_value
    return 0.0003


def _set_ent_coef(agent, value: float):
    if hasattr(agent, 'ent_coef'):
        try:
            agent.ent_coef = (lambda _: value) if callable(agent.ent_coef) else value
        except Exception as e:
            print(f"Warning: could not set ent_coef: {e}")


def _set_learning_rate(agent, value: float):
    if hasattr(agent, 'policy') and hasattr(agent.policy, 'optimizer'):
        for pg in agent.policy.optimizer.param_groups:
            pg['lr'] = value
    if hasattr(agent, 'lr_schedule') and hasattr(agent.lr_schedule, 'initial_value'):
        agent.lr_schedule.initial_value = value


def run_experiment_with_visualization(config: Dict[str, Any], pretrained_model_path: str = None,
                                     transfer_learning: bool = False, fine_tune_steps: int = 0,
                                     exploration_boost: float = 1.5, eval_seed: Optional[int] = None,
                                     eval_mode: str = "auto", run_name: Optional[str] = None,
                                     output_dir: Optional[str] = None):
    agent_cfg = config.get('agent')
    agent_name = agent_cfg['name']
    is_recurrent = agent_name in RECURRENT_ALGORITHMS

    if eval_mode == "auto":
        eval_mode = "final" if is_recurrent else "best"
        print(f"eval_mode='{eval_mode}' (auto for {agent_name})")

    # Папка для результатов
    if output_dir:
        log_dir = output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"{ts}_{run_name}" if run_name else ts
        log_dir = os.path.join("logs", agent_name, folder)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output: {log_dir}")

    backend_cfg = config.get('backend')
    backend = build_backend(backend_cfg)

    if backend.dimensions != 2:
        print(f"Warning: {backend.dimensions}D task, visualization is for 2D")

    hp_space = config.get('hp_space') or generate_benchmark_hp_space(backend)
    env_cfg = config.get('environment')

    # Transfer learning настройки
    if transfer_learning and pretrained_model_path:
        env_cfg.setdefault('params', {})
        if is_recurrent:
            env_cfg['params']['use_history'] = False
            print(f"Forced use_history=False for {agent_name}")

        # Авто-определение reward_mode
        if 'per_cycle' in pretrained_model_path.lower():
            env_cfg['params']['reward_mode'] = 'per_cycle'
        else:
            env_cfg['params'].setdefault('reward_mode', 'per_step')

    # TD3/SAC требуют continuous actions
    if agent_name in ["TD3", "SAC"]:
        env_cfg.setdefault('params', {})
        env_cfg['params']['action_type'] = "continuous"
        env_cfg['params'].setdefault('max_step_bins', 30)
        agent_cfg['policy'] = "MultiInputPolicy"
        print(f"{agent_name}: forced continuous actions + MultiInputPolicy")

    # Создание среды
    n_envs = agent_cfg.get('params', {}).get('n_envs', 1)
    if n_envs > 1:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
        env = DummyVecEnv([lambda: build_env(env_cfg, backend=backend, hp_space=hp_space) for _ in range(n_envs)])
        print(f"Created {n_envs} parallel envs")
    else:
        env = build_env(env_cfg, backend=backend, hp_space=hp_space)

    function_name = backend_cfg.get('params', {}).get('function_name', 'unknown')
    if transfer_learning:
        log_suffix = f"{function_name}_transfer_finetune{fine_tune_steps}" if fine_tune_steps else f"{function_name}_transfer"
    else:
        log_suffix = function_name

    log_path = os.path.join(log_dir, log_suffix)
    monitor_kwargs = {"info_keywords": ("best_metric", "best_config", "current_metric", "current_config")}

    if n_envs > 1:
        from stable_baselines3.common.vec_env import VecMonitor
        env = VecMonitor(env, filename=log_path, **monitor_kwargs)
    else:
        env = Monitor(env, filename=log_path, **monitor_kwargs)

    agent_class = AGENT_REGISTRY[agent_name]

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading: {pretrained_model_path}")
        agent = agent_class.load(pretrained_model_path, env=env)

        if transfer_learning:
            print(f"Transfer to: {function_name}")

            orig_ent = _get_ent_coef(agent) or 0.01
            orig_lr = _get_learning_rate(agent)
            new_ent = orig_ent * exploration_boost
            new_lr = orig_lr * 2.0

            print(f"ent_coef: {orig_ent:.4f} -> {new_ent:.4f}, lr: {orig_lr:.6f} -> {new_lr:.6f}")

            if fine_tune_steps > 0:
                _set_ent_coef(agent, new_ent)
                _set_learning_rate(agent, new_lr)
                print(f"Fine-tuning {fine_tune_steps} steps...")
                agent.learn(total_timesteps=fine_tune_steps, reset_num_timesteps=False, progress_bar=False)
            else:
                print("Zero-shot transfer (no fine-tuning)")
    else:
        if pretrained_model_path:
            print(f"Model not found: {pretrained_model_path}, training new")

        import torch
        device = agent_cfg.get('params', {}).get('device')
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device.startswith('cuda') and not torch.cuda.is_available():
            print(f"Warning: {device} unavailable, using cpu")
            device = 'cpu'
        print(f"Device: {device}")

        agent_params = agent_cfg.get('params', {}).copy()
        agent_params['device'] = device
        agent_params.pop('n_envs', None)

        agent = agent_class(policy=agent_cfg['policy'], env=env, **agent_params)

        train_steps = config.get('training', {}).get('total_timesteps', 5000)
        print(f"Training {train_steps} steps...")
        agent.learn(total_timesteps=train_steps)

        save_path = config.get('save_path')
        if save_path:
            if not save_path.endswith('.zip'):
                save_path += '.zip'
            model_path = os.path.join(log_dir, save_path)
            agent.save(model_path)
            print(f"Model saved: {model_path}")

    # Сбор траектории
    print("Collecting trajectory...")
    if eval_seed is not None:
        print(f"eval_seed={eval_seed}")

    trajectory, final_point = collect_trajectory(agent, env, agent_name,
                                                 num_episodes=1, eval_seed=eval_seed)
    print(f"Points: {len(trajectory)}")

    if trajectory:
        rewards = [t[2] for t in trajectory]
        if backend.maximize:
            values = rewards
            best_value = max(values)
        else:
            values = [-r for r in rewards]
            best_value = min(values)

        if final_point:
            final_value = final_point[2] if backend.maximize else -final_point[2]
            print(f"Final: {final_value:.6f}")

        print(f"Best: {best_value:.6f}")

        if hasattr(backend, 'global_optimum_value'):
            print(f"Optimum: {backend.global_optimum_value:.6f}, gap: {abs(best_value - backend.global_optimum_value):.6f}")

        # Визуализация
        seed_suffix = f"_seed{eval_seed}" if eval_seed else ""
        if pretrained_model_path:
            suffix = f"_transfer_finetune{fine_tune_steps}" if fine_tune_steps else "_transfer"
            vis_path = os.path.join(log_dir, f"trajectory_{function_name}{suffix}{seed_suffix}.png")
        else:
            vis_path = os.path.join(log_dir, f"trajectory_{function_name}{seed_suffix}.png")

        visualize_2d_trajectory(backend, trajectory, eval_mode=eval_mode,
                               final_point=final_point, save_path=vis_path)
    else:
        print("Failed to collect trajectory")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HPO-RL 2D Optimization with Visualization and Transfer Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --config configs/function_2d_test.yaml
  python run_experiment.py --config configs/sphere.yaml --pretrained-model model.zip --transfer-learning
  python run_experiment.py --config configs/sphere.yaml --pretrained-model model.zip --fine-tune-steps 2000
"""
    )
    parser.add_argument("--config", type=str, default="configs/function_2d_test.yaml")
    parser.add_argument("--pretrained-model", type=str, default=None)
    parser.add_argument("--transfer-learning", action="store_true")
    parser.add_argument("--fine-tune-steps", type=int, default=0)
    parser.add_argument("--exploration-boost", type=float, default=1.5)
    parser.add_argument("--agent", type=str, default=None,
                       help="Override agent (PPO, TD3, SAC, RecurrentPPO, etc.)")
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--eval-mode", type=str, default="auto", choices=["best", "final", "auto"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)

        if args.agent:
            if args.agent not in AGENT_REGISTRY:
                print(f"Unknown agent '{args.agent}'. Available: {', '.join(AGENT_REGISTRY.keys())}")
                exit(1)
            main_config.setdefault('agent', {})['name'] = args.agent
            print(f"Agent: {args.agent}")

        if args.pretrained_model and not args.transfer_learning:
            print("Note: enabling transfer-learning (pretrained-model specified)")
            args.transfer_learning = True

        run_experiment_with_visualization(
            main_config,
            pretrained_model_path=args.pretrained_model,
            transfer_learning=args.transfer_learning,
            fine_tune_steps=args.fine_tune_steps,
            exploration_boost=args.exploration_boost,
            eval_seed=args.eval_seed,
            eval_mode=args.eval_mode,
            run_name=args.run_name,
            output_dir=args.output_dir
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
