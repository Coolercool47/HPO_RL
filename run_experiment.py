"""
Unified script for HPO-RL experiments with 2D visualization and transfer learning support.
Supports all algorithms: PPO, RecurrentPPO, TD3, SAC, A2C, DQN, TRPO, MaskablePPO.

Usage examples:
    # Train new model
    python run_experiment.py --config configs/function_2d_test.yaml
    
    # Transfer learning (zero-shot)
    python run_experiment.py --config configs/function_2d_sphere.yaml \
        --pretrained-model logs/PPO/model_2d.zip --transfer-learning
    
    # Transfer learning with fine-tuning
    python run_experiment.py --config configs/function_2d_sphere.yaml \
        --pretrained-model logs/PPO/model_2d.zip --transfer-learning \
        --fine-tune-steps 2000 --exploration-boost 2.5
    
    # RecurrentPPO with final point evaluation
    python run_experiment.py --config configs/function_2d_recurrent_ppo.yaml \
        --agent RecurrentPPO --eval-mode final
"""
import yaml
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

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

# Algorithms that require special handling for recurrent state
RECURRENT_ALGORITHMS = {"RecurrentPPO"}


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


def collect_trajectory(agent, env, agent_name: str, num_episodes: int = 1, 
                       eval_seed: Optional[int] = None) -> Tuple[List[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
    """
    Collects search trajectory: (x0, x1, metric)
    
    Args:
        agent: Trained agent
        env: Environment
        agent_name: Name of the algorithm (for detecting recurrent models)
        num_episodes: Number of episodes to collect
        eval_seed: Seed for starting point initialization (None = random)
    
    Returns:
        (trajectory, final_point) where final_point is the final configuration of the episode
    """
    trajectory = []
    step_sizes_used = []
    final_point = None
    is_recurrent = agent_name in RECURRENT_ALGORITHMS
    
    vec_env = agent.get_env()
    
    for episode in range(num_episodes):
        if eval_seed is not None:
            vec_env.seed(eval_seed + episode)
        
        obs = vec_env.reset()
        done = False
        step_count = 0
        _states = None  # For RecurrentPPO
        
        # First step
        if is_recurrent:
            action, _states = agent.predict(obs, state=_states, deterministic=True, 
                                           episode_start=np.array([True]))
        else:
            action, _states = agent.predict(obs, deterministic=True)
        
        obs, reward, dones, infos = vec_env.step(action)
        
        # Process info from first step
        if isinstance(infos, (list, tuple)) and len(infos) > 0:
            info = infos[0]
        else:
            info = infos if isinstance(infos, dict) else {}
        
        current_config = info.get("current_config", info.get("best_config", {}))
        current_metric = info.get("current_metric", info.get("best_metric", -float('inf')))
        
        if len(current_config) >= 2:
            x0 = current_config.get("x0", 0.0)
            x1 = current_config.get("x1", 0.0)
            trajectory.append((x0, x1, current_metric))
        
        done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
        
        while not done:
            if is_recurrent:
                action, _states = agent.predict(obs, state=_states, deterministic=True,
                                               episode_start=np.array([False]))
            else:
                action, _states = agent.predict(obs, deterministic=True)
            
            obs, reward, dones, infos = vec_env.step(action)
            done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
            
            if done and is_recurrent:
                _states = None  # Reset LSTM state between episodes
            
            if isinstance(infos, (list, tuple)) and len(infos) > 0:
                info = infos[0]
            else:
                info = infos if isinstance(infos, dict) else {}
            
            current_config = info.get("current_config", info.get("best_config", {}))
            current_metric = info.get("current_metric", info.get("best_metric", -float('inf')))
            
            # Save final configuration
            if done and len(current_config) >= 2:
                x0 = current_config.get("x0", 0.0)
                x1 = current_config.get("x1", 0.0)
                final_point = (x0, x1, current_metric)
            
            # Step size statistics
            step_size = info.get("step_size", 0)
            if step_size > 0:
                step_sizes_used.append(step_size)
            
            if len(current_config) >= 2:
                x0 = current_config.get("x0", 0.0)
                x1 = current_config.get("x1", 0.0)
                trajectory.append((x0, x1, current_metric))
            
            step_count += 1
            if step_count > 1000:
                if final_point is None and len(trajectory) > 0:
                    final_point = trajectory[-1]
                break
        
        if final_point is None and len(trajectory) > 0:
            final_point = trajectory[-1]
    
    # Print step size statistics
    if step_sizes_used:
        from collections import Counter
        step_size_counts = Counter(step_sizes_used)
        print(f"\n=== Step size statistics ===")
        print(f"Total steps: {len(step_sizes_used)}")
        print(f"Mean step size: {np.mean(step_sizes_used):.2f}")
        print(f"Median step size: {np.median(step_sizes_used):.2f}")
        print(f"Distribution:")
        for size, count in sorted(step_size_counts.items(), reverse=True):
            percentage = (count / len(step_sizes_used)) * 100
            print(f"  Size {size:2d}: {count:4d} times ({percentage:5.1f}%)")
    
    return trajectory, final_point


def visualize_2d_trajectory(backend, trajectory: List[Tuple[float, float, float]], 
                           eval_mode: str = "best",
                           final_point: Optional[Tuple[float, float, float]] = None,
                           save_path: str = "trajectory_2d.png"):
    """
    Visualizes search trajectory on 2D plane with function contour lines
    
    Args:
        backend: Backend with function to optimize
        trajectory: List of (x0, x1, metric) points
        eval_mode: "best" (mark best intermediate point) or "final" (mark final point)
        final_point: Final configuration (used when eval_mode="final")
        save_path: Path to save the plot
    """
    if len(trajectory) == 0:
        print("Trajectory is empty, nothing to visualize")
        return
    
    # Extract coordinates
    x0_vals = [t[0] for t in trajectory]
    x1_vals = [t[1] for t in trajectory]
    rewards = [t[2] for t in trajectory]
    
    if backend.maximize:
        metrics = rewards
    else:
        metrics = [-r for r in rewards]
    
    # Create grid for contour lines
    bounds = backend.bounds
    x0_range = np.linspace(bounds[0], bounds[1], 100)
    x1_range = np.linspace(bounds[0], bounds[1], 100)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    # Compute function values on grid
    Z = np.zeros_like(X0)
    for i in range(X0.shape[0]):
        for j in range(X0.shape[1]):
            config = {"x0": X0[i, j], "x1": X1[i, j]}
            reward = backend.evaluate(config)
            if backend.maximize:
                Z[i, j] = reward
            else:
                Z[i, j] = -reward
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Contour lines + trajectory
    contour = ax1.contour(X0, X1, Z, levels=20, cmap='viridis', alpha=0.6)
    ax1.clabel(contour, inline=True, fontsize=8)
    ax1.contourf(X0, X1, Z, levels=20, cmap='viridis', alpha=0.3)
    
    # Draw trajectory
    ax1.plot(x0_vals, x1_vals, 'r-', linewidth=2, alpha=0.7, label='Trajectory')
    ax1.scatter(x0_vals[0], x1_vals[0], c='green', s=100, marker='o', 
                label='Start', zorder=5, edgecolors='black', linewidths=2)
    ax1.scatter(x0_vals[-1], x1_vals[-1], c='red', s=100, marker='*', 
                label='End', zorder=5, edgecolors='black', linewidths=2)
    
    # Mark result point based on eval_mode
    if eval_mode == "final" and final_point is not None:
        final_x0, final_x1, final_metric = final_point
        if backend.maximize:
            final_value = final_metric
        else:
            final_value = -final_metric
        ax1.scatter(final_x0, final_x1, c='yellow', s=150, 
                    marker='X', label='Final point (result)', zorder=5, 
                    edgecolors='black', linewidths=2)
        marker_idx = len(trajectory) - 1  # For 3D
        marker_value = final_value
        marker_coords = (final_x0, final_x1)
    else:
        # Mark best intermediate point
        best_idx = np.argmin(metrics) if not backend.maximize else np.argmax(metrics)
        ax1.scatter(x0_vals[best_idx], x1_vals[best_idx], c='yellow', s=150, 
                    marker='X', label='Best point', zorder=5, 
                    edgecolors='black', linewidths=2)
        marker_idx = best_idx
        marker_value = metrics[best_idx]
        marker_coords = (x0_vals[best_idx], x1_vals[best_idx])
    
    opt_type = "maximization" if backend.maximize else "minimization"
    ax1.set_xlabel('x0', fontsize=12)
    ax1.set_ylabel('x1', fontsize=12)
    ax1.set_title(f'Search trajectory on function contour map ({opt_type})', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Subplot 2: 3D surface + trajectory
    ax2 = fig.add_subplot(122, projection='3d')
    surf = ax2.plot_surface(X0, X1, Z, cmap='viridis', alpha=0.6, 
                           linewidth=0, antialiased=True)
    
    # Draw trajectory in 3D
    ax2.plot(x0_vals, x1_vals, metrics, 'r-', linewidth=2, alpha=0.8, label='Trajectory')
    ax2.scatter(x0_vals[0], x1_vals[0], metrics[0], c='green', s=100, 
                marker='o', label='Start', edgecolors='black', linewidths=2)
    ax2.scatter(x0_vals[-1], x1_vals[-1], metrics[-1], c='red', s=100, 
                marker='*', label='End', edgecolors='black', linewidths=2)
    
    # Mark result point in 3D
    if eval_mode == "final" and final_point is not None:
        ax2.scatter(marker_coords[0], marker_coords[1], marker_value, 
                    c='yellow', s=150, marker='X', label='Final point (result)', 
                    edgecolors='black', linewidths=2)
    else:
        ax2.scatter(marker_coords[0], marker_coords[1], marker_value, 
                    c='yellow', s=150, marker='X', label='Best point', 
                    edgecolors='black', linewidths=2)
    
    ax2.set_xlabel('x0', fontsize=12)
    ax2.set_ylabel('x1', fontsize=12)
    ax2.set_zlabel(f'Function value ({opt_type})', fontsize=12)
    ax2.set_title(f'3D trajectory visualization ({opt_type})', fontsize=14)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()


def run_experiment_with_visualization(config: Dict[str, Any], pretrained_model_path: str = None, 
                                     transfer_learning: bool = False, fine_tune_steps: int = 0,
                                     exploration_boost: float = 1.5, eval_seed: Optional[int] = None,
                                     eval_mode: str = "auto", run_name: Optional[str] = None,
                                     output_dir: Optional[str] = None):
    agent_cfg = config.get('agent')
    agent_name = agent_cfg['name']
    is_recurrent = agent_name in RECURRENT_ALGORITHMS
    
    # Auto-detect eval_mode
    if eval_mode == "auto":
        eval_mode = "final" if is_recurrent else "best"
        print(f"Auto-detected eval_mode='{eval_mode}' for {agent_name}")
    
    # Use provided output_dir or generate new folder with timestamp
    if output_dir:
        log_dir_root = output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            folder_name = f"{timestamp}_{run_name}"
        else:
            folder_name = timestamp
        log_dir_root = os.path.join("logs", agent_name, folder_name)
    os.makedirs(log_dir_root, exist_ok=True)
    print(f"Experiment folder: {log_dir_root}")
    
    print("\n=== Experiment Setup ===")
    backend_cfg = config.get('backend')
    backend = build_backend(backend_cfg)
    
    if backend.dimensions != 2:
        print(f"WARNING: Task is {backend.dimensions}D, visualization is designed for 2D")
    
    hp_space = config.get('hp_space', {})
    if not hp_space:
        hp_space = generate_benchmark_hp_space(backend)
    
    env_cfg = config.get('environment')
    
    # Sync environment parameters for transfer learning
    if transfer_learning and pretrained_model_path:
        if 'params' not in env_cfg:
            env_cfg['params'] = {}
        
        # For RecurrentPPO, force use_history=False
        if is_recurrent:
            env_cfg['params']['use_history'] = False
            print(f"[OK] Forced use_history=False for {agent_name}")
        
        # Detect reward_mode from model filename
        if 'per_cycle' in pretrained_model_path.lower():
            if env_cfg['params'].get('reward_mode') != 'per_cycle':
                env_cfg['params']['reward_mode'] = 'per_cycle'
                print(f"[OK] Auto-set reward_mode='per_cycle' for compatibility")
        elif env_cfg['params'].get('reward_mode') is None:
            env_cfg['params']['reward_mode'] = 'per_step'
            print(f"[OK] Using reward_mode='per_step' (default)")
        
        # Sync penalty parameters
        if env_cfg['params'].get('step_size_penalty_coef') is None:
            env_cfg['params']['step_size_penalty_coef'] = 0.15
            print(f"[OK] Auto-set step_size_penalty_coef=0.15")
        if env_cfg['params'].get('adaptive_step_penalty') is None:
            env_cfg['params']['adaptive_step_penalty'] = True
            print(f"[OK] Auto-enabled adaptive_step_penalty=True")
    
    # For TD3 and SAC, auto-switch to continuous actions
    if agent_name in ["TD3", "SAC"] and env_cfg.get('params', {}).get('action_type') != "continuous":
        if 'params' not in env_cfg:
            env_cfg['params'] = {}
        env_cfg['params']['action_type'] = "continuous"
        if 'max_step_bins' not in env_cfg['params']:
            env_cfg['params']['max_step_bins'] = 30
        print(f"WARNING: {agent_name} requires continuous actions. Auto-set action_type='continuous'")
    
    # Check if parallel environments are needed
    n_envs = agent_cfg.get('params', {}).get('n_envs', 1)
    if n_envs > 1:
        from stable_baselines3.common.vec_env import DummyVecEnv
        print(f"Creating {n_envs} parallel environments...")
        
        def make_env():
            return build_env(env_cfg, backend=backend, hp_space=hp_space)
        
        env = DummyVecEnv([make_env for _ in range(n_envs)])
        print(f"[OK] Created {n_envs} parallel environments")
    else:
        env = build_env(env_cfg, backend=backend, hp_space=hp_space)
    
    # Create unique log path
    function_name = backend_cfg.get('params', {}).get('function_name', 'unknown')
    if transfer_learning:
        if fine_tune_steps > 0:
            log_suffix = f"{function_name}_transfer_finetune{fine_tune_steps}"
        else:
            log_suffix = f"{function_name}_transfer"
    else:
        log_suffix = function_name
    
    log_path = os.path.join(log_dir_root, log_suffix)
    
    if n_envs > 1:
        from stable_baselines3.common.vec_env import VecMonitor
        env = VecMonitor(env, filename=log_path, 
                        info_keywords=("best_metric", "best_config", "current_metric", "current_config"))
    else:
        env = Monitor(env, filename=log_path, 
                     info_keywords=("best_metric", "best_config", "current_metric", "current_config"))
    
    print("\n=== Creating/Loading Agent ===")
    agent_class = AGENT_REGISTRY[agent_name]
    
    # For TD3 and SAC, auto-use MultiInputPolicy
    if agent_name in ["TD3", "SAC"] and agent_cfg.get('policy') != "MultiInputPolicy":
        agent_cfg['policy'] = "MultiInputPolicy"
        print(f"WARNING: {agent_name} requires MultiInputPolicy. Auto-set policy='MultiInputPolicy'")
    
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Loading pretrained model from: {pretrained_model_path}")
        print(f"Target function: {function_name}")
        agent = agent_class.load(pretrained_model_path, env=env)
        print("[OK] Model loaded successfully")
        
        if transfer_learning:
            print(f"\n=== Transfer Learning Mode ===")
            print(f"Testing on new function: {function_name}")
            
            # Get original ent_coef
            original_ent_coef = _get_ent_coef(agent)
            if original_ent_coef is None or original_ent_coef < 1e-6:
                original_ent_coef = 0.01
                print(f"[!]  ent_coef not found or zero, using default: {original_ent_coef:.4f}")
            
            new_ent_coef = original_ent_coef * exploration_boost
            print(f"Increased entropy for exploration: {original_ent_coef:.4f} -> {new_ent_coef:.4f}")
            
            # Get original learning rate
            original_lr = _get_learning_rate(agent)
            new_lr = original_lr * 2.0
            print(f"Increased learning rate for adaptation: {original_lr:.6f} -> {new_lr:.6f}")
            
            if fine_tune_steps > 0:
                print(f"Fine-tuning for {fine_tune_steps} steps with increased exploration...")
                _set_ent_coef(agent, new_ent_coef)
                _set_learning_rate(agent, new_lr)
                
                agent.learn(
                    total_timesteps=fine_tune_steps, 
                    reset_num_timesteps=False,
                    progress_bar=False
                )
                print("[OK] Fine-tuning completed")
            else:
                print("Testing without fine-tuning (zero-shot transfer)")
                print("[!]  Consider using --fine-tune-steps for better adaptation")
    else:
        if pretrained_model_path:
            print(f"WARNING: Model file not found: {pretrained_model_path}")
            print("Creating new model...")
        
        # Determine device
        import torch
        device = agent_cfg.get('params', {}).get('device', None)
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"[OK] GPU available: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                print("[!]  GPU not available, using CPU")
        else:
            if device.startswith('cuda') and not torch.cuda.is_available():
                print(f"[!]  WARNING: {device} specified but GPU not available. Using CPU")
                device = 'cpu'
            else:
                print(f"Using device: {device}")
        
        # Create agent
        agent_params = agent_cfg.get('params', {}).copy()
        agent_params['device'] = device
        agent_params.pop('n_envs', None)  # Already handled
        
        agent = agent_class(
            policy=agent_cfg['policy'],
            env=env,
            **agent_params
        )
        
        train_steps = config.get('training', {}).get('total_timesteps', 5000)
        print(f"\n=== Training ({train_steps} steps) ===")
        agent.learn(total_timesteps=train_steps)
        
        # Save model if path specified
        save_path = config.get('save_path')
        if save_path:
            if not save_path.endswith('.zip'):
                save_path = save_path + '.zip'
            model_path = os.path.join(log_dir_root, save_path)
            agent.save(model_path)
            print(f"[OK] Model saved to: {model_path}")
    
    print("\n=== Collecting Trajectory ===")
    if eval_seed is not None:
        print(f"Using eval_seed={eval_seed} for starting point")
    
    trajectory, final_point = collect_trajectory(agent, env, agent_name, 
                                                  num_episodes=1, eval_seed=eval_seed)
    print(f"Collected {len(trajectory)} trajectory points")
    
    if len(trajectory) > 0:
        # Compute metrics
        rewards = [t[2] for t in trajectory]
        if backend.maximize:
            metrics = rewards
            best_intermediate = max(metrics)
        else:
            metrics = [-r for r in rewards]
            best_intermediate = min(metrics)
        
        # Get result based on eval_mode
        if eval_mode == "final" and final_point is not None:
            final_metric_reward = final_point[2]
            if backend.maximize:
                result_metric = final_metric_reward
            else:
                result_metric = -final_metric_reward
            print(f"Final metric (end of episode): {result_metric:.6f}")
            if eval_mode == "final":
                print("  [!]  NOTE: Evaluation uses final configuration, not best intermediate!")
                print("  This is correct for recurrent algorithms (RecurrentPPO).")
        else:
            result_metric = best_intermediate
            print(f"Best metric: {result_metric:.6f}")
        
        if hasattr(backend, 'global_optimum_value'):
            print(f"Global optimum: {backend.global_optimum_value:.6f}")
            print(f"Deviation from optimum: {abs(result_metric - backend.global_optimum_value):.6f}")
        
        # Show best intermediate for reference if using final mode
        if eval_mode == "final":
            print(f"Best intermediate metric (for reference): {best_intermediate:.6f}")
        
        print("\n=== Visualization ===")
        seed_suffix = f"_seed{eval_seed}" if eval_seed is not None else ""
        if pretrained_model_path:
            if fine_tune_steps > 0:
                save_path = os.path.join(log_dir_root, 
                    f"trajectory_{function_name}_transfer_finetune{fine_tune_steps}{seed_suffix}.png")
            else:
                save_path = os.path.join(log_dir_root, 
                    f"trajectory_{function_name}_transfer{seed_suffix}.png")
        else:
            save_path = os.path.join(log_dir_root, f"trajectory_{function_name}{seed_suffix}.png")
        
        visualize_2d_trajectory(backend, trajectory, eval_mode=eval_mode,
                               final_point=final_point, save_path=save_path)
    else:
        print("Failed to collect trajectory")


def _get_ent_coef(agent) -> Optional[float]:
    """Helper to get entropy coefficient from agent"""
    if hasattr(agent, 'ent_coef'):
        if isinstance(agent.ent_coef, float):
            return agent.ent_coef
        elif hasattr(agent.ent_coef, 'value'):
            return agent.ent_coef.value()
        else:
            try:
                return float(agent.ent_coef)
            except:
                pass
    if hasattr(agent, 'policy') and hasattr(agent.policy, 'ent_coef'):
        if isinstance(agent.policy.ent_coef, float):
            return agent.policy.ent_coef
        elif hasattr(agent.policy.ent_coef, 'value'):
            return agent.policy.ent_coef.value()
        else:
            try:
                return float(agent.policy.ent_coef)
            except:
                pass
    return 0.01  # Default for PPO


def _get_learning_rate(agent) -> float:
    """Helper to get learning rate from agent"""
    if hasattr(agent, 'learning_rate'):
        if isinstance(agent.learning_rate, float):
            return agent.learning_rate
        elif hasattr(agent.learning_rate, 'value'):
            return agent.learning_rate.value()
        else:
            try:
                return float(agent.learning_rate)
            except:
                pass
    if hasattr(agent, 'lr_schedule'):
        if hasattr(agent.lr_schedule, 'initial_value'):
            return agent.lr_schedule.initial_value
    return 0.0003  # Default for PPO


def _set_ent_coef(agent, value: float):
    """Helper to set entropy coefficient"""
    if hasattr(agent, 'ent_coef'):
        try:
            if hasattr(agent.ent_coef, '__call__'):
                agent.ent_coef = lambda _: value
            else:
                agent.ent_coef = value
        except Exception as e:
            print(f"[!]  Could not set ent_coef: {e}")


def _set_learning_rate(agent, value: float):
    """Helper to set learning rate"""
    if hasattr(agent, 'policy') and hasattr(agent.policy, 'optimizer'):
        for param_group in agent.policy.optimizer.param_groups:
            param_group['lr'] = value
    if hasattr(agent, 'lr_schedule'):
        if hasattr(agent.lr_schedule, 'initial_value'):
            agent.lr_schedule.initial_value = value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HPO-RL 2D Optimization with Visualization and Transfer Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train new model:
    python run_experiment.py --config configs/function_2d_test.yaml

  Transfer learning (zero-shot):
    python run_experiment.py --config configs/function_2d_sphere.yaml \\
        --pretrained-model logs/PPO/model_2d.zip --transfer-learning

  Fine-tuning:
    python run_experiment.py --config configs/function_2d_sphere.yaml \\
        --pretrained-model logs/PPO/model_2d.zip --transfer-learning \\
        --fine-tune-steps 2000 --exploration-boost 2.5

  RecurrentPPO:
    python run_experiment.py --config configs/function_2d_recurrent_ppo.yaml \\
        --agent RecurrentPPO --eval-mode final
        """
    )
    parser.add_argument("--config", type=str, default="configs/function_2d_test.yaml",
                       help="Path to configuration file")
    parser.add_argument("--pretrained-model", type=str, default=None,
                       help="Path to pretrained model for transfer learning")
    parser.add_argument("--transfer-learning", action="store_true",
                       help="Enable transfer learning mode")
    parser.add_argument("--fine-tune-steps", type=int, default=0,
                       help="Number of fine-tuning steps (0 = zero-shot transfer)")
    parser.add_argument("--exploration-boost", type=float, default=1.5,
                       help="Multiplier for exploration during transfer learning (default: 1.5)")
    parser.add_argument("--agent", type=str, default=None,
                       help="Algorithm to use (PPO, TD3, SAC, RecurrentPPO, etc.). Overrides config")
    parser.add_argument("--eval-seed", type=int, default=None,
                       help="Seed for starting point during inference (None = random)")
    parser.add_argument("--eval-mode", type=str, default="auto",
                       choices=["best", "final", "auto"],
                       help="Evaluation mode: 'best' (best intermediate), 'final' (end of episode), "
                            "'auto' (final for RecurrentPPO, best for others)")
    parser.add_argument("--run-name", type=str, default=None,
                       help="Custom run name suffix (e.g., 'ultra_low_budget', 'full'). Added to folder name")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for all experiment files. If specified, all files go to this folder")
    
    args = parser.parse_args()
    
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        
        # Override algorithm if specified
        if args.agent:
            if args.agent not in AGENT_REGISTRY:
                print(f"ERROR: Unknown algorithm '{args.agent}'")
                print(f"Available algorithms: {', '.join(AGENT_REGISTRY.keys())}")
                exit(1)
            if 'agent' not in main_config:
                main_config['agent'] = {}
            main_config['agent']['name'] = args.agent
            print(f"Using algorithm: {args.agent}")
        
        if args.pretrained_model and not args.transfer_learning:
            print("WARNING: --pretrained-model specified but not --transfer-learning")
            print("Enabling transfer learning mode by default")
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

