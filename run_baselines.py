import yaml
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from hpo_rl.core.factory import build_backend, build_env
from hpo_rl.core.register import initialize_framework

from hpo_rl.baselines.TPE import TPE
from hpo_rl.baselines.BOHB import BOHB
from hpo_rl.baselines.hyperband import hyperband


initialize_framework()

ALGORITHM_REGISTRY = {
    "TPE": TPE,  "BOHB": BOHB, "hyperband": hyperband
}

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

def collect_trajectory_alg(algorithm, eval_seed):
    result = algorithm.main_loop()
    data = algorithm.data
    return data, result

def run_experiment_with_visualization(config: Dict[str, Any], eval_seed: Optional[int] = None,
                                     eval_mode: str = "auto", run_name: Optional[str] = None,
                                     output_dir: Optional[str] = None):
    algorithm_cfg = config.get('algorithm')
    algorithm_name = algorithm_cfg['name']
    algorithm = ALGORITHM_REGISTRY[algorithm_name]
    algorithm_params = algorithm_cfg.get('params', {})
    algorithm = algorithm(**algorithm_params)
    
    # Папка для результатов
    if output_dir:
        log_dir = output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = f"{ts}_{run_name}" if run_name else ts
        log_dir = os.path.join("logs", algorithm_cfg, folder)
    os.makedirs(log_dir, exist_ok=True)
    print(f"Output: {log_dir}")

    backend_cfg = config.get('backend')
    backend = build_backend(backend_cfg)
        
    # Сбор траектории
    print("Collecting trajectory...")
    if eval_seed is not None:
        print(f"eval_seed={eval_seed}")
    
    trajectory, final_point = collect_trajectory_alg(algorithm, eval_seed=eval_seed)
    print(f"Points: {len(trajectory)}")

    function_name = backend_cfg.get('params', {}).get('function_name', 'unknown')

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
        vis_path = os.path.join(log_dir, f"trajectory_{function_name}{seed_suffix}.png")

        visualize_2d_trajectory(backend, trajectory, eval_mode=eval_mode,
                               final_point=final_point, save_path=vis_path)
    else:
        print("Failed to collect trajectory")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline algorithms",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiment.py --config configs/function_2d_test.yaml
  python run_experiment.py --config configs/sphere.yaml --pretrained-model model.zip --transfer-learning
  python run_experiment.py --config configs/sphere.yaml --pretrained-model model.zip --fine-tune-steps 2000
"""
    )
    parser.add_argument("--config", type=str, default="configs/function_2d_test.yaml")
    parser.add_argument("--algorithm", type=str, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--eval-mode", type=str, default="auto", choices=["best", "final", "auto"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)

        if args.algorithm:
            if args.algorithm not in ALGORITHM_REGISTRY:
                print(f"Unknown algorithm '{args.algorithm}'. Available: {', '.join(ALGORITHM_REGISTRY.keys())}")
                exit(1)
            main_config.setdefault('algorithm', {})['name'] = args.algorithm
            print(f"Algorithm: {args.algorithm}")

        run_experiment_with_visualization(
            main_config,
            eval_seed=args.eval_seed,
            eval_mode=args.eval_mode,
            run_name=args.run_name,
            output_dir=args.output_dir
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        exit(1)
