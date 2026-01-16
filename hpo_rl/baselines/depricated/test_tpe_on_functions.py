import numpy as np
from TPE import TPE
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_optimization_landscape(optimizer, param_x, param_y, mode='heatmap'):
    x_bounds = optimizer.dict_to_optimize[param_x]["values"]
    y_bounds = optimizer.dict_to_optimize[param_y]["values"]
    
    resolution = 200 
    gx = np.linspace(x_bounds[0], x_bounds[1], resolution)
    gy = np.linspace(y_bounds[0], y_bounds[1], resolution)
    X, Y = np.meshgrid(gx, gy)
    
    print("Computing landscape heatmap...")
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            p = {param_x: X[i, j], param_y: Y[i, j]}
            Z[i, j] = optimizer.objective_func(p)
            
    plt.figure(figsize=(10, 8))
    
    if mode == 'heatmap':
        plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis', norm=LogNorm())
        plt.colorbar(label='Objective Score (Log Scale)')
    
    history_x = [d[0][param_x] for d in optimizer.data]
    history_y = [d[0][param_y] for d in optimizer.data]
    
    plt.scatter(history_x, history_y, c=range(len(history_x)), cmap='cool', 
                edgecolors='white', linewidth=0.5, s=60, label='TPE Samples')
    
    best_res = min(optimizer.data, key=lambda x: x[1])
    plt.scatter(best_res[0][param_x], best_res[0][param_y], c='red', marker='*', s=300, 
                edgecolors='black', label='Best Found', zorder=10)

    plt.title(f"Optimization Landscape\nBest Score: {best_res[1]:.4f}")
    plt.xlabel(param_x)
    plt.ylabel(param_y)
    plt.legend()
    plt.show()
    
def plot_score_history(optimizer):
    history_scores = [d[1] for d in optimizer.data]
    iterations = range(1, len(history_scores) + 1)
    
    best_so_far = np.minimum.accumulate(history_scores)
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(iterations, history_scores, marker='o', linestyle='-', color='blue', 
             alpha=0.3, label='Current Iteration Score')
    
    plt.plot(iterations, best_so_far, color='red', linewidth=2, label='Best Score So Far')
    
    plt.title("Optimization History (Convergence Plot)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective Score")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()
    
    plt.yscale('symlog') 
    
    plt.show()
    
def test_TPE_func(function_registry):
    optimization_dict = {
        "x": {"type": "float", "values": [-10, 10]},
        "y": {"type": "float", "values": [-10, 10]}
    }

    for func_name, func_obj in function_registry.items():
        print("="*30)
        print(f"Starting optimization for: {func_name}")
        print("="*30)

        tpe = TPE(
            objective_func=func_obj,  
            N_init=10,
            N_s=10,
            budget=100,
            dict_to_optimize=optimization_dict,
            gamma_func=lambda n: 0.2,
        )

        best = tpe.optimize()
        print(f"Best found for {func_name}: {best}")

        plt.figure(figsize=(10, 5))
        plt.suptitle(f"Optimization Landscape: {func_name}")
        
        plot_optimization_landscape(tpe, 'x', 'y', mode='heatmap')
        plt.show()

        plot_score_history(tpe)
        plt.show()