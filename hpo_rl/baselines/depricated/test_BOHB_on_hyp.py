import matplotlib.pyplot as plt
from BOHB import BOHB

def test_BOHB_hyp(history, objective_function, dict_config):
    bohb = BOHB(R=9, nu=3, objective_function=objective_function, dict_config=dict_config, min_points_in_model=5, num_samples=64)

    best_config = bohb.main_loop()
    print("Лучшая конфигурация:", best_config)

    losses = [h["loss"] for h in history]
    accuracies = [h["accuracy"] for h in history]
    iterations = [h["iteration"] for h in history]
    budgets = [h["epochs"] for h in history]

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sizes = [b * 10 for b in budgets] 
    plt.scatter(iterations, losses, s=sizes, c='red', alpha=0.6, edgecolors='black')
    plt.plot(iterations, losses, linestyle='--', alpha=0.3, color='gray')
    plt.xlabel("Номер итерации")
    plt.ylabel("Validation Loss")
    plt.title("История Loss по итерациям")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(iterations, accuracies, s=sizes, c='blue', alpha=0.6, edgecolors='black')
    plt.plot(iterations, accuracies, linestyle='--', alpha=0.3, color='gray')
    plt.xlabel("Номер итерации")
    plt.ylabel("Validation Accuracy")
    plt.title("История Accuracy по итерациям")
    plt.grid(True)

    plt.tight_layout()
    plt.show()