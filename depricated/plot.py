import pandas as pd
import matplotlib.pyplot as plt
import os


def load_data(log_dir, task_file_prefix):
    file_path = os.path.join(log_dir, f"{task_file_prefix}.monitor.csv")
    if not os.path.exists(file_path):
        print(f"Warning: File not found {file_path}")
        return None

    df = pd.read_csv(file_path, skiprows=1)

    df['steps'] = df['l'].cumsum()

    window = 50
    if 'best_metric' in df.columns:
        df['metric_smooth'] = df['best_metric'].rolling(window=window).mean()
        df['metric_smooth'] = df['metric_smooth'].fillna(df['best_metric'])

    return df


def plot_comparison():
    agents = ["PPO", "A2C", "DQN", "TRPO", "recurrentPPO"]
    colors = {"PPO": "blue", "A2C": "orange", "DQN": "green", "TRPO": "red", "recurrentPPO": "brown"}

    plt.figure(figsize=(10, 6))
    plt.title("Task A: Rastrigin Optimization (Absolute Metric)")
    plt.xlabel("Total Timesteps")
    plt.ylabel("Best Metric Found (Higher is Better, Max=0)")
    plt.grid(True, alpha=0.3)

    for agent in agents:
        df = load_data(f"logs/{agent}", "task_a_rastrigin")
        if df is not None:
            plt.plot(df['steps'], df['metric_smooth'], label=f"{agent}", color=colors.get(agent, "black"))
            plt.plot(df['steps'], df['best_metric'], color=colors.get(agent), alpha=0.1)

    plt.legend()
    plt.savefig("plot_task_a_metric.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.title("Task B: Sphere Finetuning (Absolute Metric)")
    plt.xlabel("Finetuning Steps")
    plt.ylabel("Best Metric Found (Higher is Better, Max=0)")
    plt.grid(True, alpha=0.3)

    for agent in agents:
        df = load_data(f"logs/{agent}", "task_b_sphere_finetune")
        if df is not None:
            plt.plot(df['steps'], df['metric_smooth'], label=f"{agent}", color=colors.get(agent, "black"))

    plt.legend()
    plt.savefig("plot_task_b_metric.png")
    plt.show()


if __name__ == "__main__":
    plot_comparison()
