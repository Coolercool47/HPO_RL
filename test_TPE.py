from hpo_rl.experiments.run_experiment import run_n_experiments


config_HMM = {
    "full_args": {
        "algorithm": {
            "name": "TPE",
            "N_init": 20,
            "N_s": 100,
            "budget": 500,
            "separation_value": 0.2
        }
    },
    "backend": {
        "name": "function", "function": "schwefel", "dimensions": 10, "noise_std": 100
    }
}

if __name__ == "__main__":
    run_n_experiments(config_HMM, n_experiments=3, inference_only=False)
