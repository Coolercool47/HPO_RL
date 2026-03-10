from hpo_rl.experiments.run_experiment import run_n_experiments

if __name__ == "__main__":
    config_cma_es = {
        "backend": {
            "name": "function",
            "function": "rastrigin",
            "dimensions": 2
        },
        "full_args": {
            "algorithm": {
                "name": "CMA_ES",
                "N_pop": None,  # None means it will use the default formula 4 + 3*ln(N)
                "budget": 2000,
                "initial_step_size": 0.5
            }
        }
    }

    config_simple_ga = {
        "backend": {
            "name": "function",
            "function": "rastrigin",
            "dimensions": 2
        },
        "full_args": {
            "algorithm": {
                "name": "SimpleGA",
                "N_pop": 20,
                "budget": 2000,
                "mutation_prob": 0.1,
                "crossover_prob": 0.8,
                "tournament_size": 5,
                "elitism": True
            }
        }
    }

    # Запускаем 3 независимых эксперимента с CMA-ES на функции Растригина
    # run_n_experiments(config_cma_es, 3, inference_only=False)

    # Запускаем 3 независимых эксперимента с SimpleGA на функции Растригина
    run_n_experiments(config_simple_ga, 3, inference_only=False)
