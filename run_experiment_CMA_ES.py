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
                "budget": 200,
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
                "budget": 200,
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
    # run_n_experiments(config_simple_ga, 3, inference_only=False)

    funcs = [
        "sphere", "rosenbrock", "rastrigin", "ackley", "griewank",
        "schwefel", "levy", "michalewicz", "booth", "beale",
        "goldstein_price", "bukin_n6", "cross_in_tray", "drop_wave",
        "eggholder", "holder_table", "schaffer_n2", "schaffer_n4",
        "shubert", "dejong_n5", "easom", "levy_n13", "langermann"
    ]

    for func in funcs:
        print(f"--- Running experiments for {func} ---")
        config_cma_es["backend"]["function"] = func
        config_simple_ga["backend"]["function"] = func

        # По одному запуску на каждую функцию
        run_n_experiments(config_cma_es, 2, inference_only=False)
        run_n_experiments(config_simple_ga, 2, inference_only=False)
