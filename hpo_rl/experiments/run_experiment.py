from hpo_rl.controller.plot import plot_and_save
from hpo_rl.controller.check import check
from hpo_rl.controller.controller import controller
from hpo_rl.backends.sequential import SequentialBackend
from hpo_rl.backends.function import OptimizationBenchmarkBackend
import torch
from pathlib import Path
from datetime import datetime
import numpy as np
import json

def run_n_experiments(config, n_experiments, inference_only=False):
    """Функция запускающая полный пайплайн несколько раз, начиная с конфигурации пользователя, заканчивая отображением изображений и истории поиска.
    
    Args: 
        config: необработанная конфигурация
        n_experiments: количество экспериментов
        inference_only: если True — пропускает обучение и сразу запускает инференс.
            Требует ``load_checkpoint`` в конфиге для загрузки обученной модели.

    """
    parsed_config = check(config)
    # print(config, parsed_config, sep="\n\n", end="\n\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = parsed_config.get("mode")

    algorithm_name = config["full_args"].get("algorithm").get("name")
    backend_name = config.get("backend").get("name")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / algorithm_name / timestamp 
    log_dir.mkdir(parents=True, exist_ok=True)
    save_path = parsed_config.get("log_save_path", log_dir)

    expreiment_controller = controller(**parsed_config)
    if mode == "RL" and not inference_only:
        expreiment_controller.train()
    elif mode == "RL" and inference_only:
        if not getattr(expreiment_controller, 'load_loc', None):
            raise ValueError(
                "inference_only=True requires 'load_checkpoint' in config to load trained model weights. "
            )

    full_history = []

    for i_experiment in range(n_experiments):
        if mode == "baseline" and i_experiment > 0:
            if hasattr(expreiment_controller.algorithm, "reset"):
                expreiment_controller.algorithm.reset()
        best_result = expreiment_controller.inference()
        history = expreiment_controller.return_history()
        outputs = plot_and_save(history, best_result, save_path, expreiment_controller.backend, experiment_number=i_experiment)
        if backend_name == "function" and expreiment_controller.backend.dimensions == 2:
            outputs.plot_3d()
        elif backend_name == "sequential":
            backend = expreiment_controller.backend
            merged_bounds = getattr(backend, '_merged_bounds', None)
            # Для каждого дочернего бэкенда — отдельный inference
            seen_names = {}
            for child_idx, child in enumerate(backend.backends):
                fn_name = getattr(child, 'function_name', type(child).__name__)
                # Пропускаем дубликаты (schwefel встречается дважды — достаточно одного)
                if fn_name in seen_names:
                    continue
                seen_names[fn_name] = child_idx

                # Переключаем SequentialBackend на этого child
                backend.set_active_backend(child_idx)
                # Отдельный inference на этом child
                child_best = expreiment_controller.inference()
                child_history = expreiment_controller.return_history()

                suffix = f"_{child_idx}_{fn_name}"
                child_outputs = plot_and_save(
                    child_history, child_best, save_path, child,
                    experiment_number=i_experiment,
                    merged_bounds=merged_bounds,
                )
                if isinstance(child, OptimizationBenchmarkBackend) and child.dimensions == 2:
                    child_outputs.plot_3d(suffix=suffix)
                child_outputs.plot_trajectory(suffix=suffix)
                child_outputs.save_history(as_latex=True, suffix=suffix)
                child_outputs.save_history(as_latex=False, suffix=suffix)
        outputs.plot_trajectory()
        outputs.save_history(as_latex=True)
        outputs.save_history(as_latex=False)
        full_history.append(history)

    is_maximize = expreiment_controller.backend.maximize

    results = np.array([[trial[1] for trial in inference] for inference in full_history])
    best_trial_indices = np.argmax(results, axis=1) if is_maximize else np.argmin(results, axis=1)
    best = np.array([full_history[i][best_trial_indices[i]] for i in range(len(full_history))], dtype=object)

    # Сортируем так, чтобы worst был первым (idx 0), best — последним (idx -1)
    best_scores = [item[1] for item in best]
    if is_maximize:
        sorted_idx = np.argsort(best_scores)
    else:
        sorted_idx = np.argsort(best_scores)[::-1]

    worst_of_best, best_of_best, median_of_best = best[sorted_idx[0]],  best[sorted_idx[-1]], best[sorted_idx[len(sorted_idx) // 2]]

    last = np.array([inference[-1] for inference in full_history], dtype=object)
    last_scores = [item[1] for item in last]
    if is_maximize:
        last_idxs = np.argsort(last_scores)
    else:
        last_idxs = np.argsort(last_scores)[::-1]

    worst_of_last, best_of_last, median_of_last = last[last_idxs[0]], last[last_idxs[-1]], last[last_idxs[len(last_idxs)//2]]

    data_to_save = {
        "best_of_each_inference": {
            "worst": worst_of_best.tolist(),
            "best": best_of_best.tolist(),
            "median": median_of_best.tolist()
        },
        "final_of_each_inference": {
            "worst": worst_of_last.tolist(),
            "best": best_of_last.tolist(),
            "median": median_of_last.tolist()
        }
    }

    def serialize(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()

    file_name = save_path / "inference_results.json"
    with open(file_name, "w") as f:
        json.dump(data_to_save, f, indent=4, default=serialize)
    print(f"Saved median/best/worst: {file_name}")
