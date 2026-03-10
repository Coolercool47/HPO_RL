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
        if not getattr(expreiment_controller, '_checkpoint_loaded', False):
            raise RuntimeError(
                f"inference_only=True but checkpoint was NOT loaded from: "
                f"{expreiment_controller.load_loc!r}\n"
                f"Hint: если путь содержит backslash, используйте r\"...\" или '/' "
                f"(Python интерпретирует \\f как form-feed, \\n как newline и т.д.)"
            )

    full_history = []

    for i_experiment in range(n_experiments):
        if mode == "baseline" and i_experiment > 0:
            if hasattr(expreiment_controller.algorithm, "reset"):
                expreiment_controller.algorithm.reset()
        best_result = expreiment_controller.inference()
        history = expreiment_controller.return_history()
        rewards = expreiment_controller.return_rewards()
        outputs = plot_and_save(history, best_result, save_path, expreiment_controller.backend, experiment_number=i_experiment)
        if backend_name == "function" and expreiment_controller.backend.dimensions == 2:
            outputs.plot_3d()
        elif backend_name == "sequential":
            backend = expreiment_controller.backend
            merged_bounds = getattr(backend, '_merged_bounds', None)

            # Если env использует sync_bounds_to_backend (InstantContinuousPipelineEnv),
            # координаты в history уже в native bounds дочернего бэкенда — ремап не нужен.
            raw_env = getattr(expreiment_controller.env, 'unwrapped', expreiment_controller.env)
            env_has_native_bounds = hasattr(raw_env, 'sync_bounds_to_backend')
            child_merged_bounds = None if env_has_native_bounds else merged_bounds

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
                child_rewards = expreiment_controller.return_rewards()

                suffix = f"_{child_idx}_{fn_name}"
                child_outputs = plot_and_save(
                    child_history, child_best, save_path, child,
                    experiment_number=i_experiment,
                    merged_bounds=child_merged_bounds,
                )
                if isinstance(child, OptimizationBenchmarkBackend) and child.dimensions == 2:
                    child_outputs.plot_3d(suffix=suffix)
                child_outputs.plot_trajectory(suffix=suffix)
                child_outputs.plot_reward(child_rewards, suffix=suffix)
                child_outputs.save_history(as_latex=True, suffix=suffix)
                child_outputs.save_history(as_latex=False, suffix=suffix)
        elif not backend_name == "sequential":
            outputs.plot_trajectory()
            outputs.save_history(as_latex=True)
            outputs.save_history(as_latex=False)
            outputs.plot_reward(rewards)

        full_history.append(history)

    is_maximize = expreiment_controller.backend.maximize

    # Эпизоды могут иметь разную длину из-за early termination,
    # поэтому нельзя сложить в np.array напрямую.
    # Извлекаем лучший trial из каждого эпизода отдельно.
    best = []
    for history in full_history:
        scores = [trial[1] for trial in history]
        if is_maximize:
            idx = int(np.argmax(scores))
        else:
            idx = int(np.argmin(scores))
        best.append(history[idx])

    # Сортируем так, чтобы worst был первым (idx 0), best — последним (idx -1)
    best_scores = [item[1] for item in best]
    if is_maximize:
        sorted_idx = np.argsort(best_scores).tolist()
    else:
        sorted_idx = np.argsort(best_scores)[::-1].tolist()

    worst_of_best, best_of_best, median_of_best = best[sorted_idx[0]],  best[sorted_idx[-1]], best[sorted_idx[len(sorted_idx) // 2]]

    last = [inference[-1] for inference in full_history]
    last_scores = [item[1] for item in last]
    if is_maximize:
        last_idxs = np.argsort(last_scores).tolist()
    else:
        last_idxs = np.argsort(last_scores)[::-1].tolist()

    worst_of_last, best_of_last, median_of_last = last[last_idxs[0]], last[last_idxs[-1]], last[last_idxs[len(last_idxs)//2]]

    def _trial_to_serializable(trial):
        """Конвертирует trial (config, metric) в JSON-сериализуемый формат."""
        config, metric = trial
        if isinstance(config, dict):
            serialized_config = {k: float(v) if isinstance(v, (np.floating,)) else v for k, v in config.items()}
        elif isinstance(config, np.ndarray):
            serialized_config = config.tolist()
        elif config is None:
            serialized_config = None
        else:
            serialized_config = config
        if isinstance(metric, (np.floating,)):
            metric = float(metric)
        return [serialized_config, metric]

    data_to_save = {
        "best_of_each_inference": {
            "worst": _trial_to_serializable(worst_of_best),
            "best": _trial_to_serializable(best_of_best),
            "median": _trial_to_serializable(median_of_best)
        },
        "final_of_each_inference": {
            "worst": _trial_to_serializable(worst_of_last),
            "best": _trial_to_serializable(best_of_last),
            "median": _trial_to_serializable(median_of_last)
        }
    }

    def serialize(obj):
        if isinstance(obj, (np.float32, np.float64)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return str(obj)

    file_name = save_path / "inference_results.json"
    with open(file_name, "w") as f:
        json.dump(data_to_save, f, indent=4, default=serialize)
    print(f"Saved median/best/worst: {file_name}")

    config_file_name = save_path / "config.json"
    with open(config_file_name, "w") as f:
        json.dump(config, f, indent=4, default=serialize)
    print(f"Saved config: {config_file_name}")
