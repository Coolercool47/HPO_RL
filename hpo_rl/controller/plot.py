import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from datetime import datetime
import pandas as pd
import re

class plot_and_save():
    """Класс для создания таблиц и изображений.
    
    Args:
        history: история сгенерированных гиперпараметров
        best_result: лучший результат 
        save_path: папка для сохранения таблиц и изображений
        backend: выбранный `backend`

    Attributes:
        history: история сгенерированных гиперпараметров
        best_result: лучший результат 
        save_path: папка для сохранения таблиц и изображений
        backend: выбранный `backend`
    
    """
    def __init__(self, history, best_result, save_path, backend, experiment_number = 0, merged_bounds = None): #Сделать experiment_number - optional
        """Инициализация plot_and_save

        Args:
            history: история сгенерированных гиперпараметров
            best_result: лучший результат 
            save_path: папка для сохранения таблиц и изображений
            backend: выбранный `backend`
            experiment_number: номер экперимента
            merged_bounds: объединённые границы из SequentialBackend для ремаппинга координат
        
        """
        self.best_result = best_result
        self.save_path = save_path
        self.backend = backend
        self.experiment_number = experiment_number
        self.merged_bounds = merged_bounds

        # Если переданы merged_bounds и backend имеет свои bounds —
        # ремапим координаты из merged в child bounds и пересчитываем метрики
        if merged_bounds and hasattr(backend, 'bounds') and hasattr(backend, 'dimensions'):
            self.history = self._remap_history(history)
        else:
            self.history = history

    def _remap_history(self, history):
        """Ремапит координаты из merged bounds в bounds текущего backend и пересчитывает метрики.

        Returns:
            Новая history с ремапленными координатами и пересчитанными метриками.
        """
        child_bounds_map = {}
        for i in range(self.backend.dimensions):
            key = f"x{i}"
            child_bounds_map[key] = self.backend.bounds

        remapped_history = []
        for config, _metric in history:
            new_config = {}
            for key, val in config.items():
                if key in self.merged_bounds and key in child_bounds_map:
                    m_lo, m_hi = self.merged_bounds[key]
                    c_lo, c_hi = child_bounds_map[key]
                    if m_hi - m_lo > 1e-12 and (m_lo != c_lo or m_hi != c_hi):
                        t = (val - m_lo) / (m_hi - m_lo)
                        new_config[key] = c_lo + t * (c_hi - c_lo)
                    else:
                        new_config[key] = val
                else:
                    new_config[key] = val
            # Пересчитываем метрику через child backend
            new_metric = self.backend.evaluate(new_config)
            remapped_history.append((new_config, new_metric))
        return remapped_history

    def plot_3d(self, suffix=""):
        """Функция, создающая изображение функции на плоскости и в трехмерии.
        
        Args:
            suffix: дополнительный суффикс для имени файла (например, имя функции).
        """
        x0_vals = np.array([t[0]["x0"] for t in self.history])
        x1_vals = np.array([t[0]["x1"] for t in self.history])
        metrics = np.array([t[-1] for t in self.history])
        
        n_points = len(x0_vals)
        colors = np.linspace(0, 1, n_points)

        original_plasma = plt.get_cmap('plasma')
        truncated_plasma = mcolors.LinearSegmentedColormap.from_list(
            'truncated_plasma', original_plasma(np.linspace(0, 0.85, 256))
        )

        bounds = self.backend.bounds
        grid = np.linspace(bounds[0], bounds[1], 100)
        X0, X1 = np.meshgrid(grid, grid)
        Z = np.zeros_like(X0)
        for i in range(X0.shape[0]):
            for j in range(X0.shape[1]):
                Z[i, j] = self.backend.evaluate({"x0": X0[i, j], "x1": X1[i, j]})

        fig, (ax1, _) = plt.subplots(1, 2, figsize=(16, 7))

        ax1.contourf(X0, X1, Z, levels=20, cmap='viridis', alpha=0.15)
        contour = ax1.contour(X0, X1, Z, levels=20, cmap='viridis', alpha=0.3)
        ax1.clabel(contour, inline=True, fontsize=8)

        points = np.array([x0_vals, x1_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=truncated_plasma, array=colors, linewidth=2.5, zorder=3)
        ax1.add_collection(lc)

        ax1.scatter(x0_vals, x1_vals, c=colors, cmap=truncated_plasma, s=25, edgecolors='none', alpha=0.8, zorder=4)
        
        ax1.scatter(x0_vals[0], x1_vals[0], c='green', s=100, marker='o', label='Start', zorder=5, edgecolors='white')
        ax1.scatter(x0_vals[-1], x1_vals[-1], c='red', s=120, marker='*', label='End', zorder=5, edgecolors='white')

        best_idx = np.argmax(metrics) if self.backend.maximize else np.argmin(metrics)
        ax1.scatter(x0_vals[best_idx], x1_vals[best_idx], c='cyan', s=150, marker='X', label='Best', zorder=6, edgecolors='black')

        ax1.set_title('2D Trajectory (Plasma Truncated)')
        ax1.set_aspect('equal')
        ax1.legend()

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot_surface(X0, X1, Z, cmap='viridis', alpha=0.3, linewidth=0, antialiased=True)

        points3d = np.array([x0_vals, x1_vals, metrics]).T.reshape(-1, 1, 3)
        segments3d = np.concatenate([points3d[:-1], points3d[1:]], axis=1)
        lc3d = Line3DCollection(segments3d, cmap=truncated_plasma, array=colors, linewidth=3)
        ax2.add_collection3d(lc3d)

        ax2.scatter(x0_vals, x1_vals, metrics, c=colors, cmap=truncated_plasma, s=30, depthshade=False)

        ax2.scatter(x0_vals[0], x1_vals[0], metrics[0], c='green', s=100, marker='o', edgecolors='white')
        ax2.scatter(x0_vals[-1], x1_vals[-1], metrics[-1], c='red', s=130, marker='*', edgecolors='white')
        ax2.scatter(x0_vals[best_idx], x1_vals[best_idx], metrics[best_idx], c='cyan', s=150, marker='X', edgecolors='black')

        ax2.set_title('3D View')

        plt.tight_layout()
        file_label = f"3d_{self.experiment_number}{suffix}"
        temp_path = self.save_path / f"{file_label}.png"
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        temp_path_pgf = self.save_path / f"{file_label}.pgf"
        plt.savefig(temp_path_pgf, dpi=150, bbox_inches='tight')
        print(f"Saved: {temp_path}, {temp_path_pgf}")
        plt.close()


    def plot_trajectory(self, suffix=""):
        """Функция, создающая изображение с историей наград
        
        Args:
            suffix: дополнительный суффикс для имени файла.
        """
        is_maximize = self.backend.maximize
        history_scores = [d[-1] for d in self.history]
        iterations = range(1, len(history_scores) + 1)
        
        if not is_maximize:
            best_so_far = np.minimum.accumulate(history_scores)
            label_best = 'Best Score (Min)'
        else:
            best_so_far = np.maximum.accumulate(history_scores)
            label_best = 'Best Score (Max)'
        
        plt.figure(figsize=(10, 6))

        plt.plot(iterations, history_scores, marker='o', markersize=4, linestyle='-', color='blue', 
                alpha=0.3, label='Iteration Score $f(x)$')

        plt.plot(iterations, best_so_far, color='red', linewidth=2, label=label_best)

        plt.title(f"Optimization History ({'Minimization' if not is_maximize else 'Maximization'})", fontsize=14)
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Objective function", fontsize=12) 

        plt.yscale('linear') 

        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        y_formatter = ScalarFormatter(useOffset=False)
        y_formatter.set_scientific(False) 
        plt.gca().yaxis.set_major_formatter(y_formatter)

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.legend(frameon=True, loc='upper right')
        plt.tight_layout()

        # Сохранение
        file_label = f"trajectory_{self.experiment_number}{suffix}"
        temp_path = self.save_path / f"{file_label}.png"
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        temp_path_pgf = self.save_path / f"{file_label}.pgf"
        plt.savefig(temp_path_pgf, dpi=150, bbox_inches='tight')
        print(f"Saved: {temp_path}, {temp_path_pgf}")
        plt.close()
    
    def save_history(self, as_latex=True, suffix=""):
        """Функция, сохраняющая историю в виде таблицы и Latex кода
        
        Args:
            as_latex: если True — сохраняет .tex, иначе .csv
            suffix: дополнительный суффикс для имени файла.
        """
        data = []
        for i, (params, score) in enumerate(self.history, 1):
            row = {"Iteration": i}
            row.update(params)
            row["Objective"] = score
            data.append(row)

        df = pd.DataFrame(data).set_index("Iteration")
        
        if not as_latex:
            out_path = self.save_path / f"history_{self.experiment_number}{suffix}.csv"
            df.to_csv(out_path)
            print(f"Saved CSV history: {out_path}")
            return

        obj_col = "Objective"
        min_idx = df[obj_col].idxmin()
        max_idx = df[obj_col].idxmax()

        df_latex = df.copy().astype(object) 
        
        for idx in df_latex.index:
            val = df.loc[idx, obj_col]
            formatted_val = f"{val:.5g}"
            if idx == min_idx:
                df_latex.loc[idx, obj_col] = f"\\cellcolor{{cyan!30}}{formatted_val}"
            elif idx == max_idx:
                df_latex.loc[idx, obj_col] = f"\\cellcolor{{orange!30}}{formatted_val}"
            else:
                df_latex.loc[idx, obj_col] = formatted_val

        col_format = 'r' * (len(df.columns)) + 'l' 
        
        latex_table = df_latex.to_latex(
            index=True,
            longtable=True,
            escape=False,
            caption=("Optimization History", "tab:opt_history"),
            column_format=col_format,
            label="tab:opt_history"
        )

        
        latex_table = re.sub(r'\\multicolumn\{\d+\}\{r\}\{Continued on next page\} \\\\', '', latex_table)
        
        # По желанию: если вы хотите, чтобы шапка повторялась на каждой странице (стандарт longtable),
        # Pandas уже это сделал через \endhead. 
        # Если вам нужно подправить конкретно разделители, делаем это точечно:
        latex_table = latex_table.replace(r'\bottomrule', r'\midrule') # Чтобы в конце промежуточных страниц была линия
        latex_table = latex_table.replace(r'\endlastfoot', r'\bottomrule' + '\n' + r'\endlastfoot')

        out_path = self.save_path / f"history_table_{self.experiment_number}{suffix}.tex"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"Saved TEX history: {out_path}")
