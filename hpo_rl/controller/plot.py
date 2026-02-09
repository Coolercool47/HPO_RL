import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
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
    def __init__(self, history, best_result, save_path, backend, experiment_number = 0): #Сделать experiment_number - optional
        """Инициализация plot_and_save

        Args:
            history: история сгенерированных гиперпараметров
            best_result: лучший результат 
            save_path: папка для сохранения таблиц и изображений
            backend: выбранный `backend`
            experiment_number: номер экперимента
        
        """
        self.history = history
        self.best_result = best_result
        self.save_path = save_path
        self.backend = backend
        self.experiment_number = experiment_number

    def plot_3d(self):
        """Функция, создающая изображение функции на плоскости и в трехмерии"""
        # print(self.history)
        x0_vals = [t[0]["x0"] for t in self.history]
        x1_vals = [t[0]["x1"] for t in self.history]
        rewards = [t[-1] for t in self.history]
        metrics = rewards if self.backend.maximize else [-r for r in rewards]

        # Сетка для contour/surface
        bounds = self.backend.bounds
        grid = np.linspace(bounds[0], bounds[1], 100)
        X0, X1 = np.meshgrid(grid, grid)

        Z = np.zeros_like(X0)
        for i in range(X0.shape[0]):
            for j in range(X0.shape[1]):
                val = self.backend.evaluate({"x0": X0[i, j], "x1": X1[i, j]})
                Z[i, j] = val if self.backend.maximize else -val

        fig, (ax1, _) = plt.subplots(1, 2, figsize=(16, 6))

        contour = ax1.contour(X0, X1, Z, levels=20, cmap='viridis', alpha=0.6)
        ax1.clabel(contour, inline=True, fontsize=8)
        ax1.contourf(X0, X1, Z, levels=20, cmap='viridis', alpha=0.3)

        ax1.plot(x0_vals, x1_vals, 'r-', linewidth=2, alpha=0.7, label='Trajectory')
        ax1.scatter(x0_vals[0], x1_vals[0], c='green', s=100, marker='o',
                    label='Start', zorder=5, edgecolors='black', linewidths=2)
        ax1.scatter(x0_vals[-1], x1_vals[-1], c='red', s=100, marker='*',
                    label='End', zorder=5, edgecolors='black', linewidths=2)

        best_idx = np.argmax(metrics) if self.backend.maximize else np.argmin(metrics)
        ax1.scatter(x0_vals[best_idx], x1_vals[best_idx], c='yellow', s=150,
                    marker='X', label='Best', zorder=5, edgecolors='black', linewidths=2)
        marker_coords = (x0_vals[best_idx], x1_vals[best_idx])
        marker_value = metrics[best_idx]

        opt_type = "max" if self.backend.maximize else "min"
        # print(opt_type)
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
        temp_path = self.save_path / f"3d_{self.experiment_number}.png"
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        temp_path_pgf = self.save_path / f"3d_{self.experiment_number}.pgf"
        plt.savefig(temp_path_pgf, dpi=150, bbox_inches='tight')
        print(f"Saved: {temp_path}, {temp_path_pgf}")
        plt.close()


    def plot_trajectory(self):
        """Функция, создающая изображение с историей наград"""
        is_maximize = self.backend.maximize
        history_scores = [d[-1] if is_maximize else -d[-1] for d in self.history]
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
        temp_path = self.save_path / f"trajectory_{self.experiment_number}.png"
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        temp_path_pgf = self.save_path / f"trajectory_{self.experiment_number}.pgf"
        plt.savefig(temp_path_pgf, dpi=150, bbox_inches='tight')
        print(f"Saved: {temp_path}, {temp_path_pgf}")
        plt.close()
        # добавить сохранение
    
    def save_history(self, as_latex=True):
        """Функция, сохраняющая историю в виде таблицы и Latex кода"""
        data = []
        for i, (params, score) in enumerate(self.history, 1):
            row = {"Iteration": i}
            row.update(params)
            row["Objective"] = -score
            data.append(row)

        df = pd.DataFrame(data).set_index("Iteration")
        
        if not as_latex:
            out_path = self.save_path / f"history_{self.experiment_number}.csv"
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

        out_path = self.save_path / f"history_table_{self.experiment_number}.tex"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"Saved TEX history: {out_path}")

