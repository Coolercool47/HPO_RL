"""Визуализация и сохранение результатов экспериментов HPO.

Модуль содержит :class:`plot_and_save` — построение графиков траектории,
3D-ландшафта, наград и экспорт истории в CSV/LaTeX.
"""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, ScalarFormatter
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from datetime import datetime
import pandas as pd
import re

# Default font sizes for publication-ready figures
_TITLE_FONTSIZE = 18
_LABEL_FONTSIZE = 15
_TICK_FONTSIZE = 13
_LEGEND_FONTSIZE = 13
_CLABEL_FONTSIZE = 11

# Larger fonts for 2D Trajectory / 3D View panels in plot_3d
_TRAJ3D_TITLE_FONTSIZE = 24
_TRAJ3D_LABEL_FONTSIZE = 20
_TRAJ3D_TICK_FONTSIZE = 18
_TRAJ3D_LEGEND_FONTSIZE = 18

# 3D View panel: slightly smaller labels, tick numbers on 3D axes only
_VIEW3D_TITLE_FONTSIZE = 20
_VIEW3D_LABEL_FONTSIZE = 16
_VIEW3D_TICK_FONTSIZE = 13


def _format_sci_tick(val, _pos):
    """Формат делений оси: 160000 -> 1.6e5; малые значения — без e-нотации."""
    if abs(val) < 1e-15:
        return '0'
    av = abs(val)
    if av >= 1e4 or (av < 1e-2 and av > 0):
        exp = int(np.floor(np.log10(av)))
        mant = val / (10 ** exp)
        if abs(mant - round(mant)) < 0.05:
            m = int(round(mant))
            sign = '-' if m < 0 else ''
            m = abs(m)
            return f'{sign}{m}e{exp}' if m != 1 else f'{sign}1e{exp}'
        s = f'{mant:.1f}e{exp}'
        return s.replace('.0e', 'e')
    if av >= 100:
        return f'{val:.0f}'
    if av >= 10:
        return f'{val:.0f}'
    return f'{val:g}'


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
        experiment_number: номер эксперимента в серии запусков

    Пример::

        outputs = plot_and_save(history, best, save_path, backend, experiment_number=0)
        outputs.plot_trajectory()
        outputs.save_history(as_latex=True)

    """
    def __init__(self, history, best_result, save_path, backend, experiment_number=0):
        """Инициализация plot_and_save

        Args:
            history: история сгенерированных гиперпараметров
            best_result: лучший результат
            save_path: папка для сохранения таблиц и изображений
            backend: выбранный `backend`
            experiment_number: номер эксперимента

        """
        self.best_result = best_result
        self.save_path = save_path
        self.backend = backend
        self.experiment_number = experiment_number
        self.history = history

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
        grid_x0 = np.linspace(bounds[0][0], bounds[0][1], 100)
        grid_x1 = np.linspace(bounds[1][0], bounds[1][1], 100) if len(bounds) > 1 else grid_x0
        X0, X1 = np.meshgrid(grid_x0, grid_x1)
        Z = np.zeros_like(X0)
        for i in range(X0.shape[0]):
            for j in range(X0.shape[1]):
                Z[i, j] = self.backend.evaluate({"x0": X0[i, j], "x1": X1[i, j]})

        fig = plt.figure(figsize=(16, 7))
        # Без tight: фиксированные отступы, ~10% справа под подписи оси Z
        ax1 = fig.add_axes([0.07, 0.12, 0.42, 0.76])
        ax2 = fig.add_axes([0.50, 0.12, 0.38, 0.76], projection='3d')

        ax1.contourf(X0, X1, Z, levels=20, cmap='viridis', alpha=0.15)
        ax1.contour(X0, X1, Z, levels=20, cmap='viridis', alpha=0.3)

        points = np.array([x0_vals, x1_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=truncated_plasma, array=colors, linewidth=2.5, zorder=3)
        ax1.add_collection(lc)

        ax1.scatter(x0_vals, x1_vals, c=colors, cmap=truncated_plasma, s=25, edgecolors='none', alpha=0.8, zorder=4)

        ax1.scatter(x0_vals[0], x1_vals[0], c='green', s=100, marker='o', label='Start', zorder=5, edgecolors='white')
        ax1.scatter(x0_vals[-1], x1_vals[-1], c='red', s=120, marker='*', label='End', zorder=5, edgecolors='white')

        best_idx = np.argmax(metrics) if self.backend.maximize else np.argmin(metrics)
        ax1.scatter(x0_vals[best_idx], x1_vals[best_idx], c='cyan', s=150, marker='X', label='Best', zorder=6, edgecolors='black')

        ax1.set_title('2D Trajectory', fontsize=_TRAJ3D_TITLE_FONTSIZE)
        ax1.set_xlabel('x0', fontsize=_TRAJ3D_LABEL_FONTSIZE)
        ax1.set_ylabel('x1', fontsize=_TRAJ3D_LABEL_FONTSIZE)
        ax1.tick_params(axis='both', labelsize=_TRAJ3D_TICK_FONTSIZE)
        ax1.set_xlim(bounds[0][0], bounds[0][1])
        ax1.set_ylim(bounds[1][0], bounds[1][1])
        ax1.set_aspect('equal')
        ax1.legend(fontsize=_TRAJ3D_LEGEND_FONTSIZE)

        ax2.plot_surface(X0, X1, Z, cmap='viridis', alpha=0.3, linewidth=0, antialiased=True)

        points3d = np.array([x0_vals, x1_vals, metrics]).T.reshape(-1, 1, 3)
        segments3d = np.concatenate([points3d[:-1], points3d[1:]], axis=1)
        lc3d = Line3DCollection(segments3d, cmap=truncated_plasma, array=colors, linewidth=3)
        ax2.add_collection3d(lc3d)

        ax2.scatter(x0_vals, x1_vals, metrics, c=colors, cmap=truncated_plasma, s=30, depthshade=False)

        ax2.scatter(x0_vals[0], x1_vals[0], metrics[0], c='green', s=100, marker='o', edgecolors='white')
        ax2.scatter(x0_vals[-1], x1_vals[-1], metrics[-1], c='red', s=130, marker='*', edgecolors='white')
        ax2.scatter(x0_vals[best_idx], x1_vals[best_idx], metrics[best_idx], c='cyan', s=150, marker='X', edgecolors='black')

        ax2.set_title('3D View', fontsize=_VIEW3D_TITLE_FONTSIZE)
        ax2.set_xlim(bounds[0][0], bounds[0][1])
        ax2.set_ylim(bounds[1][0], bounds[1][1])
        ax2.set_xlabel('x0', fontsize=_VIEW3D_LABEL_FONTSIZE, labelpad=6)
        ax2.set_ylabel('x1', fontsize=_VIEW3D_LABEL_FONTSIZE, labelpad=6)
        ax2.set_zlabel('Objective', fontsize=_VIEW3D_LABEL_FONTSIZE, labelpad=10)
        ax2.zaxis.set_major_locator(MaxNLocator(nbins=5, integer=False))
        ax2.zaxis.set_major_formatter(FuncFormatter(_format_sci_tick))
        ax2.zaxis.get_offset_text().set_visible(False)
        ax2.tick_params(axis='x', labelsize=_VIEW3D_TICK_FONTSIZE, pad=1)
        ax2.tick_params(axis='y', labelsize=_VIEW3D_TICK_FONTSIZE, pad=1)
        ax2.tick_params(axis='z', labelsize=_VIEW3D_TICK_FONTSIZE, pad=1)
        ax2.zaxis.set_rotate_label(True)

        file_label = f"3d_{self.experiment_number}{suffix}"
        temp_path = self.save_path / f"{file_label}.png"
        plt.savefig(temp_path, dpi=150)
        temp_path_pgf = self.save_path / f"{file_label}.pgf"
        plt.savefig(temp_path_pgf, dpi=150)
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

        plt.title(
            f"Optimization History ({'Minimization' if not is_maximize else 'Maximization'})",
            fontsize=_TITLE_FONTSIZE,
        )
        plt.xlabel("Iteration", fontsize=_LABEL_FONTSIZE)
        plt.ylabel("Objective function", fontsize=_LABEL_FONTSIZE)

        plt.yscale('linear')

        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        y_formatter = ScalarFormatter(useOffset=False)
        y_formatter.set_scientific(False)
        plt.gca().yaxis.set_major_formatter(y_formatter)

        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.gca().tick_params(axis='both', labelsize=_TICK_FONTSIZE)

        plt.legend(frameon=True, loc='upper right', fontsize=_LEGEND_FONTSIZE)
        plt.tight_layout()


        file_label = f"trajectory_{self.experiment_number}{suffix}"
        temp_path = self.save_path / f"{file_label}.png"
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        temp_path_pgf = self.save_path / f"{file_label}.pgf"
        plt.savefig(temp_path_pgf, dpi=150, bbox_inches='tight')
        print(f"Saved: {temp_path}, {temp_path_pgf}")
        plt.close()

    def plot_reward(self, rewards, suffix=""):
        """Строит график пошаговой награды и кумулятивной награды.

        Args:
            rewards: список наград за каждый шаг эпизода.
            suffix: дополнительный суффикс для имени файла.
        """
        if not rewards:
            return

        rewards = np.array(rewards, dtype=np.float64)
        steps = np.arange(1, len(rewards) + 1)
        cumulative = np.cumsum(rewards)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)


        ax1.plot(steps, rewards, linewidth=1.0, color='steelblue', alpha=0.7, label='Per-step reward')

        if len(rewards) >= 10:
            window = max(5, len(rewards) // 20)
            kernel = np.ones(window) / window
            smoothed = np.convolve(rewards, kernel, mode='valid')
            offset = window // 2
            ax1.plot(steps[offset:offset + len(smoothed)], smoothed,
                     linewidth=2.0, color='darkblue', label=f'Moving avg (w={window})')
        ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax1.set_ylabel('Reward', fontsize=_LABEL_FONTSIZE)
        ax1.set_title('Per-step Reward', fontsize=_TITLE_FONTSIZE)
        ax1.tick_params(axis='both', labelsize=_TICK_FONTSIZE)
        ax1.legend(loc='upper right', frameon=True, fontsize=_LEGEND_FONTSIZE)
        ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)


        ax2.plot(steps, cumulative, linewidth=2.0, color='darkorange', label='Cumulative reward')
        ax2.fill_between(steps, 0, cumulative, alpha=0.15, color='orange')
        ax2.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax2.set_xlabel('Step', fontsize=_LABEL_FONTSIZE)
        ax2.set_ylabel('Cumulative Reward', fontsize=_LABEL_FONTSIZE)
        ax2.set_title('Cumulative Reward', fontsize=_TITLE_FONTSIZE)
        ax2.tick_params(axis='both', labelsize=_TICK_FONTSIZE)
        ax2.legend(loc='upper left', frameon=True, fontsize=_LEGEND_FONTSIZE)
        ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        file_label = f"reward_{self.experiment_number}{suffix}"
        temp_path = self.save_path / f"{file_label}.png"
        plt.savefig(temp_path, dpi=150, bbox_inches='tight')
        temp_path_pgf = self.save_path / f"{file_label}.pgf"
        plt.savefig(temp_path_pgf, dpi=150, bbox_inches='tight')
        print(f"Saved: {temp_path}, {temp_path_pgf}")
        plt.close()

    def save_history(self, as_latex=True, suffix=""):
        """Функция, сохраняющая историю в виде таблицы и LaTeX-кода

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


        latex_table = latex_table.replace(r'\bottomrule', r'\midrule')
        latex_table = latex_table.replace(r'\endlastfoot', r'\bottomrule' + '\n' + r'\endlastfoot')

        out_path = self.save_path / f"history_table_{self.experiment_number}{suffix}.tex"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"Saved TEX history: {out_path}")

    def save_intermediate_csv(self, suffix=""):
        """Сохраняет промежуточные результаты в CSV (без LaTeX/графиков).

        Используется для промежуточных сохранений во время оптимизации
        и при сохранении после KeyboardInterrupt.

        Args:
            suffix: дополнительный суффикс для имени файла.
        """
        if not self.history:
            return
        data = []
        for i, (params, score) in enumerate(self.history, 1):
            row = {"Iteration": i}
            row.update(params)
            row["Objective"] = score
            data.append(row)
        df = pd.DataFrame(data).set_index("Iteration")
        out_path = self.save_path / f"intermediate_results_{self.experiment_number}{suffix}.csv"
        df.to_csv(out_path)
        print(f"Saved intermediate CSV: {out_path}")

    def save_hmm_history(self, history_table, suffix=""):
        """Сохраняет таблицу состояний HMM MCMC в CSV.

        Args:
            history_table: list[dict] из :class:`HMM_MCMC`.history_table.
            suffix: дополнительный суффикс для имени файла.
        """
        if not history_table:
            return
        df = pd.DataFrame(history_table)
        out_path = self.save_path / f"hmm_mcmc_history_{self.experiment_number}{suffix}.csv"
        df.to_csv(out_path, index=False)
        print(f"Saved HMM history: {out_path}")
