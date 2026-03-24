"""
Сравнение HMM_MCMC и TPE на четырёх типах задач:
  1) Много непрерывных + мало категориальных гиперпараметров
  2) Мало непрерывных + много категориальных гиперпараметров
  3) Только непрерывные гиперпараметры
  4) Только категориальные гиперпараметры

Параметры алгоритмов (кроме бюджета) взяты из run_exp_HMM_MCMC.py.
Результаты выводятся в таблицу и сохраняются в comparison_results.txt.
"""

import numpy as np
import sys
import io
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hpo_rl.baselines.HMM_MCMC import HMM_MCMC
from hpo_rl.baselines.TPE import TPE


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario 1: 8 continuous + 2 categorical
# ═══════════════════════════════════════════════════════════════════════════

def objective_many_continuous(cfg):
    """Schwefel 8D + 2 categorical.

    8 непрерывных параметров x0..x7 (Schwefel landscape).
    scale ∈ {small, medium, large} — масштабирование функции.
    shift ∈ {none, left, right} — сдвиг аргументов на 50.

    Оптимум: scale=medium, shift=none, x_i ≈ 420.9687 → f ≈ 0.0
    """
    xs = np.array([cfg[f'x{i}'] for i in range(8)])
    scale = cfg['scale']
    shift = cfg['shift']

    if shift == 'left':
        xs = xs - 50.0
    elif shift == 'right':
        xs = xs + 50.0

    val = 418.9829 * 8 - np.sum(xs * np.sin(np.sqrt(np.abs(xs))))

    if scale == 'small':
        val *= 0.5
    elif scale == 'large':
        val *= 2.0

    return float(val)


SPACE_MANY_CONTINUOUS = {
    'x0': {'type': 'float', 'values': [-500.0, 500.0]},
    'x1': {'type': 'float', 'values': [-500.0, 500.0]},
    'x2': {'type': 'float', 'values': [-500.0, 500.0]},
    'x3': {'type': 'float', 'values': [-500.0, 500.0]},
    'x4': {'type': 'float', 'values': [-500.0, 500.0]},
    'x5': {'type': 'float', 'values': [-500.0, 500.0]},
    'x6': {'type': 'float', 'values': [-500.0, 500.0]},
    'x7': {'type': 'float', 'values': [-500.0, 500.0]},
    'scale': {'type': 'categorical', 'values': ['small', 'medium', 'large']},
    'shift': {'type': 'categorical', 'values': ['none', 'left', 'right']},
}


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario 2: 2 continuous + 6 categorical
# ═══════════════════════════════════════════════════════════════════════════

def objective_many_categorical(cfg):
    """Имитация HPO-задачи: 2 continuous + 6 categorical.

    lr ∈ [1e-5, 1e-1], momentum ∈ [0.0, 0.99]
    optimizer ∈ {SGD, Adam, AdamW}
    activation ∈ {relu, gelu, tanh, swish}
    scheduler ∈ {none, cosine, step, exponential}
    dropout ∈ {none, light, heavy}
    norm ∈ {none, batch, layer, group}
    init ∈ {default, xavier, kaiming}

    Глубокий ландшафт с зависимостями между категориальными параметрами.
    Оптимум: AdamW, gelu, cosine, light, layer, kaiming, lr≈0.001, momentum≈0 → ~0.01
    """
    lr = cfg['lr']
    momentum = cfg['momentum']
    optimizer = cfg['optimizer']
    activation = cfg['activation']
    scheduler = cfg['scheduler']
    dropout = cfg['dropout']
    norm = cfg['norm']
    init = cfg['init']

    # Base loss from optimizer choice
    opt_map = {'Adam': (0.001, 0.5, 0.0), 'AdamW': (0.001, 0.3, 0.0),
               'SGD': (0.01, 0.8, 0.9)}
    lr_opt, base_penalty, mom_opt = opt_map[optimizer]

    lr_loss = 2.0 * (np.log10(lr) - np.log10(lr_opt))**2
    mom_loss = 0.5 * (momentum - mom_opt)**2

    # Activation interaction
    act_table = {
        ('Adam', 'gelu'): -0.15, ('Adam', 'swish'): -0.10,
        ('Adam', 'relu'): 0.0, ('Adam', 'tanh'): 0.2,
        ('AdamW', 'gelu'): -0.20, ('AdamW', 'swish'): -0.12,
        ('AdamW', 'relu'): 0.0, ('AdamW', 'tanh'): 0.25,
        ('SGD', 'tanh'): -0.10, ('SGD', 'relu'): 0.0,
        ('SGD', 'gelu'): 0.15, ('SGD', 'swish'): 0.10,
    }
    act_bonus = act_table.get((optimizer, activation), 0.0)

    # Scheduler interaction
    sched_table = {
        'cosine': -0.15, 'exponential': -0.05, 'step': 0.10, 'none': 0.0
    }
    sched_bonus = sched_table[scheduler]
    if scheduler == 'step' and optimizer in ('Adam', 'AdamW'):
        sched_bonus += 0.10

    # Dropout effect
    drop_table = {'none': 0.05, 'light': -0.10, 'heavy': 0.15}
    drop_bonus = drop_table[dropout]

    # Normalization effect
    norm_table = {'none': 0.10, 'batch': -0.05, 'layer': -0.12, 'group': -0.03}
    norm_bonus = norm_table[norm]

    # Init effect
    init_table = {'default': 0.0, 'xavier': -0.05, 'kaiming': -0.08}
    init_bonus = init_table[init]

    noise = 0.01 * np.random.randn()
    total = base_penalty + lr_loss + mom_loss + act_bonus + sched_bonus + drop_bonus + norm_bonus + init_bonus + noise
    return float(max(total, 0.01))


SPACE_MANY_CATEGORICAL = {
    'lr': {'type': 'float', 'values': [1e-5, 1e-1]},
    'momentum': {'type': 'float', 'values': [0.0, 0.99]},
    'optimizer': {'type': 'categorical', 'values': ['SGD', 'Adam', 'AdamW']},
    'activation': {'type': 'categorical', 'values': ['relu', 'gelu', 'tanh', 'swish']},
    'scheduler': {'type': 'categorical', 'values': ['none', 'cosine', 'step', 'exponential']},
    'dropout': {'type': 'categorical', 'values': ['none', 'light', 'heavy']},
    'norm': {'type': 'categorical', 'values': ['none', 'batch', 'layer', 'group']},
    'init': {'type': 'categorical', 'values': ['default', 'xavier', 'kaiming']},
}


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario 3: Only continuous (10 float, 0 categorical)
# ═══════════════════════════════════════════════════════════════════════════

def objective_only_continuous(cfg):
    """Schwefel 10D, только непрерывные параметры.

    10 непрерывных x0..x9.
    Оптимум: x_i ≈ 420.9687 → f ≈ 0.0
    """
    xs = np.array([cfg[f'x{i}'] for i in range(10)])
    return float(418.9829 * 10 - np.sum(xs * np.sin(np.sqrt(np.abs(xs)))))


SPACE_ONLY_CONTINUOUS = {
    f'x{i}': {'type': 'float', 'values': [-500.0, 500.0]}
    for i in range(10)
}


# ═══════════════════════════════════════════════════════════════════════════
#  Scenario 4: Only categorical (0 float, 8 categorical)
# ═══════════════════════════════════════════════════════════════════════════

def objective_only_categorical(cfg):
    """Комбинаторная задача: 8 категориальных параметров.

    Имитация выбора архитектуры нейросети без непрерывных параметров.
    optimizer ∈ {SGD, Adam, AdamW, RMSprop}
    activation ∈ {relu, gelu, tanh, swish, mish}
    scheduler ∈ {none, cosine, step, exponential, plateau}
    dropout ∈ {none, light, medium, heavy}
    norm ∈ {none, batch, layer, group}
    init ∈ {default, xavier, kaiming, orthogonal}
    pool ∈ {max, avg, adaptive}
    loss_fn ∈ {ce, focal, label_smooth}

    Оптимум: AdamW, gelu, cosine, light, layer, kaiming, adaptive, label_smooth → 0.01
    """
    optimizer = cfg['optimizer']
    activation = cfg['activation']
    scheduler = cfg['scheduler']
    dropout = cfg['dropout']
    norm = cfg['norm']
    init = cfg['init']
    pool = cfg['pool']
    loss_fn = cfg['loss_fn']

    # Base from optimizer
    opt_score = {'Adam': 0.4, 'AdamW': 0.25, 'SGD': 0.7, 'RMSprop': 0.5}[optimizer]

    # Activation interaction with optimizer
    act_table = {
        ('AdamW', 'gelu'): -0.20, ('AdamW', 'swish'): -0.12, ('AdamW', 'mish'): -0.10,
        ('AdamW', 'relu'): 0.0, ('AdamW', 'tanh'): 0.20,
        ('Adam', 'gelu'): -0.15, ('Adam', 'swish'): -0.08,  ('Adam', 'mish'): -0.06,
        ('Adam', 'relu'): 0.0,  ('Adam', 'tanh'): 0.15,
        ('SGD', 'tanh'): -0.08, ('SGD', 'relu'): 0.0, ('SGD', 'gelu'): 0.10,
        ('SGD', 'swish'): 0.08, ('SGD', 'mish'): 0.05,
        ('RMSprop', 'gelu'): -0.05, ('RMSprop', 'relu'): 0.0,
        ('RMSprop', 'tanh'): 0.10, ('RMSprop', 'swish'): -0.03, ('RMSprop', 'mish'): -0.02,
    }
    act_bonus = act_table.get((optimizer, activation), 0.0)

    sched_score = {'cosine': -0.15, 'exponential': -0.05, 'plateau': -0.08,
                   'step': 0.10, 'none': 0.0}[scheduler]
    if scheduler == 'step' and optimizer in ('Adam', 'AdamW'):
        sched_score += 0.10

    drop_score = {'none': 0.05, 'light': -0.10, 'medium': 0.0, 'heavy': 0.15}[dropout]
    norm_score = {'none': 0.10, 'batch': -0.05, 'layer': -0.12, 'group': -0.03}[norm]
    init_score = {'default': 0.0, 'xavier': -0.05, 'kaiming': -0.08, 'orthogonal': -0.04}[init]
    pool_score = {'max': 0.0, 'avg': -0.03, 'adaptive': -0.06}[pool]
    loss_score = {'ce': 0.0, 'focal': -0.03, 'label_smooth': -0.05}[loss_fn]

    noise = 0.01 * np.random.randn()
    total = opt_score + act_bonus + sched_score + drop_score + norm_score + init_score + pool_score + loss_score + noise
    return float(max(total, 0.01))


SPACE_ONLY_CATEGORICAL = {
    'optimizer':  {'type': 'categorical', 'values': ['SGD', 'Adam', 'AdamW', 'RMSprop']},
    'activation': {'type': 'categorical', 'values': ['relu', 'gelu', 'tanh', 'swish', 'mish']},
    'scheduler':  {'type': 'categorical', 'values': ['none', 'cosine', 'step', 'exponential', 'plateau']},
    'dropout':    {'type': 'categorical', 'values': ['none', 'light', 'medium', 'heavy']},
    'norm':       {'type': 'categorical', 'values': ['none', 'batch', 'layer', 'group']},
    'init':       {'type': 'categorical', 'values': ['default', 'xavier', 'kaiming', 'orthogonal']},
    'pool':       {'type': 'categorical', 'values': ['max', 'avg', 'adaptive']},
    'loss_fn':    {'type': 'categorical', 'values': ['ce', 'focal', 'label_smooth']},
}


# ═══════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════

# Параметры HMM_MCMC из run_exp_HMM_MCMC.py (без budget)
HMM_KWARGS = dict(
    n_init=5,
    n_chains=1,
    orchestrate_every=15,
    T_mcmc=0.01,
    sigma_fraction=0.005,
    wide_sigma_fraction=0.8,
    temperature=0.30,
    hmm_window=8,
    hmm_obs_epsilon=1e-8,
    hmm_lambda_noise=0.01,
    clone_noise=0.05,
    burnin_fraction=0.10,
    p_cat_step=0.05,
    anneal_T=True,
)

# Параметры TPE из run_exp_HMM_MCMC.py (без budget)
TPE_KWARGS = dict(
    N_init=5,
    N_s=20,
    separation_value=0.2,
)


def _cumulative_best(data):
    """Из списка (config, score) строит массив кумулятивного лучшего."""
    scores = [s for _, s in data]
    cum = np.minimum.accumulate(scores)
    return cum


def run_hmm_mcmc(obj_func, space, budget, n_seeds, **extra_kwargs):
    kwargs = {**HMM_KWARGS, **extra_kwargs}
    results = []
    histories = []
    for seed in range(n_seeds):
        np.random.seed(seed * 42 + 7)
        algo = HMM_MCMC(
            objective_func=obj_func,
            budget=budget,
            dict_to_optimize=space,
            **kwargs,
        )
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            _, best_loss = algo.main_loop()
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
        results.append(best_loss)
        histories.append(_cumulative_best(algo.data))
    return np.array(results), histories


def run_tpe(obj_func, space, budget, n_seeds, **extra_kwargs):
    kwargs = {**TPE_KWARGS, **extra_kwargs}
    results = []
    histories = []
    for seed in range(n_seeds):
        np.random.seed(seed * 42 + 7)
        algo = TPE(
            objective_func=obj_func,
            N_init=kwargs['N_init'],
            N_s=kwargs['N_s'],
            budget=budget,
            dict_to_optimize=space,
            separation_value=kwargs['separation_value'],
        )
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            _, best_loss = algo.main_loop()
        finally:
            sys.stdout = old_out
            sys.stderr = old_err
        results.append(best_loss)
        histories.append(_cumulative_best(algo.data))
    return np.array(results), histories


def compute_stats(losses):
    return {
        'mean': np.mean(losses),
        'median': np.median(losses),
        'std': np.std(losses),
        'best': np.min(losses),
        'worst': np.max(losses),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HMM_MCMC vs TPE comparison")
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds per experiment")
    parser.add_argument("--budget", type=int, default=400, help="Evaluation budget for both algorithms")
    args = parser.parse_args()

    N_SEEDS = args.seeds
    BUDGET = args.budget

    scenarios = [
        {
            'name': 'Only continuous (10F)',
            'obj': objective_only_continuous,
            'space': SPACE_ONLY_CONTINUOUS,
        },
        {
            'name': 'Many continuous (8F + 2C)',
            'obj': objective_many_continuous,
            'space': SPACE_MANY_CONTINUOUS,
        },
        {
            'name': 'Many categorical (2F + 6C)',
            'obj': objective_many_categorical,
            'space': SPACE_MANY_CATEGORICAL,
        },
        {
            'name': 'Only categorical (8C)',
            'obj': objective_only_categorical,
            'space': SPACE_ONLY_CATEGORICAL,
        },
    ]

    all_rows = []

    for sc in scenarios:
        print(f"\n{'='*70}")
        print(f"  Scenario: {sc['name']}  |  budget={BUDGET}  seeds={N_SEEDS}")
        print(f"{'='*70}")

        print("  Running HMM_MCMC ...")
        hmm_losses, hmm_histories = run_hmm_mcmc(sc['obj'], sc['space'], BUDGET, N_SEEDS)
        hmm_stats = compute_stats(hmm_losses)
        print(f"    done — mean={hmm_stats['mean']:.4f}")

        print("  Running TPE ...")
        tpe_losses, tpe_histories = run_tpe(sc['obj'], sc['space'], BUDGET, N_SEEDS)
        tpe_stats = compute_stats(tpe_losses)
        print(f"    done — mean={tpe_stats['mean']:.4f}")

        all_rows.append((sc['name'], 'HMM_MCMC', hmm_stats, hmm_losses))
        all_rows.append((sc['name'], 'TPE', tpe_stats, tpe_losses))
        sc['hmm_histories'] = hmm_histories
        sc['tpe_histories'] = tpe_histories

    # ─── Format table ───────────────────────────────────────────────────
    header = f"{'Scenario':<28} {'Algorithm':<12} {'Mean':>10} {'Median':>10} {'Std':>10} {'Best':>10} {'Worst':>10}"
    sep = '-' * len(header)

    lines = []
    lines.append("=" * len(header))
    lines.append("  HMM_MCMC vs TPE — Hyperparameter Optimization Comparison")
    lines.append(f"  Budget: {BUDGET}   Seeds: {N_SEEDS}")
    lines.append("=" * len(header))
    lines.append("")
    lines.append(header)
    lines.append(sep)

    prev_scenario = None
    for scenario_name, algo_name, stats, losses in all_rows:
        if prev_scenario and prev_scenario != scenario_name:
            lines.append(sep)
        prev_scenario = scenario_name
        row = (
            f"{scenario_name:<28} {algo_name:<12} "
            f"{stats['mean']:>10.4f} {stats['median']:>10.4f} {stats['std']:>10.4f} "
            f"{stats['best']:>10.4f} {stats['worst']:>10.4f}"
        )
        lines.append(row)

    lines.append(sep)
    lines.append("")

    # Per-seed details
    lines.append("Per-seed best losses:")
    lines.append("")
    for scenario_name, algo_name, stats, losses in all_rows:
        seed_str = ", ".join(f"{v:.4f}" for v in losses)
        lines.append(f"  {scenario_name} | {algo_name}: [{seed_str}]")

    lines.append("")

    table_text = "\n".join(lines)
    print("\n" + table_text)

    out_path = "comparison_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(table_text)
    print(f"\nResults saved to {out_path}")

    # ─── Plot convergence histories ─────────────────────────────────────
    n_scenarios = len(scenarios)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(6 * n_scenarios, 5), squeeze=False)
    axes = axes[0]

    hmm_color = '#2176AE'
    tpe_color = '#E84855'

    for ax, sc in zip(axes, scenarios):
        for i, h in enumerate(sc['hmm_histories']):
            ax.plot(range(1, len(h) + 1), h, color=hmm_color, alpha=0.35, linewidth=0.8,
                    label='HMM_MCMC' if i == 0 else None)
        for i, h in enumerate(sc['tpe_histories']):
            ax.plot(range(1, len(h) + 1), h, color=tpe_color, alpha=0.35, linewidth=0.8,
                    label='TPE' if i == 0 else None)

        # Mean convergence curve
        max_len = max(len(h) for h in sc['hmm_histories'])
        hmm_mean = np.mean([np.pad(h, (0, max_len - len(h)), constant_values=h[-1])
                            for h in sc['hmm_histories']], axis=0)
        max_len_t = max(len(h) for h in sc['tpe_histories'])
        tpe_mean = np.mean([np.pad(h, (0, max_len_t - len(h)), constant_values=h[-1])
                            for h in sc['tpe_histories']], axis=0)
        ax.plot(range(1, len(hmm_mean) + 1), hmm_mean, color=hmm_color, linewidth=2.5,
                linestyle='--', label='HMM_MCMC (mean)')
        ax.plot(range(1, len(tpe_mean) + 1), tpe_mean, color=tpe_color, linewidth=2.5,
                linestyle='--', label='TPE (mean)')

        ax.set_title(sc['name'], fontsize=12)
        ax.set_xlabel('Evaluation #')
        ax.set_ylabel('Best loss so far')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'HMM_MCMC vs TPE — Convergence (budget={BUDGET}, seeds={N_SEEDS})',
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig_path = 'comparison_convergence.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Convergence plot saved to {fig_path}")
