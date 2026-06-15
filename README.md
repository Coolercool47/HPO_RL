# HPO_RL: Гибридные Методы Оптимизации Гиперпараметров c использованием методов Обучения с Подкреплением

Фреймворк для решения задачи оптимизации гиперпараметров (Hyperparameter optimization, HPO).
Позволяет использовать классические алгоритмы, а также алгоритмы на основе методов обучения с подкреплением (Reinforcement learning, RL) для подбора гиперпараметров.

## Руководитель проекта 

- Парфенов Денис Васильевич, promasterden@yandex.ru

## Участники проекта

- Петерс Е. А., cool47.cool@yandex.ru - Тимлид, RL алгоритмы и фреймворк
- Матков Н. К., nikgenom@mail.ru - Разработчик, генетические и RL алгоритмы
- Тимошин Э. К., edik.timoshin@gmail.com - Разработчик, классические алгоритмы и фреймворк

## Описание проекта

Данный проект посвящен исследованию возможности применения методов RL для создания алгоритмов оптимизации гиперпараметров.

### Цель исследования

Разработка и экспериментальная оценка применения методов обучения с подкреплением для оптимизации гиперпараметров в различных нейросетевых алгоритмах.

### Основные задачи

- **Исследование современных методов оптимизации гиперпараметров**
- **Разработка алгоритмов использующих RL агентов для оптимизации гиперпараметров и сред для них** 
- **Анализ и оценка работы HPO RL в сравнении с классическими методами оптимизации** 

## Функциональность фреймворка

Фреймворк предлагает решение различных возможных задач оптимизации:

- **Тестовые функции** — набор бенчмарков различной сложности для проверки алгоритмов оптимизации (например, Rastrigin, Sphere, Rosenbrock, Ackley, Griewank, Schwefel, Levy, Michalewicz, Booth, Beale, Goldstein-Price и др.). Полный список задаётся в `hpo_rl/controller/check.py`.
- **Последовательный бэкенд (`SequentialBackend`)** — оптимизация нескольких целевых функций или objective-подзадач в одном эксперименте с режимами `random` и `shuffle`.
- **Тестовые и добавляемые модели (PyTorch)** — модели, обучаемые на стандартном цикле обучения, написанные на фреймворке PyTorch.
- **Целевая функция (`ObjectiveBackend`)** — гибкий контракт: пользователь сам определяет функцию, которую необходимо оптимизировать, что позволяет использовать различные ML-фреймворки и сценарии обучения.

Эти задачи можно решать при помощи следующих алгоритмов.

### Классические методы оптимизации гиперпараметров

- **Tree-Structured Parzen Estimator (TPE)** — [ссылка на статью](https://proceedings.neurips.cc/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf)
- **Hyperband** — [ссылка на статью](https://arxiv.org/abs/1603.06560)
- **Bayesian Optimization HyperBand (BOHB)** — [ссылка на статью](https://arxiv.org/abs/1807.01774)
- **SimpleGA (SGA)** — простой генетический алгоритм — [ссылка на книгу](https://www2.fiit.stuba.sk/~kvasnicka/Free%20books/Goldberg_Genetic_Algorithms_in_Search.pdf)
- **CMA-ES** — Covariance Matrix Adaptation Evolution Strategy — [ссылка на статью](https://arxiv.org/pdf/1604.00772)
- **HMM_MCMC** — метод на основе HMM и MCMC

### Алгоритмы на основе RL

RL-алгоритмы реализованы поверх [Tianshou](https://tianshou.org/) и делятся на on-policy и off-policy.

**On-policy:** A2C, NPG, PPO, RecurrentPPO, REINFORCE, TRPO.

**Off-policy:** BDQN, C51, DDPG, DiscreteSAC, DQN, RecurrentDQN, FQF, IQN, QRDQN, Rainbow, REDQ, SAC, TD3.

Дополнительно поддерживается **ICM (Intrinsic Curiosity Module)** — обёртка для исследования среды при обучении RL-агента.

#### Среды для RL-алгоритмов

- **`new_cycle_move_pipeline`** (`CyclicPipelineEnvNew`) — дискретная среда с циклическим покоординатным перебором: за один шаг агента изменяется один гиперпараметр. Float-параметры дискретизируются сеткой `num_bins` и шагами `step_sizes`; categorical-параметры выбираются напрямую. Постановка соответствует покоординатному спуску без требования дифференцируемости целевой функции.
- **`instant_continuous_pipeline`** (`InstantContinuousPipelineEnv`) — непрерывная среда: все float/int-параметры изменяются одновременно за один шаг. Вектор действий в `[-1, 1]` масштабируется через `max_delta_frac` и добавляется к текущим значениям с клипом в допустимые границы.

## Визуализация и логгирование

- Для 2D-задач — графики поверхности функции (2D и 3D) и траектории оптимизации.
- Для всех экспериментов — график изменения целевой метрики (функция потерь или метрика качества модели).
- Для RL — график пошаговых наград.
- Логирование истории проб в `.csv` и `.tex` (таблица исследованных точек).

Результаты сохраняются в каталог `logs/<algorithm>/<timestamp>/`: графики, истории, `inference_results.json` (агрегированные best/median/worst по повторным инференсам) и `config.json`.

Пример фрагмента CSV-истории:

```
Iteration,x0,x1,Objective
1,0.9197326,3.7290974,14.752075
2,0.9197326,3.695652,14.503752
3,1.086956,3.695652,14.839317
4,1.086956,3.6622066,14.593231
5,0.2508359,3.6622066,13.474676
6,0.2508359,3.6287622,13.230834
7,0.21739101,3.6287622,13.215174
8,0.21739101,3.5953178,12.97357
9,0.3846159,3.5953178,13.07424
10,0.3846159,3.5618725,12.834865
```

## Установка

```bash
pip install git+https://github.com/Coolercool47/HPO_RL.git@dev_SAC
```

## Удаление 
```bash
pip uninstall hpo_rl
```

## Структура

```
HPO_RL/
├── hpo_rl/
│   ├── alg/            # RecurrentPPO, RecurrentDQN, ICM-обёртки
│   ├── backends/       # function, objective, sequential, dummy
│   ├── baselines/      # TPE, BOHB, hyperband, SimpleGA, CMA_ES, HMM_MCMC
│   ├── controller/     # controller, check, plot
│   ├── environments/   # new_cycle_move_pipeline, instant_continuous_pipeline
│   ├── experiments/    # run_experiment (run_n_experiments)
│   └── nets/           # сети актора/критика, masked/recurrent, rainbow, ICM
├── experiments/        # запускаемые скрипты и ноутбук
├── docs/
├── tests/
├── requirements.txt
├── pyproject.toml
├── setup.py
└── README.md
```

## Запуск

Эксперимент с настраиваемой конфигурацией: обучение RL-алгоритма и оптимизация гиперпараметров заданной функции.

```bash
python experiments/run_experiment.py
```

Скрипт вызывает `run_n_experiments(config, n_experiments)` из `hpo_rl.experiments.run_experiment`. Внутри `experiments/run_experiment.py` задаётся конфигурация и число повторов инференса.

Пример конфигурации RL (PPO + дискретная среда + последовательный бэкенд):

```python
from hpo_rl.experiments.run_experiment import run_n_experiments
import torch
from tianshou.algorithm.modelfree.reinforce import ProbabilisticActorPolicy
from tianshou.utils.net.discrete import DiscreteCritic
from hpo_rl.nets.base_net import BaseNet
from hpo_rl.nets.masked_actor import MaskedDiscreteActor

config_ppo = {
    "full_args": {
        "algorithm": {
            "name": "ppo",
            "gamma": 0.97,
            "gae_lambda": 0.95,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "max_grad_norm": 0.5,
            "value_clip": True,
            "return_scaling": True,
            "recompute_advantage": True,
        },
        "optim": {
            "name": "TorchOptimizerFactory",
            "optim_class": torch.optim.Adam,
            "lr": 3e-4,
        },
        "net": {
            "actor": MaskedDiscreteActor,
            "critic": DiscreteCritic,
            "net": BaseNet,
            "hidden_sizes": [256, 256, 256],
        },
        "trainer": {
            "max_epochs": 100,
            "epoch_num_steps": 4000,
            "batch_size": 20,
            "collection_step_num_env_steps": 2000,
            "update_step_num_repetitions": 8,
            "test_step_num_episodes": 20,
        },
        "policy": {
            "class": ProbabilisticActorPolicy,
            "dist_fn": lambda x: torch.distributions.Categorical(logits=x),
            "action_scaling": False,
        },
        "inference": {
            "n_episode": 1,
            "reset_before_collect": True,
        },
        "num_training_envs": 20,
        "num_test_envs": 20,
    },
    "env": {
        "name": "new_cycle_move_pipeline",
        "num_bins": 500,
        "max_steps": 200,
        "step_sizes": [1, 2, 5, 10, 25, 50],
        "history_window": 3,
        "reward_mode": "absolute",
        "obs_mode": "ohe",
    },
    "backend": {
        "name": "sequential",
        "mode": "shuffle",
        "backends": [
            {"name": "function", "function": "rastrigin", "dimensions": 2},
            {"name": "function", "function": "rosenbrock", "dimensions": 2},
            {"name": "function", "function": "schwefel", "dimensions": 2},
        ],
    },
}

run_n_experiments(config_ppo, n_experiments=3, inference_only=False)
```

Эксперимент с baseline-алгоритмом: оптимизация гиперпараметров простой свёрточной нейронной сети (CNN), обучаемой на датасете CIFAR-100.

```bash
python experiments/run_experiment_objective.py
```

Пример конфигурации baseline (TPE):

```python
config_TPE = {
    "backend": {
        "name": "objective",
        "objective_function": objective_function,
        "hp_space": {
            "lr": {
                "type": "float",
                "values": [1e-6, 1e-2],
            },
            "batch_size": {
                "type": "categorical",
                "values": [32, 64, 128],
            },
            "optimizer": {
                "type": "categorical",
                "values": ["Adam", "SGD"],
            },
            "n_params": {
                "type": "categorical",
                "values": [16, 32, 64],
            },
        },
    },
    "full_args": {
        "algorithm": {
            "name": "TPE",
            "N_init": 20,
            "N_s": 100,
            "budget": 209,
            "separation_value": 0.2,
        },
    },
}
```

Пример `objective_function`:

```python
def objective_function(config, dict_config):
    param_values = {}
    for name in dict_config.keys():
        param_values[name] = config[name]

    n_params = param_values["n_params"]
    lr = param_values["lr"]
    batch_size = int(param_values["batch_size"])
    optimizer_name = param_values["optimizer"]

    transform = transforms.ToTensor()
    
    try:
        dataset = datasets.CIFAR100(root='./tmp_data', train=True, download=True, transform=transform)
    except:
        dataset = datasets.CIFAR100(root='./tmp_data', train=True, download=False, transform=transform)
        
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SimpleCNN(num_classes=100, n_params=n_params) 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    model.train()
    
    sub_bar = tqdm(total=int(2), desc="Model training", position=1, leave=False)
    
    for epoch in range(int(2)):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        sub_bar.update(1)
    
    sub_bar.close()

    model.eval()
    val_loss, correct = 0.0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            val_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / len(val_dataset)

    print(f"Config: {param_values}, ValLoss: {avg_val_loss:.4f}, ValAcc: {val_accuracy:.4f}")

    return avg_val_loss
```

## Подробная документация

Документация API генерируется через Sphinx и находится в каталоге `docs/`. Исходные `.rst`-файлы — в `docs/source/`, собранная HTML-версия — в `docs/build/`.
