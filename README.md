# HPO_RL

RL-агент для оптимизации гиперпараметров. Агент учится стратегии поиска на benchmark-функциях, затем переносит знания на реальные задачи.

Текущий пайплайн работает на циклической постановке: агент последовательно выбирает значение для каждого гиперпараметра, проходя по ним по кругу.

## Запуск

### Windows (.bat)

Полный эксперимент с обучением и transfer learning:

```batch
run_transfer_learning.bat [ALGORITHM] [BUDGET]
```

Примеры:

```batch
run_transfer_learning.bat PPO full     # PPO, полный бюджет (20k fine-tune steps)
run_transfer_learning.bat PPO low      # PPO, низкий бюджет (2k steps)
run_transfer_learning.bat PPO ultra    # PPO, ультра-низкий (600 steps)
run_transfer_learning.bat RecurrentPPO full
run_transfer_learning.bat TD3 low
```

Скрипт последовательно: обучает модель на Rastrigin → тестирует на других функциях (zero-shot) → fine-tuning.

### Ручной запуск

```bash
# Обучение
python run_experiment.py --config configs/function_2d_test.yaml

# Transfer learning (zero-shot)
python run_experiment.py --config configs/sphere.yaml \
    --pretrained-model model.zip --transfer-learning

# Fine-tuning
python run_experiment.py --config configs/sphere.yaml \
    --pretrained-model model.zip --fine-tune-steps 2000
```

## Агенты

PPO, TD3, SAC, RecurrentPPO и другие из stable-baselines3.
