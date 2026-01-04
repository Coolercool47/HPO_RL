# HPO_RL

Фреймворк для использования RL агента для подбора гиперпараметров. Текущий пайплайн работает на циклической постановке: агент последовательно выбирает значение для каждого гиперпараметра, проходя по ним по кругу.

## Запуск

Полный эксперимент с обучением и тестированием на других функциях:

```batch
run_transfer_learning.bat [Алгоритм] [Бюджет]
```

Примеры:

```batch
run_transfer_learning.bat PPO full     # PPO, полный бюджет 20k шагов
run_transfer_learning.bat PPO low      # PPO, низкий бюджет 2k шагов
run_transfer_learning.bat PPO ultra    # PPO, ультра-низкий бюджет 600 шагов
run_transfer_learning.bat RecurrentPPO full
run_transfer_learning.bat TD3 low
```

Скрипт работает так: обучает модель на фукнции Растригина → тестирует на других функциях → дообучает на других функциях и тестирует на них .

### Ручной запуск

```bash
# Обучение
python run_experiment.py --config configs/function_2d_test.yaml

# Тестирование
python run_experiment.py --config configs/sphere.yaml \
    --pretrained-model model.zip --transfer-learning

# Дообучение и тестирование
python run_experiment.py --config configs/sphere.yaml \
    --pretrained-model model.zip --fine-tune-steps 2000
```
