#!/bin/bash
# Пример скрипта для демонстрации transfer learning

echo "=== Шаг 1: Обучение модели на Rastrigin ==="
python run_experiment_2d_visualize.py --config configs/function_2d_test.yaml

echo ""
echo "=== Шаг 2: Тестирование на Sphere (zero-shot) ==="
python run_experiment_2d_visualize.py \
    --config configs/function_2d_sphere.yaml \
    --pretrained-model logs/PPO/model_2d.zip \
    --transfer-learning

echo ""
echo "=== Шаг 3: Тестирование на Rosenbrock (zero-shot) ==="
python run_experiment_2d_visualize.py \
    --config configs/function_2d_rosenbrock.yaml \
    --pretrained-model logs/PPO/model_2d.zip \
    --transfer-learning

echo ""
echo "=== Шаг 4: Дообучение на Sphere (fine-tuning с повышенным exploration) ==="
python run_experiment_2d_visualize.py \
    --config configs/function_2d_sphere.yaml \
    --pretrained-model logs/PPO/model_2d.zip \
    --transfer-learning \
    --fine-tune-steps 20000 \
    --exploration-boost 2.5

echo ""
echo "=== Шаг 5: Дообучение на Rosenbrock (fine-tuning с повышенным exploration) ==="
python run_experiment_2d_visualize.py \
    --config configs/function_2d_rosenbrock.yaml \
    --pretrained-model logs/PPO/model_2d.zip \
    --transfer-learning \
    --fine-tune-steps 20000 \
    --exploration-boost 2.5

echo ""
echo "=== Шаг 6: Дообучение на Ackley (fine-tuning с повышенным exploration) ==="
python run_experiment_2d_visualize.py \
    --config configs/function_2d_ackley.yaml \
    --pretrained-model logs/PPO/model_2d.zip \
    --transfer-learning \
    --fine-tune-steps 20000 \
    --exploration-boost 2.5

echo ""
echo "=== Готово! Проверьте графики в logs/PPO/ ==="
echo "Логи сохранены с уникальными именами для каждой функции и режима"

