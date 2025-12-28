@echo off
setlocal enabledelayedexpansion
REM Отдельный файл для экспериментов с RecurrentPPO
REM Использование: run_recurrent_ppo_experiments.bat [BUDGET]
REM Пример: run_recurrent_ppo_experiments.bat (полный бюджет)
REM Пример: run_recurrent_ppo_experiments.bat low (малый бюджет)
REM Пример: run_recurrent_ppo_experiments.bat ultra (ультрамалый бюджет)

set AGENT=RecurrentPPO
set BUDGET=%1

REM Устанавливаем значения по умолчанию (полный бюджет)
set "CONFIG=configs/function_2d_recurrent_ppo.yaml"
set "MODEL_FILE=model_2d_recurrent_ppo.zip"
set "FINE_TUNE_STEPS=20000"
set BUDGET_TYPE=полный бюджет

REM Проверяем аргумент и переопределяем при необходимости
if /i "%BUDGET%"=="low" (
    set "CONFIG=configs/function_2d_recurrent_ppo_low_budget.yaml"
    set "MODEL_FILE=model_2d_recurrent_ppo_low_budget.zip"
    set "FINE_TUNE_STEPS=2000"
    set BUDGET_TYPE=малый бюджет
) else if /i "%BUDGET%"=="ultra" (
    set "CONFIG=configs/function_2d_recurrent_ppo_ultra_low_budget.yaml"
    set "MODEL_FILE=model_2d_recurrent_ppo_ultra_low_budget.zip"
    set "FINE_TUNE_STEPS=500"
    set BUDGET_TYPE=ультрамалый бюджет
) else (
    REM Полный бюджет (по умолчанию или если передан "full")
    set "CONFIG=configs/function_2d_recurrent_ppo.yaml"
    set "MODEL_FILE=model_2d_recurrent_ppo.zip"
    set "FINE_TUNE_STEPS=20000"
    set BUDGET_TYPE=полный бюджет
)

echo Используется алгоритм: %AGENT% (!BUDGET_TYPE!)
if /i "%BUDGET%"=="low" (
    echo ВНИМАНИЕ: Используются малые бюджеты (10000 шагов обучения = 100 эпизодов, 100 шагов в эпизоде)
) else if /i "%BUDGET%"=="ultra" (
    echo ВНИМАНИЕ: Используются ультрамалые бюджеты (2000 шагов обучения = 100 эпизодов, 20 шагов в эпизоде)
) else (
    echo Используется полный бюджет (200000 шагов обучения, 1000 шагов в эпизоде)
)

echo.
echo DEBUG: Используется конфиг: !CONFIG!
echo DEBUG: Файл модели: !MODEL_FILE!
echo DEBUG: Бюджет: %BUDGET%
echo DEBUG: Fine-tune steps: !FINE_TUNE_STEPS!
echo.

REM Проверяем, что переменные установлены
if not defined CONFIG (
    echo ОШИБКА: CONFIG не установлен!
    pause
    exit /b 1
)
if not defined MODEL_FILE (
    echo ОШИБКА: MODEL_FILE не установлен!
    pause
    exit /b 1
)

echo === Шаг 1: Обучение модели на Rastrigin ===
python run_recurrent_ppo_2d_visualize.py --config !CONFIG! --agent %AGENT%
if errorlevel 1 (
    echo ОШИБКА при обучении модели!
    pause
    exit /b 1
)

echo.
echo === Шаг 2: Тестирование обученной модели с разными стартовыми точками ===
echo Запуск с 3 разными seed'ами для стартовой точки...
for %%S in (42 123 777) do (
    echo.
    echo --- Тест на Rastrigin, seed=%%S ---
    python run_recurrent_ppo_2d_visualize.py --config !CONFIG! --pretrained-model logs/%AGENT%/!MODEL_FILE! --transfer-learning --agent %AGENT% --eval-seed %%S
)

echo.
echo === Шаг 3: Тестирование на Sphere (zero-shot) с разными стартовыми точками ===
for %%S in (42 123 777) do (
    echo.
    echo --- Тест на Sphere, seed=%%S ---
    python run_recurrent_ppo_2d_visualize.py --config configs/function_2d_sphere.yaml --pretrained-model logs/%AGENT%/!MODEL_FILE! --transfer-learning --agent %AGENT% --eval-seed %%S
)

echo.
echo === Шаг 4: Тестирование на Rosenbrock (zero-shot) с разными стартовыми точками ===
for %%S in (42 123 777) do (
    echo.
    echo --- Тест на Rosenbrock, seed=%%S ---
    python run_recurrent_ppo_2d_visualize.py --config configs/function_2d_rosenbrock.yaml --pretrained-model logs/%AGENT%/!MODEL_FILE! --transfer-learning --agent %AGENT% --eval-seed %%S
)

echo.
echo === Шаг 5: Дообучение на Sphere (fine-tuning) ===
python run_recurrent_ppo_2d_visualize.py --config configs/function_2d_sphere.yaml --pretrained-model logs/%AGENT%/!MODEL_FILE! --transfer-learning --fine-tune-steps !FINE_TUNE_STEPS! --exploration-boost 2.5 --agent %AGENT% --eval-seed 42

echo.
echo === Шаг 6: Дообучение на Rosenbrock (fine-tuning) ===
python run_recurrent_ppo_2d_visualize.py --config configs/function_2d_rosenbrock.yaml --pretrained-model logs/%AGENT%/!MODEL_FILE! --transfer-learning --fine-tune-steps !FINE_TUNE_STEPS! --exploration-boost 2.5 --agent %AGENT% --eval-seed 42

echo.
echo === Шаг 7: Дообучение на Ackley (fine-tuning) ===
python run_recurrent_ppo_2d_visualize.py --config configs/function_2d_ackley.yaml --pretrained-model logs/%AGENT%/!MODEL_FILE! --transfer-learning --fine-tune-steps !FINE_TUNE_STEPS! --exploration-boost 2.5 --agent %AGENT% --eval-seed 42

echo.
echo === Готово! Проверьте графики в logs/%AGENT%/ ===
echo Графики сохранены с суффиксом _seedN для разных стартовых точек
pause
