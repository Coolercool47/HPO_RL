@echo off
REM Пример скрипта для демонстрации transfer learning (Windows)
REM Использование: run_transfer_learning_example.bat [ALGORITHM]
REM Пример: run_transfer_learning_example.bat PPO
REM Пример: run_transfer_learning_example.bat PPO_PER_CYCLE
REM Пример: run_transfer_learning_example.bat TD3
REM Пример: run_transfer_learning_example.bat SAC
REM Пример: run_transfer_learning_example.bat RecurrentPPO
REM
REM Для малых бюджетов используйте: run_transfer_learning_example_low_budget.bat
REM Для ультрамалых бюджетов используйте: run_transfer_learning_example_ultra_low_budget.bat

REM Устанавливаем алгоритм по умолчанию или из параметра
set AGENT=%1
if "%AGENT%"=="" set AGENT=PPO

echo Используется алгоритм: %AGENT%
echo Используется полный бюджет (200000 шагов обучения, 1000 шагов в эпизоде)
echo.

REM Определяем конфиг в зависимости от алгоритма
if "%AGENT%"=="TD3" (
    set CONFIG=configs/function_2d_td3.yaml
    set MODEL_FILE=model_2d_td3.zip
) else if "%AGENT%"=="SAC" (
    set CONFIG=configs/function_2d_sac.yaml
    set MODEL_FILE=model_2d_sac.zip
) else if "%AGENT%"=="PPO_PER_CYCLE" (
    set CONFIG=configs/function_2d_ppo_per_cycle.yaml
    set MODEL_FILE=model_2d_ppo_per_cycle.zip
    set AGENT=PPO
) else if "%AGENT%"=="RecurrentPPO" (
    set CONFIG=configs/function_2d_recurrent_ppo.yaml
    set MODEL_FILE=model_2d_recurrent_ppo.zip
) else (
    set CONFIG=configs/function_2d_test.yaml
    set MODEL_FILE=model_2d.zip
)

echo === Шаг 1: Обучение модели на Rastrigin ===
python run_experiment_2d_visualize.py --config %CONFIG% --agent %AGENT%

echo.
echo === Шаг 2: Тестирование обученной модели с разными стартовыми точками ===
for %%S in (42 123 777) do (
    echo.
    echo --- Тест на Rastrigin, seed=%%S ---
    python run_experiment_2d_visualize.py --config %CONFIG% --pretrained-model logs/%AGENT%/%MODEL_FILE% --transfer-learning --agent %AGENT% --eval-seed %%S
)

echo.
echo === Шаг 3: Тестирование на Sphere (zero-shot) с разными стартовыми точками ===
for %%S in (42 123 777) do (
    echo.
    echo --- Тест на Sphere, seed=%%S ---
    python run_experiment_2d_visualize.py --config configs/function_2d_sphere.yaml --pretrained-model logs/%AGENT%/%MODEL_FILE% --transfer-learning --agent %AGENT% --eval-seed %%S
)

echo.
echo === Шаг 4: Тестирование на Rosenbrock (zero-shot) с разными стартовыми точками ===
for %%S in (42 123 777) do (
    echo.
    echo --- Тест на Rosenbrock, seed=%%S ---
    python run_experiment_2d_visualize.py --config configs/function_2d_rosenbrock.yaml --pretrained-model logs/%AGENT%/%MODEL_FILE% --transfer-learning --agent %AGENT% --eval-seed %%S
)

echo.
echo === Шаг 5: Тестирование на функциях с минимумом НЕ в центре ===
echo --- Booth: минимум в (1, 3) ---
for %%S in (42 123 777) do (
    python run_experiment_2d_visualize.py --config configs/function_2d_booth.yaml --pretrained-model logs/%AGENT%/%MODEL_FILE% --transfer-learning --agent %AGENT% --eval-seed %%S
)
echo --- Beale: минимум в (3, 0.5) ---
for %%S in (42 123 777) do (
    python run_experiment_2d_visualize.py --config configs/function_2d_beale.yaml --pretrained-model logs/%AGENT%/%MODEL_FILE% --transfer-learning --agent %AGENT% --eval-seed %%S
)
echo --- Shifted Sphere: минимум в (2, 2) ---
for %%S in (42 123 777) do (
    python run_experiment_2d_visualize.py --config configs/function_2d_shifted_sphere.yaml --pretrained-model logs/%AGENT%/%MODEL_FILE% --transfer-learning --agent %AGENT% --eval-seed %%S
)

echo.
echo === Шаг 7: Дообучение на Sphere (fine-tuning) ===
python run_experiment_2d_visualize.py --config configs/function_2d_sphere.yaml --pretrained-model logs/%AGENT%/%MODEL_FILE% --transfer-learning --fine-tune-steps 20000 --exploration-boost 2.5 --agent %AGENT% --eval-seed 42

echo.
echo === Шаг 8: Дообучение на Rosenbrock (fine-tuning) ===
python run_experiment_2d_visualize.py --config configs/function_2d_rosenbrock.yaml --pretrained-model logs/%AGENT%/%MODEL_FILE% --transfer-learning --fine-tune-steps 20000 --exploration-boost 2.5 --agent %AGENT% --eval-seed 42

echo.
echo === Шаг 9: Дообучение на Ackley (fine-tuning) ===
python run_experiment_2d_visualize.py --config configs/function_2d_ackley.yaml --pretrained-model logs/%AGENT%/%MODEL_FILE% --transfer-learning --fine-tune-steps 20000 --exploration-boost 2.5 --agent %AGENT% --eval-seed 42

echo.
echo === Готово! Проверьте графики в logs/%AGENT%/ ===
echo Графики сохранены с суффиксом _seedN для разных стартовых точек
pause

