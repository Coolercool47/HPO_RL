@echo off
setlocal enabledelayedexpansion
REM Unified script for transfer learning experiments (Windows)
REM Usage: run_transfer_learning.bat [ALGORITHM] [BUDGET]

set AGENT=%1
set BUDGET=%2
if "%AGENT%"=="" set AGENT=PPO
if "%BUDGET%"=="" set BUDGET=full

REM Set budget-specific parameters
if /i "%BUDGET%"=="ultra" goto budget_ultra
if /i "%BUDGET%"=="low" goto budget_low
goto budget_full

:budget_ultra
set BUDGET_SUFFIX=_ultra_low_budget
set RUN_NAME=ultra
set FINE_TUNE_STEPS=600
set BUDGET_DESC=ultra-low budget
goto budget_done

:budget_low
set BUDGET_SUFFIX=_low_budget
set RUN_NAME=low
set FINE_TUNE_STEPS=2000
set BUDGET_DESC=low budget
goto budget_done

:budget_full
set BUDGET_SUFFIX=
set RUN_NAME=full
set FINE_TUNE_STEPS=20000
set BUDGET_DESC=full budget
goto budget_done

:budget_done

REM Set algorithm-specific config
set ACTUAL_AGENT=%AGENT%
if /i "%AGENT%"=="TD3" goto agent_td3
if /i "%AGENT%"=="SAC" goto agent_sac
if /i "%AGENT%"=="PPO_PER_CYCLE" goto agent_ppo_per_cycle
if /i "%AGENT%"=="RecurrentPPO" goto agent_rppo
if /i "%AGENT%"=="PPO" goto agent_ppo
goto agent_default

:agent_td3
set CONFIG=configs/function_2d_td3%BUDGET_SUFFIX%.yaml
set MODEL_FILE=model_2d_td3%BUDGET_SUFFIX%.zip
goto agent_done

:agent_sac
set CONFIG=configs/function_2d_sac%BUDGET_SUFFIX%.yaml
set MODEL_FILE=model_2d_sac%BUDGET_SUFFIX%.zip
goto agent_done

:agent_ppo_per_cycle
set CONFIG=configs/function_2d_ppo_per_cycle%BUDGET_SUFFIX%.yaml
set MODEL_FILE=model_2d_ppo_per_cycle%BUDGET_SUFFIX%.zip
set ACTUAL_AGENT=PPO
goto agent_done

:agent_rppo
set CONFIG=configs/function_2d_recurrent_ppo%BUDGET_SUFFIX%.yaml
set MODEL_FILE=model_2d_recurrent_ppo%BUDGET_SUFFIX%.zip
set ACTUAL_AGENT=RecurrentPPO
goto agent_done

:agent_ppo
set CONFIG=configs/function_2d_test%BUDGET_SUFFIX%.yaml
set MODEL_FILE=model_2d%BUDGET_SUFFIX%.zip
set ACTUAL_AGENT=PPO
goto agent_done

:agent_default
set CONFIG=configs/function_2d_test%BUDGET_SUFFIX%.yaml
set MODEL_FILE=model_2d%BUDGET_SUFFIX%.zip
goto agent_done

:agent_done

REM Generate timestamp
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

REM Create experiment folder
set EXPERIMENT_DIR=logs\%ACTUAL_AGENT%\%TIMESTAMP%_%RUN_NAME%
mkdir "%EXPERIMENT_DIR%" 2>nul

echo ============================================
echo HPO-RL Transfer Learning Experiment
echo ============================================
echo Algorithm: %AGENT%
echo Budget: %BUDGET_DESC%
echo Config: %CONFIG%
echo Experiment folder: %EXPERIMENT_DIR%
echo Fine-tune steps: %FINE_TUNE_STEPS%
echo ============================================
echo.

if not exist "%CONFIG%" (
    echo ERROR: Config file not found: %CONFIG%
    pause
    exit /b 1
)

echo === Step 1: Training model on Rastrigin ===
python run_experiment.py --config %CONFIG% --agent %ACTUAL_AGENT% --output-dir "%EXPERIMENT_DIR%"
if errorlevel 1 (
    echo ERROR during model training!
    pause
    exit /b 1
)

set MODEL_PATH=%EXPERIMENT_DIR%\%MODEL_FILE%

echo.
echo === Step 2: Testing on Rastrigin with different seeds ===
for %%S in (42 123 777) do (
    echo --- Rastrigin seed=%%S ---
    python run_experiment.py --config %CONFIG% --pretrained-model "%MODEL_PATH%" --transfer-learning --agent %ACTUAL_AGENT% --eval-seed %%S --output-dir "%EXPERIMENT_DIR%"
)

echo.
echo === Step 3: Testing on Sphere (zero-shot) ===
for %%S in (42 123 777) do (
    echo --- Sphere seed=%%S ---
    python run_experiment.py --config configs/function_2d_sphere.yaml --pretrained-model "%MODEL_PATH%" --transfer-learning --agent %ACTUAL_AGENT% --eval-seed %%S --output-dir "%EXPERIMENT_DIR%"
)

echo.
echo === Step 4: Testing on Rosenbrock (zero-shot) ===
for %%S in (42 123 777) do (
    echo --- Rosenbrock seed=%%S ---
    python run_experiment.py --config configs/function_2d_rosenbrock.yaml --pretrained-model "%MODEL_PATH%" --transfer-learning --agent %ACTUAL_AGENT% --eval-seed %%S --output-dir "%EXPERIMENT_DIR%"
)

echo.
echo === Step 5: Testing on non-center minimum functions ===
echo --- Booth ---
for %%S in (42 123 777) do python run_experiment.py --config configs/function_2d_booth.yaml --pretrained-model "%MODEL_PATH%" --transfer-learning --agent %ACTUAL_AGENT% --eval-seed %%S --output-dir "%EXPERIMENT_DIR%"
echo --- Beale ---
for %%S in (42 123 777) do python run_experiment.py --config configs/function_2d_beale.yaml --pretrained-model "%MODEL_PATH%" --transfer-learning --agent %ACTUAL_AGENT% --eval-seed %%S --output-dir "%EXPERIMENT_DIR%"
echo --- Shifted Sphere ---
for %%S in (42 123 777) do python run_experiment.py --config configs/function_2d_shifted_sphere.yaml --pretrained-model "%MODEL_PATH%" --transfer-learning --agent %ACTUAL_AGENT% --eval-seed %%S --output-dir "%EXPERIMENT_DIR%"

echo.
echo === Step 6: Fine-tuning on Sphere ===
python run_experiment.py --config configs/function_2d_sphere.yaml --pretrained-model "%MODEL_PATH%" --transfer-learning --fine-tune-steps %FINE_TUNE_STEPS% --exploration-boost 2.5 --agent %ACTUAL_AGENT% --eval-seed 42 --output-dir "%EXPERIMENT_DIR%"

echo.
echo === Step 7: Fine-tuning on Rosenbrock ===
python run_experiment.py --config configs/function_2d_rosenbrock.yaml --pretrained-model "%MODEL_PATH%" --transfer-learning --fine-tune-steps %FINE_TUNE_STEPS% --exploration-boost 2.5 --agent %ACTUAL_AGENT% --eval-seed 42 --output-dir "%EXPERIMENT_DIR%"

echo.
echo === Step 8: Fine-tuning on Ackley ===
python run_experiment.py --config configs/function_2d_ackley.yaml --pretrained-model "%MODEL_PATH%" --transfer-learning --fine-tune-steps %FINE_TUNE_STEPS% --exploration-boost 2.5 --agent %ACTUAL_AGENT% --eval-seed 42 --output-dir "%EXPERIMENT_DIR%"

echo.
echo ============================================
echo Done! All results in: %EXPERIMENT_DIR%
echo ============================================
pause
