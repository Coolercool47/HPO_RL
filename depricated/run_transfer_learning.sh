#!/bin/bash
# Transfer learning experiments
# Usage: ./run_transfer_learning.sh [ALGORITHM] [BUDGET]

AGENT="${1:-PPO}"
BUDGET="${2:-full}"
BUDGET=$(echo "$BUDGET" | tr '[:upper:]' '[:lower:]')

case "$BUDGET" in
    ultra)
        BUDGET_SUFFIX="_ultra_low_budget"
        RUN_NAME="ultra"
        FINE_TUNE_STEPS=600
        ;;
    low)
        BUDGET_SUFFIX="_low_budget"
        RUN_NAME="low"
        FINE_TUNE_STEPS=2000
        ;;
    *)
        BUDGET_SUFFIX=""
        RUN_NAME="full"
        FINE_TUNE_STEPS=20000
        ;;
esac

ACTUAL_AGENT="$AGENT"
case "$AGENT" in
    TD3)
        CONFIG="configs/function_2d_td3${BUDGET_SUFFIX}.yaml"
        MODEL_FILE="model_2d_td3${BUDGET_SUFFIX}.zip"
        ;;
    SAC)
        CONFIG="configs/function_2d_sac${BUDGET_SUFFIX}.yaml"
        MODEL_FILE="model_2d_sac${BUDGET_SUFFIX}.zip"
        ;;
    PPO_PER_CYCLE)
        CONFIG="configs/function_2d_ppo_per_cycle${BUDGET_SUFFIX}.yaml"
        MODEL_FILE="model_2d_ppo_per_cycle${BUDGET_SUFFIX}.zip"
        ACTUAL_AGENT="PPO"
        ;;
    RecurrentPPO)
        CONFIG="configs/function_2d_recurrent_ppo${BUDGET_SUFFIX}.yaml"
        MODEL_FILE="model_2d_recurrent_ppo${BUDGET_SUFFIX}.zip"
        ;;
    *)
        CONFIG="configs/function_2d_test${BUDGET_SUFFIX}.yaml"
        MODEL_FILE="model_2d${BUDGET_SUFFIX}.zip"
        ;;
esac

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
EXPERIMENT_DIR="logs/${ACTUAL_AGENT}/${TIMESTAMP}_${RUN_NAME}"
mkdir -p "$EXPERIMENT_DIR"

echo ""
echo "$AGENT / $RUN_NAME budget / fine-tune=$FINE_TUNE_STEPS"
echo "Config: $CONFIG"
echo "Output: $EXPERIMENT_DIR"
echo ""

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

echo "[1/8] Training on Rastrigin..."
python run_experiment.py --config "$CONFIG" --agent "$ACTUAL_AGENT" --output-dir "$EXPERIMENT_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR during training!"
    exit 1
fi

MODEL_PATH="${EXPERIMENT_DIR}/${MODEL_FILE}"

echo "[2/8] Testing on Rastrigin (seeds: 42, 123, 777)"
for SEED in 42 123 777; do
    python run_experiment.py --config "$CONFIG" --pretrained-model "$MODEL_PATH" --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done

echo "[3/8] Zero-shot: Sphere"
for SEED in 42 123 777; do
    python run_experiment.py --config configs/function_2d_sphere.yaml --pretrained-model "$MODEL_PATH" --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done

echo "[4/8] Zero-shot: Rosenbrock"
for SEED in 42 123 777; do
    python run_experiment.py --config configs/function_2d_rosenbrock.yaml --pretrained-model "$MODEL_PATH" --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done

echo "[5/8] Zero-shot: Booth, Beale, Shifted Sphere"
for SEED in 42 123 777; do
    python run_experiment.py --config configs/function_2d_booth.yaml --pretrained-model "$MODEL_PATH" --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done
for SEED in 42 123 777; do
    python run_experiment.py --config configs/function_2d_beale.yaml --pretrained-model "$MODEL_PATH" --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done
for SEED in 42 123 777; do
    python run_experiment.py --config configs/function_2d_shifted_sphere.yaml --pretrained-model "$MODEL_PATH" --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done

echo "[6/8] Fine-tuning: Sphere"
python run_experiment.py --config configs/function_2d_sphere.yaml --pretrained-model "$MODEL_PATH" --transfer-learning --fine-tune-steps "$FINE_TUNE_STEPS" --exploration-boost 2.5 --agent "$ACTUAL_AGENT" --eval-seed 42 --output-dir "$EXPERIMENT_DIR"

echo "[7/8] Fine-tuning: Rosenbrock"
python run_experiment.py --config configs/function_2d_rosenbrock.yaml --pretrained-model "$MODEL_PATH" --transfer-learning --fine-tune-steps "$FINE_TUNE_STEPS" --exploration-boost 2.5 --agent "$ACTUAL_AGENT" --eval-seed 42 --output-dir "$EXPERIMENT_DIR"

echo "[8/8] Fine-tuning: Ackley"
python run_experiment.py --config configs/function_2d_ackley.yaml --pretrained-model "$MODEL_PATH" --transfer-learning --fine-tune-steps "$FINE_TUNE_STEPS" --exploration-boost 2.5 --agent "$ACTUAL_AGENT" --eval-seed 42 --output-dir "$EXPERIMENT_DIR"

echo ""
echo "Done: $EXPERIMENT_DIR"
