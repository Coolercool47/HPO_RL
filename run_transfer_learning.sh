#!/bin/bash
# Unified script for transfer learning experiments (Linux/Mac)
# Usage: ./run_transfer_learning.sh [ALGORITHM] [BUDGET]
#
# ALGORITHM: PPO (default), PPO_PER_CYCLE, TD3, SAC, RecurrentPPO
# BUDGET: full (default), low, ultra
#
# Examples:
#   ./run_transfer_learning.sh                    # PPO, full budget
#   ./run_transfer_learning.sh PPO low            # PPO, low budget
#   ./run_transfer_learning.sh RecurrentPPO ultra # RecurrentPPO, ultra low budget
#   ./run_transfer_learning.sh TD3 full           # TD3, full budget

AGENT="${1:-PPO}"
BUDGET="${2:-full}"

# Normalize budget to lowercase
BUDGET=$(echo "$BUDGET" | tr '[:upper:]' '[:lower:]')

# Set budget-specific parameters
case "$BUDGET" in
    ultra)
        BUDGET_SUFFIX="_ultra_low_budget"
        RUN_NAME="ultra"
        FINE_TUNE_STEPS=600
        BUDGET_DESC="ultra-low budget (2000 training steps, 20 steps per episode)"
        ;;
    low)
        BUDGET_SUFFIX="_low_budget"
        RUN_NAME="low"
        FINE_TUNE_STEPS=2000
        BUDGET_DESC="low budget (10000 training steps, 100 steps per episode)"
        ;;
    *)
        BUDGET_SUFFIX=""
        RUN_NAME="full"
        FINE_TUNE_STEPS=20000
        BUDGET_DESC="full budget (200000 training steps, 1000 steps per episode)"
        ;;
esac

# Set algorithm-specific config and model file
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
        # Default: PPO
        CONFIG="configs/function_2d_test${BUDGET_SUFFIX}.yaml"
        MODEL_FILE="model_2d${BUDGET_SUFFIX}.zip"
        ;;
esac

# Generate timestamp ONCE for the entire experiment
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create single experiment folder for ALL results
EXPERIMENT_DIR="logs/${ACTUAL_AGENT}/${TIMESTAMP}_${RUN_NAME}"
mkdir -p "$EXPERIMENT_DIR"

echo "============================================"
echo "HPO-RL Transfer Learning Experiment"
echo "============================================"
echo "Algorithm: $AGENT"
echo "Budget: $BUDGET_DESC"
echo "Config: $CONFIG"
echo "Experiment folder: $EXPERIMENT_DIR"
echo "Fine-tune steps: $FINE_TUNE_STEPS"
echo "============================================"
echo ""

# Verify config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    echo "Please check that the config file exists."
    exit 1
fi

echo "=== Step 1: Training model on Rastrigin ==="
python run_experiment.py --config "$CONFIG" --agent "$ACTUAL_AGENT" --output-dir "$EXPERIMENT_DIR"
if [ $? -ne 0 ]; then
    echo "ERROR during model training!"
    exit 1
fi

# Model is saved in experiment folder
MODEL_PATH="${EXPERIMENT_DIR}/${MODEL_FILE}"

echo ""
echo "=== Step 2: Testing trained model with different starting points ==="
for SEED in 42 123 777; do
    echo ""
    echo "--- Test on Rastrigin, seed=$SEED ---"
    python run_experiment.py --config "$CONFIG" \
        --pretrained-model "$MODEL_PATH" \
        --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done

echo ""
echo "=== Step 3: Testing on Sphere (zero-shot) with different starting points ==="
for SEED in 42 123 777; do
    echo ""
    echo "--- Test on Sphere, seed=$SEED ---"
    python run_experiment.py --config configs/function_2d_sphere.yaml \
        --pretrained-model "$MODEL_PATH" \
        --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done

echo ""
echo "=== Step 4: Testing on Rosenbrock (zero-shot) with different starting points ==="
for SEED in 42 123 777; do
    echo ""
    echo "--- Test on Rosenbrock, seed=$SEED ---"
    python run_experiment.py --config configs/function_2d_rosenbrock.yaml \
        --pretrained-model "$MODEL_PATH" \
        --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done

echo ""
echo "=== Step 5: Testing on functions with non-center minimum ==="
echo "--- Booth: minimum at (1, 3) ---"
for SEED in 42 123 777; do
    python run_experiment.py --config configs/function_2d_booth.yaml \
        --pretrained-model "$MODEL_PATH" \
        --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done

echo "--- Beale: minimum at (3, 0.5) ---"
for SEED in 42 123 777; do
    python run_experiment.py --config configs/function_2d_beale.yaml \
        --pretrained-model "$MODEL_PATH" \
        --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done

echo "--- Shifted Sphere: minimum at (2, 2) ---"
for SEED in 42 123 777; do
    python run_experiment.py --config configs/function_2d_shifted_sphere.yaml \
        --pretrained-model "$MODEL_PATH" \
        --transfer-learning --agent "$ACTUAL_AGENT" --eval-seed "$SEED" --output-dir "$EXPERIMENT_DIR"
done

echo ""
echo "=== Step 6: Fine-tuning on Sphere ==="
python run_experiment.py --config configs/function_2d_sphere.yaml \
    --pretrained-model "$MODEL_PATH" \
    --transfer-learning --fine-tune-steps "$FINE_TUNE_STEPS" \
    --exploration-boost 2.5 --agent "$ACTUAL_AGENT" --eval-seed 42 --output-dir "$EXPERIMENT_DIR"

echo ""
echo "=== Step 7: Fine-tuning on Rosenbrock ==="
python run_experiment.py --config configs/function_2d_rosenbrock.yaml \
    --pretrained-model "$MODEL_PATH" \
    --transfer-learning --fine-tune-steps "$FINE_TUNE_STEPS" \
    --exploration-boost 2.5 --agent "$ACTUAL_AGENT" --eval-seed 42 --output-dir "$EXPERIMENT_DIR"

echo ""
echo "=== Step 8: Fine-tuning on Ackley ==="
python run_experiment.py --config configs/function_2d_ackley.yaml \
    --pretrained-model "$MODEL_PATH" \
    --transfer-learning --fine-tune-steps "$FINE_TUNE_STEPS" \
    --exploration-boost 2.5 --agent "$ACTUAL_AGENT" --eval-seed 42 --output-dir "$EXPERIMENT_DIR"

echo ""
echo "============================================"
echo "Done! All results saved to: $EXPERIMENT_DIR"
echo "============================================"
