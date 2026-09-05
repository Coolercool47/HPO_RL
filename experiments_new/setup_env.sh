#!/usr/bin/env bash
# Linux / macOS / WSL: create the environment for experiments_new/ from the repo root.
#   bash experiments_new/setup_env.sh
# Optional: PYTHON=python3.11 bash experiments_new/setup_env.sh
set -euo pipefail
PY="${PYTHON:-python3}"
[ -d .venv ] || "$PY" -m venv .venv
VPY=".venv/bin/python"
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install "ConfigSpace>=1.2,<2"
"$VPY" -m pip install -r experiments_new/requirements.txt
"$VPY" -m pip install --no-deps yahpo-gym==1.0.2
"$VPY" -m pip install -e . --no-deps
# SMAC baseline (Linux only): uncomment to enable
# "$VPY" -m pip install "smac>=2.1"
if [ ! -f yahpo_data/lcbench/encoding.json ]; then
  echo "Cloning YAHPO surrogate data (lcbench only, sparse checkout)..."
  git clone --filter=blob:none --no-checkout https://github.com/slds-lmu/yahpo_data.git yahpo_data
  (cd yahpo_data && git sparse-checkout init --cone && git sparse-checkout set lcbench && git checkout)
fi
"$VPY" -c "import optuna, cmaes, yahpo_gym, ConfigSpace, onnxruntime, torch, scikit_posthocs; print('environment OK')"
"$VPY" experiments_new/synt_functions/run.py --dry-run --methods RS FMP
