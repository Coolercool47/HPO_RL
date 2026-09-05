# Windows PowerShell: create the environment for experiments_new/ from the repo root.
#   powershell -ExecutionPolicy Bypass -File experiments_new\setup_env.ps1
# Optional: set $env:PYTHON to a specific interpreter (3.11 or 3.12) before running.
$ErrorActionPreference = "Stop"
$py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
if (-not (Test-Path ".venv")) { & $py -m venv .venv }
$venvPy = ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip
& $venvPy -m pip install --only-binary=:all: "ConfigSpace>=1.2,<2"
& $venvPy -m pip install -r experiments_new\requirements.txt
& $venvPy -m pip install --no-deps yahpo-gym==1.0.2
& $venvPy -m pip install -e . --no-deps
if (-not (Test-Path "yahpo_data\lcbench\encoding.json")) {
  Write-Host "Cloning YAHPO surrogate data (lcbench only, sparse checkout)..."
  git clone --filter=blob:none --no-checkout https://github.com/slds-lmu/yahpo_data.git yahpo_data
  Push-Location yahpo_data
  git sparse-checkout init --cone
  git sparse-checkout set lcbench
  git checkout
  Pop-Location
}
& $venvPy -c "import optuna, cmaes, yahpo_gym, ConfigSpace, onnxruntime, torch, scikit_posthocs; print('environment OK')"
& $venvPy experiments_new\synt_functions\run.py --dry-run --methods RS FMP
