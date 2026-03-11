#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/vison-ai-service}"
DATA_ROOT="${TRAINING_DATA_ROOT:-/workspace/data}"
RUNS_ROOT="${TRAINING_RUNS_ROOT:-/workspace/runs}"
CACHE_ROOT="${TRAINING_CACHE_ROOT:-/workspace/cache}"
TRAINING_VENV="${TRAINING_VENV:-${ROOT_DIR}/.venv-training}"
TRAINING_PYTHON="${TRAINING_PYTHON:-${TRAINING_VENV}/bin/python}"

mkdir -p "${DATA_ROOT}" "${RUNS_ROOT}" "${CACHE_ROOT}" "${ROOT_DIR}"
cd "${ROOT_DIR}"

if [ -f ".env.training.example" ] && [ ! -f ".env.training" ]; then
  cp .env.training.example .env.training
fi

if [ -f ".env.training" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env.training"
  set +a
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

python3 -m venv "${TRAINING_VENV}"
"${TRAINING_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${TRAINING_PYTHON}" -m pip install -r requirements-training.lock.txt

echo "If Kaggle credentials are not already configured, add either:"
echo "  export KAGGLE_USERNAME=..."
echo "  export KAGGLE_KEY=..."
echo "or place kaggle.json under ~/.kaggle/"
echo "Workspace prepared."
echo "Data root: ${DATA_ROOT}"
echo "Runs root: ${RUNS_ROOT}"
echo "Artifacts root: ${ARTIFACT_ROOT:-/workspace/artifacts}"
echo "Training venv: ${TRAINING_VENV}"
echo "Training python: ${TRAINING_PYTHON}"
echo "Example:"
echo "  ${TRAINING_PYTHON} -m training.vison_train fit --config configs/training/passive_pad.json --override optimization.checkpoint_dir=\"${RUNS_ROOT}/passive_pad/checkpoints\""
