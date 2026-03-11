#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/vison-ai-service}"
DATA_ROOT="${TRAINING_DATA_ROOT:-/workspace/data}"
RUNS_ROOT="${TRAINING_RUNS_ROOT:-/workspace/runs}"
CACHE_ROOT="${TRAINING_CACHE_ROOT:-/workspace/cache}"

mkdir -p "${DATA_ROOT}" "${RUNS_ROOT}" "${CACHE_ROOT}" "${ROOT_DIR}"
cd "${ROOT_DIR}"

if [ -f ".env.training.example" ] && [ ! -f ".env.training" ]; then
  cp .env.training.example .env.training
fi

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

python -m pip install --upgrade pip
python -m pip install -r requirements-training.lock.txt

echo "If Kaggle credentials are not already configured, add either:"
echo "  export KAGGLE_USERNAME=..."
echo "  export KAGGLE_KEY=..."
echo "or place kaggle.json under ~/.kaggle/"
echo "Workspace prepared."
echo "Data root: ${DATA_ROOT}"
echo "Runs root: ${RUNS_ROOT}"
echo "Artifacts root: ${ARTIFACT_ROOT:-/workspace/artifacts}"
echo "Example:"
echo "  python -m training.vison_train fit --config configs/training/passive_pad.json --override optimization.checkpoint_dir=\"${RUNS_ROOT}/passive_pad/checkpoints\""
