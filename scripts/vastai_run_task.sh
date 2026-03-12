#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
if [ -z "${TASK}" ]; then
  echo "Usage: bash scripts/vastai_run_task.sh <deepfake|passive_pad|verification|age_gender|face_attributes|face_quality|face_parser>"
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-/workspace/vison-ai-service}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts}"
PREFERRED_REGION="${PREFERRED_REGION:-indonesia}"
ALLOW_RESTRICTED="${ALLOW_RESTRICTED:-true}"
ALLOW_FALLBACK="${ALLOW_FALLBACK:-true}"
TASK_ALLOW_NONCOMMERCIAL_DEFAULT="false"
TASK_ALLOW_NONMODIFIABLE_DEFAULT="false"
case "${TASK}" in
  passive_pad|verification|age_gender|face_attributes|face_quality|deepfake|face_parser)
    TASK_ALLOW_NONCOMMERCIAL_DEFAULT="true"
    TASK_ALLOW_NONMODIFIABLE_DEFAULT="true"
    ;;
esac
ALLOW_NONCOMMERCIAL="${ALLOW_NONCOMMERCIAL:-${TASK_ALLOW_NONCOMMERCIAL_DEFAULT}}"
ALLOW_NONMODIFIABLE="${ALLOW_NONMODIFIABLE:-${TASK_ALLOW_NONMODIFIABLE_DEFAULT}}"
TRAINING_VENV="${TRAINING_VENV:-${ROOT_DIR}/.venv-training}"
TRAINING_PYTHON="${TRAINING_PYTHON:-${TRAINING_VENV}/bin/python}"
PREPARE_FLAGS=(--task "${TASK}" --preferred-region "${PREFERRED_REGION}")
MAX_DATASETS="${MAX_DATASETS:-}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-false}"

if [ "${ALLOW_RESTRICTED}" = "true" ]; then
  PREPARE_FLAGS+=(--allow-restricted)
fi
if [ "${ALLOW_FALLBACK}" = "true" ]; then
  PREPARE_FLAGS+=(--allow-fallback)
fi
if [ "${ALLOW_NONCOMMERCIAL}" = "true" ]; then
  PREPARE_FLAGS+=(--allow-noncommercial)
fi
if [ "${ALLOW_NONMODIFIABLE}" = "true" ]; then
  PREPARE_FLAGS+=(--allow-nonmodifiable)
fi
if [ -n "${MAX_DATASETS}" ]; then
  PREPARE_FLAGS+=(--max-datasets "${MAX_DATASETS}")
fi
if [ "${FORCE_DOWNLOAD}" = "true" ]; then
  PREPARE_FLAGS+=(--force-download)
fi

cd "${ROOT_DIR}"

if [ -f ".env.training" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env.training"
  set +a
fi

if [ ! -x "${TRAINING_PYTHON}" ]; then
  TRAINING_PYTHON="$(command -v python3 || command -v python)"
fi

"${TRAINING_PYTHON}" scripts/vastai_prepare_task.py "${PREPARE_FLAGS[@]}"
"${TRAINING_PYTHON}" scripts/train_pipeline.py --task "${TASK}"
"${TRAINING_PYTHON}" scripts/evaluate_pipeline.py --task "${TASK}"
"${TRAINING_PYTHON}" scripts/export_pipeline.py --task "${TASK}"
"${TRAINING_PYTHON}" scripts/collect_training_artifacts.py --task "${TASK}" --output-dir "${ARTIFACT_ROOT}/${TASK}"

echo "Artifacts ready under ${ARTIFACT_ROOT}/${TASK}"
