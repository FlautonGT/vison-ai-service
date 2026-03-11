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
PREPARE_FLAGS=(--task "${TASK}" --preferred-region "${PREFERRED_REGION}")

if [ "${ALLOW_RESTRICTED}" = "true" ]; then
  PREPARE_FLAGS+=(--allow-restricted)
fi
if [ "${ALLOW_FALLBACK}" = "true" ]; then
  PREPARE_FLAGS+=(--allow-fallback)
fi

cd "${ROOT_DIR}"

python scripts/vastai_prepare_task.py "${PREPARE_FLAGS[@]}"
python scripts/train_pipeline.py --task "${TASK}"
python scripts/evaluate_pipeline.py --task "${TASK}"
python scripts/export_pipeline.py --task "${TASK}"
python scripts/collect_training_artifacts.py --task "${TASK}" --output-dir "${ARTIFACT_ROOT}/${TASK}"

echo "Artifacts ready under ${ARTIFACT_ROOT}/${TASK}"
