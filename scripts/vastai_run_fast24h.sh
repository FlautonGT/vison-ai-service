#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/vison-ai-service}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
PREFERRED_REGION="${PREFERRED_REGION:-indonesia}"
TRAINING_VENV="${TRAINING_VENV:-${ROOT_DIR}/.venv-training}"
TRAINING_PYTHON="${TRAINING_PYTHON:-${TRAINING_VENV}/bin/python}"

mkdir -p "${LOG_ROOT}" "${ARTIFACT_ROOT}"
cd "${ROOT_DIR}"

if [ -f ".env.training" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env.training"
  set +a
fi

declare -A EPOCHS=(
  [verification]=8
  [passive_pad]=8
  [age_gender]=6
  [face_attributes]=6
  [face_quality]=5
  [deepfake]=5
  [face_parser]=4
)

declare -A EXTRA_PREPARE_FLAGS=(
  [verification]="--allow-noncommercial --allow-nonmodifiable --allow-restricted --allow-fallback"
  [passive_pad]="--allow-noncommercial --allow-nonmodifiable --allow-restricted --allow-fallback"
  [age_gender]="--allow-fallback"
  [face_attributes]="--allow-restricted --allow-fallback"
  [face_quality]="--allow-fallback"
  [deepfake]="--allow-fallback"
  [face_parser]="--allow-noncommercial --allow-nonmodifiable --allow-restricted --allow-fallback"
)

TASKS=(
  verification
  passive_pad
  age_gender
  face_attributes
  face_quality
  deepfake
  face_parser
)

for task in "${TASKS[@]}"; do
  task_log="${LOG_ROOT}/${task}.log"
  echo "=== $(date -Iseconds) Starting ${task} ===" | tee -a "${task_log}"

  # shellcheck disable=SC2206
  prepare_flags=( ${EXTRA_PREPARE_FLAGS[$task]} )
  "${TRAINING_PYTHON}" scripts/vastai_prepare_task.py \
    --task "${task}" \
    --preferred-region "${PREFERRED_REGION}" \
    "${prepare_flags[@]}" 2>&1 | tee -a "${task_log}"

  "${TRAINING_PYTHON}" scripts/train_pipeline.py \
    --task "${task}" \
    --override "optimization.epochs=${EPOCHS[$task]}" 2>&1 | tee -a "${task_log}"

  "${TRAINING_PYTHON}" scripts/evaluate_pipeline.py --task "${task}" 2>&1 | tee -a "${task_log}"
  "${TRAINING_PYTHON}" scripts/export_pipeline.py --task "${task}" 2>&1 | tee -a "${task_log}"
  "${TRAINING_PYTHON}" scripts/collect_training_artifacts.py \
    --task "${task}" \
    --output-dir "${ARTIFACT_ROOT}/${task}" 2>&1 | tee -a "${task_log}"

  echo "=== $(date -Iseconds) Finished ${task} ===" | tee -a "${task_log}"
done

echo "All fast24h tasks finished. Logs are under ${LOG_ROOT} and artifacts are under ${ARTIFACT_ROOT}."
