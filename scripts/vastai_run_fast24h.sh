#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/vison-ai-service}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
PREFERRED_REGION="${PREFERRED_REGION:-indonesia}"
TRAINING_VENV="${TRAINING_VENV:-${ROOT_DIR}/.venv-training}"
TRAINING_PYTHON="${TRAINING_PYTHON:-${TRAINING_VENV}/bin/python}"
S3_UPLOAD_ENABLED="${S3_UPLOAD_ENABLED:-false}"
S3_BUCKET="${S3_BUCKET:-}"
S3_REGION="${S3_REGION:-${AWS_DEFAULT_REGION:-ap-southeast-1}}"
S3_PREFIX="${S3_PREFIX:-vison-training}"
S3_SYNC_ROOT="${S3_SYNC_ROOT:-/workspace}"
RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
S3_RUN_PREFIX="${S3_PREFIX%/}/${RUN_STAMP}"

mkdir -p "${LOG_ROOT}" "${ARTIFACT_ROOT}"
cd "${ROOT_DIR}"

SUMMARY_PATH="${LOG_ROOT}/run_fast24h_summary.tsv"
MAIN_LOG="${LOG_ROOT}/run_fast24h.log"
LOCK_FILE="${LOG_ROOT}/run_fast24h.lock"

if [ -f "${LOCK_FILE}" ]; then
  existing_pid="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
  if [ -n "${existing_pid}" ] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Another fast24h queue is already running with PID ${existing_pid}. Remove ${LOCK_FILE} only if that process is gone."
    exit 1
  fi
fi
echo "$$" > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

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

run_logged() {
  local task_log="$1"
  shift
  "$@" 2>&1 | tee -a "${task_log}"
  return ${PIPESTATUS[0]}
}

s3_enabled() {
  [ "${S3_UPLOAD_ENABLED}" = "true" ] && [ -n "${S3_BUCKET}" ]
}

sync_s3() {
  local task="$1"
  local task_log="$2"
  if ! s3_enabled; then
    return 0
  fi

  local task_prefix="${S3_RUN_PREFIX}/${task}"
  run_logged "${task_log}" "${TRAINING_PYTHON}" scripts/s3_upload.py \
    --bucket "${S3_BUCKET}" \
    --region "${S3_REGION}" \
    --root "${S3_SYNC_ROOT}" \
    --key-prefix "${task_prefix}" \
    --skip-missing \
    --path "${task_log}" \
    --path "${MAIN_LOG}" \
    --path "${SUMMARY_PATH}" \
    --path "${ARTIFACT_ROOT}/${task}" \
    --path "${ROOT_DIR}/runs/${task}"
}

echo -e "task\tstatus\tfailed_steps" > "${SUMMARY_PATH}"

overall_failures=0
for task in "${TASKS[@]}"; do
  task_log="${LOG_ROOT}/${task}.log"
  failed_steps=()
  echo "=== $(date -Iseconds) Starting ${task} ===" | tee -a "${task_log}" "${MAIN_LOG}"

  # shellcheck disable=SC2206
  prepare_flags=( ${EXTRA_PREPARE_FLAGS[$task]} )
  if ! run_logged "${task_log}" "${TRAINING_PYTHON}" scripts/vastai_prepare_task.py \
    --task "${task}" \
    --preferred-region "${PREFERRED_REGION}" \
    "${prepare_flags[@]}"; then
    failed_steps+=("prepare")
  fi

  if ! run_logged "${task_log}" "${TRAINING_PYTHON}" scripts/train_pipeline.py \
    --task "${task}" \
    --override "optimization.epochs=${EPOCHS[$task]}"; then
    failed_steps+=("train")
  fi

  if ! run_logged "${task_log}" "${TRAINING_PYTHON}" scripts/evaluate_pipeline.py --task "${task}"; then
    failed_steps+=("evaluate")
  fi
  if ! run_logged "${task_log}" "${TRAINING_PYTHON}" scripts/export_pipeline.py --task "${task}"; then
    failed_steps+=("export")
  fi
  if ! run_logged "${task_log}" "${TRAINING_PYTHON}" scripts/collect_training_artifacts.py \
    --task "${task}" \
    --output-dir "${ARTIFACT_ROOT}/${task}"; then
    failed_steps+=("collect")
  fi

  if ! sync_s3 "${task}" "${task_log}"; then
    failed_steps+=("s3_sync")
    echo "=== $(date -Iseconds) S3 sync failed for ${task} ===" | tee -a "${task_log}" "${MAIN_LOG}"
  fi

  if [ "${#failed_steps[@]}" -eq 0 ]; then
    echo -e "${task}\tOK\t-" | tee -a "${SUMMARY_PATH}" >/dev/null
    echo "=== $(date -Iseconds) Finished ${task} [OK] ===" | tee -a "${task_log}" "${MAIN_LOG}"
  else
    overall_failures=$((overall_failures + 1))
    failed_joined="$(IFS=,; echo "${failed_steps[*]}")"
    echo -e "${task}\tFAILED\t${failed_joined}" | tee -a "${SUMMARY_PATH}" >/dev/null
    echo "=== $(date -Iseconds) Finished ${task} [FAILED: ${failed_joined}] ===" | tee -a "${task_log}" "${MAIN_LOG}"
  fi
done

echo "All fast24h tasks finished. Summary: ${SUMMARY_PATH}. Logs: ${LOG_ROOT}. Artifacts: ${ARTIFACT_ROOT}." | tee -a "${MAIN_LOG}"
exit "${overall_failures}"
