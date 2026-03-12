#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/vison-ai-service}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
TRAINING_VENV="${TRAINING_VENV:-${ROOT_DIR}/.venv-training}"
TRAINING_PYTHON="${TRAINING_PYTHON:-${TRAINING_VENV}/bin/python}"
PREFERRED_REGION="${PREFERRED_REGION:-indonesia}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-false}"
LOCK_FILE="${LOG_ROOT}/prepare_all_manifests.lock"
MAIN_LOG="${LOG_ROOT}/prepare_all_manifests.log"
SUMMARY_PATH="${LOG_ROOT}/prepare_all_manifests_summary.tsv"
COUNTS_PATH="${LOG_ROOT}/manifest_counts.json"

mkdir -p "${LOG_ROOT}"
cd "${ROOT_DIR}"

if [ -f ".env.training" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env.training"
  set +a
fi

if [ -f "${LOCK_FILE}" ]; then
  existing_pid="$(cat "${LOCK_FILE}" 2>/dev/null || true)"
  if [ -n "${existing_pid}" ] && kill -0 "${existing_pid}" 2>/dev/null; then
    echo "Another manifest-prepare queue is already running with PID ${existing_pid}."
    exit 1
  fi
fi
echo "$$" > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

TASKS=(
  verification
  passive_pad
  age_gender
  face_attributes
  face_quality
  deepfake
  face_parser
)

declare -A EXTRA_PREPARE_FLAGS=(
  [verification]="--allow-noncommercial --allow-nonmodifiable --allow-restricted --allow-fallback"
  [passive_pad]="--allow-noncommercial --allow-nonmodifiable --allow-restricted --allow-fallback"
  [age_gender]="--allow-noncommercial --allow-nonmodifiable --allow-restricted --allow-fallback"
  [face_attributes]="--allow-noncommercial --allow-nonmodifiable --allow-restricted --allow-fallback"
  [face_quality]="--allow-noncommercial --allow-nonmodifiable --allow-restricted --allow-fallback"
  [deepfake]="--allow-noncommercial --allow-nonmodifiable --allow-restricted --allow-fallback"
  [face_parser]="--allow-noncommercial --allow-nonmodifiable --allow-restricted --allow-fallback"
)

run_logged() {
  local task_log="$1"
  shift
  "$@" 2>&1 | tee -a "${task_log}"
  return ${PIPESTATUS[0]}
}

echo -e "task\tstatus" > "${SUMMARY_PATH}"

for task in "${TASKS[@]}"; do
  task_log="${LOG_ROOT}/${task}_prepare.log"
  echo "=== $(date -Iseconds) Preparing ${task} ===" | tee -a "${task_log}" "${MAIN_LOG}"

  # shellcheck disable=SC2206
  prepare_flags=( ${EXTRA_PREPARE_FLAGS[$task]} )
  if [ "${FORCE_DOWNLOAD}" = "true" ]; then
    prepare_flags+=(--force-download)
  fi

  if run_logged "${task_log}" "${TRAINING_PYTHON}" scripts/vastai_prepare_task.py \
    --task "${task}" \
    --preferred-region "${PREFERRED_REGION}" \
    "${prepare_flags[@]}"; then
    echo -e "${task}\tOK" | tee -a "${SUMMARY_PATH}" >/dev/null
    echo "=== $(date -Iseconds) Prepared ${task} [OK] ===" | tee -a "${task_log}" "${MAIN_LOG}"
  else
    echo -e "${task}\tFAILED" | tee -a "${SUMMARY_PATH}" >/dev/null
    echo "=== $(date -Iseconds) Prepared ${task} [FAILED] ===" | tee -a "${task_log}" "${MAIN_LOG}"
  fi
done

"${TRAINING_PYTHON}" scripts/report_manifest_counts.py > "${COUNTS_PATH}"
echo "Manifest preparation finished. Summary: ${SUMMARY_PATH}. Counts: ${COUNTS_PATH}." | tee -a "${MAIN_LOG}"
