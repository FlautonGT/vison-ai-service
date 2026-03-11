#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/vison-ai-service}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/workspace/artifacts}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
RAW_ROOT="${RAW_ROOT:-${ROOT_DIR}/data/raw}"
MANIFEST_ROOT="${MANIFEST_ROOT:-${ROOT_DIR}/data/manifests}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/runs}"
RESET_RAW="${RESET_RAW:-false}"
TRAINING_VENV="${TRAINING_VENV:-${ROOT_DIR}/.venv-training}"
TRAINING_PYTHON="${TRAINING_PYTHON:-${TRAINING_VENV}/bin/python}"

cd "${ROOT_DIR}"

if [ -f ".env.training" ]; then
  set -a
  # shellcheck disable=SC1091
  source ".env.training"
  set +a
fi

pkill -f 'scripts/vastai_run_fast24h.sh' || true
pkill -f 'scripts/vastai_run_full_queue.sh' || true
pkill -f 'scripts/train_pipeline.py' || true
pkill -f 'scripts/evaluate_pipeline.py' || true
pkill -f 'scripts/export_pipeline.py' || true

rm -rf "${ARTIFACT_ROOT}" "${LOG_ROOT}" "${MANIFEST_ROOT}" "${RUN_ROOT}"
mkdir -p "${ARTIFACT_ROOT}" "${LOG_ROOT}" "${MANIFEST_ROOT}" "${RUN_ROOT}"

if [ "${RESET_RAW}" = "true" ]; then
  rm -rf "${RAW_ROOT}"
  mkdir -p "${RAW_ROOT}"
fi

if [ "${S3_UPLOAD_ENABLED:-false}" = "true" ] && [ -n "${S3_BUCKET:-}" ] && [ -n "${S3_PREFIX:-}" ]; then
  if [ ! -x "${TRAINING_PYTHON}" ]; then
    TRAINING_PYTHON="$(command -v python3 || command -v python)"
  fi
  "${TRAINING_PYTHON}" scripts/s3_delete_prefix.py --bucket "${S3_BUCKET}" --region "${S3_REGION:-ap-southeast-1}" --prefix "${S3_PREFIX%/}/"
fi

echo "Reset complete."
