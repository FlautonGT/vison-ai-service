#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/workspace/vison-ai-service}"
LOG_ROOT="${LOG_ROOT:-/workspace/logs}"
PREPARE_LOCK="${LOG_ROOT}/prepare_all_manifests.lock"
PREPARE_PID_FILE="${LOG_ROOT}/prepare_all_manifests.pid"
WATCH_LOG="${LOG_ROOT}/train_after_prepare.log"

mkdir -p "${LOG_ROOT}"
cd "${ROOT_DIR}"

wait_for_prepare() {
  while true; do
    if [ -f "${PREPARE_PID_FILE}" ]; then
      prepare_pid="$(cat "${PREPARE_PID_FILE}" 2>/dev/null || true)"
      if [ -n "${prepare_pid}" ] && kill -0 "${prepare_pid}" 2>/dev/null; then
        sleep 30
        continue
      fi
    fi
    if [ -f "${PREPARE_LOCK}" ]; then
      lock_pid="$(cat "${PREPARE_LOCK}" 2>/dev/null || true)"
      if [ -n "${lock_pid}" ] && kill -0 "${lock_pid}" 2>/dev/null; then
        sleep 30
        continue
      fi
    fi
    break
  done
}

{
  echo "=== $(date -Iseconds) Waiting for manifest prep to finish ==="
  wait_for_prepare
  echo "=== $(date -Iseconds) Manifest prep finished; starting training queue ==="
  SKIP_PREPARE=true bash scripts/vastai_run_full_queue.sh
} 2>&1 | tee -a "${WATCH_LOG}"
