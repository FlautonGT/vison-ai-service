#!/usr/bin/env bash
set -euo pipefail

TASKS=(
  deepfake
  passive_pad
  verification
  age_gender
  face_attributes
  face_quality
  face_parser
)

for task in "${TASKS[@]}"; do
  echo "=== Running ${task} ==="
  bash scripts/vastai_run_task.sh "${task}"
done
