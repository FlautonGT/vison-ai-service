# Vast.ai Training Runbook

## Scope

- Vast.ai is used only for dataset download, manifest preparation, training, evaluation, export, and artifact collection.
- Local inference stays in the existing `vison-ai-service` stack.

## Instance prerequisites

- GPU instance with enough disk for raw datasets and artifacts
- Mounted persistent volume for:
  - `/workspace/data`
  - `/workspace/runs`
  - `/workspace/artifacts`
  - optional `/workspace/cache`
- Kaggle credentials configured as either:
  - `KAGGLE_USERNAME` and `KAGGLE_KEY`
  - or `~/.kaggle/kaggle.json`

## Bootstrap

```bash
git clone <your-repo-url> /workspace/vison-ai-service
cd /workspace/vison-ai-service
cp .env.training.example .env.training
bash scripts/vastai_train_setup.sh
set -a
source .env.training
set +a
```

The setup script creates a repo-local virtualenv and installs the training stack into:

```bash
/workspace/vison-ai-service/.venv-training
```

## Kaggle-first dataset selection

Approved-only selection:

```bash
$TRAINING_PYTHON scripts/select_training_datasets.py --task face_attributes
```

Restricted and fallback inclusion when gaps exist:

```bash
$TRAINING_PYTHON scripts/select_training_datasets.py \
  --task verification \
  --allow-noncommercial \
  --allow-restricted \
  --allow-fallback
```

Regional priority is enforced in selection as:

- Indonesia first
- then Southeast Asia
- then broader Asia
- then global fallback only when label coverage would otherwise be missing

Task-specific dataset preference currently favors Asia-heavy sources where Kaggle availability allows:

- `passive_pad`: `trainingdatapro/asian-people-liveness-detection-video-dataset` first
- `verification`: `trainingdatapro/asian-kyc-photo-dataset` first, then global selfie-ID fallback
- `age_gender` and `face_quality`: `ghaidaalatoum/fairface` first because it includes `Southeast Asian` and broader Asia labels for slice reporting
- `deepfake`, `face_attributes`, and `face_parser`: global fallback remains necessary because SEA-specific Kaggle coverage is weak or legally unclear

To inspect the exact regional mix before any run, use:

```bash
$TRAINING_PYTHON scripts/vastai_prepare_task.py \
  --task verification \
  --preferred-region indonesia \
  --allow-noncommercial \
  --allow-restricted \
  --allow-fallback \
  --skip-download
```

The JSON summary includes `regional_assessment` warnings. Treat a warning about missing Indonesia/SEA coverage as a deployment limitation, not something to hide.

## Dataset download and manifest preparation

```bash
$TRAINING_PYTHON scripts/vastai_prepare_task.py \
  --task verification \
  --preferred-region indonesia \
  --allow-noncommercial \
  --allow-restricted \
  --allow-fallback
```

Task names:

- `deepfake`
- `passive_pad`
- `verification`
- `age_gender`
- `face_attributes`
- `face_quality`
- `face_parser`

## One-command run per task

```bash
bash scripts/vastai_run_task.sh deepfake
bash scripts/vastai_run_task.sh passive_pad
bash scripts/vastai_run_task.sh verification
bash scripts/vastai_run_task.sh age_gender
bash scripts/vastai_run_task.sh face_attributes
bash scripts/vastai_run_task.sh face_quality
bash scripts/vastai_run_task.sh face_parser
```

## Manual staged flow per task

```bash
$TRAINING_PYTHON scripts/vastai_prepare_task.py --task passive_pad --allow-restricted --allow-fallback
$TRAINING_PYTHON scripts/train_pipeline.py --task passive_pad
$TRAINING_PYTHON scripts/evaluate_pipeline.py --task passive_pad
$TRAINING_PYTHON scripts/export_pipeline.py --task passive_pad
$TRAINING_PYTHON scripts/collect_training_artifacts.py --task passive_pad --output-dir /workspace/artifacts/passive_pad
```

## Artifact locations

During a run:

- checkpoints: `runs/<task>/checkpoints/`
- reports: `runs/<task>/reports/`
- ONNX exports and handoff manifest: `runs/<task>/artifacts/`

Collected handoff bundle:

- `/workspace/artifacts/<task>/checkpoints`
- `/workspace/artifacts/<task>/reports`
- `/workspace/artifacts/<task>/exports`

## Optional S3 sync

If you want logs and artifacts copied to S3 during the run, set these in `.env.training`:

```bash
S3_UPLOAD_ENABLED=true
S3_BUCKET=<your-bucket>
S3_REGION=ap-southeast-1
S3_PREFIX=vison-training
AWS_ACCESS_KEY_ID=<your-key>
AWS_SECRET_ACCESS_KEY=<your-secret>
```

The fast unattended queue uploads task logs, the main queue log, the summary TSV, `runs/<task>`, and `/workspace/artifacts/<task>` under a UTC-stamped run prefix:

```text
${S3_PREFIX}/${RUN_STAMP}/<task>/...
```

## Local inference handoff

Use the exported ONNX file plus the generated export manifest to map the artifact into the existing local env vars:

- deepfake: `DEEPFAKE_MODELS`, optionally `AI_FACE_DETECTOR_MODEL`
- passive PAD: `LIVENESS_MODELS`
- verification: `ARCFACE_MODEL` or `ARCFACE_EXTRA_MODEL`
- age/gender: `AGE_GENDER_MODEL`
- face parser: `FACE_PARSING_MODEL`

Quality and attribute learned models are exported for offline use and future optional local integration, but the current local inference path still primarily uses the existing heuristic/parser-based logic.

## Known regional data gaps

- Verification and PAD still depend heavily on restricted non-commercial Asia datasets if you want SEA-relevant coverage.
- Indonesia-only datasets remain weak in licensing and provenance.
- Exact-age SEA datasets are limited; fallback global age data is still required.
- Deepfake and parser training are still global-fallback-heavy because a legally clearer SEA-focused Kaggle alternative was not identified in this pass.
- Attribute coverage for `hat/cap` still relies on global academic mirrors if you need trainable supervision beyond heuristics.
- Parser/hat-support data is available via Kaggle mirrors of CelebA and CelebAMask-HQ, but they should stay fail-closed unless official terms are cleared.

## External validation boundary

- These runs align internal reports with ISO/IEC 30107, ISO/IEC 29794-5, ISO/IEC 19795, ISO/IEC 20059, and FIDO/NIST-style concepts.
- They do not establish external certification or conformance.
