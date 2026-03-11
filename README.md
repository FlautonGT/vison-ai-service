# Vison AI Service

`vison-ai-service` remains a stateless Python FastAPI inference service, and now also includes a separate PyTorch training/evaluation surface for reproducible face-model development.

- No PostgreSQL access
- No S3 access
- No enrollment/search persistence in this service
- Called by Go backend via HTTP multipart

Go backend remains source of truth for auth, billing, object storage, and pgvector search.

## Architecture

### VPS 1 (API + DB)
- Go API service
- PostgreSQL + pgvector
- S3 integration

### VPS 2 (AI service, this repo)
- FastAPI + ONNX Runtime (CPUExecutionProvider)
- Receives image(s), returns JSON inference result
- Stateless
- Training pipelines live alongside the inference code but do not run inside the inference service process

## Docs

- Audit summary: `docs/repository_audit.md`
- Architecture and training design: `docs/architecture_and_training.md`
- Vast.ai training runbook: `docs/vastai_runbook.md`
- API examples: `docs/api_examples.md`
- Model cards: `docs/model_cards/`
- Dataset inventory: `configs/datasets/dataset_inventory.json`
- Inference endpoint-to-model registry: `configs/model_registry.json`

## Endpoints

### Stable response format (unchanged vs existing Go contract)
- `POST /api/face/compare`
- `POST /api/face/liveness`
- `POST /api/face/deepfake`
- `POST /api/face/analyze`
- `POST /api/face/verify-live` (combined liveness + deepfake + quality)

### New stateless endpoints
- `POST /api/face/embed` (replacement for enroll)
- `POST /api/face/similarity` (replacement for verify)
- `POST /api/face/quality`
- `POST /api/face/attributes`
- `GET /api/face/capabilities`

### Health
- `GET /health`

Removed from Python service:
- `/api/face/enroll`
- `/api/face/verify`
- `/api/face/search`
- `/api/face/person/{id}`
- `/api/face/persons`
- `DELETE /api/face/person/{id}`

## Tier Model System

The service supports model-tier switching by env vars. Default is Tier 3 (highest accuracy CPU-viable).

### Tier 1 (lightweight)
- Lower memory / latency
- Lower accuracy

### Tier 2 (balanced)
- Mid memory / latency
- Better accuracy

### Tier 3 (default in this repo)
- SCRFD 10G detector
- ArcFace R100 (`glintr100`) + ArcFace ResNet50 (`w600k_r50`) embedding ensemble
- Optional stronger recognizer: ArcFace R100 (`glintr100`) from `antelopev2` archive
- Liveness ensemble: `MiniFASNetV2` + `MiniFASNetV1SE`
- Deepfake ensemble: EfficientNet model + ViT model
- AI-generated face detector: `deep_fake_detector_v2.onnx` (used in deepfake fusion)
- BiSeNet face parsing
- InsightFace genderage

Switch tier by changing environment model filenames (`SCRFD_MODEL`, `ARCFACE_MODEL`, `LIVENESS_MODELS`, `DEEPFAKE_MODELS`, etc.).

## Resource Budget (CPU VPS 4 core / 8 GB)

Approximate per worker model RAM:
- Tier 3 model set: ~1.7 - 2.0 GB

With `FACE_AI_WORKERS=2`:
- ~3.4 - 4.0 GB for models
- Remaining RAM for OS + Python runtime + request buffers

## VPS Deployment From Zero

This section is a full runbook from first login to production run on a fresh Ubuntu VPS.

### 1. Login to VPS

```bash
ssh root@YOUR_VPS_IP
```

### 2. Create non-root deploy user

```bash
adduser vison
usermod -aG sudo vison
rsync --archive --chown=vison:vison ~/.ssh /home/vison
```

Reconnect as deploy user:

```bash
ssh vison@YOUR_VPS_IP
```

### 3. Install system packages

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget unzip build-essential python3 python3-venv python3-pip
```

Optional hardening:

```bash
sudo timedatectl set-timezone Asia/Jakarta
sudo ufw allow OpenSSH
# only allow API VPS to reach AI service
sudo ufw allow from YOUR_API_VPS_IP to any port 8000
sudo ufw --force enable
```

### 4. Clone repository

```bash
cd /opt
sudo mkdir -p /opt/vison
sudo chown vison:vison /opt/vison
cd /opt/vison
git clone https://github.com/FlautonGT/vison-ai-service.git
cd vison-ai-service
```

### 5. Prepare Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Configure environment

```bash
cp .env.production .env
```

Minimum required settings in `.env`:
- `MODEL_DIR=/opt/models`
- `ONNX_PROVIDERS=CPUExecutionProvider`
- `AI_SERVICE_SECRET=<same-secret-as-go-backend>`
- `ALLOWED_IPS=<GO_API_VPS_IP>`
- `FACE_AI_WORKERS=2`

### 7. Download models

```bash
mkdir -p /opt/models
python scripts/download_models.py --model-dir /opt/models
```

### 8. Smoke run (manual)

```bash
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2 --env-file .env
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

### 9. Run as systemd service (recommended)

Create unit file:

```bash
sudo tee /etc/systemd/system/vison-ai.service > /dev/null << 'EOF'
[Unit]
Description=Vison AI Service
After=network.target

[Service]
User=vison
Group=vison
WorkingDirectory=/opt/vison/vison-ai-service
EnvironmentFile=/opt/vison/vison-ai-service/.env
ExecStart=/opt/vison/vison-ai-service/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=3
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable vison-ai
sudo systemctl start vison-ai
sudo systemctl status vison-ai --no-pager
```

Logs:

```bash
sudo journalctl -u vison-ai -f
```

### 10. Production verification checklist

```bash
curl http://127.0.0.1:8000/health
python scripts/test_e2e.py
```

Expected:
- health `status=ok`
- all required models loaded
- no `FACE_NOT_DETECTED` for valid face images
- stable latency under your load target

## Setup

```bash
make setup
```

This installs Python dependencies and downloads ONNX models to `MODEL_DIR` (default `/opt/models`).

## Run

```bash
make run
```

Or directly:

```bash
MODEL_DIR=/opt/models \
ONNX_PROVIDERS=CPUExecutionProvider \
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
```

Windows local dev:
```powershell
venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --env-file .env
```
Use `MODEL_DIR=./models` in `.env` for local Windows run.

## Testing

### E2E API test
```bash
make test
```

### Unit tests
```bash
make unit
```

### Inference benchmark / model checks
```bash
python scripts/test_inference.py
```

### Security benchmark (200 real vs 200 AI)
```bash
make benchmark
```

### NIST-style PAD metrics (APCER/BPCER/ACER)
```bash
make nist
```

### FRVT-style FNMR@FMR benchmark (compare)
```bash
python scripts/benchmark_frvt.py --dataset-dir ./benchmark_data/lfw --max-pairs 500
```

### Asian/custom dataset compare benchmark
```bash
python scripts/benchmark_asian.py --dataset-dir ./benchmark_data/asian_faces --max-pairs 200
```

### Fairness snapshot (UTKFace: race/gender/age-group)
```bash
python scripts/benchmark_fairness.py --dataset-dir ./benchmark_data/UTKFace --max-tests 1000
```

### Build INT8 models (recommended for 4 core / 8 GB VPS)
```bash
python scripts/quantize_models.py --model-dir ./models
```

### Auto-calibrate deepfake thresholds from local dataset
```bash
make calibrate
```

### Compile check
```bash
python -m compileall app scripts training tests
```

## Training and Evaluation

The supported production training path is the shared config-driven CLI under `training/vison_train`, not the older ad hoc fine-tune scripts.

### Task configs

- `configs/training/deepfake_detection.json`
- `configs/training/passive_pad.json`
- `configs/training/face_quality.json`
- `configs/training/age_gender.json`
- `configs/training/face_attributes.json`
- `configs/training/verification.json`
- `configs/training/face_parser.json`

### Dataset selection

```bash
python scripts/select_training_datasets.py --task verification --allow-noncommercial
```

### Leakage-safe manifest split

```bash
python scripts/build_training_manifests.py \
  --manifest data/raw/verification_manifest.csv \
  --group-cols subject_id \
  --train-output data/manifests/verification/train.csv \
  --val-output data/manifests/verification/val.csv \
  --test-output data/manifests/verification/test.csv \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

### Train

```bash
python scripts/train_pipeline.py --task passive_pad
python scripts/train_pipeline.py --task verification --override optimization.epochs=24
```

### Evaluate

```bash
python scripts/evaluate_pipeline.py --task passive_pad
python scripts/evaluate_pipeline.py --task verification
```

### Direct CLI usage

```bash
python -m training.vison_train fit --config configs/training/deepfake_detection.json
python -m training.vison_train evaluate --config configs/training/verification.json
```

## Vast.ai Training

Training is designed to run on Vast.ai GPU instances only. Inference does not depend on Vast.ai.

Recommended assets:

- `Dockerfile.training`
- `requirements-training.lock.txt`
- `.env.training.example`
- `scripts/vastai_train_setup.sh`

Example setup on a Vast.ai instance:

```bash
cp .env.training.example .env.training
bash scripts/vastai_train_setup.sh
bash scripts/vastai_run_task.sh verification
```

Use a mounted persistent workspace for:

- `/workspace/data`
- `/workspace/runs`
- `/workspace/cache`

Checkpoint resume is handled through `optimization.resume_from` in the JSON configs. Mixed precision and multi-GPU training are available through the config and standard `torch.distributed` environment variables.

See `docs/vastai_runbook.md` for the exact Kaggle setup, dataset preparation flow, per-task commands, and artifact retrieval paths.

## Standards and Reporting Notes

- The repo aligns internal reporting with ISO/IEC 30107, ISO/IEC 29794-5, ISO/IEC 19795, ISO/IEC 20059, and FIDO-style concepts where relevant.
- It does not claim external ISO, NIST, or FIDO certification.
- Indonesian demographic coverage is currently limited by dataset availability and licensing; see `configs/datasets/dataset_inventory.json` and `docs/repository_audit.md`.

## Security

Middleware supports:
- Shared secret header: `X-AI-Service-Key`
- Optional IP allowlist from `ALLOWED_IPS`
- In-memory rate limiting per source IP (`RATE_LIMIT_RPS`, default `100`)

Health endpoint (`/health`) is excluded from auth checks.

## Environment Variables

See `.env.example`.

Important:
- `ONNX_PROVIDERS=CPUExecutionProvider`
- INT8 preference + SCRFD input sizing:
  `PREFER_INT8_MODELS=true`, `SCRFD_INPUT_SIZE=640`
- `ARCFACE_FLIP_AUG=true` for better selfie-vs-ID robustness
- ArcFace ensemble for compare/embed/similarity:
  `ARCFACE_MODEL=glintr100.onnx`, `ARCFACE_EXTRA_MODEL=w600k_r50.onnx`,
  `ARCFACE_PRIMARY_WEIGHT=0.65`, `ARCFACE_EXTRA_WEIGHT=0.35`
- Compare default threshold tuning:
  `COMPARE_THRESHOLD_DEFAULT=74` (can still be overridden per request via `similarityThreshold`)
- Compare throughput/score tuning:
  `COMPARE_PARALLEL_EMBEDDING=true`,
  `SIMILARITY_CALIBRATION_ENABLED=true`,
  `SIMILARITY_CALIBRATION_START=75`,
  `SIMILARITY_CALIBRATION_GAIN=1.6`,
  `SIMILARITY_CALIBRATION_POWER=0.75`,
  `SIMILARITY_CALIBRATION_CAP=99.99`
- ONNX CPU tuning for VPS 4 core / 8 GB:
  `ONNX_INTRA_OP_THREADS=2`,
  `ONNX_INTER_OP_THREADS=1`,
  `ONNX_OPT_LEVEL=all`,
  `ONNX_EXECUTION_MODE=parallel`
- Optional age/gender secondary model:
  `AGE_GENDER_VIT_MODEL=age_gender_vit.onnx`,
  `AGE_GENDER_PRIMARY_WEIGHT=0.7`, `AGE_GENDER_VIT_WEIGHT=0.3`,
  `AGE_GENDER_MALE_THRESHOLD=0.45`
- `FACE_MIN_AREA_RATIO=0.05` sets preferred face-area baseline
- Pre-cropped fallback hardening:
  `PRE_CROPPED_MIN_DIM=40`,
  `PRE_CROPPED_ASPECT_MIN=0.5`,
  `PRE_CROPPED_ASPECT_MAX=2.0`
- Small-face behavior (full ID card / far face):
  `ALLOW_SMALL_FACE_AUTOCROP=true` accepts faces below 5% area,
  and only rejects extremely tiny faces via
  `FACE_MIN_AREA_RATIO_HARD=0.003` and `FACE_MIN_PIXELS_HARD=20`
- Quality metrics include ISO-style fields:
  sharpness, brightness, pose (yaw/pitch), inter-eye distance,
  illumination uniformity, and contrast.
- Deepfake fusion knobs for production calibration:
  `AI_FACE_THRESHOLD=68`, `AI_FACE_LOW_CONF_THRESHOLD=12`,
  `AI_FACE_LOW_CONF_FACE_CONF=70`, `DEEPFAKE_FACE_SWAP_STRONG_THRESHOLD=95`,
  `AI_FACE_PRIMARY_WEIGHT=0.7`, `AI_FACE_EXTRA_WEIGHT=0.3`,
  `AI_FACE_CALIBRATION_ALPHA=1.0`, `AI_FACE_CALIBRATION_BETA=0.0`,
  `AI_FACE_CONSENSUS_AI_THRESHOLD=101` (default off),
  `AI_FACE_CONSENSUS_FACE_SWAP_THRESHOLD=55`,
  `AI_FACE_ALWAYS_CROP_CHECK=true` (run AI detector on full-frame + face-crop),
  `AI_FACE_HARD_BLOCK_THRESHOLD=85`,
  `AI_FACE_VOTE_THRESHOLD=55`, `AI_FACE_VOTE_MIN_COUNT=2`,
  `AI_FACE_ANY_TRIGGER_THRESHOLD=42`,
  `AI_FACE_REAL_SUPPRESS_ENABLED=false` (recommended for strict anti-AI mode),
  `AI_FACE_REAL_SUPPRESS_FACE_CONF=72`,
  `AI_FACE_REAL_SUPPRESS_AI_MAX=88`,
  `AI_FACE_REAL_SUPPRESS_FACE_SWAP_MAX=62`
- Optional second AI-generated detector model:
  `AI_FACE_EXTRA_DETECTOR_MODEL=ai_vs_deepfake_vs_real.onnx`
- `AI_FACE_DETECTOR_MODEL=deepfake_efficientnet_b0.onnx` shares the same
  ONNX session with deepfake detector when file path is identical (RAM saving).
- For static-image e-KYC decisions, use `/api/face/verify-live` as final gate
  (`isLive = liveness AND NOT deepfake`), not liveness-only score.
- No `DATABASE_*` env variables in this service

## Go Integration

Go client in `go-client/aiclient.go` now provides:
- `Embed(...)`
- `Similarity(...)`
- `ParseEmbedding(...)`

Flow:
1. Go calls `/api/face/embed` and stores embedding in PostgreSQL/pgvector.
2. Go calls `/api/face/similarity` with live image + stored embedding for verification.
3. Go handles 1:N search in DB directly using pgvector.
