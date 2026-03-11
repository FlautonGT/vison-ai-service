# Repository Audit

## Current inference endpoints

- `GET /health`
- `GET /api/face/capabilities`
- `POST /api/face/compare`
- `POST /api/face/liveness`
- `POST /api/face/deepfake`
- `POST /api/face/analyze`
- `POST /api/face/quality`
- `POST /api/face/attributes`
- `POST /api/face/verify-live`
- `POST /api/face/embed`
- `POST /api/face/similarity`

## Current inference model wrappers

- Shared detection and alignment: `SCRFDDetector`
- Verification embeddings: `ArcFaceRecognizer`, optional `AdaFaceRecognizer`
- Passive PAD: `LivenessChecker`, optional `CDCNLiveness`
- Manipulated/synthetic face detection: `DeepfakeDetector`, `AIFaceDetector`, `DeepfakeVitV2Detector`, `NPRDetector`, `CLIPFakeDetector`
- Demographic estimation: `AgeGenderEstimator`, `AgeGenderVitEstimator`, `FairFaceEstimator`, `MiVOLOEstimator`
- Attribute and segmentation support: `FaceParser`
- Quality and attribute fusion: heuristic services in `app/services/quality.py` and `app/services/attributes.py`

## Current checkpoints in `models/`

- Detection: `scrfd_10g_bnkps.onnx`, `scrfd_2.5g_bnkps.onnx`
- Verification: `glintr100.onnx`, `w600k_r50.onnx`, `w600k_mbf.onnx`
- PAD: `MiniFASNetV1SE.onnx`, `MiniFASNetV2.onnx`
- Manipulated/synthetic detection: `deepfake_efficientnet_b0.onnx`, `community_forensics_vit.onnx`, `ai_vs_deepfake_vs_real.onnx`, `deep_fake_detector_v2.onnx`
- Parsing: `bisenet_face_parsing.onnx`
- Age/gender/demographic proxies: `genderage.onnx`, `age_gender_vit.onnx`, `fairface.onnx`
- Other probes: `clip_indonesian_probe.onnx`

## Existing training and fine-tuning code before refactor

- Ad hoc scripts under `scripts/` for deepfake, liveness, age/gender, compare/parsing, FairFace, and Indonesian experiments
- No shared config system
- No shared manifest schema
- No reproducible grouped split utility
- No single task-aware training or evaluation CLI

## Current dataset loading pattern before refactor

- Dataset reading was embedded directly inside the fine-tune scripts
- Splits were mostly file-level or folder-level, not consistently subject-disjoint
- No central dataset inventory or licensing gate
- No consistent attack-type holdout workflow for PAD or deepfake evaluation

## Current deployment/runtime assumptions

- FastAPI inference service
- ONNX Runtime focused, CPU-first default runtime
- Stateless service, no database or object storage responsibilities
- Docker runtime designed for inference only
- Existing production contract primarily expects multipart image uploads

## Gap analysis

### What already existed

- Strong inference-oriented ONNX service for detection, verification, liveness, deepfake, and age/gender
- Shared face detection and crop handling already used across endpoints
- Baseline benchmark scripts for compare, PAD-style checks, and fairness snapshots

### What was missing

- Dedicated quality and attribute endpoints
- Endpoint-to-model registry/config surface
- Reusable PyTorch training/evaluation stack
- Reproducible grouped manifest splitting
- Dataset inventory with licensing and region notes
- Per-task configs for deepfake, passive PAD, verification, quality, age/gender, and attributes
- Vast.ai training assets
- Model cards and standards-alignment documentation

### What should be refactored

- Replace script-specific dataset crawling with manifest-driven training
- Separate inference model selection from endpoint definitions using a service catalog
- Centralize metrics, plots, threshold search, and slice reporting
- Keep experimental scripts as references, but treat the new CLI/config flow as the production training path

### What can be shared across tasks

- Face detection and alignment in inference
- Image transforms, manifest loading, grouped splits, and inventory selection in training
- Evaluation primitives: confusion matrix, ROC, DET, threshold search, slice reports
- Model registry metadata and capability discovery for endpoint documentation and runtime checks
