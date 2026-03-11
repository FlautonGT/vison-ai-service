# Architecture and Training Plan

## Inference architecture

- Shared face detection and crop extraction remains separate from task heads.
- Verification stays embedding-based rather than being collapsed into a classifier.
- Passive liveness stays modeled as PAD, not generic fake-image classification.
- Quality and attributes are optional inference surfaces and can be requested independently.
- Endpoint-to-model usage is now declared in `configs/model_registry.json` and exposed through `/api/face/capabilities`.

## Training architecture

- All new training flows use JSON configs plus the shared CLI in `training/vison_train`.
- Supported task types:
  - `binary_classification` for deepfake/manipulated-face detection and PAD
  - `metric_learning` for verification embeddings with ArcMargin head
  - `regression` for face quality
  - `age_gender_multitask` for age + gender
  - `multilabel_classification` for attributes
- Checkpoints are resumable through `optimization.resume_from`.
- Mixed precision is supported through `optimization.mixed_precision`.
- Multi-GPU training is supported through standard `torch.distributed` environment variables.
- Optional tracking backends: `wandb`, `mlflow`, or `none`

## Dataset workflow

1. Search candidate datasets in `configs/datasets/dataset_inventory.json`.
2. Rank them with `python scripts/select_training_datasets.py`.
3. Build a normalized manifest with dataset-specific preprocessing outside the trainer.
4. Split the manifest with `python scripts/build_training_manifests.py` using grouped columns like `subject_id`, `session_id`, `video_id`, or `attack_type`.
5. Train with `python scripts/train_pipeline.py --task <task>`.
6. Evaluate with `python scripts/evaluate_pipeline.py --task <task>`.

## Region and licensing guidance

- Indonesia-first selection is attempted, but the audit found weak licensing and provenance for the most obvious Indonesia-only public dataset.
- Southeast Asia or broader Asia datasets are the current practical fallback for legally safer internal development.
- Several attractive Kaggle datasets are non-commercial or no-derivatives. They should not be silently promoted into production training.

## Standards alignment boundaries

- PAD reporting is aligned to ISO/IEC 30107 concepts through APCER, BPCER, and ACER style metrics.
- Quality reporting is aligned to ISO/IEC 29794-5 style image utility concepts, not a conformance claim.
- Verification reporting is aligned to ISO/IEC 19795 and FIDO-style biometric metrics through ROC, DET, EER, and operating points.
- Morphing-attack resistance should only be discussed in relation to internal research unless a dedicated protocol is run.
- None of the above should be described as certified conformance without external testing.

## Vast.ai training path

- Use `Dockerfile.training` for GPU training containers.
- Use `.env.training.example` as the baseline environment file.
- Use `scripts/vastai_train_setup.sh` to prepare `/workspace/data`, `/workspace/runs`, and `/workspace/cache`.
- Store checkpoints and reports on a mounted persistent volume, not only on ephemeral instance storage.
- Datasets should be downloaded to local NVMe scratch or a mounted volume, then referenced through manifests to avoid repeated network reads.
