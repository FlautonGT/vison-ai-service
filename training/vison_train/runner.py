"""Task-aware PyTorch runner with checkpointing and evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import dump_json
from .data.datasets import (
    AgeGenderDataset,
    IdentityImageDataset,
    ManifestImageDataset,
    MultiLabelDataset,
    SegmentationDataset,
    build_segmentation_transforms,
    build_transform,
)
from .data.manifests import load_manifest, require_columns, resolve_image_paths
from .evaluation.metrics import (
    age_gender_report,
    binary_classification_report,
    demographic_slice_report,
    multilabel_report,
    pad_report,
    regression_report,
    search_binary_threshold,
    segmentation_report,
    verification_report,
)
from .evaluation.plots import save_confusion_matrix, save_det_curve, save_roc_curve
from .models.factory import AgeGenderModel, ArcMarginProduct, ClassificationModel, EmbeddingModel, SegmentationModel


class _NullTracker:
    def log(self, *_args, **_kwargs) -> None:
        return None

    def close(self) -> None:
        return None


class _WandbTracker:
    def __init__(self, config: dict[str, Any]):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            project=config.get("logging", {}).get("project", "vison-ai"),
            name=config.get("logging", {}).get("run_name"),
            config=config,
        )

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        self.wandb.log(payload, step=step)

    def close(self) -> None:
        if self.run is not None:
            self.run.finish()


class _MlflowTracker:
    def __init__(self, config: dict[str, Any]):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_experiment(config.get("logging", {}).get("project", "vison-ai"))
        self.run = mlflow.start_run(run_name=config.get("logging", {}).get("run_name"))
        mlflow.log_dict(config, "config.json")

    def log(self, payload: dict[str, Any], step: int | None = None) -> None:
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                self.mlflow.log_metric(key, float(value), step=step)

    def close(self) -> None:
        if self.run is not None:
            self.mlflow.end_run()


def _tracker(config: dict[str, Any], enabled: bool):
    backend = str(config.get("logging", {}).get("report_to", "none")).lower()
    if not enabled or backend in {"", "none"}:
        return _NullTracker()
    if backend == "wandb":
        try:
            return _WandbTracker(config)
        except Exception:
            return _NullTracker()
    if backend == "mlflow":
        try:
            return _MlflowTracker(config)
        except Exception:
            return _NullTracker()
    return _NullTracker()


def _setup_runtime() -> tuple[torch.device, bool, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1
    if distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return device, distributed, rank, world_size


def _cleanup_runtime(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _resolve_frames(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_cfg = config["data"]
    root = data_cfg.get("manifest_root")
    image_columns = [data_cfg["image_col"]]
    if config["task"]["type"] == "segmentation":
        image_columns.append(data_cfg["mask_col"])
    train_frame = resolve_image_paths(load_manifest(data_cfg["train_manifest"]), image_columns, root=root)
    val_frame = resolve_image_paths(load_manifest(data_cfg["val_manifest"]), image_columns, root=root)
    required = list(image_columns)
    task_type = config["task"]["type"]
    if task_type == "age_gender_multitask":
        required.extend([data_cfg["age_col"], data_cfg["gender_col"]])
    elif task_type == "metric_learning":
        required.append(data_cfg["identity_col"])
    elif task_type == "multilabel_classification":
        required.extend(list(data_cfg["label_cols"]))
    else:
        required.append(data_cfg["label_col"])
    require_columns(train_frame, required)
    require_columns(val_frame, required)
    return train_frame, val_frame


def _dataloaders(config: dict[str, Any], distributed: bool) -> tuple[Any, Any, Any, Any]:
    task_type = config["task"]["type"]
    input_size = int(config["model"].get("input_size", 224))
    train_frame, val_frame = _resolve_frames(config)
    image_col = config["data"]["image_col"]

    if task_type == "age_gender_multitask":
        train_ds = AgeGenderDataset(
            train_frame,
            image_col=image_col,
            age_col=config["data"]["age_col"],
            gender_col=config["data"]["gender_col"],
            transform=build_transform(input_size, augment=True),
        )
        val_ds = AgeGenderDataset(
            val_frame,
            image_col=image_col,
            age_col=config["data"]["age_col"],
            gender_col=config["data"]["gender_col"],
            transform=build_transform(input_size, augment=False),
        )
    elif task_type == "metric_learning":
        train_ds = IdentityImageDataset(
            train_frame,
            image_col=image_col,
            identity_col=config["data"]["identity_col"],
            transform=build_transform(input_size, augment=True),
        )
        val_ds = IdentityImageDataset(
            val_frame,
            image_col=image_col,
            identity_col=config["data"]["identity_col"],
            transform=build_transform(input_size, augment=False),
        )
    elif task_type == "multilabel_classification":
        train_ds = MultiLabelDataset(
            train_frame,
            image_col=image_col,
            transform=build_transform(input_size, augment=True),
            label_cols=list(config["data"]["label_cols"]),
        )
        val_ds = MultiLabelDataset(
            val_frame,
            image_col=image_col,
            transform=build_transform(input_size, augment=False),
            label_cols=list(config["data"]["label_cols"]),
        )
    elif task_type == "segmentation":
        train_img_tf, train_mask_tf = build_segmentation_transforms(input_size, augment=True)
        val_img_tf, val_mask_tf = build_segmentation_transforms(input_size, augment=False)
        train_ds = SegmentationDataset(
            train_frame,
            image_col=image_col,
            mask_col=config["data"]["mask_col"],
            image_transform=train_img_tf,
            mask_transform=train_mask_tf,
        )
        val_ds = SegmentationDataset(
            val_frame,
            image_col=image_col,
            mask_col=config["data"]["mask_col"],
            image_transform=val_img_tf,
            mask_transform=val_mask_tf,
        )
    elif task_type == "regression":
        train_ds = ManifestImageDataset(
            train_frame,
            image_col=image_col,
            target_cols=[config["data"]["label_col"]],
            transform=build_transform(input_size, augment=True),
            target_mode="regression",
        )
        val_ds = ManifestImageDataset(
            val_frame,
            image_col=image_col,
            target_cols=[config["data"]["label_col"]],
            transform=build_transform(input_size, augment=False),
            target_mode="regression",
        )
    else:
        train_ds = ManifestImageDataset(
            train_frame,
            image_col=image_col,
            target_cols=[config["data"]["label_col"]],
            transform=build_transform(input_size, augment=True),
            target_mode="binary",
        )
        val_ds = ManifestImageDataset(
            val_frame,
            image_col=image_col,
            target_cols=[config["data"]["label_col"]],
            transform=build_transform(input_size, augment=False),
            target_mode="binary",
        )

    batch_size = int(config["optimization"].get("batch_size", 32))
    num_workers = int(config["optimization"].get("num_workers", 4))
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_frame, val_frame, train_loader, val_loader


def _build_model_and_loss(config: dict[str, Any], train_frame: pd.DataFrame, device: torch.device):
    task_type = config["task"]["type"]
    backbone = config["model"].get("backbone", "efficientnet_b0")
    pretrained = bool(config["model"].get("pretrained", True))
    dropout = float(config["model"].get("dropout", 0.2))

    if task_type == "age_gender_multitask":
        model = AgeGenderModel(backbone=backbone, dropout=dropout, pretrained=pretrained).to(device)
        losses = {
            "age": nn.SmoothL1Loss(),
            "gender": nn.CrossEntropyLoss(),
        }
        return model, losses
    if task_type == "metric_learning":
        num_classes = int(train_frame[config["data"]["identity_col"]].nunique())
        model = EmbeddingModel(
            backbone=backbone,
            embedding_dim=int(config["model"].get("embedding_dim", 512)),
            dropout=dropout,
            pretrained=pretrained,
        ).to(device)
        head = ArcMarginProduct(int(config["model"].get("embedding_dim", 512)), num_classes).to(device)
        return {"backbone": model, "head": head}, {"classification": nn.CrossEntropyLoss()}
    if task_type == "multilabel_classification":
        model = ClassificationModel(
            backbone=backbone,
            num_outputs=len(config["data"]["label_cols"]),
            dropout=dropout,
            pretrained=pretrained,
        ).to(device)
        return model, {"labels": "masked_bce"}
    if task_type == "segmentation":
        model = SegmentationModel(
            backbone=backbone,
            num_classes=int(config["data"].get("num_classes", 19)),
            pretrained=pretrained,
        ).to(device)
        return model, {"segmentation": nn.CrossEntropyLoss()}
    if task_type == "regression":
        model = ClassificationModel(backbone=backbone, num_outputs=1, dropout=dropout, pretrained=pretrained).to(device)
        return model, {"score": nn.SmoothL1Loss()}
    model = ClassificationModel(backbone=backbone, num_outputs=1, dropout=dropout, pretrained=pretrained).to(device)
    return model, {"label": nn.BCEWithLogitsLoss()}


def _wrap_model(model, distributed: bool, device: torch.device):
    if not distributed:
        return model
    if isinstance(model, dict):
        for key, module in model.items():
            model[key] = DistributedDataParallel(module, device_ids=[device.index] if device.type == "cuda" else None)
        return model
    return DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)


def _checkpoint_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    ckpt_dir = Path(config["optimization"].get("checkpoint_dir", "runs/checkpoints")).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ckpt_dir / "last.pt", ckpt_dir / "best.pt"


def _state_dict(model):
    if isinstance(model, dict):
        return {key: (value.module if hasattr(value, "module") else value).state_dict() for key, value in model.items()}
    return (model.module if hasattr(model, "module") else model).state_dict()


def _load_state(model, state: dict[str, Any]) -> None:
    if isinstance(model, dict):
        for key, module in model.items():
            target = module.module if hasattr(module, "module") else module
            target.load_state_dict(state[key])
        return
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(state)


def _save_checkpoint(path: Path, epoch: int, model, optimizer, scheduler, best_metric: float, config: dict[str, Any]) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": _state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_metric": best_metric,
            "config": config,
        },
        path,
    )


def _load_checkpoint(path: str | Path, model, optimizer, scheduler) -> tuple[int, float]:
    checkpoint = torch.load(Path(path), map_location="cpu")
    _load_state(model, checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return int(checkpoint.get("epoch", 0)), float(checkpoint.get("best_metric", float("-inf")))


def _masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    raw = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked = raw * mask
    denom = torch.clamp(mask.sum(), min=1.0)
    return masked.sum() / denom


def _evaluate(config: dict[str, Any], model, val_frame: pd.DataFrame, val_loader: DataLoader, device: torch.device) -> tuple[dict[str, Any], float]:
    task_type = config["task"]["type"]
    base_model = model
    if isinstance(model, dict):
        base_model = {key: (value.module if hasattr(value, "module") else value) for key, value in model.items()}
    elif hasattr(model, "module"):
        base_model = model.module

    if task_type == "metric_learning":
        pair_frame = load_manifest(config["data"]["pair_manifest"])
        pair_frame = resolve_image_paths(pair_frame, [config["data"]["left_image_col"], config["data"]["right_image_col"]], root=config["data"].get("manifest_root"))
        transform = build_transform(int(config["model"].get("input_size", 224)), augment=False)
        embeddings: dict[str, np.ndarray] = {}
        base_model["backbone"].eval()
        with torch.no_grad():
            for image_path in sorted(set(pair_frame[config["data"]["left_image_col"]]).union(set(pair_frame[config["data"]["right_image_col"]]))):
                tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                embeddings[image_path] = base_model["backbone"](tensor).cpu().numpy()[0]
        scores = []
        labels = []
        for _, row in pair_frame.iterrows():
            emb_left = embeddings[str(row[config["data"]["left_image_col"]])]
            emb_right = embeddings[str(row[config["data"]["right_image_col"]])]
            score = float(np.dot(emb_left, emb_right) / (np.linalg.norm(emb_left) * np.linalg.norm(emb_right) + 1e-9))
            scores.append(score)
            labels.append(int(row[config["data"]["pair_label_col"]]))
        report = verification_report(np.asarray(labels, dtype=np.int32), np.asarray(scores, dtype=np.float32))
        return report, -float(report["EER"])

    labels = []
    scores = []
    masks = []
    age_labels = []
    age_preds = []
    gender_labels = []
    gender_logits = []
    seg_labels = []
    seg_preds = []
    base_model.eval()
    with torch.no_grad():
        for batch in val_loader:
            if task_type == "age_gender_multitask":
                images, ages, genders = batch
                age_out, gender_out = base_model(images.to(device))
                age_labels.extend(ages.numpy().tolist())
                age_preds.extend(age_out.cpu().numpy().tolist())
                gender_labels.extend(genders.numpy().tolist())
                gender_logits.extend(gender_out.cpu().numpy().tolist())
            elif task_type == "segmentation":
                images, targets = batch
                logits = base_model(images.to(device))
                seg_labels.append(targets.numpy())
                seg_preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            else:
                if task_type == "multilabel_classification":
                    images, targets, label_mask = batch
                else:
                    images, targets = batch
                logits = base_model(images.to(device))
                if task_type == "multilabel_classification":
                    labels.append(targets.numpy())
                    scores.append(torch.sigmoid(logits).cpu().numpy())
                    masks.append(label_mask.numpy())
                elif task_type == "regression":
                    labels.extend(targets.numpy().tolist())
                    scores.extend(logits.squeeze(1).cpu().numpy().tolist())
                else:
                    labels.extend(targets.numpy().astype(int).tolist())
                    scores.extend(torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist())

    if task_type == "age_gender_multitask":
        report = age_gender_report(
            np.asarray(age_labels, dtype=np.float32),
            np.asarray(age_preds, dtype=np.float32),
            np.asarray(gender_labels, dtype=np.int32),
            np.asarray(gender_logits, dtype=np.float32),
        )
        return report, -(float(report["age_mae"]) + (1.0 - float(report["gender_accuracy"])))
    if task_type == "multilabel_classification":
        report = multilabel_report(np.vstack(labels), np.vstack(scores), mask=np.vstack(masks) if masks else None)
        return report, float(report["micro_f1"])
    if task_type == "segmentation":
        label_arr = np.concatenate(seg_labels, axis=0)
        pred_arr = np.concatenate(seg_preds, axis=0)
        report = segmentation_report(label_arr, pred_arr, num_classes=int(config["data"].get("num_classes", 19)))
        return report, float(report["mean_iou"])
    if task_type == "regression":
        threshold = config.get("evaluation", {}).get("accept_threshold")
        report = regression_report(np.asarray(labels, dtype=np.float32), np.asarray(scores, dtype=np.float32), threshold)
        return report, -float(report["mae"])

    label_arr = np.asarray(labels, dtype=np.int32)
    score_arr = np.asarray(scores, dtype=np.float32)
    threshold = search_binary_threshold(label_arr, score_arr, objective=config.get("evaluation", {}).get("threshold_objective", "balanced_accuracy"))
    if config["task"].get("reporting") == "pad":
        attack_col = config["data"].get("attack_type_col")
        attack_types = val_frame[attack_col].astype(str).tolist() if attack_col and attack_col in val_frame.columns else None
        report = pad_report(label_arr, score_arr, threshold, attack_types=attack_types)
    else:
        report = binary_classification_report(label_arr, score_arr, threshold)
    return report, float(report.get("balanced_accuracy", report.get("f1", 0.0)))


def _collect_binary_outputs(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    scores: list[float] = []
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()
    with torch.no_grad():
        for images, targets in loader:
            logits = base_model(images.to(device))
            labels.extend(targets.numpy().astype(int).tolist())
            scores.extend(torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist())
    return np.asarray(labels, dtype=np.int32), np.asarray(scores, dtype=np.float32)


def _collect_regression_outputs(model, loader: DataLoader, device: torch.device) -> np.ndarray:
    preds: list[float] = []
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()
    with torch.no_grad():
        for images, _targets in loader:
            preds.extend(base_model(images.to(device)).squeeze(1).cpu().numpy().tolist())
    return np.asarray(preds, dtype=np.float32)


def _collect_multilabel_outputs(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()
    with torch.no_grad():
        for images, targets, label_mask in loader:
            logits = base_model(images.to(device))
            labels.append(targets.cpu().numpy())
            scores.append(torch.sigmoid(logits).cpu().numpy())
            masks.append(label_mask.cpu().numpy())
    return np.vstack(labels), np.vstack(scores), np.vstack(masks)


def _save_binary_artifacts(
    report_dir: Path,
    task_name: str,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> None:
    pred_arr = (scores >= threshold).astype(np.int32)
    save_confusion_matrix(labels, pred_arr, report_dir / f"{task_name}_confusion.png", task_name)
    save_roc_curve(labels, scores, report_dir / f"{task_name}_roc.png", task_name)
    save_det_curve(labels, scores, report_dir / f"{task_name}_det.png", task_name)
    dump_json(
        {
            "task": task_name,
            "recommended_threshold": float(threshold),
            "score_range": {"min": float(np.min(scores)), "max": float(np.max(scores))},
        },
        report_dir / f"{task_name}_thresholds.json",
    )


def _save_demographic_slices(
    config: dict[str, Any],
    report: dict[str, Any],
    val_frame: pd.DataFrame,
    report_dir: Path,
    labels: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    age_labels: np.ndarray | None = None,
    age_preds: np.ndarray | None = None,
    gender_labels: np.ndarray | None = None,
    gender_logits: np.ndarray | None = None,
    multilabel_labels: np.ndarray | None = None,
    multilabel_scores: np.ndarray | None = None,
    multilabel_mask: np.ndarray | None = None,
) -> None:
    slice_columns = list(config.get("evaluation", {}).get("slice_columns", []))
    if not slice_columns:
        return

    task_type = config["task"]["type"]
    slice_frame = val_frame.reset_index(drop=True).copy()
    if task_type in {"binary_classification", "metric_learning", "regression"} and labels is not None and scores is not None:
        if len(slice_frame) != len(labels):
            return
        slice_frame["_label"] = labels
        slice_frame["_score"] = scores
        threshold = float(report.get("threshold", report.get("threshold_at_EER", 0.5)))

        if config["task"].get("reporting") == "pad":
            attack_col = config["data"].get("attack_type_col")

            def metric_fn(group: pd.DataFrame) -> dict[str, Any]:
                attack_types = group[attack_col].astype(str).tolist() if attack_col and attack_col in group.columns else None
                return pad_report(
                    group["_label"].astype(np.int32).to_numpy(),
                    group["_score"].astype(np.float32).to_numpy(),
                    threshold=threshold,
                    attack_types=attack_types,
                )
        elif task_type == "regression":
            accept_threshold = config.get("evaluation", {}).get("accept_threshold")

            def metric_fn(group: pd.DataFrame) -> dict[str, Any]:
                return regression_report(
                    group["_label"].astype(np.float32).to_numpy(),
                    group["_score"].astype(np.float32).to_numpy(),
                    accept_threshold=accept_threshold,
                )
        else:
            def metric_fn(group: pd.DataFrame) -> dict[str, Any]:
                return binary_classification_report(
                    group["_label"].astype(np.int32).to_numpy(),
                    group["_score"].astype(np.float32).to_numpy(),
                    threshold=threshold,
                )
    elif task_type == "age_gender_multitask" and all(item is not None for item in [age_labels, age_preds, gender_labels, gender_logits]):
        if len(slice_frame) != len(age_labels):
            return
        slice_frame["_age_label"] = age_labels
        slice_frame["_age_pred"] = age_preds
        slice_frame["_gender_label"] = gender_labels
        slice_frame["_gender_pred"] = np.argmax(gender_logits, axis=1)

        def metric_fn(group: pd.DataFrame) -> dict[str, Any]:
            return {
                "age_mae": float(np.mean(np.abs(group["_age_label"] - group["_age_pred"]))),
                "gender_accuracy": float(np.mean(group["_gender_label"] == group["_gender_pred"])),
            }
    elif task_type == "multilabel_classification" and all(item is not None for item in [multilabel_labels, multilabel_scores]):
        if len(slice_frame) != multilabel_labels.shape[0]:
            return
        slice_frame["_row_idx"] = np.arange(len(slice_frame), dtype=np.int32)

        def metric_fn(group: pd.DataFrame) -> dict[str, Any]:
            row_indices = group["_row_idx"].astype(np.int32).to_numpy()
            group_mask = multilabel_mask[row_indices] if multilabel_mask is not None else None
            return multilabel_report(
                multilabel_labels[row_indices],
                multilabel_scores[row_indices],
                mask=group_mask,
            )
    else:
        return

    slice_report = demographic_slice_report(slice_frame, metric_fn, slice_columns=slice_columns)
    dump_json(slice_report, report_dir / f"{config['task']['name']}_slice_report.json")


def fit(config: dict[str, Any]) -> dict[str, Any]:
    device, distributed, rank, _world_size = _setup_runtime()
    main_process = _is_main_process(rank)
    tracker = _tracker(config, enabled=main_process)
    try:
        train_frame, val_frame, train_loader, val_loader = _dataloaders(config, distributed)
        model, losses = _build_model_and_loss(config, train_frame, device)
        model = _wrap_model(model, distributed, device)

        parameters = []
        if isinstance(model, dict):
            for module in model.values():
                parameters.extend(list(module.parameters()))
        else:
            parameters = list(model.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(config["optimization"].get("lr", 3e-4)),
            weight_decay=float(config["optimization"].get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config["optimization"].get("epochs", 10)),
        )
        scaler = GradScaler(enabled=bool(config["optimization"].get("mixed_precision", True)) and device.type == "cuda")

        start_epoch = 0
        best_metric = float("-inf")
        last_ckpt, best_ckpt = _checkpoint_paths(config)
        resume_from = config["optimization"].get("resume_from")
        grad_accum_steps = max(1, int(config["optimization"].get("grad_accum_steps", 1)))
        if resume_from:
            start_epoch, best_metric = _load_checkpoint(resume_from, model, optimizer, scheduler)

        for epoch in range(start_epoch + 1, int(config["optimization"].get("epochs", 10)) + 1):
            if distributed and isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            if isinstance(model, dict):
                for module in model.values():
                    module.train()
            else:
                model.train()

            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for step_index, batch in enumerate(train_loader, start=1):
                with autocast(enabled=scaler.is_enabled()):
                    if config["task"]["type"] == "age_gender_multitask":
                        images, ages, genders = batch
                        age_out, gender_out = model(images.to(device))
                        loss = losses["age"](age_out, ages.to(device)) + losses["gender"](gender_out, genders.to(device))
                    elif config["task"]["type"] == "metric_learning":
                        images, labels = batch
                        embeddings = model["backbone"](images.to(device))
                        logits = model["head"](embeddings, labels.to(device))
                        loss = losses["classification"](logits, labels.to(device))
                    elif config["task"]["type"] == "multilabel_classification":
                        images, targets, label_mask = batch
                        logits = model(images.to(device))
                        loss = _masked_bce_loss(logits, targets.to(device), label_mask.to(device))
                    elif config["task"]["type"] == "segmentation":
                        images, targets = batch
                        logits = model(images.to(device))
                        loss = losses["segmentation"](logits, targets.to(device))
                    elif config["task"]["type"] == "regression":
                        images, targets = batch
                        preds = model(images.to(device)).squeeze(1)
                        loss = losses["score"](preds, targets.to(device))
                    else:
                        images, targets = batch
                        logits = model(images.to(device)).squeeze(1)
                        loss = losses["label"](logits, targets.to(device))
                scaled_loss = loss / float(grad_accum_steps)
                scaler.scale(scaled_loss).backward()
                if step_index % grad_accum_steps == 0 or step_index == len(train_loader):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(parameters, max_norm=float(config["optimization"].get("grad_clip_norm", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                running_loss += float(loss.detach().cpu())

            scheduler.step()

            if main_process:
                report, score = _evaluate(config, model, val_frame, val_loader, device)
                epoch_payload = {"epoch": epoch, "train_loss": running_loss / max(len(train_loader), 1), "selection_score": score}
                for key, value in report.items():
                    if isinstance(value, (int, float)):
                        epoch_payload[key] = value
                tracker.log(epoch_payload, step=epoch)
                _save_checkpoint(last_ckpt, epoch, model, optimizer, scheduler, best_metric, config)
                if score >= best_metric:
                    best_metric = score
                    _save_checkpoint(best_ckpt, epoch, model, optimizer, scheduler, best_metric, config)

        if main_process:
            dump_json({"best_metric": best_metric, "checkpoint": str(best_ckpt)}, last_ckpt.parent / "fit_summary.json")
        return {"best_metric": best_metric, "checkpoint": str(best_ckpt)}
    finally:
        tracker.close()
        _cleanup_runtime(distributed)


def evaluate(config: dict[str, Any], checkpoint_path: str | None = None) -> dict[str, Any]:
    device, distributed, rank, _world_size = _setup_runtime()
    main_process = _is_main_process(rank)
    try:
        train_frame, val_frame, _train_loader, val_loader = _dataloaders(config, distributed=False)
        model, _losses = _build_model_and_loss(config, train_frame, device)
        ckpt_path = Path(checkpoint_path or _checkpoint_paths(config)[1])
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        _load_state(model, checkpoint["model"])
        report, _score = _evaluate(config, model, val_frame, val_loader, device)

        if main_process:
            report_dir = Path(config.get("evaluation", {}).get("report_dir", "runs/reports")).resolve()
            report_dir.mkdir(parents=True, exist_ok=True)
            dump_json(report, report_dir / f"{config['task']['name']}_report.json")

            task_type = config["task"]["type"]
            if task_type == "metric_learning":
                pair_frame = load_manifest(config["data"]["pair_manifest"])
                pair_frame = resolve_image_paths(
                    pair_frame,
                    [config["data"]["left_image_col"], config["data"]["right_image_col"]],
                    root=config["data"].get("manifest_root"),
                )
                transform = build_transform(int(config["model"].get("input_size", 224)), augment=False)
                backbone = model["backbone"]
                backbone.eval()
                embeddings: dict[str, np.ndarray] = {}
                with torch.no_grad():
                    for image_path in sorted(set(pair_frame[config["data"]["left_image_col"]]).union(set(pair_frame[config["data"]["right_image_col"]]))):
                        tensor = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
                        embeddings[image_path] = backbone(tensor).cpu().numpy()[0]
                scores = []
                labels = []
                for _, row in pair_frame.iterrows():
                    emb_left = embeddings[str(row[config["data"]["left_image_col"]])]
                    emb_right = embeddings[str(row[config["data"]["right_image_col"]])]
                    scores.append(float(np.dot(emb_left, emb_right) / (np.linalg.norm(emb_left) * np.linalg.norm(emb_right) + 1e-9)))
                    labels.append(int(row[config["data"]["pair_label_col"]]))
                _save_binary_artifacts(
                    report_dir,
                    config["task"]["name"],
                    np.asarray(labels, dtype=np.int32),
                    np.asarray(scores, dtype=np.float32),
                    threshold=float(report["threshold_at_EER"]),
                )
                _save_demographic_slices(
                    config,
                    report,
                    pair_frame,
                    report_dir,
                    labels=np.asarray(labels, dtype=np.int32),
                    scores=np.asarray(scores, dtype=np.float32),
                )
            elif task_type == "binary_classification":
                label_arr, score_arr = _collect_binary_outputs(model, val_loader, device)
                _save_binary_artifacts(report_dir, config["task"]["name"], label_arr, score_arr, threshold=float(report["threshold"]))
                _save_demographic_slices(config, report, val_frame, report_dir, labels=label_arr, scores=score_arr)
            elif task_type == "regression":
                label_arr = val_frame[config["data"]["label_col"]].astype(np.float32).to_numpy()
                score_arr = _collect_regression_outputs(model, val_loader, device)
                _save_demographic_slices(config, report, val_frame, report_dir, labels=label_arr, scores=score_arr)
            elif task_type == "multilabel_classification":
                multilabel_labels, multilabel_scores, multilabel_mask = _collect_multilabel_outputs(model, val_loader, device)
                _save_demographic_slices(
                    config,
                    report,
                    val_frame,
                    report_dir,
                    multilabel_labels=multilabel_labels,
                    multilabel_scores=multilabel_scores,
                    multilabel_mask=multilabel_mask,
                )
            elif task_type == "age_gender_multitask":
                age_labels = []
                age_preds = []
                gender_labels = []
                gender_logits = []
                model.eval()
                with torch.no_grad():
                    for images, ages, genders in val_loader:
                        age_out, gender_out = model(images.to(device))
                        age_labels.extend(ages.numpy().tolist())
                        age_preds.extend(age_out.cpu().numpy().tolist())
                        gender_labels.extend(genders.numpy().tolist())
                        gender_logits.extend(gender_out.cpu().numpy().tolist())
                _save_demographic_slices(
                    config,
                    report,
                    val_frame,
                    report_dir,
                    age_labels=np.asarray(age_labels, dtype=np.float32),
                    age_preds=np.asarray(age_preds, dtype=np.float32),
                    gender_labels=np.asarray(gender_labels, dtype=np.int32),
                    gender_logits=np.asarray(gender_logits, dtype=np.float32),
                )

        return report
    finally:
        _cleanup_runtime(distributed)
