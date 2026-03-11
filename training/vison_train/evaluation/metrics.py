"""Task metrics for detection, PAD, quality, and verification."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np
from sklearn import metrics


def search_binary_threshold(labels: np.ndarray, scores: np.ndarray, objective: str = "balanced_accuracy") -> float:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.0, 1.0, 401):
        preds = (scores >= threshold).astype(np.int32)
        score = metrics.f1_score(labels, preds, zero_division=0) if objective == "f1" else metrics.balanced_accuracy_score(labels, preds)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold


def binary_classification_report(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(np.int32)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    roc_auc = metrics.roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else None
    return {
        "threshold": threshold,
        "accuracy": float(metrics.accuracy_score(labels, preds)),
        "balanced_accuracy": float(metrics.balanced_accuracy_score(labels, preds)),
        "precision": float(metrics.precision_score(labels, preds, zero_division=0)),
        "recall": float(metrics.recall_score(labels, preds, zero_division=0)),
        "f1": float(metrics.f1_score(labels, preds, zero_division=0)),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }


def pad_report(labels: np.ndarray, scores: np.ndarray, threshold: float, attack_types: Optional[list[str]] = None) -> dict:
    preds = (scores >= threshold).astype(np.int32)
    attack_mask = labels == 1
    bona_fide_mask = labels == 0
    apcer = float(np.mean(preds[attack_mask] == 0)) if np.any(attack_mask) else 0.0
    bpcer = float(np.mean(preds[bona_fide_mask] == 1)) if np.any(bona_fide_mask) else 0.0
    result = {
        **binary_classification_report(labels, scores, threshold),
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": (apcer + bpcer) * 0.5,
    }
    if attack_types is not None:
        per_attack = defaultdict(list)
        for attack_type, label, pred in zip(attack_types, labels, preds):
            if int(label) == 1:
                per_attack[str(attack_type)].append(int(pred))
        result["per_attack_APCER"] = {
            key: float(np.mean(np.asarray(values) == 0)) if values else None
            for key, values in sorted(per_attack.items())
        }
    return result


def multilabel_report(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5, mask: np.ndarray | None = None) -> dict:
    preds = (scores >= threshold).astype(np.int32)
    if mask is not None:
        valid = mask.astype(bool)
        if not np.any(valid):
            return {"subset_accuracy": 0.0, "micro_f1": 0.0, "macro_f1": 0.0}
        flat_labels = labels[valid]
        flat_preds = preds[valid]
        per_label_f1 = []
        for index in range(labels.shape[1]):
            column_mask = valid[:, index]
            if not np.any(column_mask):
                continue
            per_label_f1.append(
                metrics.f1_score(labels[column_mask, index], preds[column_mask, index], zero_division=0)
            )
        subset_values = []
        for row_index in range(labels.shape[0]):
            row_mask = valid[row_index]
            if not np.any(row_mask):
                continue
            subset_values.append(bool(np.all(labels[row_index, row_mask] == preds[row_index, row_mask])))
        return {
            "subset_accuracy": float(np.mean(subset_values)) if subset_values else 0.0,
            "micro_f1": float(metrics.f1_score(flat_labels, flat_preds, average="binary", zero_division=0)),
            "macro_f1": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        }
    return {
        "subset_accuracy": float(metrics.accuracy_score(labels, preds)),
        "micro_f1": float(metrics.f1_score(labels, preds, average="micro", zero_division=0)),
        "macro_f1": float(metrics.f1_score(labels, preds, average="macro", zero_division=0)),
    }


def regression_report(labels: np.ndarray, preds: np.ndarray, accept_threshold: float | None = None) -> dict:
    result = {
        "mae": float(metrics.mean_absolute_error(labels, preds)),
        "rmse": float(np.sqrt(metrics.mean_squared_error(labels, preds))),
    }
    if accept_threshold is not None:
        bin_labels = (labels >= accept_threshold).astype(np.int32)
        bin_preds = (preds >= accept_threshold).astype(np.int32)
        tn, fp, fn, tp = metrics.confusion_matrix(bin_labels, bin_preds, labels=[0, 1]).ravel()
        result["accept_reject_confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return result


def age_gender_report(age_labels: np.ndarray, age_preds: np.ndarray, gender_labels: np.ndarray, gender_logits: np.ndarray) -> dict:
    gender_preds = np.argmax(gender_logits, axis=1)
    return {
        "age_mae": float(metrics.mean_absolute_error(age_labels, age_preds)),
        "age_rmse": float(np.sqrt(metrics.mean_squared_error(age_labels, age_preds))),
        "gender_accuracy": float(metrics.accuracy_score(gender_labels, gender_preds)),
    }


def segmentation_report(labels: np.ndarray, preds: np.ndarray, num_classes: int) -> dict:
    pixel_accuracy = float(np.mean(labels == preds)) if labels.size > 0 else 0.0
    ious = []
    for class_id in range(num_classes):
        label_mask = labels == class_id
        pred_mask = preds == class_id
        union = np.logical_or(label_mask, pred_mask).sum()
        if union == 0:
            continue
        intersection = np.logical_and(label_mask, pred_mask).sum()
        ious.append(float(intersection / union))
    return {
        "pixel_accuracy": pixel_accuracy,
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
    }


def demographic_slice_report(frame, metric_fn, slice_columns: list[str]) -> dict:
    report = {}
    for column in slice_columns:
        if column not in frame.columns:
            continue
        report[column] = {str(value): metric_fn(group) for value, group in frame.groupby(column)}
    return report


def verification_report(labels: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    fnr = 1.0 - tpr
    eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
    result = {
        "EER": float((fpr[eer_index] + fnr[eer_index]) * 0.5),
        "threshold_at_EER": float(thresholds[eer_index]),
        "ROC_AUC": float(metrics.auc(fpr, tpr)),
        "FNMR_FMR_operating_points": {},
    }
    for target_fmr in (0.0001, 0.001, 0.01):
        eligible = np.where(fpr <= target_fmr)[0]
        if eligible.size == 0:
            result["FNMR_FMR_operating_points"][str(target_fmr)] = None
            continue
        best_idx = int(eligible[np.argmin(fnr[eligible])])
        result["FNMR_FMR_operating_points"][str(target_fmr)] = {
            "threshold": float(thresholds[best_idx]),
            "FMR": float(fpr[best_idx]),
            "FNMR": float(fnr[best_idx]),
        }
    return result
