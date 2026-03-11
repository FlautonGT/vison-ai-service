"""Plot helpers for confusion matrices, ROC, and DET curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def save_confusion_matrix(labels: np.ndarray, preds: np.ndarray, output_path: str | Path, title: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = metrics.confusion_matrix(labels, preds, labels=[0, 1])
    fig, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, str(int(matrix[row, col])), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_roc_curve(labels: np.ndarray, scores: np.ndarray, output_path: str | Path, title: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5, 5))
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = metrics.roc_curve(labels, scores)
        auc_score = metrics.auc(fpr, tpr)
        axis.plot(fpr, tpr, label=f"AUC={auc_score:.4f}")
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.legend(loc="lower right")
    else:
        axis.text(0.5, 0.5, "ROC unavailable: single-class labels", ha="center", va="center")
    axis.set_title(title)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_det_curve(labels: np.ndarray, scores: np.ndarray, output_path: str | Path, title: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(5, 5))
    if len(np.unique(labels)) > 1:
        fpr, fnr, _ = metrics.det_curve(labels, scores)
        axis.plot(fpr, fnr)
        axis.set_xscale("log")
        axis.set_yscale("log")
    else:
        axis.text(0.5, 0.5, "DET unavailable: single-class labels", ha="center", va="center")
    axis.set_title(title)
    axis.set_xlabel("False Match / False Accept Rate")
    axis.set_ylabel("False Non-Match / False Reject Rate")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
