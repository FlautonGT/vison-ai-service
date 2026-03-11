"""ONNX export and handoff helpers for trained task models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .config import dump_json
from .runner import _build_model_and_loss, _checkpoint_paths, _load_state, _resolve_frames


class _AgeGenderConcat(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        age_out, gender_out = self.model(x)
        return torch.cat([gender_out, age_out.unsqueeze(1)], dim=1)


def _wrapped_model(config: dict[str, Any], model):
    task_type = config["task"]["type"]
    if task_type == "metric_learning":
        return model["backbone"], ["embedding"]
    if task_type == "age_gender_multitask":
        return _AgeGenderConcat(model), ["age_gender"]
    if task_type == "segmentation":
        return model, ["segmentation"]
    if task_type == "binary_classification":
        return model, ["logit"]
    if task_type == "regression":
        return model, ["score"]
    if task_type == "multilabel_classification":
        return model, ["attributes"]
    return model, ["output"]


def _env_hints(config: dict[str, Any], export_path: Path) -> dict[str, Any]:
    task_name = config["task"]["name"]
    mapping = {
        "deepfake_detection": {"env": ["DEEPFAKE_MODELS", "AI_FACE_DETECTOR_MODEL"]},
        "passive_pad": {"env": ["LIVENESS_MODELS"]},
        "verification_embedding": {"env": ["ARCFACE_EXTRA_MODEL", "ARCFACE_MODEL"]},
        "age_gender": {"env": ["AGE_GENDER_MODEL"]},
        "face_parsing": {"env": ["FACE_PARSING_MODEL"]},
        "face_attributes": {"env": []},
        "face_quality": {"env": []},
    }
    payload = mapping.get(task_name, {"env": []})
    payload["artifact"] = str(export_path)
    payload["local_inference_ready"] = bool(payload["env"])
    return payload


def export_onnx(config: dict[str, Any], checkpoint_path: str | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    device = torch.device("cpu")
    train_frame, _val_frame = _resolve_frames(config)
    model, _losses = _build_model_and_loss(config, train_frame, device)
    ckpt_path = Path(checkpoint_path or _checkpoint_paths(config)[1]).expanduser().resolve()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    _load_state(model, checkpoint["model"])
    export_model, output_names = _wrapped_model(config, model)
    export_model.eval()

    artifact_dir = Path(output_dir or Path(config["optimization"].get("checkpoint_dir", "runs/checkpoints")).resolve().parent / "artifacts").expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = artifact_dir / f"{config['task']['name']}.onnx"
    input_size = int(config["model"].get("input_size", 224))
    dummy = torch.randn(1, 3, input_size, input_size, dtype=torch.float32)

    torch.onnx.export(
        export_model,
        dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=output_names,
        dynamic_axes={"input": {0: "batch"}},
        opset_version=int(config.get("export", {}).get("opset", 17)),
    )

    parity = {"checked": False}
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        with torch.no_grad():
            torch_out = export_model(dummy).detach().cpu().numpy()
        onnx_out = session.run(None, {"input": dummy.numpy()})[0]
        parity = {
            "checked": True,
            "max_abs_diff": float(np.max(np.abs(torch_out - onnx_out))),
            "mean_abs_diff": float(np.mean(np.abs(torch_out - onnx_out))),
        }
    except Exception as exc:
        parity = {"checked": False, "error": str(exc)}

    manifest = {
        "task": config["task"]["name"],
        "checkpoint": str(ckpt_path),
        "onnx_path": str(onnx_path),
        "parity": parity,
        "handoff": _env_hints(config, onnx_path),
    }
    dump_json(manifest, artifact_dir / f"{config['task']['name']}_export_manifest.json")
    return manifest
