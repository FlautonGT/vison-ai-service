"""JSON configuration helpers for training and evaluation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_config_path"] = str(config_path)
    return payload


def dump_json(payload: dict[str, Any], path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def apply_overrides(payload: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    updated = copy.deepcopy(payload)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected dotted.path=value")
        dotted_key, raw_value = item.split("=", 1)
        value = json.loads(raw_value) if raw_value[:1] in "[{\"" else _coerce_value(raw_value)
        cursor = updated
        parts = dotted_key.split(".")
        for key in parts[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[parts[-1]] = value
    return updated


def _coerce_value(raw_value: str) -> Any:
    lowered = raw_value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        if "." in raw_value:
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value
