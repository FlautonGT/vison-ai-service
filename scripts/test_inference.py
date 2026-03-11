#!/usr/bin/env python3
"""Static inference configuration checks without loading every model session."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import settings
from app.core.service_catalog import load_service_catalog


def _resolve_setting(name: str) -> str | list[str] | None:
    return getattr(settings, name, None) if name else None


def main() -> int:
    catalog = load_service_catalog()
    model_dir = Path(settings.MODEL_DIR).expanduser().resolve()
    checks = []
    for model_key, model_spec in catalog.models.items():
        setting_name = model_spec.get("setting")
        configured = _resolve_setting(setting_name)
        if isinstance(configured, list):
            values = configured
        elif isinstance(configured, str) and "," in configured:
            values = [item.strip() for item in configured.split(",") if item.strip()]
        elif configured:
            values = [str(configured)]
        else:
            values = []

        paths = []
        for value in values:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = model_dir / candidate
            paths.append({"path": str(candidate), "exists": candidate.exists()})

        checks.append(
            {
                "model_key": model_key,
                "setting": setting_name,
                "configured": values,
                "paths": paths,
            }
        )

    print(json.dumps({"model_dir": str(model_dir), "checks": checks}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
