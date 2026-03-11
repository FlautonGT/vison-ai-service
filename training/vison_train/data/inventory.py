"""Dataset inventory loading and selection logic."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_INVENTORY = Path(__file__).resolve().parents[3] / "configs" / "datasets" / "dataset_inventory.json"

REGION_SCORES = {
    "indonesia": 4,
    "southeast_asia": 3,
    "asia": 2,
    "global": 1,
}

STATUS_SCORES = {
    "approved": 4,
    "restricted": 1,
    "fallback_only": 0,
    "rejected": -5,
}


@dataclass(frozen=True)
class DatasetRecord:
    payload: dict[str, Any]

    @property
    def id(self) -> str:
        return str(self.payload["id"])

    @property
    def task_fit(self) -> list[str]:
        return list(self.payload.get("task_fit", []))

    @property
    def region_coverage(self) -> list[str]:
        return list(self.payload.get("region_coverage", []))

    @property
    def commercial_use_ok(self) -> bool:
        return bool(self.payload.get("commercial_use_ok", False))

    @property
    def modifiable(self) -> bool:
        return bool(self.payload.get("modifiable", False))

    @property
    def status(self) -> str:
        return str(self.payload.get("status", "restricted"))

    @property
    def approx_sample_count(self) -> int:
        value = self.payload.get("approx_sample_count", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    @property
    def approx_subject_count(self) -> int:
        value = self.payload.get("approx_subject_count", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def region_score(self, preferred_region: str) -> int:
        if preferred_region in self.region_coverage:
            return REGION_SCORES.get(preferred_region, 0) + 2
        for region in self.region_coverage:
            if region in REGION_SCORES:
                return REGION_SCORES[region]
        return 0


def _scale_score(count: int) -> int:
    if count <= 0:
        return 0
    if count >= 1_000_000:
        return 6
    if count >= 500_000:
        return 5
    if count >= 100_000:
        return 4
    if count >= 50_000:
        return 3
    if count >= 10_000:
        return 2
    if count >= 1_000:
        return 1
    return 0


def load_inventory(path: str | Path | None = None) -> dict[str, Any]:
    inventory_path = Path(path or DEFAULT_INVENTORY).expanduser().resolve()
    with inventory_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["_inventory_path"] = str(inventory_path)
    return payload


def select_datasets(
    task: str,
    preferred_region: str = "indonesia",
    require_commercial: bool = True,
    require_modifiable: bool = True,
    allowed_statuses: list[str] | None = None,
    path: str | Path | None = None,
) -> list[dict[str, Any]]:
    inventory = load_inventory(path)
    allowed = set(allowed_statuses or inventory.get("selection_policy", {}).get("default_requirements", {}).get("allowed_statuses", ["approved"]))
    ranked: list[dict[str, Any]] = []
    for raw in inventory.get("datasets", []):
        record = DatasetRecord(raw)
        if task not in record.task_fit:
            continue
        if record.status not in allowed:
            continue
        if require_commercial and not record.commercial_use_ok:
            continue
        if require_modifiable and not record.modifiable:
            continue

        score = record.region_score(preferred_region) + STATUS_SCORES.get(record.status, 0)
        if record.commercial_use_ok:
            score += 3
        if record.modifiable:
            score += 2
        if "subject_disjoint" in str(raw.get("recommended_split", "")):
            score += 1
        score += _scale_score(record.approx_sample_count)
        if task == "verification":
            score += _scale_score(record.approx_subject_count)

        ranked.append(
            {
                "score": score,
                "dataset": raw,
                "selection_reasons": [
                    f"task={task}",
                    f"preferred_region={preferred_region}",
                    f"commercial_use_ok={record.commercial_use_ok}",
                    f"modifiable={record.modifiable}",
                    f"status={record.status}",
                    f"approx_sample_count={record.approx_sample_count}",
                    f"approx_subject_count={record.approx_subject_count}",
                    f"recommended_split={raw.get('recommended_split', 'unspecified')}",
                ],
            }
        )

    ranked.sort(key=lambda item: (-item["score"], item["dataset"]["id"]))
    return ranked
