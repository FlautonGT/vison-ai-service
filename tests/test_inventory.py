from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.vison_train.data.inventory import select_datasets


class InventorySelectionTests(unittest.TestCase):
    def test_select_datasets_prefers_larger_verification_fallback_when_allowed(self) -> None:
        inventory = {
            "selection_policy": {
                "default_requirements": {
                    "allowed_statuses": ["approved", "restricted", "fallback_only"],
                }
            },
            "datasets": [
                {
                    "id": "small/sea",
                    "task_fit": ["verification"],
                    "region_coverage": ["southeast_asia"],
                    "commercial_use_ok": False,
                    "modifiable": False,
                    "status": "restricted",
                    "approx_sample_count": 75,
                    "approx_subject_count": 5,
                    "recommended_split": "subject_disjoint",
                },
                {
                    "id": "large/global",
                    "task_fit": ["verification"],
                    "region_coverage": ["global"],
                    "commercial_use_ok": False,
                    "modifiable": False,
                    "status": "restricted",
                    "approx_sample_count": 202599,
                    "approx_subject_count": 10177,
                    "recommended_split": "subject_disjoint",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            inventory_path = Path(tmp_dir) / "inventory.json"
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

            ranked = select_datasets(
                task="verification",
                preferred_region="indonesia",
                require_commercial=False,
                require_modifiable=False,
                allowed_statuses=["restricted"],
                path=inventory_path,
            )

        self.assertEqual(ranked[0]["dataset"]["id"], "large/global")


if __name__ == "__main__":
    unittest.main()
