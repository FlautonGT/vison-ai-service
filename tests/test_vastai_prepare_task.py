from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT_DIR / "scripts" / "vastai_prepare_task.py"
SPEC = importlib.util.spec_from_file_location("vastai_prepare_task", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
vastai_prepare_task = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vastai_prepare_task)


class VastaiPrepareTaskTests(unittest.TestCase):
    def test_verification_selection_prefers_asian_kyc_when_allowed(self) -> None:
        args = argparse.Namespace(
            preferred_region="indonesia",
            allow_noncommercial=True,
            allow_nonmodifiable=True,
            allow_restricted=True,
            allow_fallback=True,
            allow_rejected=False,
            inventory=str(ROOT_DIR / "configs" / "datasets" / "dataset_inventory.json"),
            max_datasets=4,
        )

        selected = vastai_prepare_task._select("verification", args)

        self.assertGreaterEqual(len(selected), 1)
        self.assertEqual(selected[0]["dataset"]["id"], "trainingdatapro/asian-kyc-photo-dataset")

    def test_deepfake_regional_assessment_warns_when_only_global_fallbacks_exist(self) -> None:
        args = argparse.Namespace(
            preferred_region="indonesia",
            allow_noncommercial=False,
            allow_nonmodifiable=False,
            allow_restricted=False,
            allow_fallback=True,
            allow_rejected=False,
            inventory=str(ROOT_DIR / "configs" / "datasets" / "dataset_inventory.json"),
            max_datasets=4,
        )

        selected = vastai_prepare_task._select("deepfake", args)
        assessment = vastai_prepare_task._regional_assessment("deepfake", selected)

        self.assertGreaterEqual(assessment["counts"]["global_only"], 1)
        self.assertTrue(any("fallback" in warning.lower() for warning in assessment["warnings"]))


if __name__ == "__main__":
    unittest.main()
