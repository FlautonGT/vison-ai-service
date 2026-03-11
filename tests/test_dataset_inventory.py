from __future__ import annotations

import unittest

from training.vison_train.data.inventory import select_datasets


class DatasetInventoryTests(unittest.TestCase):
    def test_deepfake_selection_prefers_commercial_modifiable_option(self) -> None:
        ranked = select_datasets("deepfake_detection", allowed_statuses=["approved", "fallback_only"])
        self.assertGreaterEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["dataset"]["id"], "kshitizbhargava/deepfake-face-images")

    def test_verification_returns_candidates_when_noncommercial_is_allowed(self) -> None:
        ranked = select_datasets(
            "verification",
            require_commercial=False,
            require_modifiable=False,
            allowed_statuses=["approved", "restricted", "fallback_only"],
        )
        self.assertGreaterEqual(len(ranked), 1)


if __name__ == "__main__":
    unittest.main()
