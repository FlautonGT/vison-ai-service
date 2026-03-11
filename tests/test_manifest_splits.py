from __future__ import annotations

import unittest

import pandas as pd

from training.vison_train.data.manifests import split_manifest


class ManifestSplitTests(unittest.TestCase):
    def test_split_manifest_is_group_disjoint(self) -> None:
        frame = pd.DataFrame(
            {
                "image_path": [f"img_{idx}.jpg" for idx in range(12)],
                "subject_id": ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"],
                "label": [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            }
        )
        splits = split_manifest(frame, group_cols=["subject_id"], val_ratio=0.17, test_ratio=0.17, seed=7, stratify_col="label")

        train_ids = set(splits["train"]["subject_id"])
        val_ids = set(splits["val"]["subject_id"])
        test_ids = set(splits["test"]["subject_id"])

        self.assertFalse(train_ids & val_ids)
        self.assertFalse(train_ids & test_ids)
        self.assertFalse(val_ids & test_ids)

    def test_holdout_values_are_reserved_for_test(self) -> None:
        frame = pd.DataFrame(
            {
                "image_path": [f"img_{idx}.jpg" for idx in range(6)],
                "subject_id": ["a", "b", "c", "d", "e", "f"],
                "label": [0, 0, 1, 1, 0, 1],
                "attack_type": ["print", "print", "replay", "replay", "mask", "mask"],
            }
        )
        splits = split_manifest(
            frame,
            group_cols=["subject_id"],
            holdout_column="attack_type",
            holdout_values=["mask"],
        )
        self.assertTrue(all(value == "mask" for value in splits["test"].loc[splits["test"]["attack_type"] == "mask", "attack_type"]))


if __name__ == "__main__":
    unittest.main()
