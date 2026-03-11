from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from training.vison_train.data.builders import (
    build_age_gender_manifest,
    build_celebamaskhq_parser_manifest,
    build_identity_manifest,
    generate_verification_pairs,
)


class ManifestBuilderTests(unittest.TestCase):
    def test_build_age_gender_manifest_resolves_fairface_nested_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "ghaidaalatoum__fairface"
            fairface_root = dataset_dir / "fairface"
            train_dir = fairface_root / "train"
            train_dir.mkdir(parents=True, exist_ok=True)

            image_path = train_dir / "1.jpg"
            Image.fromarray(np.full((8, 8, 3), 127, dtype=np.uint8)).save(image_path)

            pd.DataFrame(
                [
                    {
                        "file": "train/1.jpg",
                        "age": "20-29",
                        "gender": "Female",
                        "race": "Southeast Asian",
                    }
                ]
            ).to_csv(fairface_root / "fairface_label_train.csv", index=False)

            frame = build_age_gender_manifest([dataset_dir])

            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["image_path"], str(image_path.resolve()))
            self.assertEqual(frame.iloc[0]["gender_label"], "female")
            self.assertEqual(frame.iloc[0]["race_label"], "Southeast Asian")
            self.assertEqual(frame.iloc[0]["region_proxy"], "southeast_asia")

    def test_build_celebamaskhq_parser_manifest_matches_zero_padded_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "minipromax__celebamask_hq"
            images_dir = dataset_dir / "CelebAMask-HQ" / "CelebAMask-HQ" / "CelebA-HQ-img"
            masks_dir = dataset_dir / "CelebAMask-HQ" / "CelebAMask-HQ" / "CelebAMask-HQ-mask-anno" / "0"
            images_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)

            image_path = images_dir / "0.jpg"
            Image.fromarray(np.full((4, 4, 3), 200, dtype=np.uint8)).save(image_path)

            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[1:3, 1:3] = 255
            Image.fromarray(mask).save(masks_dir / "00000_hat.png")

            output_dir = Path(tmp_dir) / "generated_masks"
            frame = build_celebamaskhq_parser_manifest([dataset_dir], output_dir)

            self.assertEqual(len(frame), 1)
            saved_mask = Path(frame.iloc[0]["mask_path"])
            self.assertTrue(saved_mask.exists())
            saved_mask_arr = np.array(Image.open(saved_mask))
            self.assertEqual(int(saved_mask_arr.max()), 18)

    def test_generate_verification_pairs_preserves_region_and_capture_metadata(self) -> None:
        frame = pd.DataFrame(
            {
                "image_path": ["a1.jpg", "a2.jpg", "b1.jpg", "b2.jpg"],
                "subject_id": ["a", "a", "b", "b"],
                "region_proxy": ["southeast_asia", "southeast_asia", "global", "global"],
                "capture_type": ["selfie", "document", "selfie", "document"],
                "source_dataset": ["asian_kyc", "asian_kyc", "fallback", "fallback"],
            }
        )

        pairs = generate_verification_pairs(frame, positives_per_subject=1, negatives_per_subject=1, seed=7)

        self.assertIn("pair_region_proxy", pairs.columns)
        self.assertIn("left_capture_type", pairs.columns)
        self.assertTrue((pairs.loc[pairs["is_match"] == 1, "pair_region_proxy"] != "mixed").all())
        self.assertTrue((pairs.loc[pairs["is_match"] == 0, "pair_region_proxy"] == "mixed").any())

    def test_build_identity_manifest_uses_subject_dirs_and_skips_corrupt_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_dir = Path(tmp_dir) / "trainingdatapro__asian_kyc_photo_dataset"
            subject_dir = dataset_dir / "files" / "12"
            subject_dir.mkdir(parents=True, exist_ok=True)

            Image.fromarray(np.full((6, 6, 3), 120, dtype=np.uint8)).save(subject_dir / "Selfie_1.jpg")
            Image.fromarray(np.full((6, 6, 3), 140, dtype=np.uint8)).save(subject_dir / "ID_1.jpg")
            (subject_dir / "broken.jpg").write_bytes(b"not_an_image")

            frame = build_identity_manifest([dataset_dir])

            self.assertEqual(len(frame), 2)
            self.assertEqual(set(frame["capture_type"]), {"selfie", "document"})
            self.assertEqual(set(frame["region_proxy"]), {"southeast_asia"})
            self.assertEqual(len(frame["subject_id"].unique()), 1)


if __name__ == "__main__":
    unittest.main()
