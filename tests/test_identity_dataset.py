from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from training.vison_train.data.datasets import IdentityImageDataset, build_transform


class IdentityDatasetTests(unittest.TestCase):
    def test_identity_dataset_maps_string_subject_ids_to_class_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            image_paths = []
            for index in range(2):
                image_path = Path(tmp_dir) / f"{index}.jpg"
                Image.fromarray(np.full((8, 8, 3), 100 + index, dtype=np.uint8)).save(image_path)
                image_paths.append(str(image_path))

            frame = pd.DataFrame(
                {
                    "image_path": image_paths,
                    "subject_id": ["subject_a", "subject_b"],
                }
            )

            dataset = IdentityImageDataset(
                frame,
                image_col="image_path",
                identity_col="subject_id",
                transform=build_transform(16, augment=False),
            )

            _image_a, label_a = dataset[0]
            _image_b, label_b = dataset[1]

            self.assertEqual(int(label_a), 0)
            self.assertEqual(int(label_b), 1)


if __name__ == "__main__":
    unittest.main()
