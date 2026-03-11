from __future__ import annotations

import unittest

import numpy as np

from app.services.attributes import build_attribute_report


class AttributeReportTests(unittest.TestCase):
    def test_build_attribute_report_combines_parser_and_quality(self) -> None:
        face_crop = np.full((64, 64, 3), 120, dtype=np.uint8)
        landmarks = np.array([[18, 20], [46, 20], [32, 32], [22, 46], [42, 46]], dtype=np.float32)
        parser_attributes = {
            "hasGlasses": True,
            "hasMask": False,
            "hasHat": True,
            "hasBeard": True,
            "faceVisibleRatio": 0.72,
            "hatCoverage": 0.18,
            "glassesCoverage": 0.35,
        }
        quality_payload = {
            "brightness": 58.0,
            "sharpness": 67.0,
            "illumination": {"asymmetry": 12.0, "isUniform": True},
        }

        result = build_attribute_report(face_crop, landmarks, parser_attributes, quality_payload)

        self.assertIn("eyeglasses", result)
        self.assertIn("hatCap", result)
        self.assertTrue(result["hatCap"])
        self.assertTrue(result["facialHair"])
        self.assertEqual(result["brightness"]["label"], "GOOD")
        self.assertEqual(result["blurSharpness"]["label"], "SHARP")


if __name__ == "__main__":
    unittest.main()
