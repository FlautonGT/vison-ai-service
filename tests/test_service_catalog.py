from __future__ import annotations

import unittest

from app.core.service_catalog import load_service_catalog


class ServiceCatalogTests(unittest.TestCase):
    def test_service_catalog_contains_new_endpoints(self) -> None:
        catalog = load_service_catalog()
        endpoints = set(catalog.endpoint_names())
        self.assertIn("quality", endpoints)
        self.assertIn("attributes", endpoints)
        self.assertIn("capabilities", endpoints)

    def test_compare_endpoint_declares_verification_models(self) -> None:
        catalog = load_service_catalog()
        model_keys = set(catalog.endpoint_model_keys("compare"))
        self.assertIn("face_detector", model_keys)
        self.assertIn("face_recognition_primary", model_keys)


if __name__ == "__main__":
    unittest.main()
