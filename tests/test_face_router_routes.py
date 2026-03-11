from __future__ import annotations

import unittest

from app.api.face_router import router


class FaceRouterRouteTests(unittest.TestCase):
    def test_router_exposes_new_routes(self) -> None:
        paths = {route.path for route in router.routes}
        self.assertIn("/capabilities", paths)
        self.assertIn("/quality", paths)
        self.assertIn("/attributes", paths)


if __name__ == "__main__":
    unittest.main()
