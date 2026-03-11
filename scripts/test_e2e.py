#!/usr/bin/env python3
"""Minimal HTTP smoke checks for a running service instance."""

from __future__ import annotations

import json
import os
import sys

import requests


def _base_url() -> str:
    host = os.getenv("E2E_HOST", "127.0.0.1")
    port = os.getenv("APP_PORT", os.getenv("FACE_AI_PORT", "8000"))
    if host.startswith("http://") or host.startswith("https://"):
        return host.rstrip("/")
    return f"http://{host}:{port}"


def main() -> int:
    base_url = _base_url()
    secret = os.getenv("AI_SERVICE_SECRET", "").strip()
    headers = {"X-AI-Service-Key": secret} if secret else {}

    health = requests.get(f"{base_url}/health", timeout=20)
    health.raise_for_status()
    capabilities = requests.get(f"{base_url}/api/face/capabilities", headers=headers, timeout=20)
    capabilities.raise_for_status()

    payload = {
        "health": health.json(),
        "capabilities": capabilities.json(),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.RequestException as exc:
        print(f"Smoke test failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
