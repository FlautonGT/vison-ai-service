"""End-to-end tests for pure inference API endpoints."""

from __future__ import annotations

import json
import os
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@dataclass
class TestResult:
    name: str
    passed: bool
    elapsed_ms: float
    detail: str = ""
    kind: str = "success"


class BackgroundServer:
    def __init__(self, host: str, port: int, model_dir: Optional[str] = None):
        self.host = host
        self.port = port
        self.model_dir = model_dir
        self.process: Optional[subprocess.Popen] = None
        self._logs: list[str] = []

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self):
        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        if self.model_dir:
            env["MODEL_DIR"] = self.model_dir

        cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        self.process = subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def _reader():
            assert self.process is not None
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self._logs.append(line.rstrip())
                if len(self._logs) > 400:
                    self._logs = self._logs[-400:]

        threading.Thread(target=_reader, daemon=True).start()

    def wait_ready(self, timeout_sec: int = 300):
        deadline = time.time() + timeout_sec
        health_url = f"{self.base_url}/health"
        while time.time() < deadline:
            assert self.process is not None
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"uvicorn exited early ({self.process.returncode})\n"
                    + "\n".join(self._logs[-80:])
                )
            try:
                response = requests.get(health_url, timeout=3)
                if response.status_code == 200:
                    payload = response.json()
                    if payload.get("status") == "ok" and payload.get("models_loaded") is True:
                        return
            except Exception:
                pass
            time.sleep(1.0)
        raise TimeoutError("Timed out waiting for /health")

    def stop(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _download_test_image(target_path: Path):
    urls = [
        "https://thispersondoesnotexist.com/image",
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
    ]
    for url in urls:
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200 and response.content:
                target_path.write_bytes(response.content)
                return
        except Exception:
            pass
    raise RuntimeError("Failed to download sample image")


def _run_test(name: str, fn: Callable[[], None], kind: str = "success") -> TestResult:
    started = time.perf_counter()
    try:
        fn()
        return TestResult(
            name=name,
            passed=True,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            kind=kind,
        )
    except Exception as exc:
        return TestResult(
            name=name,
            passed=False,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            detail=str(exc),
            kind=kind,
        )


def _assert_keys(payload: dict, keys: list[str]):
    missing = [k for k in keys if k not in payload]
    if missing:
        raise AssertionError(f"Missing keys: {missing}")


def _assert_ok(response: requests.Response):
    if response.status_code != 200:
        try:
            body = response.json()
        except Exception:
            body = response.text[:500]
        raise AssertionError(f"HTTP {response.status_code}: {body}")


def _assert_error(response: requests.Response, expected_status: int, expected_code: str):
    if response.status_code != expected_status:
        try:
            body = response.json()
        except Exception:
            body = response.text[:500]
        raise AssertionError(f"Expected HTTP {expected_status}, got {response.status_code}: {body}")
    try:
        payload = response.json()
    except Exception:
        raise AssertionError("Error response is not JSON")
    code = payload.get("error", {}).get("code")
    if code != expected_code:
        raise AssertionError(f"Expected error code {expected_code}, got {code}: {payload}")


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * max(0.0, min(100.0, p)) / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * weight


def main():
    host = os.getenv("E2E_HOST", "127.0.0.1")
    port = int(os.getenv("E2E_PORT", "0")) or _pick_free_port()
    model_dir = os.getenv("E2E_MODEL_DIR")
    if not model_dir:
        local_models = ROOT_DIR / "models"
        if local_models.exists():
            model_dir = str(local_models)
    health_timeout = int(os.getenv("E2E_HEALTH_TIMEOUT", "300"))
    service_secret = os.getenv("AI_SERVICE_SECRET", "")

    temp_dir = Path(tempfile.mkdtemp(prefix="vison-e2e-"))
    image_path = temp_dir / "sample.jpg"
    noface_path = temp_dir / "noface.jpg"
    text_path = temp_dir / "not-image.txt"
    corrupt_jpeg_path = temp_dir / "corrupt.jpg"
    _download_test_image(image_path)
    blank = np.full((768, 1024, 3), 255, dtype=np.uint8)
    cv2.imwrite(str(noface_path), blank)
    text_path.write_text("this is not an image", encoding="utf-8")
    corrupt_jpeg_path.write_bytes(b"\xFF\xD8" + os.urandom(1024))
    oversized_bytes = b"x" * (5 * 1024 * 1024 + 1024)

    server = BackgroundServer(host=host, port=port, model_dir=model_dir)
    session = requests.Session()
    if service_secret:
        session.headers.update({"X-AI-Service-Key": service_secret})

    results: list[TestResult] = []
    embed_vector: Optional[list[float]] = None

    def post_multipart(path: str, files: dict, data: Optional[dict] = None) -> dict:
        response = session.post(f"{server.base_url}{path}", files=files, data=data or {}, timeout=120)
        _assert_ok(response)
        return response.json()

    def post_raw(path: str, files: dict, data: Optional[dict] = None) -> requests.Response:
        return session.post(f"{server.base_url}{path}", files=files, data=data or {}, timeout=120)

    def test_liveness():
        with image_path.open("rb") as fh:
            payload = post_multipart("/api/face/liveness", files={"image": ("sample.jpg", fh, "image/jpeg")})
        _assert_keys(payload, ["isLive", "liveScore", "face", "validation"])

    def test_deepfake():
        with image_path.open("rb") as fh:
            payload = post_multipart("/api/face/deepfake", files={"image": ("sample.jpg", fh, "image/jpeg")})
        _assert_keys(payload, ["isDeepfake", "attackRiskLevel", "attackTypes", "face", "validation"])

    def test_verify_live():
        with image_path.open("rb") as fh:
            payload = post_multipart("/api/face/verify-live", files={"image": ("sample.jpg", fh, "image/jpeg")})
        _assert_keys(
            payload,
            ["isLive", "liveScore", "isDeepfake", "attackRiskLevel", "attackTypes", "face", "validation"],
        )
        if "quality" not in payload.get("validation", {}):
            raise AssertionError("validation.quality missing")

    def test_analyze():
        with image_path.open("rb") as fh:
            payload = post_multipart("/api/face/analyze", files={"image": ("sample.jpg", fh, "image/jpeg")})
        _assert_keys(payload, ["gender", "genderConfidence", "ageRange", "face", "validation"])

    def test_embed():
        nonlocal embed_vector
        with image_path.open("rb") as fh:
            payload = post_multipart("/api/face/embed", files={"image": ("sample.jpg", fh, "image/jpeg")})
        _assert_keys(payload, ["embedding", "face", "validation"])
        if not isinstance(payload["embedding"], list) or len(payload["embedding"]) != 512:
            raise AssertionError("embedding must contain 512 numbers")
        embed_vector = payload["embedding"]

    def test_similarity():
        if embed_vector is None:
            raise AssertionError("embed vector is not available")
        with image_path.open("rb") as fh:
            payload = post_multipart(
                "/api/face/similarity",
                files={"image": ("sample.jpg", fh, "image/jpeg")},
                data={"embeddingStored": json.dumps(embed_vector)},
            )
        _assert_keys(payload, ["similarity", "queryEmbedding", "face", "validation"])
        if float(payload["similarity"]) <= 90.0:
            raise AssertionError(f"similarity expected > 90, got {payload['similarity']}")

    def test_compare():
        with image_path.open("rb") as src, image_path.open("rb") as tgt:
            payload = post_multipart(
                "/api/face/compare",
                files={
                    "sourceImage": ("source.jpg", src, "image/jpeg"),
                    "targetImage": ("target.jpg", tgt, "image/jpeg"),
                },
            )
        _assert_keys(payload, ["matched", "similarity", "threshold", "sourceFace", "targetFace", "validation"])

    def test_health():
        response = session.get(f"{server.base_url}/health", timeout=30)
        _assert_ok(response)
        payload = response.json()
        _assert_keys(payload, ["status", "models_loaded", "models", "memoryMb"])
        if payload.get("models_loaded") is not True:
            raise AssertionError("models_loaded must be true")

    def test_no_face_errors_all():
        if embed_vector is None:
            raise AssertionError("embed vector is not available for no-face similarity test")

        single_endpoints = [
            "/api/face/liveness",
            "/api/face/deepfake",
            "/api/face/analyze",
            "/api/face/verify-live",
            "/api/face/embed",
        ]
        for path in single_endpoints:
            with noface_path.open("rb") as fh:
                resp = post_raw(path, files={"image": ("noface.jpg", fh, "image/jpeg")})
            _assert_error(resp, 422, "FACE_NOT_DETECTED")

        with noface_path.open("rb") as fh:
            resp = post_raw(
                "/api/face/similarity",
                files={"image": ("noface.jpg", fh, "image/jpeg")},
                data={"embeddingStored": json.dumps(embed_vector)},
            )
        _assert_error(resp, 422, "FACE_NOT_DETECTED")

        with noface_path.open("rb") as src, noface_path.open("rb") as tgt:
            resp = post_raw(
                "/api/face/compare",
                files={
                    "sourceImage": ("source.jpg", src, "image/jpeg"),
                    "targetImage": ("target.jpg", tgt, "image/jpeg"),
                },
            )
        _assert_error(resp, 422, "FACE_NOT_DETECTED")

    def test_unsupported_format_error():
        with text_path.open("rb") as fh:
            resp = post_raw("/api/face/liveness", files={"image": ("not-image.txt", fh, "text/plain")})
        _assert_error(resp, 415, "UNSUPPORTED_FORMAT")

    def test_corrupt_image_error():
        with corrupt_jpeg_path.open("rb") as fh:
            resp = post_raw("/api/face/liveness", files={"image": ("corrupt.jpg", fh, "image/jpeg")})
        _assert_error(resp, 422, "IMAGE_CORRUPT")

    def test_oversized_image_error():
        resp = post_raw(
            "/api/face/liveness",
            files={"image": ("oversized.bin", oversized_bytes, "application/octet-stream")},
        )
        _assert_error(resp, 413, "IMAGE_TOO_LARGE")

    try:
        print(f"Starting server on {server.base_url}")
        server.start()
        server.wait_ready(timeout_sec=health_timeout)
        print("Server is healthy")

        tests = [
            ("POST /api/face/liveness", "success", test_liveness),
            ("POST /api/face/deepfake", "success", test_deepfake),
            ("POST /api/face/verify-live", "success", test_verify_live),
            ("POST /api/face/analyze", "success", test_analyze),
            ("POST /api/face/embed", "success", test_embed),
            ("POST /api/face/similarity", "success", test_similarity),
            ("POST /api/face/compare", "success", test_compare),
            ("GET /health", "success", test_health),
            ("ERROR no-face on all endpoints", "error", test_no_face_errors_all),
            ("ERROR unsupported format", "error", test_unsupported_format_error),
            ("ERROR corrupt image", "error", test_corrupt_image_error),
            ("ERROR oversized image", "error", test_oversized_image_error),
        ]

        for name, kind, fn in tests:
            result = _run_test(name, fn, kind=kind)
            results.append(result)
            status = "PASS" if result.passed else "FAIL"
            suffix = f" - {result.detail}" if result.detail else ""
            print(f"[{status}] {name} ({result.elapsed_ms:.1f} ms){suffix}")

        health = session.get(f"{server.base_url}/health", timeout=20).json()
        print("\nModel details:")
        print(json.dumps(health.get("models", {}), indent=2))
        print(f"Total RAM usage (process): {health.get('memoryMb', 0)} MB")

    finally:
        session.close()
        server.stop()

    passed = sum(1 for result in results if result.passed)
    failed = len(results) - passed
    success_times = [r.elapsed_ms for r in results if r.kind == "success"]
    error_times = [r.elapsed_ms for r in results if r.kind == "error"]

    latency_summary = {
        "success": {
            "count": len(success_times),
            "avg_ms": round(statistics.mean(success_times), 2) if success_times else 0.0,
            "p50_ms": round(_percentile(success_times, 50), 2),
            "p95_ms": round(_percentile(success_times, 95), 2),
            "p99_ms": round(_percentile(success_times, 99), 2),
        },
        "error": {
            "count": len(error_times),
            "avg_ms": round(statistics.mean(error_times), 2) if error_times else 0.0,
            "p50_ms": round(_percentile(error_times, 50), 2),
            "p95_ms": round(_percentile(error_times, 95), 2),
            "p99_ms": round(_percentile(error_times, 99), 2),
        },
    }
    summary = {"total": len(results), "passed": passed, "failed": failed, "server": server.base_url}

    print("\nSummary:")
    print(json.dumps(summary, indent=2))
    print("\nLatency (ms):")
    print(json.dumps(latency_summary, indent=2))
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
