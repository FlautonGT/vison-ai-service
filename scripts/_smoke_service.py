from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(r"C:/Users/USER/Website Pribadi/Gerbang/Vison/vison-ai-service")


def pick_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def bench():
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["MODEL_DIR"] = str(ROOT / "models")
    env["AI_SERVICE_SECRET"] = "test-123"
    env["FACE_AI_DEBUG"] = "false"

    port = pick_port()
    stdout_file = ROOT / "server_smoke_stdout.log"
    stderr_file = ROOT / "server_smoke_stderr.log"
    with open(stdout_file, "w", encoding="utf-8") as out, open(stderr_file, "w", encoding="utf-8") as err:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
            cwd=str(ROOT),
            env=env,
            stdout=out,
            stderr=err,
        )

        base = f"http://127.0.0.1:{port}"
        session = requests.Session()
        session.headers.update({"X-AI-Service-Key": "test-123"})

        def wait_ready(timeout=120):
            deadline = time.time() + timeout
            last = ""
            while time.time() < deadline:
                if proc.poll() is not None:
                    out.seek(0)
                    err.seek(0)
                    print("Server exited early", proc.returncode)
                    print("STDOUT:\n", out.read()[-4000:])
                    print("STDERR:\n", err.read()[-4000:])
                    raise RuntimeError("Server exited early")
                try:
                    r = session.get(f"{base}/health", timeout=2)
                    if r.status_code == 200:
                        payload = r.json()
                        if payload.get("status") == "ok" and payload.get("models_loaded") is True:
                            return
                except Exception as exc:
                    last = str(exc)
                time.sleep(0.3)
            out.seek(0)
            err.seek(0)
            raise TimeoutError(f"timeout /health. last={last} \nOUT={out.read()[-2000:]}\nERR={err.read()[-2000:]}")

        try:
            wait_ready()
            real = next((ROOT / "benchmark_data/lfw").rglob("*.jpg"))
            ai = next((ROOT / "benchmark_data/ai_faces").glob("*.jpg"))

            results = []

            def post(path: str, files: dict, data=None):
                t0 = time.perf_counter()
                r = session.post(f"{base}{path}", files=files, data=data or {}, timeout=120)
                dt = (time.perf_counter() - t0) * 1000
                return r, dt

            for name in ("liveness", "deepfake", "verify-live", "analyze"):
                with open(real, "rb") as fh:
                    r, dt = post(f"/api/face/{name}", {"image": ("img.jpg", fh, "image/jpeg")})
                results.append((name, r.status_code, dt, r.text[:120]))

            with open(real, "rb") as src, open(real, "rb") as tgt:
                r, dt = post("/api/face/compare", {
                    "sourceImage": ("a.jpg", src, "image/jpeg"),
                    "targetImage": ("b.jpg", tgt, "image/jpeg"),
                })
            results.append(("compare", r.status_code, dt, r.text[:120]))

            with open(real, "rb") as fh:
                r, dt = post("/api/face/embed", {"image": ("img.jpg", fh, "image/jpeg")})
            payload = json.loads(r.text) if r.status_code == 200 else {}
            emb = payload.get("embedding", [])
            results.append(("embed", r.status_code, dt, r.text[:120]))

            if emb:
                with open(real, "rb") as fh:
                    r, dt = post("/api/face/similarity", {"image": ("img.jpg", fh, "image/jpeg")}, data={"embeddingStored": json.dumps(emb)})
                results.append(("similarity", r.status_code, dt, r.text[:120]))

            with open(ai, "rb") as fh:
                r, dt = post("/api/face/deepfake", {"image": ("ai.jpg", fh, "image/jpeg")})
            results.append(("deepfake_ai", r.status_code, dt, r.text[:120]))

            # error paths
            t0 = time.perf_counter()
            r = session.post(f"{base}/api/face/liveness", files={"image": ("bad.txt", b"not an image", "text/plain")}, timeout=20)
            dt = (time.perf_counter() - t0) * 1000
            results.append(("err_bad_format", r.status_code, dt, r.text[:120]))

            import numpy as np
            import cv2
            blank = (np.zeros((300, 300, 3), dtype=np.uint8) + 255)
            _, buf = cv2.imencode(".jpg", blank)
            t0 = time.perf_counter()
            r = session.post(f"{base}/api/face/analyze", files={"image": ("blank.jpg", buf.tobytes(), "image/jpeg")}, timeout=20)
            dt2 = (time.perf_counter() - t0) * 1000
            results.append(("err_no_face", r.status_code, dt2, r.text[:120]))

            small = (np.zeros((20, 20, 3), dtype=np.uint8))
            _, buf2 = cv2.imencode(".jpg", small)
            t0 = time.perf_counter()
            r = session.post(f"{base}/api/face/analyze", files={"image": ("small.jpg", buf2.tobytes(), "image/jpeg")}, timeout=20)
            dt3 = (time.perf_counter() - t0) * 1000
            results.append(("err_small_img", r.status_code, dt3, r.text[:120]))

            for name, status, dt, snippet in results:
                print(f"{name}: {status} in {dt:.1f}ms | {snippet}")
            oks = [r for r in results if r[1] == 200]
            print(f"PASS_COUNT={len(oks)}/{len(results)}")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            out.close(); err.close()

if __name__ == "__main__":
    bench()
