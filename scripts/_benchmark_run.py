from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

from pathlib import Path

ROOT = Path(r"C:/Users/USER/Website Pribadi/Gerbang/Vison/vison-ai-service")


def pick_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]

port = pick_port()

env = os.environ.copy()
env.setdefault("PYTHONUNBUFFERED", "1")
env["AI_SERVICE_SECRET"] = "test-123"
env["MODEL_DIR"] = str(ROOT / "models")
env["BENCHMARK_URL"] = f"http://127.0.0.1:{port}"
env["OMP_NUM_THREADS"] = "1"
env["OPENBLAS_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"

log_file = ROOT / "benchmark_run_stdout.log"
err_file = ROOT / "benchmark_run_stderr.log"
stdout = open(log_file, "w", encoding="utf-8")
stderr = open(err_file, "w", encoding="utf-8")

proc = subprocess.Popen(
    [str(ROOT / "venv/Scripts/python.exe"), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
    cwd=str(ROOT),
    env=env,
    stdout=stdout,
    stderr=stderr,
)

start = time.time()
base_url = f"http://127.0.0.1:{port}"
import requests
while time.time() - start < 120:
    try:
        r = requests.get(f"{base_url}/health", timeout=2)
        if r.status_code == 200 and r.json().get("models_loaded") is True:
            break
    except Exception:
        pass
    time.sleep(0.3)
else:
    stdout.flush(); stderr.flush()
    raise SystemExit("Service failed to become ready")

# import benchmark module after setting env/url
sys.path.insert(0, str(ROOT))
import scripts.benchmark_real_vs_ai as bench

sys.argv = ["benchmark_real_vs_ai.py", "--ai-count", "80", "--max-tests", "40"]
bench.BASE_URL = base_url
bench.HEADERS = {"X-AI-Service-Key": env["AI_SERVICE_SECRET"]}
bench.DATA_DIR = ROOT / "benchmark_data"
bench.REPORT_DIR = ROOT / "benchmark_reports"

bench.main()

stdout.close(); stderr.close()
proc.terminate()
try:
    proc.wait(timeout=20)
except subprocess.TimeoutExpired:
    proc.kill()
