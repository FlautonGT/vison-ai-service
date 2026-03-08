"""API middleware: auth, rate limiting, and structured request logging."""

from __future__ import annotations

import hmac
import json
import logging
import threading
import time
import uuid
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

logger = logging.getLogger(__name__)


def _error_payload(code: str, message: str, detail: str) -> dict:
    return {
        "success": False,
        "message": message,
        "error": {"code": code, "detail": detail},
    }


_RATE_LIMITER_MAX_IPS = 10000
_RATE_LIMITER_CLEANUP_INTERVAL = 60


class InMemoryRateLimiter:
    """Fixed-window per-IP limiter: max N requests per 1-second bucket."""

    def __init__(self, max_rps: int):
        self.max_rps = max(1, int(max_rps))
        self._lock = threading.Lock()
        self._buckets: dict[str, dict[int, int]] = defaultdict(dict)
        self._last_full_cleanup = int(time.time())

    def allow(self, ip: str) -> bool:
        now_sec = int(time.time())
        with self._lock:
            # Periodic full cleanup to prevent memory leak from rotating IPs.
            if now_sec - self._last_full_cleanup > _RATE_LIMITER_CLEANUP_INTERVAL:
                stale_ips = [
                    k for k, v in self._buckets.items()
                    if not v or max(v.keys()) < now_sec - 5
                ]
                for k in stale_ips:
                    del self._buckets[k]
                self._last_full_cleanup = now_sec
            # Hard cap on tracked IPs to prevent memory exhaustion under DDoS.
            if ip not in self._buckets and len(self._buckets) >= _RATE_LIMITER_MAX_IPS:
                return False

            bucket = self._buckets[ip]
            count = bucket.get(now_sec, 0)
            if count >= self.max_rps:
                return False
            bucket[now_sec] = count + 1
            # cleanup old seconds for this IP
            old_keys = [sec for sec in bucket if sec < now_sec - 2]
            for sec in old_keys:
                bucket.pop(sec, None)
            return True


class AIServiceAuth(BaseHTTPMiddleware):
    """Validate shared-secret header and optional source IP allowlist."""

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        if settings.AI_SERVICE_SECRET:
            key = request.headers.get("X-AI-Service-Key", "")
            if not hmac.compare_digest(key, settings.AI_SERVICE_SECRET):
                return JSONResponse(
                    status_code=401,
                    content=_error_payload("UNAUTHORIZED", "Unauthorized", "Invalid service key"),
                )

        if settings.ALLOWED_IPS:
            allowed = [ip.strip() for ip in settings.ALLOWED_IPS.split(",") if ip.strip()]
            client_ip = request.client.host if request.client else ""
            if allowed and client_ip not in allowed:
                return JSONResponse(
                    status_code=403,
                    content=_error_payload("FORBIDDEN", "Forbidden", "IP not allowed"),
                )

        return await call_next(request)


class RequestHardeningMiddleware(BaseHTTPMiddleware):
    """Request ID, per-IP rate limiting, metrics accounting, and structured logs."""

    def __init__(self, app):
        super().__init__(app)
        self._rate_limiter = InMemoryRateLimiter(settings.RATE_LIMIT_RPS)

    @staticmethod
    def _ensure_metrics_state(request: Request):
        state = request.app.state
        if not hasattr(state, "metrics_lock"):
            state.metrics_lock = threading.Lock()
        if not hasattr(state, "total_requests"):
            state.total_requests = 0
        if not hasattr(state, "total_latency_ms"):
            state.total_latency_ms = 0.0
        if not hasattr(state, "started_at"):
            state.started_at = time.time()

    @staticmethod
    def _record_metrics(request: Request, latency_ms: float):
        state = request.app.state
        with state.metrics_lock:
            state.total_requests += 1
            state.total_latency_ms += float(latency_ms)

    async def dispatch(self, request: Request, call_next):
        self._ensure_metrics_state(request)

        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        request.state.model_timings = {}
        request.state.result_summary = {}

        client_ip = request.client.host if request.client else "unknown"
        started = time.perf_counter()

        if not self._rate_limiter.allow(client_ip):
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record_metrics(request, latency_ms)
            payload = _error_payload(
                "RATE_LIMITED",
                "Too Many Requests",
                f"Rate limit exceeded ({settings.RATE_LIMIT_RPS} req/s per IP)",
            )
            response = JSONResponse(status_code=429, content=payload)
            response.headers["X-Request-Id"] = request_id
            logger.warning(
                json.dumps(
                    {
                        "request_id": request_id,
                        "endpoint": request.url.path,
                        "method": request.method,
                        "status_code": 429,
                        "latency_ms": round(latency_ms, 2),
                        "client_ip": client_ip,
                        "error": {"code": "RATE_LIMITED"},
                    }
                )
            )
            return response

        try:
            response = await call_next(request)
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record_metrics(request, latency_ms)
            logger.exception(
                json.dumps(
                    {
                        "request_id": request_id,
                        "endpoint": request.url.path,
                        "method": request.method,
                        "status_code": 500,
                        "latency_ms": round(latency_ms, 2),
                        "client_ip": client_ip,
                        "model_timings_ms": getattr(request.state, "model_timings", {}),
                        "result_summary": getattr(request.state, "result_summary", {}),
                        "error": {"code": "MODEL_ERROR", "detail": str(exc)},
                    }
                )
            )
            raise

        latency_ms = (time.perf_counter() - started) * 1000.0
        self._record_metrics(request, latency_ms)
        response.headers["X-Request-Id"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"

        logger.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "endpoint": request.url.path,
                    "method": request.method,
                    "status_code": response.status_code,
                    "latency_ms": round(latency_ms, 2),
                    "client_ip": client_ip,
                    "model_timings_ms": getattr(request.state, "model_timings", {}),
                    "result_summary": getattr(request.state, "result_summary", {}),
                }
            )
        )
        return response
