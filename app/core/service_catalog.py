"""External endpoint-to-model catalog for inference routing and reporting."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _default_catalog_path() -> str:
    configured = os.getenv("MODEL_REGISTRY_CONFIG", "").strip()
    if configured:
        return configured
    return str(Path(__file__).resolve().parents[2] / "configs" / "model_registry.json")


@dataclass(frozen=True)
class EndpointCapability:
    name: str
    route: str
    required_bundles: tuple[str, ...]
    optional_bundles: tuple[str, ...]
    tasks: tuple[str, ...]


class ServiceCatalog:
    """Read-only catalog describing endpoint-to-model usage."""

    _STATUS_MAP = {
        "face_detector": lambda status: bool(status.get("faceDetector")),
        "face_recognition_primary": lambda status: bool(status.get("faceRecognizer")),
        "face_recognition_extra": lambda status: bool(status.get("faceRecognizerExtra")),
        "adaface": lambda status: bool(status.get("adaFace")),
        "liveness_ensemble": lambda status: int(status.get("livenessModels", 0)) > 0,
        "cdcn_liveness": lambda status: bool(status.get("cdcnLiveness")),
        "deepfake_ensemble": lambda status: int(status.get("deepfakeModels", 0)) > 0,
        "ai_face_detector": lambda status: bool(status.get("aiFaceDetector")),
        "ai_face_detector_extra": lambda status: bool(status.get("aiFaceDetectorExtra")),
        "npr": lambda status: bool(status.get("nprDetector")),
        "clip_fake": lambda status: bool(status.get("clipFakeDetector")),
        "deepfake_vit_v2": lambda status: bool(status.get("deepfakeVitV2")),
        "face_parser": lambda status: bool(status.get("faceParser")),
        "age_gender": lambda status: bool(status.get("ageGender")),
        "age_gender_vit": lambda status: bool(status.get("ageGender")),
        "fairface": lambda status: bool(status.get("fairFace")),
        "mivolo": lambda status: bool(status.get("miVOLO")),
        "face_quality_proxy": lambda status: True,
        "face_attributes_proxy": lambda status: bool(status.get("faceParser")),
    }

    def __init__(self, payload: dict[str, Any], config_path: str):
        self.config_path = config_path
        self.version = str(payload.get("version", "unknown"))
        self.description = str(payload.get("description", ""))
        self.models = dict(payload.get("models", {}))
        self.bundles = {
            name: tuple(items)
            for name, items in dict(payload.get("bundles", {})).items()
        }
        self.endpoints = {
            name: EndpointCapability(
                name=name,
                route=str(spec.get("route", "")),
                required_bundles=tuple(spec.get("required_bundles", [])),
                optional_bundles=tuple(spec.get("optional_bundles", [])),
                tasks=tuple(spec.get("tasks", [])),
            )
            for name, spec in dict(payload.get("endpoints", {})).items()
        }

    def endpoint(self, name: str) -> EndpointCapability:
        return self.endpoints[name]

    def endpoint_names(self) -> list[str]:
        return sorted(self.endpoints.keys())

    def bundle_models(self, bundle_name: str) -> tuple[str, ...]:
        return self.bundles.get(bundle_name, ())

    def endpoint_model_keys(self, endpoint_name: str, include_optional: bool = True) -> list[str]:
        endpoint = self.endpoint(endpoint_name)
        keys: list[str] = []
        bundles = list(endpoint.required_bundles)
        if include_optional:
            bundles.extend(endpoint.optional_bundles)
        for bundle in bundles:
            keys.extend(self.bundle_models(bundle))
        return list(dict.fromkeys(keys))

    def describe(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "description": self.description,
            "configPath": self.config_path,
            "bundles": self.bundles,
            "endpoints": {
                name: {
                    "route": item.route,
                    "requiredBundles": list(item.required_bundles),
                    "optionalBundles": list(item.optional_bundles),
                    "tasks": list(item.tasks),
                    "modelKeys": self.endpoint_model_keys(name),
                }
                for name, item in self.endpoints.items()
            },
        }

    def model_status(self, model_registry) -> dict[str, dict[str, Any]]:
        if model_registry is None:
            return {}
        status = model_registry.get_status()
        payload: dict[str, dict[str, Any]] = {}
        for key, spec in self.models.items():
            checker = self._STATUS_MAP.get(key, lambda _status: False)
            payload[key] = {
                "loaded": bool(checker(status)),
                "setting": spec.get("setting"),
                "loader": spec.get("loader"),
                "role": spec.get("role"),
            }
        return payload

    def capabilities_payload(self, model_registry=None) -> dict[str, Any]:
        payload = self.describe()
        payload["models"] = self.model_status(model_registry)
        return payload


def load_service_catalog(config_path: Optional[str] = None) -> ServiceCatalog:
    path = Path(config_path or _default_catalog_path()).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ServiceCatalog(payload=payload, config_path=str(path))
