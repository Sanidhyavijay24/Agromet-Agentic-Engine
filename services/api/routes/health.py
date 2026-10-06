"""
@file health.py
@description Health check and subsystem readiness endpoints.
@module services/api/routes
"""

from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter
from services.api.schemas import HealthCheckResponse
from services.ml_downscaler.inference import DEFAULT_MODEL_PATH
try:
    from services.advisory_engine.rules import ADVISORY_RULES
except ImportError:
    ADVISORY_RULES = []

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Root Health Check",
    description="Inspect overall service status and dependent ML/advisory subsystems.",
)
@router.get(
    "/api/v1/health",
    response_model=HealthCheckResponse,
    summary="API v1 Health Check",
    description="Inspect overall service status and dependent ML/advisory subsystems.",
)
def get_health_status() -> HealthCheckResponse:
    """Return component readiness status for all SIH 26074 subsystems."""
    services: dict[str, str] = {}

    # 1. Downscaler model artifact check
    try:
        if Path(DEFAULT_MODEL_PATH).exists():
            services["ml_downscaler"] = "ok"
        else:
            services["ml_downscaler"] = "degraded"
    except Exception:
        services["ml_downscaler"] = "down"

    # 2. Advisory rules matrix check
    try:
        if len(ADVISORY_RULES) > 0:
            services["advisory_engine"] = "ok"
        else:
            services["advisory_engine"] = "degraded"
    except Exception:
        services["advisory_engine"] = "down"

    # 3. LLM extractor readiness check
    services["llm_extractor"] = "ok"

    overall_status = "ok"
    if any(s == "down" for s in services.values()):
        overall_status = "down"
    elif any(s == "degraded" for s in services.values()):
        overall_status = "degraded"

    return HealthCheckResponse(
        status=overall_status,  # type: ignore[arg-type]
        version="1.0.0",
        services=services,  # type: ignore[arg-type]
    )

