"""
@file panchayats.py
@description Gram Panchayat registry directory endpoints.
@module services/api/routes
"""

from __future__ import annotations

from fastapi import APIRouter

from services.api.panchayat_registry import list_panchayats
from services.api.schemas import APIResponse, GeoLocation, PanchayatSummary

router = APIRouter(prefix="/api/v1/panchayats", tags=["Panchayats"])


@router.get(
    "",
    response_model=APIResponse,
    summary="List Registered Pilot Panchayats",
    description="Retrieve directory of all registered Gram Panchayats with geographic coordinates and crop metadata.",
)
def get_registered_panchayats() -> APIResponse:
    """List all registered pilot panchayats from canonical registry."""
    raw_list = list_panchayats()
    summaries = [
        PanchayatSummary(
            panchayat_id=p.panchayat_id,
            name=p.name,
            name_hi=p.name_hi,
            district=p.district,
            block=p.block,
            coordinates=GeoLocation(
                latitude=p.latitude,
                longitude=p.longitude,
                elevation_m=p.elevation_m,
                district=p.district,
                panchayat=p.name,
            ),
            crop=p.crop,
            crop_hi=p.crop_hi,
            crop_stage=p.crop_stage,
        )
        for p in raw_list
    ]

    return APIResponse(
        success=True,
        message="Registered panchayats retrieved successfully",
        data=[s.model_dump() for s in summaries],
        errors=[],
    )
