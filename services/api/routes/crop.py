"""
@file crop.py
@description FastAPI route handlers for deep CNN crop disease diagnosis and weather fusion.
@module services/api/routes
"""

import base64
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from loguru import logger

from services.api.panchayat_registry import get_panchayat
from services.api.routes.forecast import get_cached_forecast
from services.api.schemas import APIResponse, ErrorDetail
from services.crop_doctor.classifier import get_crop_vision_engine
from services.crop_doctor.agromet_fusion import compute_agromet_crop_advisory
from services.crop_doctor.disease_registry import PLANT_VILLAGE_CLASSES, DISEASE_PROFILES

router = APIRouter(prefix="/crop", tags=["Crop Doctor (AI Vision & Agromet Fusion)"])


class CropDiagnoseJSONRequest(BaseModel):
    image_base64: str
    panchayat_id: Optional[str] = "KADERA_001"
    crop_hint: Optional[str] = None


@router.post(
    "/diagnose",
    response_model=APIResponse,
    summary="Diagnose Crop Leaf Photo with CNN + 1-km Weather Fusion",
    description=(
        "Accepts leaf photo (multipart file or base64 data URI). Runs deep MobileNetV3 CNN inference "
        "and correlates findings with live 1-km VATA microclimate telemetry to compute spore explosion risk "
        "and optimal fungicide spray window."
    ),
)
async def diagnose_crop_leaf(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    panchayat_id: Optional[str] = Form("KADERA_001"),
    crop_hint: Optional[str] = Form(None),
) -> APIResponse:
    """Run leaf disease diagnosis with Deep CNN + 1km Weather Fusion."""
    try:
        raw_image: bytes | str
        if file is not None:
            raw_image = await file.read()
        elif image_base64 is not None and len(image_base64.strip()) > 0:
            raw_image = image_base64
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": "MISSING_IMAGE", "message": "Either 'file' or 'image_base64' must be provided."},
            )

        # 1. Execute Deep CNN Visual Inference
        engine = get_crop_vision_engine()
        cnn_result = engine.predict(raw_image, crop_hint=crop_hint)

        # 2. Extract 1-km Microclimate Telemetry Context for Panchayat
        weather_ctx: Dict[str, Any] = {
            "temperature_2m_c": 31.2,
            "relative_humidity_2m_pct": 78.0,
            "wind_speed_10m_kmh": 12.0,
            "precipitation_pop": 15.0,
            "soil_moisture_m3_m3": 0.22,
        }

        panch = get_panchayat(panchayat_id or "KADERA_001")
        if panch is not None:
            cached_fc = get_cached_forecast(panch.panchayat_id)
            if cached_fc is not None:
                weather_ctx["temperature_2m_c"] = cached_fc.downscaled_weather.downscaled_value
                weather_ctx["relative_humidity_2m_pct"] = cached_fc.baseline_weather.relative_humidity_2m_pct or 78.0
                weather_ctx["wind_speed_10m_kmh"] = cached_fc.baseline_weather.wind_speed_10m_kmh or 12.0
                weather_ctx["precipitation_pop"] = 15.0 if (cached_fc.baseline_weather.precipitation_mm or 0) == 0 else 65.0
                if cached_fc.bhoonidhi and "soil_moisture" in cached_fc.bhoonidhi:
                    weather_ctx["soil_moisture_m3_m3"] = cached_fc.bhoonidhi["soil_moisture"].get("soil_moisture_m3_m3", 0.22)

        # 3. Fuse Visual Pathology with 1-km Microclimate
        fused_report = compute_agromet_crop_advisory(cnn_result, weather_ctx)
        fused_report["panchayat_id"] = panch.panchayat_id if panch else "KADERA_001"
        fused_report["panchayat_name"] = panch.name if panch else "Kadera"

        return APIResponse(
            success=True,
            message="Crop leaf diagnosis and agromet spray fusion completed successfully",
            data=fused_report,
            errors=[],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[crop_doctor] Diagnosis API failure: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DIAGNOSIS_FAILURE", "message": f"CNN Inference failed: {str(exc)}"},
        )


@router.post(
    "/diagnose-json",
    response_model=APIResponse,
    summary="Diagnose Crop Leaf Photo via JSON Body (Base64)",
)
async def diagnose_crop_leaf_json(request: CropDiagnoseJSONRequest) -> APIResponse:
    """JSON API wrapper for base64 leaf diagnosis."""
    return await diagnose_crop_leaf(
        file=None,
        image_base64=request.image_base64,
        panchayat_id=request.panchayat_id,
        crop_hint=request.crop_hint,
    )


@router.get(
    "/catalog",
    response_model=APIResponse,
    summary="Get 38-Class Plant Pathology Catalog",
)
def get_pathology_catalog() -> APIResponse:
    """List all supported crops, diseases, and chemical profiles."""
    catalog = []
    for cls_name in PLANT_VILLAGE_CLASSES:
        parts = cls_name.split("___")
        crop = parts[0].replace("_", " ")
        dis = parts[1].replace("_", " ") if len(parts) > 1 else "Healthy"
        catalog.append({
            "class_name": cls_name,
            "crop": crop,
            "disease": dis,
            "has_detailed_profile": cls_name in DISEASE_PROFILES,
        })

    return APIResponse(
        success=True,
        message="Plant pathology catalog retrieved successfully",
        data={"total_classes": len(catalog), "classes": catalog},
        errors=[],
    )
