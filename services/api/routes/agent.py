"""
@file agent.py
@description FastAPI endpoints for Agentic AI Agromet Advisory consultations and interactive chat.
@module services/api/routes
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from services.agent.schemas import (
    AgrometAdvisoryRequest,
    AgentAdvisoryResponse,
    ToolExecution
)
from services.agent.core import run_agromet_agent
from services.agent.tools.agronomy_tool import CROP_AGRONOMY_REGISTRY

router = APIRouter(prefix="/agent", tags=["Agentic Agromet Advisory"])


class AgentChatRequest(BaseModel):
    """Conversational agronomist chat query."""
    message: str = Field(..., description="Farmer or agronomist chat message")
    latitude: float = Field(default=26.9124, ge=-90.0, le=90.0)
    longitude: float = Field(default=75.7873, ge=-180.0, le=180.0)
    crop: str = Field(default="Bajra", description="Current crop")
    crop_stage: str = Field(default="Flowering", description="Current crop stage")
    soil_type: str = Field(default="Sandy Loam")


class AgentChatResponse(BaseModel):
    """Conversational response from the Agent."""
    reply: str = Field(..., description="Conversational reply in English / Hindi")
    vernacular_hindi: str = Field(..., description="Localized Hindi advice")
    consultation: AgentAdvisoryResponse = Field(..., description="Underlying structured advisory")


@router.post("/advisory", response_model=AgentAdvisoryResponse, status_code=status.HTTP_200_OK)
async def generate_agent_advisory(request: AgrometAdvisoryRequest) -> AgentAdvisoryResponse:
    """
    Execute autonomous multi-step agentic consultation.
    Orchestrates 1km downscaling, stress indices, agronomy lookup, and Gemini synthesis.
    """
    try:
        response = run_agromet_agent(request)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agentic advisory consultation failed: {str(e)}"
        )


@router.post("/chat", response_model=AgentChatResponse, status_code=status.HTTP_200_OK)
async def chat_with_agent(request: AgentChatRequest) -> AgentChatResponse:
    """
    Interactive conversational consultation with the Agromet Agent.
    """
    try:
        advisory_req = AgrometAdvisoryRequest(
            latitude=request.latitude,
            longitude=request.longitude,
            crop=request.crop,
            crop_stage=request.crop_stage,
            soil_type=request.soil_type,
            user_query=request.message,
            forecast_hours=48
        )
        consultation = run_agromet_agent(advisory_req)
        
        reply = (
            f"Here is your agromet verdict for {consultation.crop} at {consultation.crop_stage} stage: "
            f"{consultation.action_plan.summary_headline}\n\n"
            f"• Irrigation: {consultation.action_plan.irrigation_advice}\n"
            f"• Spray Timing: {consultation.action_plan.spray_recommendation}\n"
            f"• Crop Protection: {consultation.action_plan.crop_specific_protection}"
        )
        
        return AgentChatResponse(
            reply=reply,
            vernacular_hindi=consultation.action_plan.vernacular_hindi,
            consultation=consultation
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agentic chat failed: {str(e)}"
        )


@router.get("/tools", status_code=status.HTTP_200_OK)
async def get_agent_tools() -> Dict[str, Any]:
    """
    Return metadata catalog of all specialized agent tools and registered crop profiles.
    """
    return {
        "agent_name": "Agromet Agentic Engine (A²E)",
        "framework": "Google Gemini Flash ReAct Tool-Calling Core",
        "tools": [
            {
                "name": "run_1km_downscaler",
                "description": "Physics-guided residual XGBoost downscaling (28 physical features, SRTM DEM relief, boundary layer sensible heat flux).",
                "inputs": ["latitude", "longitude", "forecast_hours", "district"]
            },
            {
                "name": "calculate_microclimate_indices",
                "description": "Computes psychrometric Vapor Pressure Deficit (VPD), Growing Degree Days (GDD), Nocturnal Inversion risk, and Spray Drift Windows.",
                "inputs": ["hourly_series", "crop_base_temp_c", "crop_max_temp_c"]
            },
            {
                "name": "lookup_crop_agronomy",
                "description": "ICAR/FAO agronomic threshold registry with stage-specific moisture sensitivity and pest trigger etiologies.",
                "inputs": ["crop_name", "stage"]
            },
            {
                "name": "get_soil_and_terrain_context",
                "description": "Extracts 30m SRTM DEM continuous ground elevation, Horn's slope, aspect azimuth, and 0-7cm surface soil moisture status.",
                "inputs": ["latitude", "longitude"]
            }
        ],
        "registered_crops": list(CROP_AGRONOMY_REGISTRY.keys())
    }
