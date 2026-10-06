"""
@file schemas.py
@description Strict Pydantic v2 schemas for Agentic AI Agromet Advisory.
@module services/agent
"""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field


class AgrometAdvisoryRequest(BaseModel):
    """Input payload requesting an agentic agromet advisory consultation."""
    latitude: float = Field(..., ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(..., ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    crop: str = Field(default="Bajra", description="Target crop name (e.g. Bajra, Guar, Groundnut, Wheat, Mustard, Cotton)")
    crop_stage: str = Field(default="Flowering", description="Current phenological stage (e.g. Sowing, Vegetative, Flowering, Grain Filling, Maturity)")
    soil_type: str = Field(default="Sandy Loam", description="Soil texture type")
    user_query: Optional[str] = Field(default=None, description="Optional specific farmer question or concern")
    forecast_hours: int = Field(default=48, ge=6, le=168, description="Forecast evaluation horizon in hours")


class SprayWindow(BaseModel):
    """Evaluated time window for chemical or bio-pesticide spraying."""
    start_time: str = Field(..., description="Start timestamp of window")
    end_time: str = Field(..., description="End timestamp of window")
    duration_hours: int = Field(..., description="Continuous duration in hours")
    suitability_score: float = Field(..., ge=0.0, le=100.0, description="Spray suitability index (0-100)")
    limiting_factor: str = Field(default="None", description="Primary constraint (Wind, Rain, Temp, etc.)")
    recommended_action: str = Field(..., description="Specific spray instruction")


class MicroclimateIndices(BaseModel):
    """Synthesized physics-based microclimate stress indicators."""
    mean_vpd_kpa: float = Field(..., description="Mean Vapor Pressure Deficit in kPa")
    max_vpd_kpa: float = Field(..., description="Peak Vapor Pressure Deficit in kPa")
    thermal_inversion_risk: bool = Field(default=False, description="Nocturnal thermal inversion detected")
    inversion_strength_c: float = Field(default=0.0, description="Inversion temperature differential in Celsius")
    frost_risk: bool = Field(default=False, description="Sub-zero or near-frost threshold breached")
    gdd_accumulated_c_days: float = Field(..., description="Growing Degree Days accumulated over horizon")
    heat_stress_hours: int = Field(default=0, description="Hours exceeding crop critical maximum temperature")
    spray_windows: List[SprayWindow] = Field(default_factory=list, description="Computed favorable spray windows")


class AgrometActionPlan(BaseModel):
    """Final structured actionable farming plan produced by the agent."""
    summary_headline: str = Field(..., description="One-line punchy agromet verdict")
    verdict_category: str = Field(..., description="Category: FAVORABLE | SPRAY_CAUTION | HEAT_STRESS_DEFENSE | FROST_WARNING | IRRIGATE_IMMEDIATELY")
    irrigation_advice: str = Field(..., description="Soil moisture & transpiration-based irrigation directive")
    spray_recommendation: str = Field(..., description="Optimal spray timing, chemical drift warnings, and dilution")
    crop_specific_protection: str = Field(..., description="Crop and stage specific agronomic interventions")
    vernacular_hindi: str = Field(..., description="Detailed, localized Hindi guidance for farmers")
    english_summary: str = Field(..., description="Executive English advisory summary")


class ToolExecution(BaseModel):
    """Log entry capturing an individual tool execution by the Agent."""
    tool_name: str = Field(..., description="Name of the tool executed")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Arguments passed to tool")
    result_summary: str = Field(..., description="Brief summary of tool output")
    execution_time_ms: float = Field(..., description="Tool execution duration in milliseconds")


class AgentAdvisoryResponse(BaseModel):
    """Complete response payload from the Agentic AI consultation."""
    status: str = Field(default="success", description="Response status (success | fallback | error)")
    location: dict[str, Any] = Field(..., description="Target coordinate & zone metadata")
    crop: str = Field(..., description="Evaluated crop")
    crop_stage: str = Field(..., description="Evaluated stage")
    indices: MicroclimateIndices = Field(..., description="Computed microclimate indices")
    action_plan: AgrometActionPlan = Field(..., description="Synthesized action plan")
    agent_trace: List[ToolExecution] = Field(default_factory=list, description="Step-by-step tool execution audit trace")
    llm_model_used: str = Field(..., description="LLM model identifier")
    total_latency_ms: float = Field(..., description="Total end-to-end agent consultation latency")
