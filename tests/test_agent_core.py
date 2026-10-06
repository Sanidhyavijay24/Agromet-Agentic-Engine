"""
@file test_agent_core.py
@description Integration tests for Gemini ReAct Agent Core and structured advisory generation.
@module tests
"""

import pytest
from services.agent.schemas import AgrometAdvisoryRequest, AgentAdvisoryResponse
from services.agent.core import run_agromet_agent


def test_agent_full_consultation_cycle():
    """Verify complete agent consultation execution with multi-step tool execution and structured plan."""
    request = AgrometAdvisoryRequest(
        latitude=26.9124,
        longitude=75.7873,
        crop="Bajra",
        crop_stage="Flowering",
        soil_type="Sandy Loam",
        user_query="Can I spray bio-pesticide for ergot disease tomorrow morning?",
        forecast_hours=24
    )
    
    response = run_agromet_agent(request)
    
    assert isinstance(response, AgentAdvisoryResponse)
    assert response.status == "success"
    assert response.location["zone_id"] in ["ACZ_14", "ZONE_14_WESTERN_DRY"]
    assert response.crop == "Pearl Millet (बाजरा)"
    assert response.crop_stage == "Flowering"
    
    # Verify 4 tool traces
    tool_names = [t.tool_name for t in response.agent_trace]
    assert "run_1km_downscaler" in tool_names
    assert "get_soil_and_terrain_context" in tool_names
    assert "lookup_crop_agronomy" in tool_names
    assert "calculate_microclimate_indices" in tool_names
    
    # Verify Action Plan
    plan = response.action_plan
    assert plan.verdict_category in [
        "FAVORABLE", "SPRAY_CAUTION", "HEAT_STRESS_DEFENSE", "FROST_WARNING", "IRRIGATE_IMMEDIATELY"
    ]
    assert len(plan.summary_headline) > 0
    assert len(plan.irrigation_advice) > 0
    assert len(plan.spray_recommendation) > 0
    assert len(plan.vernacular_hindi) > 0
    assert len(plan.english_summary) > 0
    assert response.total_latency_ms > 0

