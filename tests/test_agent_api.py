"""
@file test_agent_api.py
@description End-to-end integration tests for FastAPI agent and forecast endpoints.
@module tests
"""

import pytest
from fastapi.testclient import TestClient
from services.api.main import app

client = TestClient(app)


def test_root_ping():
    """Verify root status endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "Agromet Agentic Engine (A²E)"
    assert data["status"] == "operational"


def test_zone_catalog_endpoint():
    """Verify 15 ACZ fleet catalog endpoint."""
    response = client.get("/api/v1/forecast/zones/catalog")
    assert response.status_code == 200
    data = response.json()
    assert data["total_zones"] == 15
    assert data["active_live_zone"] == "ACZ_14"
    assert data["macro_error_reduction_pct"] == 58.9
    assert len(data["zones"]) == 15
    
    # Check Zone XIV
    z14 = next(z for z in data["zones"] if z["zone_id"] == "ACZ_14")
    assert z14["is_live_deployment_zone"] is True
    assert z14["error_reduction_pct"] == 60.6


def test_zone_routing_endpoint():
    """Verify coordinate routing to zone."""
    response = client.get("/api/v1/forecast/zones/route?lat=26.9124&lon=75.7873")
    assert response.status_code == 200
    data = response.json()
    assert data["zone_id"] == "ACZ_14"
    assert data["is_live_deployment_zone"] is True


def test_point_downscaling_endpoint():
    """Verify 1km downscaled forecast point endpoint."""
    response = client.get("/api/v1/forecast/point?latitude=26.9124&longitude=75.7873&hours=24")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["hours_count"] == 24
    assert "hourly_forecast" in data
    assert "indices" in data


def test_agent_tools_catalog_endpoint():
    """Verify agent tools metadata endpoint."""
    response = client.get("/api/v1/agent/tools")
    assert response.status_code == 200
    data = response.json()
    assert len(data["tools"]) == 4
    assert "BAJRA" in data["registered_crops"]


def test_agent_advisory_endpoint():
    """Verify full agent advisory endpoint."""
    payload = {
        "latitude": 26.9124,
        "longitude": 75.7873,
        "crop": "Bajra",
        "crop_stage": "Flowering",
        "soil_type": "Sandy Loam",
        "user_query": "Is tomorrow morning suitable for foliar nutrient spray?",
        "forecast_hours": 24
    }
    response = client.post("/api/v1/agent/advisory", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "action_plan" in data
    assert len(data["action_plan"]["vernacular_hindi"]) > 0
    assert len(data["agent_trace"]) == 4


def test_agent_chat_endpoint():
    """Verify interactive conversational agent chat endpoint."""
    payload = {
        "message": "When should I irrigate my wheat crop given the current heat?",
        "latitude": 26.9124,
        "longitude": 75.7873,
        "crop": "Wheat",
        "crop_stage": "Crown Root Initiation (CRI)",
        "soil_type": "Loam"
    }
    response = client.post("/api/v1/agent/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data
    assert "vernacular_hindi" in data
    assert "consultation" in data
