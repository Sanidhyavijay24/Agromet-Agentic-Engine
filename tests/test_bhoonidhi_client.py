"""
@file test_bhoonidhi_client.py
@description Unit and integration test suite for Bhoonidhi (NRSC/ISRO) STAC API client.
             Validates:
             - Authentication lifecycle and token refresh logic
             - STAC search with Point coordinates & bounding box
             - Soil moisture and CartoDEM telemetry extraction
             - Mandatory ISRO / Indian Space Policy 2023 attribution
@module tests
"""

import pytest
from unittest.mock import MagicMock, patch

from services.ingestion.bhoonidhi_client import (
    BHOONIDHI_ATTRIBUTION,
    COLLECTIONS,
    authenticate_bhoonidhi,
    fetch_bhoonidhi_cartodem_elevation,
    fetch_bhoonidhi_soil_moisture,
    search_bhoonidhi_catalog,
)


def test_bhoonidhi_attribution_present():
    """Verify Indian Space Policy 2023 & Bhoonidhi attribution is declared."""
    assert "Bhoonidhi" in BHOONIDHI_ATTRIBUTION
    assert "ISRO" in BHOONIDHI_ATTRIBUTION
    assert "https://bhoonidhi.nrsc.gov.in" in BHOONIDHI_ATTRIBUTION


def test_authenticate_offline_fallback():
    """Test auth falls back gracefully when credentials are not configured."""
    auth = authenticate_bhoonidhi(user_id="", password="", force_refresh=True)
    assert auth is not None
    assert "access_token" in auth
    assert auth["token_type"] == "Bearer"


def test_authenticate_mock_network_success():
    """Test standard token response when network mock returns 200."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "userId": "testuser",
        "access_token": "jwt_token_sample_123",
        "token_type": "Bearer",
        "expires_in": 1200,
        "refresh_token": "refresh_token_sample_456",
    }

    with patch("requests.post", return_value=mock_resp):
        auth = authenticate_bhoonidhi(
            user_id="testuser", password="secretpassword", force_refresh=True
        )
        assert auth["access_token"] == "jwt_token_sample_123"
        assert auth["refresh_token"] == "refresh_token_sample_456"


def test_search_bhoonidhi_catalog_offline_mock():
    """Test STAC catalog search returns GeoJSON FeatureCollection."""
    res = search_bhoonidhi_catalog(
        collections=[COLLECTIONS["SOIL_MOISTURE"], COLLECTIONS["CARTODEM_30M"]],
        lat=27.05,
        lon=76.62,
        limit=5,
    )
    assert res["type"] == "FeatureCollection"
    assert "features" in res
    assert len(res["features"]) >= 1
    assert res["features"][0]["geometry"]["type"] == "Point"
    assert "attribution" in res


def test_fetch_soil_moisture_extraction():
    """Test SAR soil moisture extractor."""
    data = fetch_bhoonidhi_soil_moisture(lat=27.05, lon=76.62)
    assert "soil_moisture_m3_m3" in data
    assert 0.0 <= data["soil_moisture_m3_m3"] <= 1.0
    assert "EOS-04" in data["satellite"]
    assert data["attribution"] == BHOONIDHI_ATTRIBUTION


def test_fetch_cartodem_elevation():
    """Test CartoDEM elevation telemetry extractor."""
    data = fetch_bhoonidhi_cartodem_elevation(lat=27.05, lon=76.62)
    assert data["collection"] == "CartoSat-1_PAN_CartoDEM_30m"
    assert data["sensor"] == "CartoSat-1 PAN Stereo"
    assert data["attribution"] == BHOONIDHI_ATTRIBUTION
