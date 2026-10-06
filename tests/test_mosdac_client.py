"""
@file test_mosdac_client.py
@description Comprehensive unit and integration tests for MOSDAC satellite ingestion client.
             Tests search, anti-lockout credential guard, caching, deterministic mock fallback,
             attribution enforcement, and FastAPI endpoint integration. All network calls
             are strictly mocked to prevent hitting live ISRO servers.
@module tests
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from services.api.main import app
from services.api.schemas import MosdacGranule, MosdacTelemetry
from services.ingestion.mosdac_client import (
    MOSDAC_ATTRIBUTION,
    MosdacAuthLockoutWarning,
    authenticate_mosdac,
    fetch_mosdac_telemetry,
    get_auth_failure_count,
    get_mock_mosdac_telemetry,
    prefetch_mosdac_panchayats,
    reset_auth_failure_counter,
    search_mosdac_datasets,
    validate_credentials_format,
)


@pytest.fixture(autouse=True)
def _reset_auth_circuit_breaker() -> None:
    """Reset auth failure counter before every test."""
    reset_auth_failure_counter()


@pytest.fixture
def client() -> TestClient:
    """Provide FastAPI test client."""
    return TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: Credential Validation & Anti-Lockout Guard
# ─────────────────────────────────────────────────────────────────────────────


def test_credential_format_validation() -> None:
    """Verify pre-flight validation catches empty or placeholder credentials."""
    valid, _ = validate_credentials_format("", "secret123")
    assert valid is False

    valid, _ = validate_credentials_format("chaawal", "")
    assert valid is False

    valid, _ = validate_credentials_format("your_username", "secret123")
    assert valid is False

    valid, _ = validate_credentials_format("chaawal", "12")
    assert valid is False

    valid, msg = validate_credentials_format("chaawal", "PimpnamedMosdac@77")
    assert valid is True
    assert "Valid" in msg


def test_auth_circuit_breaker_prevents_lockout() -> None:
    """Verify circuit breaker trips after 2 consecutive failed attempts to prevent 3-attempt lockout."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.json.return_value = {"error": "Invalid credentials"}

    with patch("requests.post", return_value=mock_resp):
        # Attempt 1: Fails, count = 1
        res1 = authenticate_mosdac("chaawal", "wrong_pass")
        assert res1 is None
        assert get_auth_failure_count() == 1

        # Attempt 2: Fails, count = 2
        res2 = authenticate_mosdac("chaawal", "wrong_pass")
        assert res2 is None
        assert get_auth_failure_count() == 2

        # Attempt 3: Circuit breaker TRIPS before sending request to protect account
        with pytest.raises(MosdacAuthLockoutWarning) as exc_info:
            authenticate_mosdac("chaawal", "wrong_pass")

        assert "circuit breaker ACTIVE" in str(exc_info.value)


def test_auth_success_resets_failure_counter() -> None:
    """Verify successful authentication resets consecutive failure counter."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "access_token": "mock_jwt_access_token",
        "refresh_token": "mock_jwt_refresh_token",
    }

    with patch("requests.post", return_value=mock_resp):
        tokens = authenticate_mosdac("chaawal", "PimpnamedMosdac@77")
        assert tokens is not None
        assert tokens["access_token"] == "mock_jwt_access_token"
        assert get_auth_failure_count() == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: OpenAPI Dataset Search
# ─────────────────────────────────────────────────────────────────────────────


def test_search_mosdac_datasets_parses_response() -> None:
    """Verify OpenAPI search parses granule metadata without requiring authentication."""
    mock_payload = {
        "totalResults": 2,
        "itemsPerPage": 2,
        "entries": [
            {
                "identifier": "3RIMG_07SEP2026_0545_L2B_CMK_V01R00.h5",
                "id": "14502891",
                "summary": "INSAT-3DR Cloud Mask Product at 4km resolution",
                "updated": "2026-09-07T05:45:00Z",
                "enclosureLink": "https://mosdac.gov.in/uops/?metaid=14502891",
                "searchLink": "https://mosdac.gov.in/apios/datasets.json?gId=14502891",
                "boundbox": [{"west": "30.0", "south": "-50.0", "east": "130.0", "north": "50.0"}],
            }
        ],
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    with patch("requests.get", return_value=mock_resp):
        granules = search_mosdac_datasets(
            dataset_id="3RIMG_L2B_CMK",
            start_time="2026-09-07",
            end_time="2026-09-07",
            count=10,
        )

        assert len(granules) == 1
        g = granules[0]
        assert isinstance(g, MosdacGranule)
        assert g.identifier == "3RIMG_07SEP2026_0545_L2B_CMK_V01R00.h5"
        assert g.granule_id == "14502891"
        assert g.dataset_id == "3RIMG_L2B_CMK"
        assert "Cloud Mask" in g.summary


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: Mock Telemetry, Caching & Attribution Enforcement
# ─────────────────────────────────────────────────────────────────────────────


def test_mock_mosdac_telemetry_contains_required_attribution() -> None:
    """Verify mock telemetry produces complete records with exact ISRO attribution."""
    for p_id in ("kadera", "bhankri", "tunga", "dadhikar"):
        tel = get_mock_mosdac_telemetry(p_id)
        assert isinstance(tel, MosdacTelemetry)
        assert tel.panchayat_id == p_id
        assert tel.satellite == "INSAT-3DR"
        assert tel.dataset_id == "3RIMG_L2B_CMK"
        assert 0.0 <= tel.cloud_fraction_pct <= 100.0
        assert tel.source_status == "mock"
        assert tel.attribution == "Data Source: MOSDAC/SAC/ISRO. https://mosdac.gov.in"


def test_prefetch_and_fetch_mosdac_telemetry(tmp_path: pytest.TempPathFactory) -> None:
    """Verify prefetch serializes records to cache file and fetch loads from cache."""
    test_cache_file = tmp_path / "test_mosdac_cache.json"

    with patch("services.ingestion.mosdac_client.search_mosdac_datasets", return_value=[]):
        prefetched = prefetch_mosdac_panchayats(cache_file_path=test_cache_file)
        assert len(prefetched) == 4
        assert test_cache_file.exists()

        with patch("services.ingestion.mosdac_client._CACHE_FILE", test_cache_file):
            t_kadera = fetch_mosdac_telemetry("kadera")
            assert t_kadera.panchayat_id == "kadera"
            assert t_kadera.source_status == "cached"
            assert t_kadera.cloud_fraction_pct == 18.5
            assert t_kadera.attribution == MOSDAC_ATTRIBUTION


# ─────────────────────────────────────────────────────────────────────────────
#  FastAPI Integration Route Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_get_panchayat_mosdac_telemetry_endpoint(client: TestClient) -> None:
    """Verify GET /api/v1/forecast/{panchayat_id}/mosdac returns 200 with attribution."""
    resp = client.get("/api/v1/forecast/KADERA_001/mosdac")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["success"] is True
    assert "data" in payload
    data = payload["data"]
    assert data["panchayat_id"] == "kadera_001"
    assert data["satellite"] == "INSAT-3DR"
    assert "cloud_fraction_pct" in data
    assert data["attribution"] == "Data Source: MOSDAC/SAC/ISRO. https://mosdac.gov.in"


def test_all_pilot_panchayats_return_distinct_mosdac_telemetry(client: TestClient) -> None:
    """Verify each of the 4 pilot panchayats returns its distinct telemetry values."""
    expected_values = {
        "KADERA_001": {"cloud": 18.5, "rain": 0.0, "temp": -12.4},
        "BHANKRI_002": {"cloud": 24.0, "rain": 0.0, "temp": -15.8},
        "TUNGA_003": {"cloud": 32.0, "rain": 1.2, "temp": -21.0},
        "DADHIKAR_004": {"cloud": 45.0, "rain": 3.8, "temp": -28.5},
    }

    cloud_values: set[float] = set()

    for pid, expected in expected_values.items():
        resp = client.get(f"/api/v1/forecast/{pid}/mosdac")
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["cloud_fraction_pct"] == expected["cloud"]
        assert data["rainfall_rate_mmh"] == expected["rain"]
        assert data["cloud_top_temp_c"] == expected["temp"]
        cloud_values.add(data["cloud_fraction_pct"])

    # Ensure all 4 sites produced distinct cloud percentages (no fallback collapse)
    assert len(cloud_values) == 4


def test_get_panchayat_mosdac_telemetry_404_for_unknown(client: TestClient) -> None:
    """Verify GET /api/v1/forecast/{invalid_id}/mosdac returns 404."""
    resp = client.get("/api/v1/forecast/INVALID_PANCHAYAT/mosdac")
    assert resp.status_code == 404
    payload = resp.json()
    assert payload["success"] is False
