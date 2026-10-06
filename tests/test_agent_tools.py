"""
@file test_agent_tools.py
@description Unit tests for specialized Agromet Agent tools.
@module tests
"""

import pytest
from services.agent.tools.downscaler_tool import run_1km_downscaler
from services.agent.tools.indices_tool import calculate_microclimate_indices, compute_vpd_kpa
from services.agent.tools.agronomy_tool import lookup_crop_agronomy, CROP_AGRONOMY_REGISTRY
from services.agent.tools.soil_tool import get_soil_and_terrain_context


def test_compute_vpd_kpa():
    """Verify VPD calculation against standard psychrometric ranges."""
    # 25°C, 50% RH -> e_s ~ 3.17 kPa, e_a ~ 1.58 kPa, VPD ~ 1.58 kPa
    vpd = compute_vpd_kpa(25.0, 50.0)
    assert 1.4 <= vpd <= 1.7
    
    # 100% RH -> VPD must be 0
    assert compute_vpd_kpa(30.0, 100.0) == 0.0


def test_lookup_crop_agronomy():
    """Verify ICAR crop registry lookup and stage sensitivity."""
    bajra = lookup_crop_agronomy("Bajra", "Flowering")
    assert bajra["crop_key"] == "BAJRA"
    assert bajra["base_temp_c"] == 10.0
    assert bajra["stage_evaluated"] == "Flowering"
    assert bajra["stage_sensitivity"] == "Extreme"
    
    wheat = lookup_crop_agronomy("Wheat", "Crown Root Initiation (CRI)")
    assert wheat["crop_key"] == "WHEAT"
    assert wheat["base_temp_c"] == 4.5
    
    # Case insensitivity & fallback
    mustard = lookup_crop_agronomy("mustard", "Flowering")
    assert mustard["crop_key"] == "MUSTARD"


def test_soil_and_terrain_tool():
    """Verify terrain retrieval and soil moisture status evaluation."""
    soil = get_soil_and_terrain_context(26.9124, 75.7873, "KADERA_001")
    assert "elevation_m" in soil
    assert soil["elevation_m"] > 0
    assert "moisture_status" in soil
    assert soil["execution_time_ms"] >= 0


def test_downscaler_and_indices_integration():
    """Verify end-to-end 1km downscaling tool and physical indices calculation."""
    # Kadera coordinates
    downscale_res = run_1km_downscaler(26.9124, 75.7873, forecast_hours=24)
    assert downscale_res["zone_id"] in ["ACZ_14", "ZONE_14_WESTERN_DRY"]
    assert len(downscale_res["hourly_series"]) == 24
    assert "mean_delta_t_c" in downscale_res
    
    # Evaluate indices on the downscaled series
    indices_res = calculate_microclimate_indices(
        hourly_series=downscale_res["hourly_series"],
        crop_base_temp_c=10.0,
        crop_max_temp_c=40.0
    )
    ind = indices_res["indices"]
    assert ind["mean_vpd_kpa"] >= 0.0
    assert ind["gdd_accumulated_c_days"] >= 0.0
    assert isinstance(ind["spray_windows"], list)

