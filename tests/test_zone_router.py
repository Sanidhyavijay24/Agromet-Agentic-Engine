"""
@file test_zone_router.py
@description Unit tests for Pan-India 15 ICAR Agro-Climatic Zone (ACZ) spatial router and grid generator.
@module tests
"""

from __future__ import annotations

import pytest

from services.ml_downscaler.zone_router import (
    ACZ_CATALOG,
    generate_zone_spatial_grid,
    get_zone_artifact_path,
    get_zone_by_id,
    route_coordinates_to_zone,
)


def test_zone_catalog_completeness():
    """Verify all 15 ICAR Agro-Climatic Zones are defined with complete bounding boxes."""
    assert len(ACZ_CATALOG) == 15

    for zid, zone in ACZ_CATALOG.items():
        assert zone["zone_id"] == zid
        assert 1 <= zone["numeric_id"] <= 15
        assert len(zone["name"]) > 0
        assert len(zone["focus_domain"]) > 0
        assert len(zone["representative_districts"]) >= 3
        assert len(zone["primary_crops"]) >= 2

        bbox = zone["bounding_box"]
        assert bbox["north"] > bbox["south"]
        assert bbox["east"] > bbox["west"]
        assert bbox["south"] <= zone["center_lat"] <= bbox["north"]
        assert bbox["west"] <= zone["center_lon"] <= bbox["east"]


def test_zone_spatial_grid_generation():
    """Verify spatial lattice generation produces expected station count and valid bounds."""
    for zid in ["ACZ_01", "ACZ_06", "ACZ_14"]:
        grid = generate_zone_spatial_grid(zid, points_count=36)
        assert len(grid) == 36

        zone = ACZ_CATALOG[zid]
        bbox = zone["bounding_box"]
        for pt in grid:
            assert pt["point_id"].startswith(zid)
            assert bbox["south"] <= pt["latitude"] <= bbox["north"]
            assert bbox["west"] <= pt["longitude"] <= bbox["east"]


def test_zone_coordinate_routing():
    """Verify known Indian district coordinates route to their respective ACZ zones."""
    test_cases = [
        # (lat, lon, expected_zone_id, description)
        (26.9124, 75.7873, "ACZ_14", "Jaipur, Rajasthan -> Western Dry"),
        (32.2190, 76.3234, "ACZ_01", "Kangra, HP -> Western Himalayan"),
        (26.1445, 91.7362, "ACZ_02", "Guwahati, Assam -> Eastern Himalayan"),
        (30.9010, 75.8573, "ACZ_06", "Ludhiana, Punjab -> Trans-Gangetic"),
        (18.5204, 73.8567, "ACZ_09", "Pune, Maharashtra -> Western Plateau"),
        (11.6234, 92.7265, "ACZ_15", "Port Blair, A&N -> Islands Region"),
    ]

    for lat, lon, expected_zid, desc in test_cases:
        matched_zone, is_exact = route_coordinates_to_zone(lat, lon)
        assert matched_zone["zone_id"] == expected_zid, f"Failed for {desc}: got {matched_zone['zone_id']}"


def test_zone_artifact_naming_convention():
    """Verify model artifact paths adhere to the standard numeric naming convention."""
    p1 = get_zone_artifact_path("ACZ_01")
    assert p1.name == "residual_model_acz_01.joblib"

    p14 = get_zone_artifact_path("ACZ_14")
    assert p14.name == "residual_model_acz_14.joblib"
