"""
@file zone_router.py
@description Pan-India 15 ICAR Agro-Climatic Zone (ACZ) registry, spatial bounding-box router,
             and dynamic zone artifact resolver.
@module services/ml_downscaler
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import numpy as np

try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger("zone_router")  # type: ignore[assignment]

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ZONES_ARTIFACTS_DIR = ROOT_DIR / "services" / "ml_downscaler" / "artifacts" / "zones"


class BoundingBox(TypedDict):
    north: float
    south: float
    west: float
    east: float


class AgroClimaticZone(TypedDict):
    zone_id: str
    numeric_id: int
    name: str
    focus_domain: str
    bounding_box: BoundingBox
    center_lat: float
    center_lon: float
    representative_districts: List[str]
    primary_crops: List[str]
    dominant_physics: str


# ─────────────────────────────────────────────────────────────────────────────
#  Canonical 15 ICAR Agro-Climatic Zone (ACZ) Specifications
# ─────────────────────────────────────────────────────────────────────────────

ACZ_CATALOG: Dict[str, AgroClimaticZone] = {
    "ACZ_01": {
        "zone_id": "ACZ_01",
        "numeric_id": 1,
        "name": "Western Himalayan",
        "focus_domain": "Kangra–Kullu–Mandi",
        "bounding_box": {"north": 33.0, "south": 31.0, "west": 76.0, "east": 78.0},
        "center_lat": 32.0,
        "center_lon": 77.0,
        "representative_districts": ["Kangra", "Mandi", "Kullu", "Shimla"],
        "primary_crops": ["Apple", "Maize", "Wheat"],
        "dominant_physics": "Steep mountain relief, katabatic drainage, slope insolation",
    },
    "ACZ_02": {
        "zone_id": "ACZ_02",
        "numeric_id": 2,
        "name": "Eastern Himalayan",
        "focus_domain": "Brahmaputra Valley",
        "bounding_box": {"north": 27.5, "south": 25.5, "west": 91.0, "east": 93.0},
        "center_lat": 26.5,
        "center_lon": 92.0,
        "representative_districts": ["Guwahati", "Tezpur", "Shillong"],
        "primary_crops": ["Tea", "Rice", "Orange"],
        "dominant_physics": "Extreme monsoon rain, valley humidity fog, orographic lift",
    },
    "ACZ_03": {
        "zone_id": "ACZ_03",
        "numeric_id": 3,
        "name": "Lower Gangetic",
        "focus_domain": "Bengal Delta Basin",
        "bounding_box": {"north": 24.0, "south": 22.0, "west": 87.5, "east": 89.5},
        "center_lat": 23.0,
        "center_lon": 88.5,
        "representative_districts": ["Burdwan", "Hooghly", "Nadia"],
        "primary_crops": ["Jute", "Rice", "Mustard"],
        "dominant_physics": "High RH, alluvial shallow water table, coastal tidal humidity",
    },
    "ACZ_04": {
        "zone_id": "ACZ_04",
        "numeric_id": 4,
        "name": "Middle Gangetic",
        "focus_domain": "Bihar–Eastern UP",
        "bounding_box": {"north": 27.0, "south": 25.0, "west": 84.0, "east": 86.0},
        "center_lat": 26.0,
        "center_lon": 85.0,
        "representative_districts": ["Patna", "Muzaffarpur", "Gaya"],
        "primary_crops": ["Rice", "Wheat", "Sugarcane"],
        "dominant_physics": "Flat basin, pre-monsoon convective storm, alluvial moisture",
    },
    "ACZ_05": {
        "zone_id": "ACZ_05",
        "numeric_id": 5,
        "name": "Upper Gangetic",
        "focus_domain": "Western UP Doab",
        "bounding_box": {"north": 29.5, "south": 27.5, "west": 77.5, "east": 79.5},
        "center_lat": 28.5,
        "center_lon": 78.5,
        "representative_districts": ["Meerut", "Aligarh", "Agra"],
        "primary_crops": ["Sugarcane", "Wheat", "Mustard"],
        "dominant_physics": "Winter radiation fog, nocturnal thermal inversion, continental extremes",
    },
    "ACZ_06": {
        "zone_id": "ACZ_06",
        "numeric_id": 6,
        "name": "Trans-Gangetic",
        "focus_domain": "Punjab–Haryana",
        "bounding_box": {"north": 31.5, "south": 29.5, "west": 75.5, "east": 77.5},
        "center_lat": 30.5,
        "center_lon": 76.5,
        "representative_districts": ["Ludhiana", "Karnal", "Patiala"],
        "primary_crops": ["Wheat", "Paddy", "Cotton"],
        "dominant_physics": "Intensive canal irrigation microclimate, high diurnal range",
    },
    "ACZ_07": {
        "zone_id": "ACZ_07",
        "numeric_id": 7,
        "name": "Eastern Plateau",
        "focus_domain": "Chota Nagpur",
        "bounding_box": {"north": 24.5, "south": 22.5, "west": 84.5, "east": 86.5},
        "center_lat": 23.5,
        "center_lon": 85.5,
        "representative_districts": ["Ranchi", "Jamshedpur", "Bokaro"],
        "primary_crops": ["Rice", "Pulses", "Millets"],
        "dominant_physics": "Undulating plateau, iron-rich laterite, convective heat islands",
    },
    "ACZ_08": {
        "zone_id": "ACZ_08",
        "numeric_id": 8,
        "name": "Central Plateau",
        "focus_domain": "Malwa Plateau",
        "bounding_box": {"north": 24.5, "south": 22.5, "west": 75.5, "east": 77.5},
        "center_lat": 23.5,
        "center_lon": 76.5,
        "representative_districts": ["Indore", "Ujjain", "Kota"],
        "primary_crops": ["Soybean", "Wheat", "Gram"],
        "dominant_physics": "Black cotton soil moisture retention, continental dry heat",
    },
    "ACZ_09": {
        "zone_id": "ACZ_09",
        "numeric_id": 9,
        "name": "Western Plateau",
        "focus_domain": "Maharashtra Deccan",
        "bounding_box": {"north": 20.5, "south": 18.5, "west": 73.5, "east": 75.5},
        "center_lat": 19.5,
        "center_lon": 74.5,
        "representative_districts": ["Pune", "Ahmednagar", "Nashik East"],
        "primary_crops": ["Jowar", "Cotton", "Sugarcane"],
        "dominant_physics": "Semi-arid rain-shadow plateau, strong lee-side subsidence",
    },
    "ACZ_10": {
        "zone_id": "ACZ_10",
        "numeric_id": 10,
        "name": "Southern Plateau",
        "focus_domain": "Telangana–Rayalaseema",
        "bounding_box": {"north": 18.5, "south": 16.5, "west": 77.5, "east": 79.5},
        "center_lat": 17.5,
        "center_lon": 78.5,
        "representative_districts": ["Hyderabad", "Mahbubnagar", "Kurnool"],
        "primary_crops": ["Groundnut", "Maize", "Cotton"],
        "dominant_physics": "Red soil rapid drainage, high solar insolation, thermal advection",
    },
    "ACZ_11": {
        "zone_id": "ACZ_11",
        "numeric_id": 11,
        "name": "East Coast Plains",
        "focus_domain": "Krishna Delta & AP",
        "bounding_box": {"north": 17.5, "south": 15.5, "west": 80.0, "east": 82.0},
        "center_lat": 16.5,
        "center_lon": 81.0,
        "representative_districts": ["Guntur", "Vijayawada", "Kakinada"],
        "primary_crops": ["Paddy", "Coconut", "Banana"],
        "dominant_physics": "Cyclonic coastal moisture, maritime boundary layer damping",
    },
    "ACZ_12": {
        "zone_id": "ACZ_12",
        "numeric_id": 12,
        "name": "West Coast & Ghats",
        "focus_domain": "Konkan–Western Ghats",
        "bounding_box": {"north": 17.0, "south": 15.0, "west": 73.0, "east": 75.0},
        "center_lat": 16.0,
        "center_lon": 74.0,
        "representative_districts": ["Ratnagiri", "Goa", "Belagavi Ghats"],
        "primary_crops": ["Coconut", "Spices", "Rubber"],
        "dominant_physics": "High orographic precipitation, diurnal sea/land breeze transition",
    },
    "ACZ_13": {
        "zone_id": "ACZ_13",
        "numeric_id": 13,
        "name": "Gujarat Plains",
        "focus_domain": "Saurashtra–C. Gujarat",
        "bounding_box": {"north": 23.5, "south": 21.5, "west": 71.0, "east": 73.0},
        "center_lat": 22.5,
        "center_lon": 72.0,
        "representative_districts": ["Ahmedabad", "Rajkot", "Anand"],
        "primary_crops": ["Cotton", "Groundnut", "Cumin"],
        "dominant_physics": "Coastal-saline transition, high insolation, arid-marine boundary",
    },
    "ACZ_14": {
        "zone_id": "ACZ_14",
        "numeric_id": 14,
        "name": "Western Dry",
        "focus_domain": "Semi-Arid Rajasthan",
        "bounding_box": {"north": 28.0, "south": 26.0, "west": 75.0, "east": 77.0},
        "center_lat": 27.0,
        "center_lon": 76.0,
        "representative_districts": ["Jaipur", "Chaksu", "Alwar", "Dausa"],
        "primary_crops": ["Bajra", "Guar", "Mustard"],
        "dominant_physics": "Extreme diurnal range, sensible heat flux, high soil thermal gradient",
    },
    "ACZ_15": {
        "zone_id": "ACZ_15",
        "numeric_id": 15,
        "name": "Islands Region",
        "focus_domain": "Andaman Maritime Arc",
        "bounding_box": {"north": 13.0, "south": 11.0, "west": 92.0, "east": 94.0},
        "center_lat": 12.0,
        "center_lon": 93.0,
        "representative_districts": ["Port Blair", "Havelock", "Diglipur"],
        "primary_crops": ["Coconut", "Arecanut", "Oil Palm"],
        "dominant_physics": "Tropical maritime boundary layer, uniform low diurnal variance",
    },
}


def get_all_zones() -> List[AgroClimaticZone]:
    """Retrieve list of all 15 canonical Agro-Climatic Zone definitions."""
    return list(ACZ_CATALOG.values())


def get_zone_by_id(zone_id: str) -> Optional[AgroClimaticZone]:
    """Retrieve zone specification by normalized ID (e.g. 'ACZ_14', 'acz_01', '14')."""
    normalized = zone_id.upper().strip()
    if not normalized.startswith("ACZ_"):
        if normalized.isdigit():
            normalized = f"ACZ_{int(normalized):02d}"
        else:
            normalized = f"ACZ_{normalized}"
    return ACZ_CATALOG.get(normalized)


def route_coordinates_to_zone(
    latitude: float,
    longitude: float,
) -> Tuple[AgroClimaticZone, bool]:
    """
    Route any geographic coordinate (lat, lon) to its matching ICAR Agro-Climatic Zone.
    
    Args:
        latitude: Target latitude in decimal degrees.
        longitude: Target longitude in decimal degrees.
        
    Returns:
        Tuple of (AgroClimaticZone, is_exact_bbox_match).
        If coordinate is outside all bounding boxes, falls back to the nearest zone centroid.
    """
    # 1. Exact Bounding Box Match
    for zone in ACZ_CATALOG.values():
        bbox = zone["bounding_box"]
        if bbox["south"] <= latitude <= bbox["north"] and bbox["west"] <= longitude <= bbox["east"]:
            return zone, True

    # 2. Nearest Centroid Fallback
    best_zone: Optional[AgroClimaticZone] = None
    min_dist_sq = float("inf")

    for zone in ACZ_CATALOG.values():
        dist_sq = (zone["center_lat"] - latitude) ** 2 + (zone["center_lon"] - longitude) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_zone = zone

    return best_zone or ACZ_CATALOG["ACZ_14"], False


def generate_zone_spatial_grid(
    zone_id: str,
    points_count: int = 36,
) -> List[Dict[str, Any]]:
    """
    Generate an evenly distributed spatial lattice of N coordinates across the zone's bounding box.
    Default: 36 stations (6x6 regular grid with ~0.35° spacing across 2° x 2°).
    """
    zone = get_zone_by_id(zone_id)
    if not zone:
        raise ValueError(f"Unknown zone_id: {zone_id}")

    bbox = zone["bounding_box"]
    grid_side = int(math.ceil(math.sqrt(points_count)))

    lats = np.linspace(bbox["south"] + 0.1, bbox["north"] - 0.1, grid_side)
    lons = np.linspace(bbox["west"] + 0.1, bbox["east"] - 0.1, grid_side)

    points: List[Dict[str, Any]] = []
    idx = 1
    for lat in lats:
        for lon in lons:
            if len(points) >= points_count:
                break
            points.append(
                {
                    "point_id": f"{zone['zone_id']}_PT_{idx:03d}",
                    "latitude": round(float(lat), 4),
                    "longitude": round(float(lon), 4),
                    "zone_id": zone["zone_id"],
                    "zone_name": zone["name"],
                }
            )
            idx += 1

    return points


def get_zone_artifact_path(zone_id: str) -> Path:
    """Return the canonical file path for a zone's serialized model artifact."""
    zone = get_zone_by_id(zone_id)
    zid_num = zone["numeric_id"] if zone else int(zone_id.replace("ACZ_", ""))
    return ZONES_ARTIFACTS_DIR / f"residual_model_acz_{zid_num:02d}.joblib"
