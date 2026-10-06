"""
@file agronomy_tool.py
@description Tool for ICAR/FAO agronomic crop knowledge base and physiological stress rules.
@module services/agent/tools
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

# ICAR / State Agricultural University agronomic profile catalog
CROP_AGRONOMY_REGISTRY: Dict[str, Dict[str, Any]] = {
    "BAJRA": {
        "common_name": "Pearl Millet (बाजरा)",
        "season": "Kharif",
        "base_temp_c": 10.0,
        "optimal_temp_range_c": [28.0, 36.0],
        "critical_max_temp_c": 42.0,
        "optimal_vpd_range_kpa": [1.0, 2.5],
        "stages": {
            "Sowing": {"water_need": "Low", "critical_pest": "Shoot fly", "sensitivity": "Moderate"},
            "Vegetative": {"water_need": "Moderate", "critical_pest": "Stem borer", "sensitivity": "Low"},
            "Flowering": {"water_need": "High", "critical_pest": "Ergot / Smut", "sensitivity": "Extreme"},
            "Grain Filling": {"water_need": "High", "critical_pest": "Bird damage, rust", "sensitivity": "High"},
            "Maturity": {"water_need": "Low", "critical_pest": "Pre-harvest rain mould", "sensitivity": "Moderate"}
        },
        "pathology_triggers": {
            "Ergot": "Relative humidity > 85% with temperatures 20-30°C during flowering",
            "Downy Mildew": "Cloudy humid weather with prolonged leaf wetness"
        },
        "advisory_notes": "Avoid nitrogen top-dressing during dry spells. Ensure light irrigation if flowering coincides with VPD > 3.0 kPa."
    },
    "GUAR": {
        "common_name": "Cluster Bean (ग्वार)",
        "season": "Kharif",
        "base_temp_c": 12.0,
        "optimal_temp_range_c": [25.0, 34.0],
        "critical_max_temp_c": 40.0,
        "optimal_vpd_range_kpa": [1.2, 2.8],
        "stages": {
            "Sowing": {"water_need": "Low", "critical_pest": "Root rot", "sensitivity": "Moderate"},
            "Vegetative": {"water_need": "Low", "critical_pest": "Jassids", "sensitivity": "Low"},
            "Flowering": {"water_need": "Moderate", "critical_pest": "Bacterial blight", "sensitivity": "High"},
            "Pod Formation": {"water_need": "Moderate", "critical_pest": "Pod borer", "sensitivity": "High"},
            "Maturity": {"water_need": "Low", "critical_pest": "Pod shattering", "sensitivity": "Moderate"}
        },
        "pathology_triggers": {
            "Bacterial Blight": "High humidity coupled with daytime heat spikes"
        },
        "advisory_notes": "Highly drought resilient. Protect against waterlogging in heavy soils."
    },
    "GROUNDNUT": {
        "common_name": "Groundnut / Peanut (मूंगफली)",
        "season": "Kharif / Zaid",
        "base_temp_c": 10.0,
        "optimal_temp_range_c": [24.0, 30.0],
        "critical_max_temp_c": 38.0,
        "optimal_vpd_range_kpa": [0.8, 2.0],
        "stages": {
            "Sowing": {"water_need": "Moderate", "critical_pest": "Collar rot", "sensitivity": "Moderate"},
            "Vegetative": {"water_need": "Moderate", "critical_pest": "Leaf miner", "sensitivity": "Low"},
            "Pegging": {"water_need": "Critical", "critical_pest": "Tikka disease", "sensitivity": "Extreme"},
            "Pod Development": {"water_need": "Critical", "critical_pest": "White grub", "sensitivity": "Extreme"},
            "Maturity": {"water_need": "Low", "critical_pest": "Aflatoxin infection", "sensitivity": "High"}
        },
        "pathology_triggers": {
            "Tikka Disease (Cercospora)": "Prolonged high humidity (>80%) with warm temperatures (25-30°C)"
        },
        "advisory_notes": "Maintain adequate topsoil moisture (0-7cm) during pegging; dry crusting prevents subterranean peg penetration."
    },
    "WHEAT": {
        "common_name": "Wheat (गेहूं)",
        "season": "Rabi",
        "base_temp_c": 4.5,
        "optimal_temp_range_c": [15.0, 24.0],
        "critical_max_temp_c": 30.0,
        "optimal_vpd_range_kpa": [0.6, 1.8],
        "stages": {
            "Crown Root Initiation (CRI)": {"water_need": "Critical", "critical_pest": "Termites", "sensitivity": "Extreme"},
            "Tillering": {"water_need": "Moderate", "critical_pest": "Aphids", "sensitivity": "Moderate"},
            "Jointing / Booting": {"water_need": "High", "critical_pest": "Rust (Yellow/Brown)", "sensitivity": "High"},
            "Flowering / Heading": {"water_need": "Critical", "critical_pest": "Karnal bunt", "sensitivity": "Extreme"},
            "Milking / Dough": {"water_need": "Moderate", "critical_pest": "Terminal heat desiccation", "sensitivity": "Extreme"}
        },
        "pathology_triggers": {
            "Yellow Rust": "Cool nights (8-13°C) and morning dew followed by mild days (15-20°C)"
        },
        "advisory_notes": "If temperatures exceed 30°C during grain filling, apply light irrigation or potassium nitrate foliar spray to combat terminal heat."
    },
    "MUSTARD": {
        "common_name": "Indian Mustard (सरसों)",
        "season": "Rabi",
        "base_temp_c": 5.0,
        "optimal_temp_range_c": [15.0, 25.0],
        "critical_max_temp_c": 32.0,
        "optimal_vpd_range_kpa": [0.6, 1.8],
        "stages": {
            "Sowing": {"water_need": "Moderate", "critical_pest": "Sawfly", "sensitivity": "Moderate"},
            "Vegetative": {"water_need": "Moderate", "critical_pest": "Painted bug", "sensitivity": "Low"},
            "Flowering": {"water_need": "Critical", "critical_pest": "Mustard aphid", "sensitivity": "Extreme"},
            "Siliqua Formation": {"water_need": "High", "critical_pest": "White rust / Alternaria", "sensitivity": "High"},
            "Maturity": {"water_need": "Low", "critical_pest": "Sclerotinia stem rot", "sensitivity": "Moderate"}
        },
        "pathology_triggers": {
            "Mustard Aphids": "Cloudy, humid weather (RH > 70%) with temperatures 15-20°C during flowering"
        },
        "advisory_notes": "Monitor aphid colonies on top 10cm twigs. Spray when population reaches 25 aphids/plant."
    },
    "COTTON": {
        "common_name": "Cotton (कपास)",
        "season": "Kharif",
        "base_temp_c": 12.0,
        "optimal_temp_range_c": [25.0, 35.0],
        "critical_max_temp_c": 40.0,
        "optimal_vpd_range_kpa": [1.0, 2.5],
        "stages": {
            "Squaring": {"water_need": "Moderate", "critical_pest": "Whitefly / Jassids", "sensitivity": "High"},
            "Flowering": {"water_need": "High", "critical_pest": "Pink bollworm", "sensitivity": "Extreme"},
            "Boll Development": {"water_need": "High", "critical_pest": "Boll rot", "sensitivity": "Extreme"},
            "Maturity / Bursting": {"water_need": "Low", "critical_pest": "Staining fungi", "sensitivity": "Moderate"}
        },
        "pathology_triggers": {
            "Pink Bollworm": "High relative humidity with evening temperatures > 24°C during flower square phase"
        },
        "advisory_notes": "Avoid excess nitrogen during square formation. Ensure good drainage to prevent leaf reddening."
    }
}


def lookup_crop_agronomy(
    crop_name: str,
    stage: str = "Flowering"
) -> Dict[str, Any]:
    """
    Lookup detailed ICAR agronomic thresholds and physiological vulnerabilities for a crop and stage.
    """
    start_time = time.perf_counter()
    key = crop_name.strip().upper()
    profile = CROP_AGRONOMY_REGISTRY.get(key)
    
    if not profile:
        # Fallback to general cereal profile
        profile = CROP_AGRONOMY_REGISTRY["BAJRA"]
        key = "BAJRA"
        
    stage_info = profile["stages"].get(stage)
    if not stage_info:
        # Find closest match
        stage_info = list(profile["stages"].values())[2]  # default to flowering/critical
        
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    return {
        "crop_key": key,
        "common_name": profile["common_name"],
        "season": profile["season"],
        "base_temp_c": profile["base_temp_c"],
        "optimal_temp_range_c": profile["optimal_temp_range_c"],
        "critical_max_temp_c": profile["critical_max_temp_c"],
        "optimal_vpd_range_kpa": profile["optimal_vpd_range_kpa"],
        "stage_evaluated": stage,
        "stage_water_demand": stage_info["water_need"],
        "stage_critical_pest": stage_info["critical_pest"],
        "stage_sensitivity": stage_info["sensitivity"],
        "pathology_triggers": profile["pathology_triggers"],
        "advisory_notes": profile["advisory_notes"],
        "execution_time_ms": round(elapsed_ms, 2)
    }
