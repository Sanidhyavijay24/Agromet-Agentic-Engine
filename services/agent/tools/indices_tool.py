"""
@file indices_tool.py
@description Tool for computing physical agrometeorological stress indices, VPD, GDD, and spray windows.
@module services/agent/tools
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List
from services.agent.schemas import MicroclimateIndices, SprayWindow


def compute_saturated_vapor_pressure_kpa(temp_c: float) -> float:
    """Tetens formula for saturated vapor pressure in kPa."""
    return 0.61078 * math.exp((17.27 * temp_c) / (temp_c + 237.3))


def compute_vpd_kpa(temp_c: float, rh_pct: float) -> float:
    """Compute Vapor Pressure Deficit in kPa."""
    e_s = compute_saturated_vapor_pressure_kpa(temp_c)
    e_a = e_s * (max(0.0, min(100.0, rh_pct)) / 100.0)
    return max(0.0, e_s - e_a)


def calculate_microclimate_indices(
    hourly_series: List[Dict[str, Any]],
    crop_base_temp_c: float = 10.0,
    crop_max_temp_c: float = 38.0
) -> Dict[str, Any]:
    """
    Compute comprehensive agrometeorological stress indicators from 1km hourly forecast series.
    """
    start_time = time.perf_counter()
    if not hourly_series:
        return {"error": "Empty hourly series"}
    
    vpd_values: List[float] = []
    temps: List[float] = []
    baseline_temps: List[float] = []
    heat_stress_hours = 0
    inversion_deltas: List[float] = []
    spray_candidates: List[Dict[str, Any]] = []
    
    for pt in hourly_series:
        t_1km = pt["downscaled_temp_1km_c"]
        t_base = pt["baseline_temp_c"]
        rh = pt["relative_humidity_pct"]
        wind = pt["wind_speed_kmh"]
        pop = pt.get("rain_probability_pct", 0.0)
        
        # 1. VPD
        vpd = compute_vpd_kpa(t_1km, rh)
        vpd_values.append(vpd)
        temps.append(t_1km)
        baseline_temps.append(t_base)
        
        # 2. Heat stress
        if t_1km > crop_max_temp_c:
            heat_stress_hours += 1
            
        # 3. Nocturnal Thermal Inversion (Lowland pooling: T_1km < T_baseline)
        if t_base - t_1km > 0.4:
            inversion_deltas.append(t_base - t_1km)
            
        # 4. Spray suitability check
        # Ideal: Wind 4-12 km/h, PoP < 20%, Temp 18-32°C, RH 35-85%
        is_sprayable = True
        limiting = "None"
        suitability = 95.0
        
        if wind > 14.0:
            is_sprayable = False
            limiting = f"High wind ({wind:.1f} km/h > 14 km/h: chemical drift risk)"
            suitability -= 50
        elif wind < 3.0:
            limiting = "Calm air (sub-optimal canopy penetration)"
            suitability -= 15
            
        if pop > 25.0:
            is_sprayable = False
            limiting = f"Rain probability ({pop:.0f}% > 25%: washout risk)"
            suitability -= 60
            
        if t_1km > 34.0:
            is_sprayable = False
            limiting = f"High temperature ({t_1km:.1f}°C: droplet evaporation risk)"
            suitability -= 45
        elif t_1km < 12.0:
            suitability -= 20
            
        if rh > 90.0:
            limiting = "High humidity (dew dilution / delayed drying)"
            suitability -= 15
            
        suitability = max(0.0, min(100.0, suitability))
        
        spray_candidates.append({
            "timestamp": pt["timestamp"],
            "is_sprayable": is_sprayable,
            "suitability": suitability,
            "limiting": limiting,
            "temp_c": t_1km,
            "wind_kmh": wind
        })
    
    # Aggregate GDD (Growing Degree Days)
    # Split into 24-hour daily chunks
    daily_gdd = 0.0
    chunk_size = 24
    for i in range(0, len(temps), chunk_size):
        chunk = temps[i:i+chunk_size]
        if chunk:
            t_max = max(chunk)
            t_min = min(chunk)
            t_mean = (t_max + t_min) / 2.0
            daily_gdd += max(0.0, t_mean - crop_base_temp_c)
            
    # Group continuous spray windows
    spray_windows: List[SprayWindow] = []
    current_window: List[Dict[str, Any]] = []
    
    for candidate in spray_candidates:
        if candidate["is_sprayable"] and candidate["suitability"] >= 65.0:
            current_window.append(candidate)
        else:
            if len(current_window) >= 2:
                start_ts = current_window[0]["timestamp"]
                end_ts = current_window[-1]["timestamp"]
                avg_score = sum(c["suitability"] for c in current_window) / len(current_window)
                spray_windows.append(SprayWindow(
                    start_time=start_ts,
                    end_time=end_ts,
                    duration_hours=len(current_window),
                    suitability_score=round(avg_score, 1),
                    limiting_factor="Optimal spray conditions",
                    recommended_action="Execute scheduled pesticide/fungicide/fertilizer foliar application"
                ))
            current_window = []
            
    if len(current_window) >= 2:
        start_ts = current_window[0]["timestamp"]
        end_ts = current_window[-1]["timestamp"]
        avg_score = sum(c["suitability"] for c in current_window) / len(current_window)
        spray_windows.append(SprayWindow(
            start_time=start_ts,
            end_time=end_ts,
            duration_hours=len(current_window),
            suitability_score=round(avg_score, 1),
            limiting_factor="Optimal spray conditions",
            recommended_action="Execute scheduled pesticide/fungicide/fertilizer foliar application"
        ))
        
    mean_vpd = sum(vpd_values) / len(vpd_values) if vpd_values else 0.0
    max_vpd = max(vpd_values) if vpd_values else 0.0
    inversion_risk = len(inversion_deltas) >= 3
    mean_inversion_str = sum(inversion_deltas) / len(inversion_deltas) if inversion_deltas else 0.0
    frost_risk = min(temps) <= 2.0 if temps else False
    
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    
    indices = MicroclimateIndices(
        mean_vpd_kpa=round(mean_vpd, 3),
        max_vpd_kpa=round(max_vpd, 3),
        thermal_inversion_risk=inversion_risk,
        inversion_strength_c=round(mean_inversion_str, 2),
        frost_risk=frost_risk,
        gdd_accumulated_c_days=round(daily_gdd, 2),
        heat_stress_hours=heat_stress_hours,
        spray_windows=spray_windows
    )
    
    return {
        "indices": indices.model_dump(),
        "execution_time_ms": round(elapsed_ms, 2)
    }
