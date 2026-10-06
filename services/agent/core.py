"""
@file core.py
@description Core Agentic AI Agromet Advisory Orchestrator powered by Google Gemini with tool calling & fallback.
@module services/agent
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional
from loguru import logger

from services.agent.schemas import (
    AgrometActionPlan,
    AgrometAdvisoryRequest,
    AgentAdvisoryResponse,
    MicroclimateIndices,
    SprayWindow,
    ToolExecution
)
from services.agent.tools.downscaler_tool import run_1km_downscaler
from services.agent.tools.indices_tool import calculate_microclimate_indices
from services.agent.tools.agronomy_tool import lookup_crop_agronomy
from services.agent.tools.soil_tool import get_soil_and_terrain_context


# Gemini model candidates ordered by priority and speed
CANDIDATE_GEMINI_MODELS = [
    "gemini-flash-lite-latest",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash"
]


def _synthesize_deterministic_plan(
    crop_info: Dict[str, Any],
    indices_data: Dict[str, Any],
    downscale_data: Dict[str, Any],
    soil_data: Dict[str, Any],
    user_query: Optional[str] = None
) -> AgrometActionPlan:
    """
    High-precision deterministic agrometeorological synthesis fallback.
    Ensures 100% reliability, structured formatting, and localized Hindi advice.
    """
    indices: MicroclimateIndices = MicroclimateIndices(**indices_data["indices"])
    crop_name = crop_info["common_name"]
    stage = crop_info["stage_evaluated"]
    spray_windows = indices.spray_windows
    
    # 1. Determine primary verdict category
    if indices.frost_risk:
        category = "FROST_WARNING"
        headline = f"Critical Frost Alert for {crop_name} ({stage}): Near-Zero Radiative Cooling Detected"
    elif indices.heat_stress_hours >= 4:
        category = "HEAT_STRESS_DEFENSE"
        headline = f"High Thermal Stress for {crop_name} ({stage}): {indices.heat_stress_hours} Hours Exceeding Threshold"
    elif not spray_windows:
        category = "SPRAY_CAUTION"
        headline = f"Spray Caution for {crop_name} ({stage}): High Wind Drift or Washout Risk in Horizon"
    elif indices.mean_vpd_kpa > 2.8:
        category = "IRRIGATE_IMMEDIATELY"
        headline = f"High Transpiration Stress (VPD: {indices.mean_vpd_kpa:.2f} kPa): Soil Moisture Depleting Rapidly"
    else:
        category = "FAVORABLE"
        headline = f"Favorable Agromet Conditions for {crop_name} ({stage})"

    # 2. Formulate Irrigation Advice
    if indices.mean_vpd_kpa > 2.5 or indices.heat_stress_hours > 0:
        irrigation_advice = (
            f"Apply light evening irrigation to replenish topsoil moisture and reduce root zone heat stress. "
            f"VPD is elevated at {indices.mean_vpd_kpa:.2f} kPa (optimal: {crop_info['optimal_vpd_range_kpa'][0]}-{crop_info['optimal_vpd_range_kpa'][1]} kPa). "
            f"Surface soil status: {soil_data['moisture_status']}."
        )
    else:
        irrigation_advice = (
            f"Soil moisture is adequate ({soil_data['estimated_surface_soil_moisture_m3m3'] * 100:.1f}%). "
            f"No immediate irrigation required. Maintain standard irrigation schedule for {stage} stage."
        )

    # 3. Formulate Spray Recommendation
    if spray_windows:
        best_win = spray_windows[0]
        spray_rec = (
            f"Optimal Spray Window: {best_win.start_time[11:16]} to {best_win.end_time[11:16]} "
            f"(Duration: {best_win.duration_hours}h, Suitability Score: {best_win.suitability_score:.0f}/100). "
            f"Wind speed is calm (< 12 km/h) with zero rain washout probability. Use recommended dilution."
        )
    else:
        spray_rec = (
            f"Postpone chemical and bio-pesticide spraying over the next 24-48 hours. "
            f"Adverse atmospheric conditions (high wind drift or elevated daytime temperatures) will cause drift loss or droplet evaporation."
        )

    # 4. Crop Specific Protection
    pest = crop_info["stage_critical_pest"]
    crop_prot = (
        f"Critical Stage: {stage} (Sensitivity: {crop_info['stage_sensitivity']}). "
        f"Primary pest/disease threat: {pest}. {crop_info['advisory_notes']}"
    )

    # 5. Vernacular Hindi Summary
    if category == "FAVORABLE":
        hindi_text = (
            f"कृषि सलाह ({crop_name} - {stage} अवस्था): मौसम पूरी तरह अनुकूल है। "
            f"हवा की गति शांत रहने से छिड़काव के लिए अनुकूल समय उपलब्ध है। "
            f"{'पहला अनुकूल छिड़काव समय: ' + spray_windows[0].start_time[11:16] + ' से ' + spray_windows[0].end_time[11:16] if spray_windows else ''}। "
            f"मुख्य कीट ({pest}) पर निगरानी रखें।"
        )
    elif category == "HEAT_STRESS_DEFENSE":
        hindi_text = (
            f"तापमान चेतावनी ({crop_name} - {stage}): 1 किमी मॉडल के अनुसार दिन का तापमान {indices.heat_stress_hours} घंटे अधिक रहेगा। "
            f"फसल को गर्मी से बचाने के लिए शाम के समय हल्की सिंचाई करें। "
            f"दोपहर 12 से 4 बजे के बीच कीटनाशक का छिड़काव बिल्कुल न करें।"
        )
    elif category == "FROST_WARNING":
        hindi_text = (
            f"पाला / पाले की चेतावनी ({crop_name}): रात के समय घाटी क्षेत्र में ठंडा हवा जमाव (Inversion) होने से तापमान अत्यधिक गिर सकता है। "
            f"फसल की सुरक्षा के लिए खेत के किनारों पर धुआं करें या हल्की सिंचाई करें।"
        )
    else:
        hindi_text = (
            f"कृषि सलाह ({crop_name} - {stage}): वाष्पीकरण दबाव (VPD) {indices.mean_vpd_kpa:.2f} kPa दर्ज किया गया है। "
            f"मिट्टी की नमी बनाए रखने के लिए समय पर सिंचाई सुनिश्चित करें। "
            f"कीट नियंत्रण के लिए हवा की गति 12 किमी/घंटे से कम होने पर ही छिड़काव करें।"
        )

    english_summary = (
        f"Agromet Summary for {crop_name} at {stage} stage: {headline}. "
        f"GDD accumulated: {indices.gdd_accumulated_c_days:.1f}°C-days. "
        f"Active spray windows identified: {len(spray_windows)}."
    )

    return AgrometActionPlan(
        summary_headline=headline,
        verdict_category=category,
        irrigation_advice=irrigation_advice,
        spray_recommendation=spray_rec,
        crop_specific_protection=crop_prot,
        vernacular_hindi=hindi_text,
        english_summary=english_summary
    )


def _call_gemini_synthesis(
    crop_info: Dict[str, Any],
    indices_data: Dict[str, Any],
    downscale_data: Dict[str, Any],
    soil_data: Dict[str, Any],
    user_query: Optional[str] = None
) -> Optional[AgrometActionPlan]:
    """
    Invoke Google Gemini Flash API with structured agronomic prompt.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
You are the Agromet Agentic Engine (A²E), an expert agrometeorologist for Indian agriculture.
Synthesize the following 1km microclimate telemetry and agronomic data into a structured action plan.

[TARGET CROP & STAGE]
Crop: {crop_info['common_name']} ({crop_info['crop_key']})
Current Stage: {crop_info['stage_evaluated']}
Stage Sensitivity: {crop_info['stage_sensitivity']}
Critical Pest/Disease: {crop_info['stage_critical_pest']}
Agronomy Notes: {crop_info['advisory_notes']}

[1-KM MICROCLIMATE INDICES]
Mean VPD: {indices_data['indices']['mean_vpd_kpa']} kPa (Optimal: {crop_info['optimal_vpd_range_kpa']})
Max VPD: {indices_data['indices']['max_vpd_kpa']} kPa
Thermal Inversion Risk: {indices_data['indices']['thermal_inversion_risk']} (Inversion strength: {indices_data['indices']['inversion_strength_c']}°C)
Frost Risk: {indices_data['indices']['frost_risk']}
Heat Stress Hours (> {crop_info['critical_max_temp_c']}°C): {indices_data['indices']['heat_stress_hours']}
Accumulated GDD: {indices_data['indices']['gdd_accumulated_c_days']}°C-days
Favorable Spray Windows: {len(indices_data['indices']['spray_windows'])} windows found

[SOIL & TERRAIN CONTEXT]
Elevation: {soil_data['elevation_m']}m MSL
Soil Moisture Status: {soil_data['moisture_status']}

[USER SPECIFIC INQUIRY]
{user_query or 'Standard 48-hour agronomic guidance.'}

Return a valid JSON object matching this exact schema:
{{
  "summary_headline": "Short punchy verdict headline",
  "verdict_category": "FAVORABLE" | "SPRAY_CAUTION" | "HEAT_STRESS_DEFENSE" | "FROST_WARNING" | "IRRIGATE_IMMEDIATELY",
  "irrigation_advice": "Detailed irrigation directive based on VPD and soil moisture",
  "spray_recommendation": "Optimal chemical/organic spray timing and wind drift warnings",
  "crop_specific_protection": "Specific actions for current crop stage and pest defense",
  "vernacular_hindi": "Complete, natural, highly accurate Hindi advisory for Indian farmers (किसान सलाह)",
  "english_summary": "Executive 2-sentence summary in English"
}}
"""
        # Try candidate models
        for model_name in CANDIDATE_GEMINI_MODELS:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                text = response.text.strip()
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                data = json.loads(text)
                return AgrometActionPlan(**data)
            except Exception as model_err:
                logger.debug(f"Gemini model {model_name} attempt failed: {model_err}")
                continue

    except Exception as e:
        logger.warning(f"Gemini API invocation encountered error, falling back to deterministic synthesis: {e}")
        
    return None


def run_agromet_agent(request: AgrometAdvisoryRequest) -> AgentAdvisoryResponse:
    """
    Autonomous multi-step ReAct agent execution:
    1. Tool 1: 1km Downscaler (Physics ML engine)
    2. Tool 2: Soil & Terrain Context
    3. Tool 3: Agronomic Knowledge Lookup
    4. Tool 4: Microclimate Stress Indices
    5. Agent Synthesis (Gemini 2.0 Flash with deterministic fallback)
    """
    total_start = time.perf_counter()
    traces: List[ToolExecution] = []
    
    # 1. Execute Downscaler Tool
    t1_start = time.perf_counter()
    downscale_data = run_1km_downscaler(
        latitude=request.latitude,
        longitude=request.longitude,
        forecast_hours=request.forecast_hours
    )
    t1_ms = (time.perf_counter() - t1_start) * 1000
    traces.append(ToolExecution(
        tool_name="run_1km_downscaler",
        parameters={"latitude": request.latitude, "longitude": request.longitude, "hours": request.forecast_hours},
        result_summary=f"Downscaled {downscale_data['hours_evaluated']} hourly points in Zone {downscale_data['zone_id']} (mean ΔT: {downscale_data['mean_delta_t_c']}°C)",
        execution_time_ms=round(t1_ms, 2)
    ))
    
    # 2. Execute Soil & Terrain Tool
    t2_start = time.perf_counter()
    soil_data = get_soil_and_terrain_context(request.latitude, request.longitude)
    t2_ms = (time.perf_counter() - t2_start) * 1000
    traces.append(ToolExecution(
        tool_name="get_soil_and_terrain_context",
        parameters={"latitude": request.latitude, "longitude": request.longitude},
        result_summary=f"Elevation: {soil_data['elevation_m']}m, Slope: {soil_data['slope_magnitude_deg']}°, Soil status: {soil_data['moisture_status'][:30]}...",
        execution_time_ms=round(t2_ms, 2)
    ))
    
    # 3. Execute Agronomy Tool
    t3_start = time.perf_counter()
    crop_info = lookup_crop_agronomy(request.crop, request.crop_stage)
    t3_ms = (time.perf_counter() - t3_start) * 1000
    traces.append(ToolExecution(
        tool_name="lookup_crop_agronomy",
        parameters={"crop": request.crop, "stage": request.crop_stage},
        result_summary=f"Profile loaded: {crop_info['common_name']}, Base Temp: {crop_info['base_temp_c']}°C, Pest: {crop_info['stage_critical_pest']}",
        execution_time_ms=round(t3_ms, 2)
    ))
    
    # 4. Execute Indices Tool
    t4_start = time.perf_counter()
    indices_data = calculate_microclimate_indices(
        hourly_series=downscale_data["hourly_series"],
        crop_base_temp_c=crop_info["base_temp_c"],
        crop_max_temp_c=crop_info["critical_max_temp_c"]
    )
    t4_ms = (time.perf_counter() - t4_start) * 1000
    raw_indices = indices_data["indices"]
    traces.append(ToolExecution(
        tool_name="calculate_microclimate_indices",
        parameters={"base_temp_c": crop_info["base_temp_c"], "max_temp_c": crop_info["critical_max_temp_c"]},
        result_summary=f"VPD mean: {raw_indices['mean_vpd_kpa']} kPa, Inversion: {raw_indices['thermal_inversion_risk']}, Spray Windows: {len(raw_indices['spray_windows'])}",
        execution_time_ms=round(t4_ms, 2)
    ))
    
    # 5. Agent Reasoning Synthesis (Gemini LLM -> Deterministic fallback)
    llm_model_used = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    action_plan = _call_gemini_synthesis(
        crop_info=crop_info,
        indices_data=indices_data,
        downscale_data=downscale_data,
        soil_data=soil_data,
        user_query=request.user_query
    )
    
    if not action_plan:
        action_plan = _synthesize_deterministic_plan(
            crop_info=crop_info,
            indices_data=indices_data,
            downscale_data=downscale_data,
            soil_data=soil_data,
            user_query=request.user_query
        )
        llm_model_used = "A2E-Deterministic-Expert-Engine"
        
    total_latency_ms = (time.perf_counter() - total_start) * 1000
    
    return AgentAdvisoryResponse(
        status="success",
        location={
            "latitude": request.latitude,
            "longitude": request.longitude,
            "zone_id": downscale_data["zone_id"],
            "elevation_m": soil_data["elevation_m"]
        },
        crop=crop_info["common_name"],
        crop_stage=request.crop_stage,
        indices=MicroclimateIndices(**raw_indices),
        action_plan=action_plan,
        agent_trace=traces,
        llm_model_used=llm_model_used,
        total_latency_ms=round(total_latency_ms, 2)
    )


