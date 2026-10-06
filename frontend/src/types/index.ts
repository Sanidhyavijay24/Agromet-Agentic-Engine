/**
 * @file index.ts
 * @description Frontend TypeScript types matching backend Pydantic contracts
 * @module frontend/types
 */

export interface SprayWindow {
  start_time: string;
  end_time: string;
  duration_hours: number;
  suitability_score: number;
  limiting_factor: string;
  recommended_action: string;
}

export interface MicroclimateIndices {
  mean_vpd_kpa: number;
  max_vpd_kpa: number;
  thermal_inversion_risk: boolean;
  inversion_strength_c: number;
  frost_risk: boolean;
  gdd_accumulated_c_days: number;
  heat_stress_hours: number;
  spray_windows: SprayWindow[];
}

export interface AgrometActionPlan {
  summary_headline: string;
  verdict_category: string;
  irrigation_advice: string;
  spray_recommendation: string;
  crop_specific_protection: string;
  vernacular_hindi: string;
  english_summary: string;
}

export interface ToolExecution {
  tool_name: string;
  parameters: Record<string, unknown>;
  result_summary: string;
  execution_time_ms: number;
}

export interface AgentAdvisoryResponse {
  status: string;
  location: Record<string, unknown>;
  crop: string;
  crop_stage: string;
  indices: MicroclimateIndices;
  action_plan: AgrometActionPlan;
  agent_trace: ToolExecution[];
  llm_model_used: string;
  total_latency_ms: number;
}

export interface ZoneCatalogItem {
  zone_id: number;
  code: string;
  name: string;
  states: string[];
  coverage_area_sqkm: number;
  lat_min: number;
  lat_max: number;
  lon_min: number;
  lon_max: number;
  center_lat: number;
  center_lon: number;
  is_flagship_live: boolean;
  model_filename: string;
  model_size_mb: number;
  rmse_reduction_percent: number;
  r2_score: number;
  kaggle_hub_url: string;
  description: string;
}

export interface HourlyForecastPoint {
  hour: number;
  coarse_temp_c: number;
  downscaled_temp_c: number;
  coarse_rh_pct: number;
  downscaled_rh_pct: number;
  wind_speed_kmh: number;
  solar_ghi_wm2: number;
  elevation_m: number;
  vpd_kpa: number;
  delta_t_c: number;
  inversion_risk: string;
  spray_suitability: string;
}

export interface PresetLocation {
  id: string;
  name: string;
  district: string;
  lat: number;
  lon: number;
  crop: string;
  stage: string;
  soil: string;
}
