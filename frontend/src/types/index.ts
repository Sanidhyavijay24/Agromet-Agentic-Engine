/**
 * @file index.ts
 * @description Core TypeScript contracts and schemas for A²E Engineering Blueprint Studio
 * @module frontend/types
 */

export interface CoordinatePreset {
  id: string;
  name: string;
  district: string;
  lat: number;
  lon: number;
  elevation_m: number;
  terrain_type: "Valley Basin" | "Rocky Ridge" | "Arid Plain" | "Sand Dune Basin";
  description: string;
}

export interface TerrainFeatures {
  elevation_m: number;
  slope_deg: number;
  aspect_deg: number;
  tpi_scale: number;
  tpi_classification: string;
  vpd_kpa: number;
  solar_ghi_w_m2: number;
}

export interface DiurnalPoint {
  hour: number;
  time_label: string;
  coarse_baseline_temp_c: number;
  downscaled_temp_c: number;
  delta_c: number;
  downscaled_rh_pct: number;
  vpd_kpa: number;
  is_inversion: boolean;
  is_safe_spray: boolean;
}

export interface ReActStep {
  step_number: number;
  thought: string;
  tool_name: "downscaler_tool" | "indices_tool" | "agronomy_tool" | "soil_tool";
  tool_args: Record<string, string | number | boolean>;
  observation: string;
  duration_ms: number;
}

export interface AdvisoryActionPlan {
  advisory_id: string;
  timestamp: string;
  location_name: string;
  lat: number;
  lon: number;
  crop: string;
  growth_stage: string;
  risk_level: "LOW" | "MODERATE" | "HIGH" | "CRITICAL";
  advisory_en: string;
  advisory_hi: string;
  spray_window_recommendation: string;
  spray_safe: boolean;
  irrigation_guidance: string;
  frost_risk: boolean;
  heat_stress_risk: boolean;
  steps: ReActStep[];
  latency_ms: number;
  model_source: string;
}

export interface ScenarioPreset {
  id: string;
  title: string;
  crop: string;
  stage: string;
  weather_condition: string;
  question: string;
  highlight_physics: string;
}

export interface ZoneFleetItem {
  zone_id: number;
  zone_name: string;
  region: string;
  raw_rmse_c: number;
  ml_rmse_c: number;
  error_drop_pct: number;
  r2_score: number;
  is_flagship_live: boolean;
  model_size_mb: number;
  trees_count: number;
  kaggle_url: string;
}

export interface SystemHealth {
  status: "ONLINE" | "STANDALONE" | "DEGRADED";
  fastapi_online: boolean;
  flagship_zone_loaded: boolean;
  model_size_mb: number;
  zone_name: string;
  tree_count: number;
  latency_ms: number;
  server_timestamp: string;
}
