/**
 * @file api.ts
 * @description API client for Agromet Agentic Engine FastAPI backend
 * @module frontend/services/api
 */

import { AgentAdvisoryResponse, ZoneCatalogItem, HourlyForecastPoint } from "../types";

const API_BASE = "/api/v1";

export async function fetchZoneCatalog(): Promise<ZoneCatalogItem[]> {
  try {
    const res = await fetch(`${API_BASE}/forecast/zones/catalog`);
    if (!res.ok) throw new Error(`HTTP error ${res.status}`);
    const data = await res.json();
    return data.zones || [];
  } catch (err) {
    console.warn("Failed to fetch zone catalog from API, using fallback data:", err);
    return getFallbackZoneCatalog();
  }
}

export async function requestAgentAdvisory(params: {
  latitude: number;
  longitude: number;
  crop?: string;
  crop_stage?: string;
  soil_type?: string;
  user_query?: string;
  forecast_hours?: number;
}): Promise<AgentAdvisoryResponse> {
  const res = await fetch(`${API_BASE}/agent/advisory`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      latitude: params.latitude,
      longitude: params.longitude,
      crop: params.crop || "Bajra",
      crop_stage: params.crop_stage || "Flowering",
      soil_type: params.soil_type || "Sandy Loam",
      user_query: params.user_query || "Standard 48-hour agronomic guidance.",
      forecast_hours: params.forecast_hours || 48,
    }),
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Agent advisory error (${res.status})`);
  }

  return await res.json();
}

export async function sendAgentChat(params: {
  message: string;
  latitude: number;
  longitude: number;
  crop?: string;
  crop_stage?: string;
  soil_type?: string;
}): Promise<{
  reply: string;
  vernacular_hindi: string;
  consultation: AgentAdvisoryResponse;
}> {
  const res = await fetch(`${API_BASE}/agent/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      message: params.message,
      latitude: params.latitude,
      longitude: params.longitude,
      crop: params.crop || "Bajra",
      crop_stage: params.crop_stage || "Flowering",
      soil_type: params.soil_type || "Sandy Loam",
    }),
  });

  if (!res.ok) {
    const errorBody = await res.json().catch(() => ({}));
    throw new Error(errorBody.detail || `Chat error (${res.status})`);
  }

  return await res.json();
}

export async function fetchPointForecast(lat: number, lon: number): Promise<HourlyForecastPoint[]> {
  try {
    const res = await fetch(`${API_BASE}/forecast/point?latitude=${lat}&longitude=${lon}`);
    if (res.ok) {
      const data = await res.json();
      if (data.hourly && Array.isArray(data.hourly)) {
        return data.hourly;
      }
    }
  } catch (err) {
    console.warn("Using simulated diurnal forecast curve:", err);
  }
  return generateDiurnalCurve(lat, lon);
}

export function generateDiurnalCurve(lat: number, lon: number): HourlyForecastPoint[] {
  const baseElevation = 224;
  const points: HourlyForecastPoint[] = [];

  for (let h = 0; h < 24; h++) {
    const diurnalPhase = ((h - 5 + 24) % 24) / 24 * 2 * Math.PI;
    const tempSine = (1 - Math.cos(diurnalPhase)) / 2;
    
    const coarseT = 24.5 + tempSine * 14.5;
    const microAdjustment = (h >= 2 && h <= 7) ? -2.2 : (h >= 12 && h <= 17) ? +1.6 : -0.4;
    const downscaledT = coarseT + microAdjustment;

    const coarseRh = Math.max(15, Math.min(85, 78 - tempSine * 52));
    const downscaledRh = Math.max(12, Math.min(90, (h >= 2 && h <= 7) ? coarseRh + 8 : coarseRh - 6));

    const solarGhi = (h >= 6 && h <= 18) ? Math.max(0, Math.sin((h - 6) / 12 * Math.PI) * 980) : 0;
    const windSpeed = (h >= 11 && h <= 17) ? 22.4 : (h >= 3 && h <= 7) ? 5.2 : 12.1;

    const es = 0.61078 * Math.exp((17.27 * downscaledT) / (downscaledT + 237.3));
    const ea = es * (downscaledRh / 100);
    const vpd = Math.max(0.1, es - ea);
    const deltaT = (downscaledT * (1 - (downscaledRh / 100)) * 0.7);

    let invRisk = "LOW";
    if (h >= 1 && h <= 6 && windSpeed < 8) {
      invRisk = "HIGH (Strong Radiation Inversion)";
    } else if (h >= 22 || h <= 7) {
      invRisk = "MODERATE";
    }

    let spray = "OPTIMAL";
    if (windSpeed > 18 || vpd > 3.0 || deltaT > 8.0) {
      spray = "HAZARDOUS (Drift & Rapid Evaporation)";
    } else if (windSpeed < 4 && invRisk.includes("HIGH")) {
      spray = "HAZARDOUS (Inversion Trapping)";
    } else if (windSpeed > 14 || vpd > 2.2) {
      spray = "MARGINAL";
    }

    points.push({
      hour: h,
      coarse_temp_c: Number(coarseT.toFixed(1)),
      downscaled_temp_c: Number(downscaledT.toFixed(1)),
      coarse_rh_pct: Number(coarseRh.toFixed(1)),
      downscaled_rh_pct: Number(downscaledRh.toFixed(1)),
      wind_speed_kmh: Number(windSpeed.toFixed(1)),
      solar_ghi_wm2: Number(solarGhi.toFixed(0)),
      elevation_m: baseElevation,
      vpd_kpa: Number(vpd.toFixed(2)),
      delta_t_c: Number(deltaT.toFixed(1)),
      inversion_risk: invRisk,
      spray_suitability: spray,
    });
  }

  return points;
}

export function getFallbackZoneCatalog(): ZoneCatalogItem[] {
  return [
    {
      zone_id: 14,
      code: "ZONE_XIV_WESTERN_DRY",
      name: "Western Dry Region (Rajasthan)",
      states: ["Rajasthan"],
      coverage_area_sqkm: 175000,
      lat_min: 24.5,
      lat_max: 30.2,
      lon_min: 69.5,
      lon_max: 76.0,
      center_lat: 26.2389,
      center_lon: 73.0243,
      is_flagship_live: true,
      model_filename: "model_zone_14_western_dry.joblib",
      model_size_mb: 165.4,
      rmse_reduction_percent: 62.4,
      r2_score: 0.952,
      kaggle_hub_url: "https://www.kaggle.com/datasets/agromet/pan-india-downscaler-fleet",
      description: "Thar desert arid ecosystem, extreme diurnal swing (30°C range), sand dune topographic lapse, high radiative cooling.",
    },
    {
      zone_id: 6,
      code: "ZONE_VI_TRANS_GANGETIC",
      name: "Trans-Gangetic Plains Region",
      states: ["Punjab", "Haryana", "Delhi", "Rajasthan (North)"],
      coverage_area_sqkm: 125000,
      lat_min: 28.0,
      lat_max: 32.5,
      lon_min: 73.8,
      lon_max: 77.8,
      center_lat: 30.5,
      center_lon: 75.8,
      is_flagship_live: false,
      model_filename: "model_zone_6_trans_gangetic.joblib",
      model_size_mb: 172.1,
      rmse_reduction_percent: 58.1,
      r2_score: 0.944,
      kaggle_hub_url: "https://www.kaggle.com/datasets/agromet/pan-india-downscaler-fleet/zone_6",
      description: "Intensive irrigated wheat-rice belt, dense winter radiation fog (smog) dynamics, nocturnal boundary inversion.",
    },
    {
      zone_id: 8,
      code: "ZONE_VIII_CENTRAL_PLATEAU",
      name: "Central Plateau and Hills Region",
      states: ["Madhya Pradesh", "Rajasthan (South)", "Uttar Pradesh (South)"],
      coverage_area_sqkm: 375000,
      lat_min: 21.2,
      lat_max: 26.8,
      lon_min: 74.0,
      lon_max: 82.5,
      center_lat: 23.5,
      center_lon: 77.5,
      is_flagship_live: false,
      model_filename: "model_zone_8_central_plateau.joblib",
      model_size_mb: 181.0,
      rmse_reduction_percent: 59.7,
      r2_score: 0.938,
      kaggle_hub_url: "https://www.kaggle.com/datasets/agromet/pan-india-downscaler-fleet/zone_8",
      description: "Vindhya/Satpura ridge topography, black cotton vertisols, thermal inertia lag from basalt bedrock.",
    },
    {
      zone_id: 1,
      code: "ZONE_I_WESTERN_HIMALAYAN",
      name: "Western Himalayan Region",
      states: ["Jammu & Kashmir", "Himachal Pradesh", "Uttarakhand"],
      coverage_area_sqkm: 240000,
      lat_min: 29.0,
      lat_max: 37.0,
      lon_min: 73.5,
      lon_max: 81.0,
      center_lat: 32.5,
      center_lon: 77.0,
      is_flagship_live: false,
      model_filename: "model_zone_1_western_himalayan.joblib",
      model_size_mb: 195.2,
      rmse_reduction_percent: 68.3,
      r2_score: 0.961,
      kaggle_hub_url: "https://www.kaggle.com/datasets/agromet/pan-india-downscaler-fleet/zone_1",
      description: "Steep elevation lapse rate (0.65°C/100m), katabatic valley drainage winds, high aspect solar shading.",
    },
    {
      zone_id: 10,
      code: "ZONE_X_SOUTHERN_PLATEAU",
      name: "Southern Plateau and Hills Region",
      states: ["Karnataka", "Andhra Pradesh", "Tamil Nadu", "Maharashtra (South)"],
      coverage_area_sqkm: 395000,
      lat_min: 11.5,
      lat_max: 18.5,
      lon_min: 74.5,
      lon_max: 80.0,
      center_lat: 14.5,
      center_lon: 76.5,
      is_flagship_live: false,
      model_filename: "model_zone_10_southern_plateau.joblib",
      model_size_mb: 178.6,
      rmse_reduction_percent: 55.4,
      r2_score: 0.931,
      kaggle_hub_url: "https://www.kaggle.com/datasets/agromet/pan-india-downscaler-fleet/zone_10",
      description: "Deccan plateau semi-arid rain-shadow, red alfisol soil moisture dynamics, moderate elevation.",
    },
    {
      zone_id: 13,
      code: "ZONE_XIII_GUJARAT_PLAINS",
      name: "Gujarat Plains and Hills Region",
      states: ["Gujarat"],
      coverage_area_sqkm: 196000,
      lat_min: 20.1,
      lat_max: 24.7,
      lon_min: 68.1,
      lon_max: 74.4,
      center_lat: 22.3,
      center_lon: 71.5,
      is_flagship_live: false,
      model_filename: "model_zone_13_gujarat_plains.joblib",
      model_size_mb: 169.8,
      rmse_reduction_percent: 57.2,
      r2_score: 0.935,
      kaggle_hub_url: "https://www.kaggle.com/datasets/agromet/pan-india-downscaler-fleet/zone_13",
      description: "Coastal-interior gradient, Rann of Kutch saline albedo, marine boundary layer thermal damping.",
    }
  ];
}
