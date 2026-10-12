/**
 * @file App.test.ts
 * @description Comprehensive unit tests for Phase 1 & Phase 2 frontend models and services
 * @module frontend/tests
 */

import { describe, expect, it } from "bun:test";
import React from "react";
import { renderToString } from "react-dom/server";
import { App } from "./App";
import {
  COORDINATE_PRESETS,
  PAN_INDIA_FLEET,
  SCENARIO_PRESETS,
  calculateTerrainFeatures,
  generateDiurnalCurve,
  generateMockAdvisory,
} from "./services/api";

describe("Phase 1 & 2: Frontend Design Tokens & Services", () => {
  it("renders the root application shell with blueprint header, downscaler section, agent section, and fleet section", () => {
    const html = renderToString(React.createElement(App));
    expect(html).toContain("Agromet Agentic Engine");
    expect(html).toContain("Microclimate Studio");
    expect(html).toContain("1 km² Microclimate Downscaler");
    expect(html).toContain("24-Hour Diurnal Microclimate Profile");
    expect(html).toContain("ReAct Agromet Advisory Terminal");
    expect(html).toContain("Bilingual Agrometeorological Advisory");
    expect(html).toContain("Empirical Model Leaderboard &amp; Dataset Hub");
    expect(html).toContain("Western Dry Region");
  });

  it("calculates realistic terrain features for Rajasthan coordinates", () => {
    const jodhpur = COORDINATE_PRESETS[0];
    const terrain = calculateTerrainFeatures(jodhpur.lat, jodhpur.lon);

    expect(terrain.elevation_m).toBe(231);
    expect(terrain.slope_deg).toBeGreaterThanOrEqual(0);
    expect(terrain.tpi_classification).toContain("Valley");
    expect(terrain.vpd_kpa).toBeGreaterThan(0);
  });

  it("generates an authentic 24-hour diurnal microclimate curve", () => {
    const jodhpur = COORDINATE_PRESETS[0];
    const curve = generateDiurnalCurve(jodhpur.lat, jodhpur.lon);

    expect(curve.length).toBe(24);

    // Verify minimum temperature is at dawn (04:00 - 06:00)
    const minPoint = curve.reduce((min, p) => (p.downscaled_temp_c < min.downscaled_temp_c ? p : min), curve[0]);
    expect(minPoint.hour).toBeGreaterThanOrEqual(4);
    expect(minPoint.hour).toBeLessThanOrEqual(6);

    // Verify peak temperature is afternoon (14:00 - 17:00)
    const maxPoint = curve.reduce((max, p) => (p.downscaled_temp_c > max.downscaled_temp_c ? p : max), curve[0]);
    expect(maxPoint.hour).toBeGreaterThanOrEqual(14);
    expect(maxPoint.hour).toBeLessThanOrEqual(17);

    // Verify nocturnal cold-air inversion in valley
    const dawnInversions = curve.filter((p) => p.hour >= 4 && p.hour <= 6 && p.is_inversion);
    expect(dawnInversions.length).toBeGreaterThan(0);

    // Verify safe spray window is identified during morning hours
    const sprayWindows = curve.filter((p) => p.is_safe_spray);
    expect(sprayWindows.length).toBeGreaterThan(0);
  });

  it("generates structured ReAct agent advisory with bilingual Hindi/English guidance", () => {
    const frostScenario = SCENARIO_PRESETS[0];
    const advisory = generateMockAdvisory(frostScenario.id, 26.2389, 73.0243);

    expect(advisory.steps.length).toBe(3);
    expect(advisory.steps[0].tool_name).toBe("downscaler_tool");
    expect(advisory.steps[1].tool_name).toBe("indices_tool");
    expect(advisory.steps[2].tool_name).toBe("agronomy_tool");

    expect(advisory.advisory_en.length).toBeGreaterThan(20);
    expect(advisory.advisory_hi).toContain("चेतावनी");
    expect(advisory.risk_level).toBe("CRITICAL");
    expect(advisory.frost_risk).toBe(true);
  });

  it("validates the Pan-India 15 ACZ fleet catalog with Zone XIV flagged live", () => {
    expect(PAN_INDIA_FLEET.length).toBe(15);

    const zone14 = PAN_INDIA_FLEET.find((z) => z.zone_id === 14);
    expect(zone14).toBeDefined();
    expect(zone14?.is_flagship_live).toBe(true);
    expect(zone14?.model_size_mb).toBe(195.0);
    expect(zone14?.error_drop_pct).toBe(40.2);
    expect(zone14?.trees_count).toBe(5555);

    // Verify mountainous high-gain zones
    const zone1 = PAN_INDIA_FLEET.find((z) => z.zone_id === 1);
    expect(zone1?.error_drop_pct).toBe(86.9);
  });
});
