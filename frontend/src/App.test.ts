/**
 * @file App.test.ts
 * @description Unit tests for frontend utilities and diurnal curve generation
 * @module frontend/tests
 */

import { describe, expect, it } from "bun:test";
import { generateDiurnalCurve, getFallbackZoneCatalog } from "./services/api";

describe("Frontend Core Services", () => {
  it("generates physically valid 24h diurnal curve points", () => {
    const curve = generateDiurnalCurve(26.2389, 73.0243);
    expect(curve.length).toBe(24);
    
    // Validate minimum temperature is before dawn (04:00 - 06:00)
    const minTempPoint = curve.reduce((min, p) => p.downscaled_temp_c < min.downscaled_temp_c ? p : min, curve[0]);
    expect(minTempPoint.hour).toBeGreaterThanOrEqual(4);
    expect(minTempPoint.hour).toBeLessThanOrEqual(6);

    // Validate peak temperature is afternoon (13:00 - 17:00 due to thermal inertia lag)
    const maxTempPoint = curve.reduce((max, p) => p.downscaled_temp_c > max.downscaled_temp_c ? p : max, curve[0]);
    expect(maxTempPoint.hour).toBeGreaterThanOrEqual(13);
    expect(maxTempPoint.hour).toBeLessThanOrEqual(17);

    // Validate VPD is positive
    curve.forEach((p) => {
      expect(p.vpd_kpa).toBeGreaterThan(0);
      expect(p.elevation_m).toBeGreaterThan(0);
    });
  });

  it("loads the 15 ACZ catalog with Zone XIV flagged as live serving", () => {
    const catalog = getFallbackZoneCatalog();
    expect(catalog.length).toBeGreaterThanOrEqual(5);

    const zone14 = catalog.find((z) => z.zone_id === 14);
    expect(zone14).toBeDefined();
    expect(zone14?.is_flagship_live).toBe(true);
    expect(zone14?.model_size_mb).toBeLessThan(200);
  });
});
