/**
 * @file TerrainFeatureBar.tsx
 * @description Physics feature chipset inspector derived from 30m USGS SRTM DEM rasters
 * @module frontend/components/section_downscaler
 */

import React from "react";
import { Droplets, Mountain, Navigation, Sun, TrendingUp } from "lucide-react";
import { TerrainFeatures } from "../../types";

interface TerrainFeatureBarProps {
  features: TerrainFeatures;
}

export const TerrainFeatureBar: React.FC<TerrainFeatureBarProps> = ({ features }) => {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "12px", marginBottom: "20px" }}>
      <div className="draft-card" style={{ padding: "14px 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--c-ink-muted)", fontSize: "11px", fontFamily: "var(--font-mono)" }}>
          <Mountain size={13} style={{ color: "var(--c-ink-heading)" }} />
          <span>30m SRTM ELEVATION</span>
        </div>
        <div style={{ marginTop: "4px" }}>
          <span className="metric-big tabular-nums">{features.elevation_m}</span>
          <span className="metric-unit">m ASL</span>
        </div>
        <div style={{ fontSize: "10.5px", color: "var(--c-ink-muted)", marginTop: "2px" }}>
          USGS 1-arc-sec raster
        </div>
      </div>

      <div className="draft-card" style={{ padding: "14px 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--c-ink-muted)", fontSize: "11px", fontFamily: "var(--font-mono)" }}>
          <TrendingUp size={13} style={{ color: "var(--c-ink-heading)" }} />
          <span>HORN'S SLOPE</span>
        </div>
        <div style={{ marginTop: "4px" }}>
          <span className="metric-big tabular-nums">{features.slope_deg.toFixed(1)}</span>
          <span className="metric-unit">deg</span>
        </div>
        <div style={{ fontSize: "10.5px", color: "var(--c-ink-muted)", marginTop: "2px" }}>
          3x3 finite difference
        </div>
      </div>

      <div className="draft-card" style={{ padding: "14px 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--c-ink-muted)", fontSize: "11px", fontFamily: "var(--font-mono)" }}>
          <Navigation size={13} style={{ color: "var(--c-ink-heading)" }} />
          <span>ASPECT AZIMUTH</span>
        </div>
        <div style={{ marginTop: "4px" }}>
          <span className="metric-big tabular-nums">{features.aspect_deg.toFixed(0)}</span>
          <span className="metric-unit">° SSW</span>
        </div>
        <div style={{ fontSize: "10.5px", color: "var(--c-ink-muted)", marginTop: "2px" }}>
          Solar face orientation
        </div>
      </div>

      <div className="draft-card" style={{ padding: "14px 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--c-ink-muted)", fontSize: "11px", fontFamily: "var(--font-mono)" }}>
          <Sun size={13} style={{ color: "var(--c-ink-heading)" }} />
          <span>TOPOGRAPHIC INDEX (TPI)</span>
        </div>
        <div style={{ marginTop: "4px" }}>
          <span className="metric-big tabular-nums" style={{ color: features.tpi_scale < 0 ? "#2563EB" : "var(--c-ink-heading)" }}>
            {features.tpi_scale > 0 ? `+${features.tpi_scale.toFixed(1)}` : features.tpi_scale.toFixed(1)}
          </span>
          <span className="metric-unit">σ</span>
        </div>
        <div style={{ fontSize: "10.5px", color: "var(--c-ink-secondary)", marginTop: "2px", fontWeight: 600 }}>
          {features.tpi_classification}
        </div>
      </div>

      <div className="draft-card" style={{ padding: "14px 16px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--c-ink-muted)", fontSize: "11px", fontFamily: "var(--font-mono)" }}>
          <Droplets size={13} style={{ color: "var(--c-curve-ml)" }} />
          <span>VAPOUR DEFICIT (VPD)</span>
        </div>
        <div style={{ marginTop: "4px" }}>
          <span className="metric-big tabular-nums">{features.vpd_kpa.toFixed(2)}</span>
          <span className="metric-unit">kPa</span>
        </div>
        <div style={{ fontSize: "10.5px", color: "var(--c-ink-muted)", marginTop: "2px" }}>
          Magnus saturation vapor
        </div>
      </div>
    </div>
  );
};
