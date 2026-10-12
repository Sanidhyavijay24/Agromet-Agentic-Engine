/**
 * @file CoordinatePicker.tsx
 * @description Rajasthan Zone XIV coordinate selector with terrain microclimate presets
 * @module frontend/components/section_downscaler
 */

import React from "react";
import { Compass, MapPin } from "lucide-react";
import { COORDINATE_PRESETS } from "../../services/api";
import { CoordinatePreset } from "../../types";

interface CoordinatePickerProps {
  selectedPreset: CoordinatePreset;
  onSelectPreset: (preset: CoordinatePreset) => void;
  lat: number;
  lon: number;
  onChangeLat: (lat: number) => void;
  onChangeLon: (lon: number) => void;
}

export const CoordinatePicker: React.FC<CoordinatePickerProps> = ({
  selectedPreset,
  onSelectPreset,
  lat,
  lon,
  onChangeLat,
  onChangeLon,
}) => {
  return (
    <div className="draft-card" style={{ marginBottom: "20px" }}>
      <div className="draft-card-header">
        <div className="draft-card-title" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <MapPin size={14} style={{ color: "var(--c-curve-ml)" }} />
          <span>Spatial Coordinate & Microclimate Target (Zone XIV)</span>
        </div>
        <span className="badge-tag safe">Western Dry Region</span>
      </div>

      <div style={{ display: "flex", gap: "8px", flexWrap: "wrap", marginBottom: "16px" }}>
        {COORDINATE_PRESETS.map((preset) => {
          const isActive = preset.id === selectedPreset.id;
          return (
            <button
              key={preset.id}
              onClick={() => {
                onSelectPreset(preset);
                onChangeLat(preset.lat);
                onChangeLon(preset.lon);
              }}
              className={`btn-paper ${isActive ? "active" : ""}`}
              style={{ fontSize: "11.5px" }}
            >
              <strong>{preset.district}:</strong> {preset.name.replace(/.*\(|\)/g, "")}
            </button>
          );
        })}
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: "12px", alignItems: "center" }}>
        <div>
          <label style={{ display: "block", fontSize: "11px", fontWeight: 700, color: "var(--c-ink-secondary)", marginBottom: "4px", fontFamily: "var(--font-mono)" }}>
            LATITUDE (°N)
          </label>
          <input
            type="number"
            step="0.0001"
            value={lat}
            onChange={(e) => onChangeLat(parseFloat(e.target.value) || 26.0)}
            className="tabular-nums"
            style={{
              width: "100%",
              padding: "8px 10px",
              border: "1px solid var(--c-paper-border-strong)",
              borderRadius: "var(--radius-xs)",
              background: "var(--c-paper-sheet)",
              color: "var(--c-ink-heading)",
              fontSize: "13px",
              fontWeight: 600,
            }}
          />
        </div>

        <div>
          <label style={{ display: "block", fontSize: "11px", fontWeight: 700, color: "var(--c-ink-secondary)", marginBottom: "4px", fontFamily: "var(--font-mono)" }}>
            LONGITUDE (°E)
          </label>
          <input
            type="number"
            step="0.0001"
            value={lon}
            onChange={(e) => onChangeLon(parseFloat(e.target.value) || 73.0)}
            className="tabular-nums"
            style={{
              width: "100%",
              padding: "8px 10px",
              border: "1px solid var(--c-paper-border-strong)",
              borderRadius: "var(--radius-xs)",
              background: "var(--c-paper-sheet)",
              color: "var(--c-ink-heading)",
              fontSize: "13px",
              fontWeight: 600,
            }}
          />
        </div>

        <div style={{ background: "var(--c-paper-sheet)", padding: "10px 14px", border: "1px dashed var(--c-paper-border-strong)", borderRadius: "var(--radius-xs)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", color: "var(--c-ink-muted)", fontFamily: "var(--font-mono)" }}>
            <Compass size={12} />
            <span>TERRAIN CLASSIFICATION</span>
          </div>
          <div style={{ fontSize: "13px", fontWeight: 750, color: "var(--c-ink-heading)", marginTop: "2px" }}>
            {selectedPreset.terrain_type} ({selectedPreset.elevation_m}m ASL)
          </div>
        </div>
      </div>

      <p style={{ margin: "12px 0 0 0", fontSize: "12.5px", color: "var(--c-ink-secondary)", fontStyle: "italic", borderLeft: "2px solid var(--c-curve-ml)", paddingLeft: "8px" }}>
        {selectedPreset.description}
      </p>
    </div>
  );
};
