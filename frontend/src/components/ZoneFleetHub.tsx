/**
 * @file ZoneFleetHub.tsx
 * @description 15 Agro-Climatic Zones Pan-India Fleet Directory & Kaggle Model Hub
 * @module frontend/components/ZoneFleetHub
 */

import React, { useState, useEffect } from "react";
import {
  Database,
  ExternalLink,
  ShieldCheck,
  CheckCircle2,
  Cpu,
  Layers,
  Sparkles,
  BarChart3,
  Server,
} from "lucide-react";
import { ZoneCatalogItem } from "../types";
import { fetchZoneCatalog } from "../services/api";

export const ZoneFleetHub: React.FC = () => {
  const [zones, setZones] = useState<ZoneCatalogItem[]>([]);
  const [selectedZone, setSelectedZone] = useState<ZoneCatalogItem | null>(null);
  const [searchQuery, setSearchQuery] = useState<string>("");

  useEffect(() => {
    fetchZoneCatalog().then((data) => {
      setZones(data);
      if (data.length > 0) {
        setSelectedZone(data.find((z) => z.is_flagship_live) || data[0]);
      }
    });
  }, []);

  const filteredZones = zones.filter(
    (z) =>
      z.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      z.code.toLowerCase().includes(searchQuery.toLowerCase()) ||
      z.states.some((s) => s.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div>
      {/* Pan-India Fleet Architecture Overview */}
      <div className="panel" style={{ borderTop: "4px solid var(--c-gold)" }}>
        <div className="panel-header">
          <div className="panel-title">
            <Database size={18} style={{ color: "var(--c-gold)" }} />
            <span>PAN-INDIA 15 AGRO-CLIMATIC ZONE MODEL FLEET</span>
          </div>
          <span className="panel-tag">ICAR AGRO-ECOLOGICAL ARCHITECTURE</span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "16px", marginBottom: "16px" }}>
          <div className="index-card safe">
            <span className="index-label">Live In-Memory Serving</span>
            <span className="index-value" style={{ fontSize: "16px", color: "var(--c-linen)" }}>
              Zone XIV: Western Dry (Rajasthan)
            </span>
            <span className="index-subtext">Active 165MB Physics Model Loaded</span>
          </div>

          <div className="index-card">
            <span className="index-label">Pan-India Fleet Error Reduction</span>
            <span className="index-value" style={{ color: "var(--c-sage)" }}>+58.9% Macro Gain</span>
            <span className="index-subtext">Verified 10-Year Multi-Zone Spatial CV</span>
          </div>

          <div className="index-card">
            <span className="index-label">Fleet Storage Optimization</span>
            <span className="index-value" style={{ fontSize: "16px" }}>Kaggle Hub Distribution</span>
            <span className="index-subtext">Remaining 14 Zones Hosted on Kaggle</span>
          </div>

          <div className="index-card">
            <span className="index-label">Physical Covariates</span>
            <span className="index-value">28 Features</span>
            <span className="index-subtext">USGS 30m DEM + Regolith Memory</span>
          </div>
        </div>

        <p style={{ fontSize: "13px", color: "var(--c-sand)", lineHeight: 1.6, background: "var(--c-bg)", padding: "12px", border: "1px solid var(--c-border)" }}>
          💡 <strong>Deployment Footprint Strategy:</strong> To ensure high responsiveness and zero server memory bloat, the live production engine serves <strong>Zone XIV (Western Dry / Thar Desert)</strong> directly in memory (~165MB). Models and 10-year training Parquets for all other 14 ICAR zones are hosted on the Kaggle Model Hub with complete spatial routing metadata.
        </p>
      </div>

      {/* Zone Directory Search & Grid */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1.5fr", gap: "21px" }}>
        
        {/* Zone List Panel */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <Server size={16} style={{ color: "var(--c-gold)" }} />
              <span>ZONE CATALOG ({filteredZones.length})</span>
            </div>
          </div>

          <div className="control-group">
            <input
              type="text"
              className="input-text"
              placeholder="Search zones, states, or codes..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>

          <div style={{ maxHeight: "460px", overflowY: "auto", display: "flex", flexDirection: "column", gap: "6px" }}>
            {filteredZones.map((z) => {
              const isSelected = selectedZone?.zone_id === z.zone_id;
              return (
                <div
                  key={z.zone_id}
                  onClick={() => setSelectedZone(z)}
                  style={{
                    padding: "10px 12px",
                    background: isSelected ? "var(--c-surface-hover)" : "var(--c-bg)",
                    border: isSelected ? "1px solid var(--c-gold)" : "1px solid var(--c-border)",
                    cursor: "pointer",
                    transition: "all 0.15s",
                    borderLeft: z.is_flagship_live ? "4px solid var(--c-sage)" : isSelected ? "4px solid var(--c-gold)" : "1px solid var(--c-border)",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "3px" }}>
                    <span style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--c-gold)", fontWeight: 700 }}>
                      ZONE {z.zone_id}
                    </span>
                    {z.is_flagship_live ? (
                      <span className="badge-safe" style={{ fontSize: "9px" }}>LIVE SERVING</span>
                    ) : (
                      <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--c-text-muted)" }}>
                        KAGGLE HUB
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: "13px", fontWeight: 600, color: "var(--c-text-primary)" }}>{z.name}</div>
                  <div style={{ fontSize: "11px", color: "var(--c-text-muted)", marginTop: "2px" }}>
                    {z.states.join(", ")}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Selected Zone Deep Dive Card */}
        {selectedZone && (
          <div className="panel" style={{ borderTop: "4px solid var(--c-wine)" }}>
            <div className="panel-header">
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span style={{ fontSize: "12px", fontFamily: "var(--font-mono)", color: "var(--c-gold)", fontWeight: 700 }}>
                    {selectedZone.code}
                  </span>
                  {selectedZone.is_flagship_live && (
                    <span className="badge-safe">LIVE IN-MEMORY ENGINE</span>
                  )}
                </div>
                <h3 style={{ fontSize: "18px", fontWeight: 700, marginTop: "4px" }}>{selectedZone.name}</h3>
              </div>
            </div>

            {/* Metrics Breakdown */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "10px", marginBottom: "16px" }}>
              <div className="index-card">
                <span className="index-label">RMSE Reduction</span>
                <span className="index-value" style={{ color: "var(--c-sage)" }}>
                  +{selectedZone.rmse_reduction_percent}%
                </span>
              </div>

              <div className="index-card">
                <span className="index-label">R² Fit Score</span>
                <span className="index-value" style={{ color: "var(--c-linen)" }}>
                  {selectedZone.r2_score}
                </span>
              </div>

              <div className="index-card">
                <span className="index-label">Area Coverage</span>
                <span className="index-value" style={{ fontSize: "15px" }}>
                  {selectedZone.coverage_area_sqkm.toLocaleString()} km²
                </span>
              </div>

              <div className="index-card">
                <span className="index-label">Model Size</span>
                <span className="index-value" style={{ fontSize: "15px" }}>
                  {selectedZone.model_size_mb} MB
                </span>
              </div>
            </div>

            {/* Ecological Characteristics */}
            <div style={{ background: "var(--c-surface-elevated)", padding: "14px", border: "1px solid var(--c-border)", marginBottom: "16px" }}>
              <div style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--c-gold)", marginBottom: "6px", textTransform: "uppercase" }}>
                Agro-Ecological Typology & Terrain Physics
              </div>
              <p style={{ fontSize: "13px", lineHeight: 1.6, color: "var(--c-text-primary)" }}>
                {selectedZone.description}
              </p>
            </div>

            {/* Spatial Bounds */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "16px" }}>
              <div style={{ background: "var(--c-bg)", padding: "10px", border: "1px solid var(--c-border)", fontSize: "12px", fontFamily: "var(--font-mono)" }}>
                <div style={{ color: "var(--c-text-muted)" }}>LATITUDE BOUNDS:</div>
                <div style={{ color: "var(--c-text-primary)", fontWeight: 600 }}>
                  {selectedZone.lat_min}°N — {selectedZone.lat_max}°N
                </div>
              </div>

              <div style={{ background: "var(--c-bg)", padding: "10px", border: "1px solid var(--c-border)", fontSize: "12px", fontFamily: "var(--font-mono)" }}>
                <div style={{ color: "var(--c-text-muted)" }}>LONGITUDE BOUNDS:</div>
                <div style={{ color: "var(--c-text-primary)", fontWeight: 600 }}>
                  {selectedZone.lon_min}°E — {selectedZone.lon_max}°E
                </div>
              </div>
            </div>

            {/* Kaggle Download / Access Link */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: "13px", borderTop: "1px solid var(--c-border)" }}>
              <div style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--c-text-muted)" }}>
                ARTIFACT: {selectedZone.model_filename}
              </div>

              <a
                href={selectedZone.kaggle_hub_url}
                target="_blank"
                rel="noopener noreferrer"
                className="btn-primary"
                style={{ textDecoration: "none" }}
              >
                <ExternalLink size={14} />
                Access on Kaggle Model Hub
              </a>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
