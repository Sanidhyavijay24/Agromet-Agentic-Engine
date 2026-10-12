/**
 * @file FleetSection.tsx
 * @description Section 3: Pan-India 15 ACZ Fleet Benchmark Leaderboard & Dataset Hub
 * @module frontend/components/section_fleet
 */

import React, { useState } from "react";
import { Award, CheckCircle2, ChevronRight, Database, ExternalLink, Filter, Layers, ShieldCheck, Zap } from "lucide-react";
import { PAN_INDIA_FLEET } from "../../services/api";
import { ZoneFleetItem } from "../../types";

export const FleetSection: React.FC = () => {
  const [filterRegion, setFilterRegion] = useState<string>("ALL");

  const filteredZones = PAN_INDIA_FLEET.filter((zone) => {
    if (filterRegion === "ALL") return true;
    if (filterRegion === "MOUNTAIN") return zone.zone_id === 1 || zone.zone_id === 2 || zone.zone_id === 12;
    if (filterRegion === "PLAINS") return zone.zone_id >= 3 && zone.zone_id <= 6;
    if (filterRegion === "PLATEAU") return zone.zone_id >= 7 && zone.zone_id <= 10;
    if (filterRegion === "ARID") return zone.zone_id === 14 || zone.zone_id === 13;
    return true;
  });

  return (
    <section className="paper-sheet" id="section-fleet" style={{ marginTop: "36px" }}>
      <div className="sheet-header">
        <div>
          <div className="sheet-number">MODULE 03 • PAN-INDIA 15 ACZ FLEET BENCHMARK</div>
          <h2 className="sheet-title">Empirical Model Leaderboard & Dataset Hub</h2>
        </div>
        <div className="sheet-caption">
          Comprehensive benchmark across all 15 Planning Commission / ICAR Agro-Climatic Zones of India, spanning 47.3 million hourly observations across 10 years (2015–2024). Zone XIV is served in-memory locally with sub-50ms inference.
        </div>
      </div>

      {/* 4 Summary Stat Cards */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "12px", marginBottom: "20px" }}>
        <div className="draft-card" style={{ padding: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--c-curve-ml)", fontSize: "11px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
            <Award size={13} />
            <span>ZONE XIV FLAGSHIP (LIVE)</span>
          </div>
          <div style={{ marginTop: "4px" }}>
            <span className="metric-big tabular-nums" style={{ color: "var(--c-curve-ml)" }}>+40.2%</span>
            <span className="metric-unit">Error Drop</span>
          </div>
          <div style={{ fontSize: "11px", color: "var(--c-ink-secondary)", marginTop: "2px" }}>
            0.746°C &rarr; <strong>0.446°C RMSE</strong> (R² = 0.9967)
          </div>
        </div>

        <div className="draft-card" style={{ padding: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--c-accent-safe)", fontSize: "11px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
            <Layers size={13} />
            <span>PAN-INDIA 15 ACZ MEAN</span>
          </div>
          <div style={{ marginTop: "4px" }}>
            <span className="metric-big tabular-nums">+60.6%</span>
            <span className="metric-unit">Macro Avg Drop</span>
          </div>
          <div style={{ fontSize: "11px", color: "var(--c-ink-secondary)", marginTop: "2px" }}>
            2.758°C &rarr; <strong>0.912°C RMSE</strong> (Pooled: 66.9%)
          </div>
        </div>

        <div className="draft-card" style={{ padding: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--c-blueprint-dark)", fontSize: "11px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
            <Database size={13} />
            <span>DATASET SCALE (KAGGLE)</span>
          </div>
          <div style={{ marginTop: "4px" }}>
            <span className="metric-big tabular-nums">47.3M</span>
            <span className="metric-unit">Hourly Rows</span>
          </div>
          <div style={{ fontSize: "11px", color: "var(--c-ink-secondary)", marginTop: "2px" }}>
            10 Years (2015–2024) • 30m SRTM DEM
          </div>
        </div>

        <div className="draft-card" style={{ padding: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--c-accent-safe)", fontSize: "11px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
            <ShieldCheck size={13} />
            <span>AUDIT INTEGRITY</span>
          </div>
          <div style={{ marginTop: "4px" }}>
            <span className="metric-big tabular-nums">0.00%</span>
            <span className="metric-unit">Target Leakage</span>
          </div>
          <div style={{ fontSize: "11px", color: "var(--c-ink-secondary)", marginTop: "2px" }}>
            Spatial LOO Cross-Validation
          </div>
        </div>
      </div>

      {/* Filter Tabs & Table Container */}
      <div className="draft-card" style={{ padding: "20px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px", marginBottom: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <Filter size={14} style={{ color: "var(--c-ink-muted)" }} />
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "12px", fontWeight: 700, textTransform: "uppercase" }}>
              Filter by Agro-Climatic Domain:
            </span>
          </div>

          <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
            {[
              { id: "ALL", label: "All 15 Zones" },
              { id: "MOUNTAIN", label: "Himalayan & Ghats" },
              { id: "PLAINS", label: "Gangetic Plains" },
              { id: "PLATEAU", label: "Central/Eastern Plateau" },
              { id: "ARID", label: "Arid & Gujarat" },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setFilterRegion(tab.id)}
                className={`btn-paper ${filterRegion === tab.id ? "active" : ""}`}
                style={{ fontSize: "11px", padding: "4px 10px" }}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>

        {/* Master Table */}
        <div style={{ overflowX: "auto" }}>
          <table
            style={{
              width: "100%",
              borderCollapse: "collapse",
              fontFamily: "var(--font-body)",
              fontSize: "12.5px",
              textAlign: "left",
            }}
          >
            <thead>
              <tr
                style={{
                  borderBottom: "2px solid var(--c-ink-heading)",
                  fontFamily: "var(--font-mono)",
                  fontSize: "11px",
                  color: "var(--c-ink-secondary)",
                  textTransform: "uppercase",
                }}
              >
                <th style={{ padding: "10px 8px" }}>Zone ID & Name</th>
                <th style={{ padding: "10px 8px" }}>Geography</th>
                <th style={{ padding: "10px 8px" }}>Baseline RMSE</th>
                <th style={{ padding: "10px 8px" }}>ML Test RMSE</th>
                <th style={{ padding: "10px 8px" }}>Error Drop</th>
                <th style={{ padding: "10px 8px" }}>Test R²</th>
                <th style={{ padding: "10px 8px" }}>Trees</th>
                <th style={{ padding: "10px 8px" }}>Serving Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredZones.map((zone) => {
                const isFlagship = zone.is_flagship_live;
                return (
                  <tr
                    key={zone.zone_id}
                    style={{
                      borderBottom: "1px solid var(--c-paper-border)",
                      background: isFlagship ? "var(--c-paper-sheet)" : "transparent",
                      fontWeight: isFlagship ? 600 : 400,
                    }}
                  >
                    <td style={{ padding: "12px 8px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <span
                          style={{
                            fontFamily: "var(--font-mono)",
                            fontSize: "11px",
                            fontWeight: 700,
                            color: isFlagship ? "var(--c-curve-ml)" : "var(--c-ink-heading)",
                          }}
                        >
                          ACZ {zone.zone_id.toString().padStart(2, "0")}
                        </span>
                        <span style={{ color: "var(--c-ink-heading)" }}>{zone.zone_name}</span>
                      </div>
                    </td>

                    <td style={{ padding: "12px 8px", color: "var(--c-ink-secondary)", fontSize: "11.5px" }}>
                      {zone.region}
                    </td>

                    <td style={{ padding: "12px 8px" }} className="tabular-nums">
                      {zone.raw_rmse_c.toFixed(3)}°C
                    </td>

                    <td style={{ padding: "12px 8px", color: "var(--c-curve-ml)", fontWeight: 700 }} className="tabular-nums">
                      {zone.ml_rmse_c.toFixed(3)}°C
                    </td>

                    <td style={{ padding: "12px 8px" }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                        <div
                          style={{
                            width: "50px",
                            height: "6px",
                            background: "var(--c-paper-border)",
                            borderRadius: "3px",
                            overflow: "hidden",
                          }}
                        >
                          <div
                            style={{
                              width: `${Math.min(100, zone.error_drop_pct)}%`,
                              height: "100%",
                              background: zone.error_drop_pct > 70 ? "var(--c-accent-safe)" : "var(--c-curve-ml)",
                            }}
                          />
                        </div>
                        <span className="tabular-nums" style={{ fontSize: "11.5px", fontWeight: 700 }}>
                          +{zone.error_drop_pct.toFixed(1)}%
                        </span>
                      </div>
                    </td>

                    <td style={{ padding: "12px 8px" }} className="tabular-nums">
                      {zone.r2_score.toFixed(4)}
                    </td>

                    <td style={{ padding: "12px 8px", fontFamily: "var(--font-mono)", fontSize: "11px" }}>
                      {zone.trees_count.toLocaleString()}
                    </td>

                    <td style={{ padding: "12px 8px" }}>
                      {isFlagship ? (
                        <span className="badge-tag safe" style={{ fontSize: "10.5px" }}>
                          <CheckCircle2 size={11} />
                          Live Serving (195MB)
                        </span>
                      ) : (
                        <a
                          href={zone.kaggle_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{
                            display: "inline-flex",
                            alignItems: "center",
                            gap: "4px",
                            fontSize: "11px",
                            fontFamily: "var(--font-mono)",
                            color: "var(--c-blueprint-bg)",
                            textDecoration: "none",
                            fontWeight: 600,
                          }}
                        >
                          <span>Kaggle Hub</span>
                          <ExternalLink size={10} />
                        </a>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer Ledger & Technical Architecture Sign-off */}
      <footer
        style={{
          marginTop: "24px",
          paddingTop: "16px",
          borderTop: "1px solid var(--c-paper-border-strong)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "12px",
          fontSize: "11.5px",
          fontFamily: "var(--font-mono)",
          color: "var(--c-ink-muted)",
        }}
      >
        <div>
          <strong>A²E ARCHITECTURE:</strong> Bun 1.3.12 Runtime • FastAPI • XGBoost • 30m USGS SRTM DEM • Google Gemini Flash ReAct
        </div>
        <div>
          <span>ICAR & Planning Commission Framework (15 ACZ)</span>
        </div>
      </footer>
    </section>
  );
};
