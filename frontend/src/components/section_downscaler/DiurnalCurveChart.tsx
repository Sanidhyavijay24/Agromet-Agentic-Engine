/**
 * @file DiurnalCurveChart.tsx
 * @description High-precision interactive SVG chart comparing Coarse Baseline vs 1 km Downscaling
 * @module frontend/components/section_downscaler
 */

import React, { useState } from "react";
import { AlertTriangle, Award, CheckCircle2, Clock, Sparkles } from "lucide-react";
import { DiurnalPoint } from "../../types";

interface DiurnalCurveChartProps {
  data: DiurnalPoint[];
  locationName: string;
}

export const DiurnalCurveChart: React.FC<DiurnalCurveChartProps> = ({ data, locationName }) => {
  const [hoveredHour, setHoveredHour] = useState<number | null>(5); // Default to dawn inversion point

  if (!data || data.length === 0) {
    return <div>No diurnal data available.</div>;
  }

  const activePoint = hoveredHour !== null ? data.find((p) => p.hour === hoveredHour) || data[5] : data[5];

  // SVG dimensions
  const svgWidth = 860;
  const svgHeight = 280;
  const padding = { top: 28, right: 30, bottom: 44, left: 45 };

  const plotWidth = svgWidth - padding.left - padding.right;
  const plotHeight = svgHeight - padding.top - padding.bottom;

  // Temperature scale
  const allTemps = data.flatMap((d) => [d.coarse_baseline_temp_c, d.downscaled_temp_c]);
  const minTemp = Math.floor(Math.min(...allTemps) - 2);
  const maxTemp = Math.ceil(Math.max(...allTemps) + 2);

  const getX = (hour: number) => padding.left + (hour / 23) * plotWidth;
  const getY = (temp: number) => padding.top + plotHeight - ((temp - minTemp) / (maxTemp - minTemp)) * plotHeight;

  // Generate SVG path strings
  const coarsePath = data
    .map((d, i) => `${i === 0 ? "M" : "L"} ${getX(d.hour).toFixed(1)} ${getY(d.coarse_baseline_temp_c).toFixed(1)}`)
    .join(" ");

  const mlPath = data
    .map((d, i) => `${i === 0 ? "M" : "L"} ${getX(d.hour).toFixed(1)} ${getY(d.downscaled_temp_c).toFixed(1)}`)
    .join(" ");

  // Y-axis grid ticks (every 4°C)
  const yTicks: number[] = [];
  for (let t = Math.ceil(minTemp / 4) * 4; t <= maxTemp; t += 4) {
    yTicks.push(t);
  }

  // Highlight regions: Inversion (03:00 to 07:00), Spray Window (07:00 to 10:00)
  const inversionXStart = getX(3);
  const inversionXEnd = getX(7);
  const sprayXStart = getX(7);
  const sprayXEnd = getX(10);

  return (
    <div className="draft-card" style={{ padding: "20px" }}>
      {/* Top Banner with Empirical Benchmark */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "12px", marginBottom: "16px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <h3 style={{ margin: 0, fontSize: "16px", fontWeight: 750, color: "var(--c-ink-heading)", fontFamily: "var(--font-display)" }}>
              24-Hour Diurnal Microclimate Profile
            </h3>
            <span className="badge-tag safe">
              <Award size={12} />
              +40.2% Error Reduction (Zone XIV)
            </span>
          </div>
          <span style={{ fontSize: "12px", color: "var(--c-ink-secondary)" }}>
            LOO Baseline (-6.5°C/km lapse) vs 1 km XGBoost Model (5,555 trees) • {locationName}
          </span>
        </div>

        {/* Legend */}
        <div style={{ display: "flex", gap: "14px", alignItems: "center", fontSize: "11.5px", fontFamily: "var(--font-mono)" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
            <span style={{ display: "inline-block", width: "16px", height: "2px", borderTop: "2px dashed var(--c-curve-coarse)" }}></span>
            <span style={{ color: "var(--c-ink-secondary)" }}>Coarse Baseline (25 km)</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "5px" }}>
            <span style={{ display: "inline-block", width: "16px", height: "3px", backgroundColor: "var(--c-curve-ml)", borderRadius: "1px" }}></span>
            <span style={{ color: "var(--c-curve-ml)", fontWeight: 700 }}>1 km A²E Downscaled</span>
          </div>
        </div>
      </div>

      {/* SVG Canvas */}
      <div style={{ width: "100%", overflowX: "auto" }}>
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          style={{ width: "100%", height: "auto", display: "block", userSelect: "none" }}
        >
          <defs>
            <linearGradient id="mlGlow" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--c-curve-ml)" stopOpacity="0.25" />
              <stop offset="100%" stopColor="var(--c-curve-ml)" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Shaded Physical Windows */}
          {/* Nocturnal Inversion Zone */}
          <rect
            x={inversionXStart}
            y={padding.top}
            width={inversionXEnd - inversionXStart}
            height={plotHeight}
            fill="#3B82F6"
            fillOpacity="0.08"
          />
          <text
            x={(inversionXStart + inversionXEnd) / 2}
            y={padding.top + 14}
            textAnchor="middle"
            fontSize="9.5"
            fontFamily="var(--font-mono)"
            fill="#2563EB"
            fontWeight="600"
          >
            COLD-AIR POOLING
          </text>

          {/* Safe Spray Window */}
          <rect
            x={sprayXStart}
            y={padding.top}
            width={sprayXEnd - sprayXStart}
            height={plotHeight}
            fill="#10B981"
            fillOpacity="0.09"
          />
          <text
            x={(sprayXStart + sprayXEnd) / 2}
            y={padding.top + 14}
            textAnchor="middle"
            fontSize="9.5"
            fontFamily="var(--font-mono)"
            fill="#059669"
            fontWeight="600"
          >
            SPRAY WINDOW
          </text>

          {/* Horizontal Grid Lines */}
          {yTicks.map((temp) => {
            const y = getY(temp);
            return (
              <g key={temp}>
                <line
                  x1={padding.left}
                  y1={y}
                  x2={svgWidth - padding.right}
                  y2={y}
                  stroke="var(--c-paper-border)"
                  strokeDasharray="2 3"
                />
                <text
                  x={padding.left - 8}
                  y={y + 3.5}
                  textAnchor="end"
                  fontSize="10"
                  fontFamily="var(--font-mono)"
                  fill="var(--c-ink-muted)"
                >
                  {temp}°C
                </text>
              </g>
            );
          })}

          {/* Vertical Hour Tick Guides */}
          {data.map((d) => {
            if (d.hour % 3 === 0) {
              const x = getX(d.hour);
              return (
                <g key={d.hour}>
                  <line
                    x1={x}
                    y1={padding.top}
                    x2={x}
                    y2={svgHeight - padding.bottom}
                    stroke="var(--c-paper-border)"
                    strokeWidth="0.5"
                  />
                  <text
                    x={x}
                    y={svgHeight - padding.bottom + 18}
                    textAnchor="middle"
                    fontSize="10"
                    fontFamily="var(--font-mono)"
                    fill="var(--c-ink-secondary)"
                    fontWeight={d.hour === activePoint.hour ? "700" : "400"}
                  >
                    {d.time_label}
                  </text>
                </g>
              );
            }
            return null;
          })}

          {/* Coarse Baseline Path (Dashed) */}
          <path
            d={coarsePath}
            fill="none"
            stroke="var(--c-curve-coarse)"
            strokeWidth="2.2"
            strokeDasharray="5 4"
          />

          {/* Downscaled ML Path (Solid Burnt Amber) */}
          <path
            d={mlPath}
            fill="none"
            stroke="var(--c-curve-ml)"
            strokeWidth="3.2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />

          {/* Active Hover Guide Line */}
          {activePoint && (
            <g>
              <line
                x1={getX(activePoint.hour)}
                y1={padding.top}
                x2={getX(activePoint.hour)}
                y2={svgHeight - padding.bottom}
                stroke="var(--c-ink-heading)"
                strokeWidth="1.2"
                strokeDasharray="3 3"
              />

              {/* Coarse Circle */}
              <circle
                cx={getX(activePoint.hour)}
                cy={getY(activePoint.coarse_baseline_temp_c)}
                r="4.5"
                fill="#FFFFFF"
                stroke="var(--c-curve-coarse)"
                strokeWidth="2"
              />

              {/* ML Circle */}
              <circle
                cx={getX(activePoint.hour)}
                cy={getY(activePoint.downscaled_temp_c)}
                r="5.5"
                fill="var(--c-curve-ml)"
                stroke="#FFFFFF"
                strokeWidth="2"
              />
            </g>
          )}

          {/* Interactive Transparent Hit-Boxes for Hover */}
          {data.map((d) => (
            <rect
              key={d.hour}
              x={getX(d.hour) - plotWidth / 48}
              y={padding.top}
              width={plotWidth / 24}
              height={plotHeight}
              fill="transparent"
              style={{ cursor: "pointer" }}
              onMouseEnter={() => setHoveredHour(d.hour)}
            />
          ))}
        </svg>
      </div>

      {/* Interactive Telemetry Readout for Hovered Hour */}
      {activePoint && (
        <div
          style={{
            marginTop: "16px",
            background: "var(--c-paper-sheet)",
            border: "1px solid var(--c-paper-border-strong)",
            borderRadius: "var(--radius-xs)",
            padding: "14px 18px",
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))",
            gap: "14px",
            alignItems: "center",
          }}
        >
          <div>
            <div style={{ fontSize: "10.5px", fontFamily: "var(--font-mono)", color: "var(--c-ink-muted)", display: "flex", alignItems: "center", gap: "4px" }}>
              <Clock size={11} />
              <span>TIME STEP</span>
            </div>
            <div className="metric-big tabular-nums" style={{ fontSize: "20px" }}>
              {activePoint.time_label}
            </div>
          </div>

          <div>
            <div style={{ fontSize: "10.5px", fontFamily: "var(--font-mono)", color: "var(--c-ink-muted)" }}>
              COARSE BASELINE
            </div>
            <div className="metric-big tabular-nums" style={{ fontSize: "20px", color: "var(--c-curve-coarse)" }}>
              {activePoint.coarse_baseline_temp_c}°C
            </div>
          </div>

          <div>
            <div style={{ fontSize: "10.5px", fontFamily: "var(--font-mono)", color: "var(--c-curve-ml)", fontWeight: 700 }}>
              1 km DOWNSCALED
            </div>
            <div className="metric-big tabular-nums" style={{ fontSize: "20px", color: "var(--c-curve-ml)" }}>
              {activePoint.downscaled_temp_c}°C
            </div>
          </div>

          <div>
            <div style={{ fontSize: "10.5px", fontFamily: "var(--font-mono)", color: "var(--c-ink-muted)" }}>
              PHYSICAL DELTA (Δ)
            </div>
            <div
              className="metric-big tabular-nums"
              style={{
                fontSize: "20px",
                color: activePoint.delta_c < 0 ? "#2563EB" : activePoint.delta_c > 0 ? "var(--c-curve-ml)" : "var(--c-ink-heading)",
              }}
            >
              {activePoint.delta_c > 0 ? `+${activePoint.delta_c}` : activePoint.delta_c}°C
            </div>
          </div>

          <div>
            <div style={{ fontSize: "10.5px", fontFamily: "var(--font-mono)", color: "var(--c-ink-muted)" }}>
              RH / VPD
            </div>
            <div className="tabular-nums" style={{ fontSize: "14px", fontWeight: 700, color: "var(--c-ink-heading)", marginTop: "2px" }}>
              {activePoint.downscaled_rh_pct}% • {activePoint.vpd_kpa} kPa
            </div>
          </div>

          <div>
            {activePoint.is_inversion ? (
              <span className="badge-tag danger" style={{ fontSize: "10.5px" }}>
                <AlertTriangle size={11} />
                Inversion / Cold Drainage
              </span>
            ) : activePoint.is_safe_spray ? (
              <span className="badge-tag safe" style={{ fontSize: "10.5px" }}>
                <CheckCircle2 size={11} />
                Safe Chemical Spray
              </span>
            ) : (
              <span className="badge-tag warn" style={{ fontSize: "10.5px" }}>
                <Sparkles size={11} />
                Standard Thermal Regime
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
