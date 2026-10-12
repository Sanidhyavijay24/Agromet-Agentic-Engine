/**
 * @file ScenarioPresets.tsx
 * @description Quick scenario triggers and crop parameter selectors for the ReAct Agent
 * @module frontend/components/section_agent
 */

import React from "react";
import { AlertOctagon, Droplet, Play, Sparkles, ThermometerSnowflake, Wind } from "lucide-react";
import { SCENARIO_PRESETS } from "../../services/api";
import { ScenarioPreset } from "../../types";

interface ScenarioPresetsProps {
  selectedScenario: ScenarioPreset;
  onSelectScenario: (scenario: ScenarioPreset) => void;
  isRunning: boolean;
  onRunAnalysis: () => void;
}

export const ScenarioPresets: React.FC<ScenarioPresetsProps> = ({
  selectedScenario,
  onSelectScenario,
  isRunning,
  onRunAnalysis,
}) => {
  return (
    <div className="draft-card" style={{ marginBottom: "20px" }}>
      <div className="draft-card-header">
        <div className="draft-card-title" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <Sparkles size={14} style={{ color: "var(--c-curve-ml)" }} />
          <span>Evaluation Scenarios (Pre-calibrated ICAR Agromet Stress Cases)</span>
        </div>
        <span className="badge-tag safe">Google Gemini Flash ReAct</span>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "10px", marginBottom: "16px" }}>
        {SCENARIO_PRESETS.map((scenario) => {
          const isActive = scenario.id === selectedScenario.id;
          return (
            <div
              key={scenario.id}
              onClick={() => onSelectScenario(scenario)}
              style={{
                border: isActive ? "2px solid var(--c-ink-heading)" : "1px solid var(--c-paper-border-strong)",
                background: isActive ? "var(--c-paper-sheet)" : "var(--c-paper-card)",
                borderRadius: "var(--radius-xs)",
                padding: "12px 14px",
                cursor: "pointer",
                transition: "all 0.15s ease",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "6px" }}>
                <span style={{ fontSize: "12.5px", fontWeight: 750, color: "var(--c-ink-heading)" }}>
                  {scenario.title}
                </span>
                {scenario.id === "frost_mustard" ? (
                  <ThermometerSnowflake size={14} style={{ color: "#2563EB" }} />
                ) : scenario.id === "spray_cumin" ? (
                  <Wind size={14} style={{ color: "var(--c-accent-safe)" }} />
                ) : (
                  <AlertOctagon size={14} style={{ color: "var(--c-accent-warn)" }} />
                )}
              </div>

              <div style={{ fontSize: "11.5px", color: "var(--c-ink-secondary)", marginBottom: "4px" }}>
                <strong>Crop:</strong> {scenario.crop} • <em>{scenario.stage}</em>
              </div>

              <div style={{ fontSize: "11px", color: "var(--c-ink-muted)", lineHeight: 1.35 }}>
                {scenario.weather_condition}
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Scenario Query & Run Button */}
      <div
        style={{
          background: "var(--c-paper-sheet)",
          border: "1px dashed var(--c-paper-border-strong)",
          borderRadius: "var(--radius-xs)",
          padding: "14px 18px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "14px",
        }}
      >
        <div style={{ maxWidth: "680px" }}>
          <div style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--c-curve-ml)", fontWeight: 700, textTransform: "uppercase" }}>
            AGENT MISSION QUERY:
          </div>
          <div style={{ fontSize: "13.5px", fontWeight: 650, color: "var(--c-ink-heading)", marginTop: "3px" }}>
            "{selectedScenario.question}"
          </div>
          <div style={{ fontSize: "11.5px", color: "var(--c-ink-secondary)", marginTop: "4px" }}>
            Physical Rationale: {selectedScenario.highlight_physics}
          </div>
        </div>

        <div>
          <button
            onClick={onRunAnalysis}
            disabled={isRunning}
            className="btn-blueprint"
            style={{ padding: "10px 20px", fontSize: "13px" }}
          >
            <Play size={14} style={{ fill: isRunning ? "none" : "#FFFFFF" }} />
            <span>{isRunning ? "Executing ReAct Loop..." : "Run Autonomous Advisory"}</span>
          </button>
        </div>
      </div>
    </div>
  );
};
