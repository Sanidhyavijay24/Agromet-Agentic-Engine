/**
 * @file BlueprintHeader.tsx
 * @description Technical blueprint poster header with live model telemetry
 * @module frontend/components
 */

import React from "react";
import { Activity, Database, ExternalLink, Layers, Zap } from "lucide-react";
import { SystemHealth } from "../types";

interface BlueprintHeaderProps {
  health: SystemHealth;
}

export const BlueprintHeader: React.FC<BlueprintHeaderProps> = ({ health }) => {
  return (
    <header className="blueprint-header">
      <div className="poster-title-band">
        <div>
          <h1 className="poster-display-heading">
            Agromet Agentic Engine
            <span>Microclimate Studio</span>
          </h1>
          <p className="poster-subtitle">
            Physics-guided 1 km² microclimate downscaler coupled with an autonomous
            Agrometeorological ReAct advisory system powered by Google Gemini Flash.
          </p>
        </div>

        <div>
          <a
            href="https://www.kaggle.com/datasets/sanidhyavijay24/pan-india-15-acz-1km-microclimate-dataset-and-models"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-blueprint"
            style={{ textDecoration: "none" }}
          >
            <Database size={14} />
            <span>Kaggle Dataset Hub (47.3M Rows)</span>
            <ExternalLink size={12} />
          </a>
        </div>
      </div>

      <div className="telemetry-strip">
        <div className="telemetry-pill live-indicator">
          <Activity size={13} style={{ color: "#22C55E" }} />
          <span>Zone XIV Flagship: {health.status}</span>
        </div>

        <div className="telemetry-pill">
          <Layers size={13} style={{ color: "#93C5FD" }} />
          <span>Artifact: {health.model_size_mb} MB (residual_model_acz_14.joblib)</span>
        </div>

        <div className="telemetry-pill">
          <Zap size={13} style={{ color: "#FBBF24" }} />
          <span>Inference Latency: ~{health.latency_ms} ms</span>
        </div>

        <div className="telemetry-pill">
          <span style={{ color: "#A7F3D0", fontWeight: 700 }}>5,555 Trees</span>
          <span>• 30m USGS SRTM DEM • Zero Target Leakage</span>
        </div>
      </div>
    </header>
  );
};
