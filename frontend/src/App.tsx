/**
 * @file App.tsx
 * @description Master application shell for A²E Engineering Blueprint Studio
 * @module frontend
 */

import React, { useEffect, useState } from "react";
import { BlueprintHeader } from "./components/BlueprintHeader";
import { SpatialRuler } from "./components/SpatialRuler";
import { AgentSection } from "./components/section_agent/AgentSection";
import { DownscalerSection } from "./components/section_downscaler/DownscalerSection";
import { FleetSection } from "./components/section_fleet/FleetSection";
import { COORDINATE_PRESETS, fetchSystemHealth } from "./services/api";
import { SystemHealth } from "./types";

export const App: React.FC = () => {
  const [lat, setLat] = useState<number>(COORDINATE_PRESETS[0].lat);
  const [lon, setLon] = useState<number>(COORDINATE_PRESETS[0].lon);

  const [health, setHealth] = useState<SystemHealth>({
    status: "ONLINE",
    fastapi_online: true,
    flagship_zone_loaded: true,
    model_size_mb: 195.0,
    zone_name: "Zone XIV (Western Dry / Rajasthan)",
    tree_count: 5555,
    latency_ms: 38,
    server_timestamp: "LIVE",
  });

  useEffect(() => {
    fetchSystemHealth().then(setHealth);
  }, []);

  return (
    <div className="blueprint-canvas">
      {/* Left-edge architectural scale ruler */}
      <SpatialRuler />

      {/* Main studio layout */}
      <main className="studio-layout">
        <BlueprintHeader health={health} />

        {/* Module 01: Microclimate Physics Downscaler Studio */}
        <DownscalerSection
          lat={lat}
          lon={lon}
          onChangeLat={setLat}
          onChangeLon={setLon}
        />

        {/* Module 02: Autonomous ReAct Agromet Agent Terminal */}
        <AgentSection lat={lat} lon={lon} />

        {/* Module 03: Pan-India 15 ACZ Fleet Benchmark & Dataset Hub */}
        <FleetSection />
      </main>
    </div>
  );
};
