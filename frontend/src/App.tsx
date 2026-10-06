/**
 * @file App.tsx
 * @description Root application component with tab navigation and theme management
 * @module frontend/App
 */

import React, { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { MicroclimateComparator } from "./components/MicroclimateComparator";
import { AgentTerminal } from "./components/AgentTerminal";
import { ZoneFleetHub } from "./components/ZoneFleetHub";

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<"comparator" | "agent" | "fleet">("comparator");
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  return (
    <div className="app-container">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        theme={theme}
        setTheme={setTheme}
      />

      <main className="main-content">
        {activeTab === "comparator" && <MicroclimateComparator />}
        {activeTab === "agent" && <AgentTerminal />}
        {activeTab === "fleet" && <ZoneFleetHub />}
      </main>

      <footer
        style={{
          padding: "16px 34px",
          background: "var(--c-surface)",
          borderTop: "1px solid var(--c-border)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          fontFamily: "var(--font-mono)",
          fontSize: "11px",
          color: "var(--c-text-muted)",
        }}
      >
        <div>
          <strong style={{ color: "var(--c-sand)" }}>A²E</strong> • Physics-Guided 1km² Microclimate Downscaling & Autonomous Krishi LLM
        </div>
        <div style={{ display: "flex", gap: "16px" }}>
          <span>ICAR ACZ-14 Testbed</span>
          <span>Google Gemini Flash Engine</span>
          <span>Pure Bun Runtime</span>
        </div>
      </footer>
    </div>
  );
};
