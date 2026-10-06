/**
 * @file Header.tsx
 * @description Main application header and navigation tab bar
 * @module frontend/components/Header
 */

import React from "react";
import { Cpu, Layers, Sparkles, Database, ShieldCheck, Sun, Moon } from "lucide-react";

interface HeaderProps {
  activeTab: "comparator" | "agent" | "fleet";
  setActiveTab: (tab: "comparator" | "agent" | "fleet") => void;
  theme: "dark" | "light";
  setTheme: (theme: "dark" | "light") => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  theme,
  setTheme,
}) => {
  return (
    <header className="top-nav">
      <div className="brand-badge">
        <div className="brand-logo-box">A²E</div>
        <div>
          <div className="brand-title">AGROMET AGENTIC ENGINE</div>
          <div className="brand-subtitle">
            1km² Physics Downscaler & Autonomous Krishi LLM
          </div>
        </div>
      </div>

      <nav className="nav-tabs">
        <button
          className={`nav-tab-btn ${activeTab === "comparator" ? "active" : ""}`}
          onClick={() => setActiveTab("comparator")}
        >
          <Layers size={15} />
          1km Diurnal Lens
        </button>
        <button
          className={`nav-tab-btn ${activeTab === "agent" ? "active" : ""}`}
          onClick={() => setActiveTab("agent")}
        >
          <Sparkles size={15} />
          Agentic Krishi Terminal
        </button>
        <button
          className={`nav-tab-btn ${activeTab === "fleet" ? "active" : ""}`}
          onClick={() => setActiveTab("fleet")}
        >
          <Database size={15} />
          15 ACZ Fleet Hub
        </button>
      </nav>

      <div style={{ display: "flex", alignItems: "center", gap: "13px" }}>
        <div className="status-indicator">
          <span className="pulse-dot"></span>
          <span style={{ color: "var(--c-sand)" }}>ZONE XIV LIVE (165MB)</span>
        </div>

        <div className="status-indicator">
          <Cpu size={13} style={{ color: "var(--c-gold)" }} />
          <span style={{ color: "var(--c-text-muted)" }}>GEMINI FLASH READY</span>
        </div>

        <button
          className="btn-secondary"
          onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          title="Toggle Light/Dark Theme"
          style={{ padding: "6px 10px" }}
        >
          {theme === "dark" ? <Sun size={14} /> : <Moon size={14} />}
        </button>
      </div>
    </header>
  );
};
