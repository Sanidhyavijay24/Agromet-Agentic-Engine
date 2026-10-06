/**
 * @file AgentTerminal.tsx
 * @description Autonomous ReAct Krishi Advisory Consultation Terminal & Live Tool Trace Visualizer
 * @module frontend/components/AgentTerminal
 */

import React, { useState } from "react";
import {
  Sparkles,
  Bot,
  Send,
  ShieldAlert,
  Droplets,
  Thermometer,
  Languages,
  CheckCircle2,
  Terminal,
  Activity,
  Layers,
} from "lucide-react";
import { AgentAdvisoryResponse } from "../types";
import { requestAgentAdvisory, sendAgentChat } from "../services/api";

interface PresetScenario {
  id: string;
  title: string;
  badge: string;
  crop_name: string;
  growth_stage: string;
  latitude: number;
  longitude: number;
  soil_type: string;
  field_notes: string;
}

const PRESET_SCENARIOS: PresetScenario[] = [
  {
    id: "bajra_heat",
    title: "Bajra Grain Filling Heat Spike (43°C)",
    badge: "CRITICAL HEAT",
    crop_name: "Bajra",
    growth_stage: "Grain Filling",
    latitude: 26.3571,
    longitude: 73.0412,
    soil_type: "Sandy Loam",
    field_notes: "Forecast indicates 43.5°C peak ambient heat. Plants showing incipient afternoon flag leaf roll.",
  },
  {
    id: "mustard_frost",
    title: "Mustard Flowering Radiative Frost Risk (3°C)",
    badge: "FROST HAZARD",
    crop_name: "Mustard",
    growth_stage: "Flowering",
    latitude: 28.0824,
    longitude: 73.3456,
    soil_type: "Thar Dune Sand",
    field_notes: "Clear nocturnal skies with calm winds (<4 km/h). Severe nocturnal radiation frost expected in low dunes.",
  },
  {
    id: "cotton_spray",
    title: "Bt-Cotton Whitefly Spray Window (Delta-T)",
    badge: "SPRAY DRIFT",
    crop_name: "Cotton",
    growth_stage: "Square Formation",
    latitude: 26.7262,
    longitude: 72.9094,
    soil_type: "Aeolian Sand",
    field_notes: "Whitefly nymph count exceeding ETL (6/leaf). Farmer intending pyriproxyfen foliar spray.",
  },
  {
    id: "guar_drought",
    title: "Guar Podding Aridisol Moisture Deficit",
    badge: "DROUGHT STRESS",
    crop_name: "Guar",
    growth_stage: "Pod Formation",
    latitude: 26.9214,
    longitude: 71.9167,
    soil_type: "Desert Calcareous Sand",
    field_notes: "21 consecutive dry days. Top 15cm soil profile moisture below 18% available water capacity.",
  },
];

export const AgentTerminal: React.FC = () => {
  const [selectedScenario, setSelectedScenario] = useState<PresetScenario>(PRESET_SCENARIOS[0]);
  const [crop, setCrop] = useState<string>(PRESET_SCENARIOS[0].crop_name);
  const [cropStage, setCropStage] = useState<string>(PRESET_SCENARIOS[0].growth_stage);
  const [latitude, setLatitude] = useState<number>(PRESET_SCENARIOS[0].latitude);
  const [longitude, setLongitude] = useState<number>(PRESET_SCENARIOS[0].longitude);
  const [soilType, setSoilType] = useState<string>(PRESET_SCENARIOS[0].soil_type);
  const [userQuery, setUserQuery] = useState<string>(PRESET_SCENARIOS[0].field_notes);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [advisoryResult, setAdvisoryResult] = useState<AgentAdvisoryResponse | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<"en" | "hi">("en");

  // Chat follow-up state
  const [chatMessages, setChatMessages] = useState<Array<{ role: string; content: string; time: string }>>([]);
  const [chatInput, setChatInput] = useState<string>("");
  const [isChatSending, setIsChatSending] = useState<boolean>(false);

  const handleSelectScenario = (scenario: PresetScenario) => {
    setSelectedScenario(scenario);
    setCrop(scenario.crop_name);
    setCropStage(scenario.growth_stage);
    setLatitude(scenario.latitude);
    setLongitude(scenario.longitude);
    setSoilType(scenario.soil_type);
    setUserQuery(scenario.field_notes);
  };

  const handleRunConsultation = async () => {
    setIsLoading(true);
    try {
      const result = await requestAgentAdvisory({
        latitude,
        longitude,
        crop,
        crop_stage: cropStage,
        soil_type: soilType,
        user_query: userQuery,
        forecast_hours: 48,
      });
      setAdvisoryResult(result);
    } catch (err: any) {
      alert(`Advisory Consultation Error: ${err.message || err}`);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendChat = async () => {
    if (!chatInput.trim() || isChatSending) return;
    const queryText = chatInput.trim();
    const newMsg = {
      role: "user",
      content: queryText,
      time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };
    setChatMessages((prev) => [...prev, newMsg]);
    setChatInput("");
    setIsChatSending(true);

    try {
      const chatRes = await sendAgentChat({
        message: queryText,
        latitude,
        longitude,
        crop,
        crop_stage: cropStage,
        soil_type: soilType,
      });

      const replyMsg = {
        role: "agent",
        content: chatRes.reply,
        time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };
      setChatMessages((prev) => [...prev, replyMsg]);
    } catch (err: any) {
      setChatMessages((prev) => [
        ...prev,
        {
          role: "agent",
          content: `⚠️ Error processing agromet chat: ${err.message || err}`,
          time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } finally {
      setIsChatSending(false);
    }
  };

  return (
    <div>
      {/* Quick Crisis Scenario Selectors */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <Sparkles size={18} style={{ color: "var(--c-gold)" }} />
            <span>AGROMET CRISIS PRESETS & FIELD PARAMETERS</span>
          </div>
          <span className="panel-tag">AUTONOMOUS ReAct ORCHESTRATOR</span>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: "10px", marginBottom: "16px" }}>
          {PRESET_SCENARIOS.map((sc) => (
            <button
              key={sc.id}
              onClick={() => handleSelectScenario(sc)}
              className="btn-secondary"
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                padding: "10px",
                borderLeft: selectedScenario.id === sc.id ? "4px solid var(--c-gold)" : "1px solid var(--c-border)",
                background: selectedScenario.id === sc.id ? "var(--c-surface-hover)" : "var(--c-surface)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", width: "100%", marginBottom: "4px" }}>
                <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--c-gold)", fontWeight: 700 }}>
                  {sc.badge}
                </span>
                <span style={{ fontSize: "10px", color: "var(--c-text-muted)" }}>{sc.crop_name}</span>
              </div>
              <span style={{ fontSize: "12px", fontWeight: 600, textAlign: "left" }}>{sc.title}</span>
            </button>
          ))}
        </div>

        {/* Form Inputs Grid */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "13px" }}>
          <div className="control-group">
            <label className="control-label">Target Crop</label>
            <input
              type="text"
              className="input-text"
              value={crop}
              onChange={(e) => setCrop(e.target.value)}
            />
          </div>

          <div className="control-group">
            <label className="control-label">Phenological Growth Stage</label>
            <input
              type="text"
              className="input-text"
              value={cropStage}
              onChange={(e) => setCropStage(e.target.value)}
            />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
            <div className="control-group">
              <label className="control-label">Latitude (°N)</label>
              <input
                type="number"
                step="0.0001"
                className="input-text"
                value={latitude}
                onChange={(e) => setLatitude(parseFloat(e.target.value) || 0)}
              />
            </div>
            <div className="control-group">
              <label className="control-label">Longitude (°E)</label>
              <input
                type="number"
                step="0.0001"
                className="input-text"
                value={longitude}
                onChange={(e) => setLongitude(parseFloat(e.target.value) || 0)}
              />
            </div>
          </div>

          <div className="control-group">
            <label className="control-label">Soil Regolith Profile</label>
            <input
              type="text"
              className="input-text"
              value={soilType}
              onChange={(e) => setSoilType(e.target.value)}
            />
          </div>
        </div>

        <div className="control-group" style={{ marginTop: "8px" }}>
          <label className="control-label">Field Observations & Symptoms</label>
          <textarea
            className="textarea-box"
            rows={2}
            value={userQuery}
            onChange={(e) => setUserQuery(e.target.value)}
          />
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end", marginTop: "13px" }}>
          <button
            className="btn-primary"
            onClick={handleRunConsultation}
            disabled={isLoading}
            style={{ padding: "10px 24px" }}
          >
            <Bot size={16} />
            {isLoading ? "Running Multi-Tool ReAct Cycle..." : "Initiate Autonomous Agromet Consultation"}
          </button>
        </div>
      </div>

      {/* Advisory Output Section */}
      {advisoryResult && (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1.2fr", gap: "21px", marginBottom: "21px" }}>
          
          {/* ReAct Tool Execution Traces Timeline */}
          <div className="panel">
            <div className="panel-header">
              <div className="panel-title">
                <Terminal size={16} style={{ color: "var(--c-gold)" }} />
                <span>ReAct EXECUTION TRACE LOGS ({advisoryResult.agent_trace.length} STEPS)</span>
              </div>
              <span className="panel-tag">{advisoryResult.llm_model_used}</span>
            </div>

            <div style={{ maxHeight: "480px", overflowY: "auto", paddingRight: "5px" }}>
              {advisoryResult.agent_trace.map((trace, idx) => (
                <div key={idx} className="trace-step">
                  <div className="trace-step-number">#{idx + 1}</div>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                      <span className="trace-tool-badge">{trace.tool_name}</span>
                      <span style={{ fontSize: "10px", color: "var(--c-text-muted)" }}>
                        {trace.execution_time_ms.toFixed(1)} ms
                      </span>
                    </div>

                    <div style={{ background: "var(--c-bg)", padding: "8px", border: "1px solid var(--c-border)", fontSize: "11px", marginTop: "4px" }}>
                      <div style={{ color: "var(--c-gold)", fontWeight: 600, marginBottom: "2px" }}>TOOL RESULT:</div>
                      <div style={{ color: "var(--c-sand)", lineHeight: 1.5 }}>
                        {trace.result_summary}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div style={{ display: "flex", justifyContent: "space-between", marginTop: "13px", fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--c-text-muted)" }}>
              <span>Total Latency: {advisoryResult.total_latency_ms.toFixed(0)} ms</span>
              <span>Status: {advisoryResult.status.toUpperCase()}</span>
            </div>
          </div>

          {/* Actionable Agromet Advisory Card */}
          <div className="panel" style={{ borderTop: "4px solid var(--c-wine)" }}>
            <div className="panel-header">
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span className={
                    advisoryResult.action_plan.verdict_category.includes("CRITICAL") || advisoryResult.action_plan.verdict_category.includes("FROST") || advisoryResult.action_plan.verdict_category.includes("HEAT")
                      ? "badge-urgent"
                      : "badge-advisory"
                  }>
                    {advisoryResult.action_plan.verdict_category}
                  </span>
                  <span style={{ fontSize: "14px", fontWeight: 700 }}>
                    {advisoryResult.crop} ({advisoryResult.crop_stage})
                  </span>
                </div>
              </div>

              {/* Language Switcher */}
              <div style={{ display: "flex", gap: "2px", background: "var(--c-bg)", border: "1px solid var(--c-border)" }}>
                <button
                  className={`btn-secondary ${selectedLanguage === "en" ? "active" : ""}`}
                  onClick={() => setSelectedLanguage("en")}
                  style={{
                    padding: "4px 10px",
                    background: selectedLanguage === "en" ? "var(--c-wine)" : "transparent",
                    color: selectedLanguage === "en" ? "#fff" : "var(--c-text-secondary)",
                  }}
                >
                  English
                </button>
                <button
                  className={`btn-secondary ${selectedLanguage === "hi" ? "active" : ""}`}
                  onClick={() => setSelectedLanguage("hi")}
                  style={{
                    padding: "4px 10px",
                    background: selectedLanguage === "hi" ? "var(--c-wine)" : "transparent",
                    color: selectedLanguage === "hi" ? "#fff" : "var(--c-text-secondary)",
                  }}
                >
                  हिंदी
                </button>
              </div>
            </div>

            {/* Headline Banner */}
            <div style={{ background: "var(--c-espresso)", padding: "12px 16px", border: "1px solid var(--c-border-accent)", marginBottom: "14px" }}>
              <div style={{ fontSize: "14px", fontWeight: 700, color: "var(--c-linen)" }}>
                {advisoryResult.action_plan.summary_headline}
              </div>
            </div>

            {/* Synthesized Agromet Guidance */}
            <div style={{ background: "var(--c-surface-elevated)", padding: "16px", border: "1px solid var(--c-border)", marginBottom: "16px" }}>
              <div style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--c-gold)", marginBottom: "8px", fontWeight: 700 }}>
                {selectedLanguage === "hi" ? "कृषि वैज्ञानिक संस्तुति (SYNTHESIZED ADVISORY)" : "EXPERT AGROMET ADVISORY"}
              </div>
              <p style={{ fontSize: "13px", lineHeight: 1.7, color: "var(--c-text-primary)", whiteSpace: "pre-line" }}>
                {selectedLanguage === "hi"
                  ? advisoryResult.action_plan.vernacular_hindi
                  : advisoryResult.action_plan.english_summary}
              </p>
            </div>

            {/* Structured Directives */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "10px", marginBottom: "16px" }}>
              {/* Spray Decision */}
              <div className="index-card">
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <Droplets size={14} style={{ color: "var(--c-cyan)" }} />
                  <span className="index-label">Spray Timing & Protocol</span>
                </div>
                <p style={{ fontSize: "12px", color: "var(--c-text-primary)", marginTop: "4px", lineHeight: 1.5 }}>
                  {advisoryResult.action_plan.spray_recommendation}
                </p>
              </div>

              {/* Irrigation Advice */}
              <div className="index-card">
                <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                  <Thermometer size={14} style={{ color: "var(--c-amber)" }} />
                  <span className="index-label">Irrigation Directive</span>
                </div>
                <p style={{ fontSize: "12px", color: "var(--c-text-primary)", marginTop: "4px", lineHeight: 1.5 }}>
                  {advisoryResult.action_plan.irrigation_advice}
                </p>
              </div>
            </div>

            {/* Crop Specific Protection */}
            <div style={{ background: "var(--c-bg)", padding: "12px", border: "1px solid var(--c-border)" }}>
              <div style={{ fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--c-sand)", marginBottom: "4px", textTransform: "uppercase" }}>
                Crop Protection & Phenological Defense
              </div>
              <p style={{ fontSize: "12px", color: "var(--c-text-secondary)", lineHeight: 1.5 }}>
                {advisoryResult.action_plan.crop_specific_protection}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Follow-up Farmer Agromet Chat */}
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            <Bot size={18} style={{ color: "var(--c-gold)" }} />
            <span>INTERACTIVE AGROMET REASONING CHAT</span>
          </div>
          <span className="panel-tag">GEMINI FLASH MULTI-TURN</span>
        </div>

        {/* Chat History Box */}
        <div style={{ minHeight: "140px", maxHeight: "280px", overflowY: "auto", background: "var(--c-bg)", border: "1px solid var(--c-border)", padding: "13px", marginBottom: "13px" }}>
          {chatMessages.length === 0 ? (
            <div style={{ textAlign: "center", color: "var(--c-text-muted)", padding: "20px", fontSize: "13px" }}>
              Ask specific field questions regarding chemical dosages, spray adjuvants, soil moisture retention, or microclimate timing.
            </div>
          ) : (
            chatMessages.map((msg, i) => (
              <div
                key={i}
                style={{
                  marginBottom: "10px",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: msg.role === "user" ? "flex-end" : "flex-start",
                }}
              >
                <div
                  style={{
                    maxWidth: "80%",
                    padding: "10px 14px",
                    background: msg.role === "user" ? "var(--c-espresso)" : "var(--c-surface-elevated)",
                    border: "1px solid var(--c-border)",
                    color: "var(--c-text-primary)",
                    fontSize: "13px",
                    lineHeight: 1.5,
                    whiteSpace: "pre-line",
                  }}
                >
                  {msg.content}
                </div>
                <span style={{ fontSize: "10px", fontFamily: "var(--font-mono)", color: "var(--c-text-muted)", marginTop: "2px" }}>
                  {msg.role === "user" ? "Farmer" : "A²E Agromet Agent"} • {msg.time}
                </span>
              </div>
            ))
          )}
        </div>

        {/* Chat Input */}
        <div style={{ display: "flex", gap: "8px" }}>
          <input
            type="text"
            className="input-text"
            placeholder="e.g. Can I mix mancozeb with insecticide in this weather, or should I spray at 07:00 tomorrow?"
            value={chatInput}
            onChange={(e) => setChatInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSendChat()}
            style={{ flex: 1 }}
          />
          <button
            className="btn-primary"
            onClick={handleSendChat}
            disabled={isChatSending || !chatInput.trim()}
          >
            <Send size={15} />
            {isChatSending ? "Reasoning..." : "Send Query"}
          </button>
        </div>
      </div>
    </div>
  );
};
