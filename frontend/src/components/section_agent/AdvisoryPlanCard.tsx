/**
 * @file AdvisoryPlanCard.tsx
 * @description Structured bilingual Agromet Action Plan card (English & Hindi)
 * @module frontend/components/section_agent
 */

import React, { useState } from "react";
import { AlertTriangle, Award, CheckCircle2, Droplets, FileText, Globe, ShieldAlert, Sparkles, Wind } from "lucide-react";
import { AdvisoryActionPlan } from "../../types";

interface AdvisoryPlanCardProps {
  plan: AdvisoryActionPlan;
}

export const AdvisoryPlanCard: React.FC<AdvisoryPlanCardProps> = ({ plan }) => {
  const [selectedLang, setSelectedLang] = useState<"en" | "hi">("en");

  const riskBadgeClass =
    plan.risk_level === "CRITICAL"
      ? "badge-tag danger"
      : plan.risk_level === "HIGH"
      ? "badge-tag warn"
      : "badge-tag safe";

  return (
    <div className="draft-card" style={{ border: "2px solid var(--c-ink-heading)", background: "var(--c-paper-card)" }}>
      {/* Action Plan Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "12px", borderBottom: "1.5px solid var(--c-paper-border-strong)", paddingBottom: "14px", marginBottom: "16px" }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
            <span className="badge-tag safe" style={{ background: "var(--c-blueprint-bg)", color: "#FFFFFF" }}>
              ACTION PLAN
            </span>
            <span style={{ fontFamily: "var(--font-mono)", fontSize: "11px", color: "var(--c-ink-muted)" }}>
              {plan.advisory_id}
            </span>
          </div>

          <h3 style={{ margin: "6px 0 0 0", fontSize: "18px", fontWeight: 750, color: "var(--c-ink-heading)", fontFamily: "var(--font-display)" }}>
            Bilingual Agrometeorological Advisory
          </h3>
          <span style={{ fontSize: "12px", color: "var(--c-ink-secondary)" }}>
            Target: {plan.crop} ({plan.growth_stage}) • {plan.location_name} • Calibrated to ICAR GKMS Standards
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
          <span className={riskBadgeClass} style={{ fontSize: "12px", padding: "4px 10px" }}>
            {plan.risk_level === "CRITICAL" && <ShieldAlert size={13} />}
            {plan.risk_level === "HIGH" && <AlertTriangle size={13} />}
            {plan.risk_level === "LOW" && <CheckCircle2 size={13} />}
            RISK: {plan.risk_level}
          </span>

          {/* Language Switcher */}
          <div style={{ display: "flex", background: "var(--c-paper-sheet)", border: "1px solid var(--c-paper-border)", borderRadius: "var(--radius-xs)", padding: "2px" }}>
            <button
              onClick={() => setSelectedLang("en")}
              style={{
                background: selectedLang === "en" ? "var(--c-ink-heading)" : "transparent",
                color: selectedLang === "en" ? "#FFFFFF" : "var(--c-ink-body)",
                border: "none",
                borderRadius: "var(--radius-xs)",
                padding: "4px 10px",
                fontSize: "11px",
                fontFamily: "var(--font-mono)",
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              ENGLISH
            </button>
            <button
              onClick={() => setSelectedLang("hi")}
              style={{
                background: selectedLang === "hi" ? "var(--c-ink-heading)" : "transparent",
                color: selectedLang === "hi" ? "#FFFFFF" : "var(--c-ink-body)",
                border: "none",
                borderRadius: "var(--radius-xs)",
                padding: "4px 10px",
                fontSize: "11px",
                fontFamily: "var(--font-mono)",
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              हिन्दी (HINDI)
            </button>
          </div>
        </div>
      </div>

      {/* Advisory Text Box */}
      <div
        style={{
          background: "var(--c-paper-sheet)",
          border: "1px solid var(--c-paper-border-strong)",
          borderRadius: "var(--radius-xs)",
          padding: "16px 20px",
          marginBottom: "16px",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "8px", color: "var(--c-ink-muted)", fontSize: "11px", fontFamily: "var(--font-mono)", fontWeight: 700 }}>
          <Globe size={12} />
          <span>{selectedLang === "en" ? "SYNTHESIZED ADVISORY (ENGLISH)" : "कृषि मौसम परामर्श (हिन्दी)"}</span>
        </div>

        <p
          style={{
            margin: 0,
            fontSize: selectedLang === "hi" ? "15.5px" : "14.5px",
            lineHeight: 1.6,
            color: "var(--c-ink-heading)",
            fontFamily: selectedLang === "hi" ? "var(--font-hindi)" : "var(--font-body)",
            fontWeight: 500,
          }}
        >
          {selectedLang === "en" ? plan.advisory_en : plan.advisory_hi}
        </p>
      </div>

      {/* Domain Directives: Spray Window & Irrigation */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: "12px", marginBottom: "16px" }}>
        {/* Spray Window Directive */}
        <div style={{ background: "var(--c-paper-sheet)", border: "1px solid var(--c-paper-border)", borderRadius: "var(--radius-xs)", padding: "14px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "6px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--c-ink-secondary)" }}>
              <Wind size={13} />
              <span>CHEMICAL SPRAY DIRECTIVE</span>
            </div>
            {plan.spray_safe ? (
              <span className="badge-tag safe" style={{ fontSize: "10px" }}>
                <CheckCircle2 size={10} />
                SAFE WINDOW
              </span>
            ) : (
              <span className="badge-tag danger" style={{ fontSize: "10px" }}>
                <AlertTriangle size={10} />
                POSTPONE
              </span>
            )}
          </div>
          <div style={{ fontSize: "12.5px", color: "var(--c-ink-body)", lineHeight: 1.45 }}>
            {plan.spray_window_recommendation}
          </div>
        </div>

        {/* Irrigation Directive */}
        <div style={{ background: "var(--c-paper-sheet)", border: "1px solid var(--c-paper-border)", borderRadius: "var(--radius-xs)", padding: "14px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "6px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "6px", fontSize: "11px", fontFamily: "var(--font-mono)", fontWeight: 700, color: "var(--c-ink-secondary)" }}>
              <Droplets size={13} style={{ color: "#2563EB" }} />
              <span>IRRIGATION SCHEDULE & THERMAL BUFFER</span>
            </div>
            <span className="badge-tag warn" style={{ fontSize: "10px" }}>
              ACTIVE GUIDANCE
            </span>
          </div>
          <div style={{ fontSize: "12.5px", color: "var(--c-ink-body)", lineHeight: 1.45 }}>
            {plan.irrigation_guidance}
          </div>
        </div>
      </div>

      {/* Provenance Footnote */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--c-ink-muted)", borderTop: "1px dashed var(--c-paper-border)", paddingTop: "10px", flexWrap: "wrap", gap: "8px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <Sparkles size={11} style={{ color: "var(--c-curve-ml)" }} />
          <span>Synthesis Model: {plan.model_source}</span>
        </div>
        <div>
          <span>Verification Latency: {plan.latency_ms} ms</span>
        </div>
      </div>
    </div>
  );
};
