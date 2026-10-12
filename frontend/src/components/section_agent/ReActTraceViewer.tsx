/**
 * @file ReActTraceViewer.tsx
 * @description Cognitive ReAct tool execution trace viewer (Thought → Tool Call → Observation)
 * @module frontend/components/section_agent
 */

import React, { useState } from "react";
import { CheckCircle2, ChevronDown, ChevronRight, Clock, Code, Cpu, Eye, HelpCircle } from "lucide-react";
import { ReActStep } from "../../types";

interface ReActTraceViewerProps {
  steps: ReActStep[];
  totalLatencyMs: number;
}

export const ReActTraceViewer: React.FC<ReActTraceViewerProps> = ({ steps, totalLatencyMs }) => {
  const [expandedStep, setExpandedStep] = useState<number | null>(1); // Expand step 1 by default

  return (
    <div className="draft-card" style={{ marginBottom: "20px" }}>
      <div className="draft-card-header">
        <div className="draft-card-title" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
          <Cpu size={14} style={{ color: "var(--c-curve-ml)" }} />
          <span>Autonomous Cognitive Trace (Multi-Step ReAct Loop)</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "8px", fontSize: "11px", fontFamily: "var(--font-mono)" }}>
          <span style={{ color: "var(--c-ink-muted)" }}>Execution Pipeline:</span>
          <span className="badge-tag safe">
            <Clock size={11} />
            Total: {totalLatencyMs} ms
          </span>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
        {steps.map((step) => {
          const isExpanded = expandedStep === step.step_number;
          return (
            <div
              key={step.step_number}
              style={{
                border: "1px solid var(--c-paper-border-strong)",
                borderRadius: "var(--radius-xs)",
                background: "var(--c-paper-sheet)",
                overflow: "hidden",
              }}
            >
              {/* Step Header */}
              <div
                onClick={() => setExpandedStep(isExpanded ? null : step.step_number)}
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  padding: "10px 14px",
                  cursor: "pointer",
                  background: isExpanded ? "var(--c-paper-card)" : "transparent",
                  borderBottom: isExpanded ? "1px solid var(--c-paper-border)" : "none",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      justifyContent: "center",
                      width: "22px",
                      height: "22px",
                      borderRadius: "50%",
                      backgroundColor: "var(--c-blueprint-dark)",
                      color: "#FFFFFF",
                      fontFamily: "var(--font-mono)",
                      fontSize: "11px",
                      fontWeight: 700,
                    }}
                  >
                    {step.step_number}
                  </span>

                  <div>
                    <span style={{ fontFamily: "var(--font-mono)", fontSize: "12px", fontWeight: 700, color: "var(--c-ink-heading)" }}>
                      TOOL: {step.tool_name}
                    </span>
                    <span style={{ marginLeft: "8px", fontSize: "11px", color: "var(--c-ink-muted)" }}>
                      ({step.duration_ms} ms)
                    </span>
                  </div>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  <span className="badge-tag safe" style={{ fontSize: "10.5px" }}>
                    <CheckCircle2 size={11} />
                    Verified
                  </span>
                  {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                </div>
              </div>

              {/* Step Body (Thought, Args, Observation) */}
              {isExpanded && (
                <div style={{ padding: "14px", display: "flex", flexDirection: "column", gap: "12px", background: "var(--c-paper-sheet)" }}>
                  {/* Thought */}
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--c-curve-ml)", fontWeight: 700, marginBottom: "3px" }}>
                      <HelpCircle size={12} />
                      <span>AGENT REASONING / THOUGHT:</span>
                    </div>
                    <div style={{ fontSize: "13px", color: "var(--c-ink-body)", paddingLeft: "17px", borderLeft: "2px solid var(--c-curve-ml)" }}>
                      {step.thought}
                    </div>
                  </div>

                  {/* Tool Arguments */}
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--c-ink-secondary)", fontWeight: 700, marginBottom: "4px" }}>
                      <Code size={12} />
                      <span>STRUCTURED ARGUMENTS (PYDANTIC V2):</span>
                    </div>
                    <pre
                      style={{
                        margin: 0,
                        padding: "8px 12px",
                        background: "#1E293B",
                        color: "#E2E8F0",
                        borderRadius: "var(--radius-xs)",
                        fontSize: "11px",
                        fontFamily: "var(--font-mono)",
                        overflowX: "auto",
                      }}
                    >
                      {JSON.stringify(step.tool_args, null, 2)}
                    </pre>
                  </div>

                  {/* Observation */}
                  <div>
                    <div style={{ display: "flex", alignItems: "center", gap: "5px", fontSize: "11px", fontFamily: "var(--font-mono)", color: "var(--c-accent-safe)", fontWeight: 700, marginBottom: "3px" }}>
                      <Eye size={12} />
                      <span>OBSERVATION & TELEMETRY RESULT:</span>
                    </div>
                    <div style={{ fontSize: "12.5px", color: "var(--c-ink-heading)", background: "var(--c-paper-card)", border: "1px solid var(--c-paper-border)", borderRadius: "var(--radius-xs)", padding: "10px 12px" }}>
                      {step.observation}
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
