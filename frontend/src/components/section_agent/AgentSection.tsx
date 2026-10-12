/**
 * @file AgentSection.tsx
 * @description Section 2: Autonomous ReAct Agromet Agent Terminal on Drafting Paper Sheet
 * @module frontend/components/section_agent
 */

import React, { useState } from "react";
import { SCENARIO_PRESETS, generateMockAdvisory } from "../../services/api";
import { AdvisoryActionPlan, ScenarioPreset } from "../../types";
import { AdvisoryPlanCard } from "./AdvisoryPlanCard";
import { ReActTraceViewer } from "./ReActTraceViewer";
import { ScenarioPresets } from "./ScenarioPresets";

interface AgentSectionProps {
  lat: number;
  lon: number;
}

export const AgentSection: React.FC<AgentSectionProps> = ({ lat, lon }) => {
  const [selectedScenario, setSelectedScenario] = useState<ScenarioPreset>(SCENARIO_PRESETS[0]);
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [advisoryPlan, setAdvisoryPlan] = useState<AdvisoryActionPlan>(() =>
    generateMockAdvisory(SCENARIO_PRESETS[0].id, lat, lon)
  );

  const handleRunAnalysis = () => {
    setIsRunning(true);
    setTimeout(() => {
      const plan = generateMockAdvisory(selectedScenario.id, lat, lon);
      setAdvisoryPlan(plan);
      setIsRunning(false);
    }, 450);
  };

  const handleSelectScenario = (scenario: ScenarioPreset) => {
    setSelectedScenario(scenario);
    const plan = generateMockAdvisory(scenario.id, lat, lon);
    setAdvisoryPlan(plan);
  };

  return (
    <section className="paper-sheet" id="section-agent" style={{ marginTop: "36px" }}>
      <div className="sheet-header">
        <div>
          <div className="sheet-number">MODULE 02 • AUTONOMOUS AGENTIC INTELLIGENCE</div>
          <h2 className="sheet-title">ReAct Agromet Advisory Terminal</h2>
        </div>
        <div className="sheet-caption">
          Autonomous decision-making powered by Google Gemini Flash. The agent dynamically queries microclimate tools (downscaler, nocturnal inversion index, VPD, and ICAR agronomy thresholds) to synthesize structured bilingual farmer action plans.
        </div>
      </div>

      {/* Scenario Presets & Mission Trigger */}
      <ScenarioPresets
        selectedScenario={selectedScenario}
        onSelectScenario={handleSelectScenario}
        isRunning={isRunning}
        onRunAnalysis={handleRunAnalysis}
      />

      {/* ReAct Tool Execution Trace */}
      <ReActTraceViewer
        steps={advisoryPlan.steps}
        totalLatencyMs={advisoryPlan.steps.reduce((sum, s) => sum + s.duration_ms, 0)}
      />

      {/* Synthesized Agromet Action Plan Card */}
      <AdvisoryPlanCard plan={advisoryPlan} />
    </section>
  );
};
