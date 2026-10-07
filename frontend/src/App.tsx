import React, { useState } from "react";
import { HeaderBand } from "./components/HeaderBand";

export const App: React.FC = () => {
  const [activeView, setActiveView] = useState<"main" | "accuracy">("main");

  return (
    <div className="app-shell">
      <HeaderBand activeView={activeView} onSelectView={setActiveView} />

      <main className="main-wrapper" id="main-content" tabIndex={-1}>
        {activeView === "main" ? (
          <div>
            {/* The single narrative flow:
                1. Map with time scrubber and coarse/downscaled toggle
                2. Detail panel for selected point (curves + advisory form)
                3. Advisory result
            */}
            <section className="flow-section" id="section-map">
              <div className="section-header">
                <div className="section-title-wrap">
                  <span className="section-number">01</span>
                  <h2 className="section-title">Microclimate Map & Hourly Field</h2>
                </div>
                <span className="section-desc">
                  Zone XIV (26°–28° N, 75°–77° E) • 1 km downscaled temperature surface
                </span>
              </div>
              <div style={{ padding: "24px", color: "var(--c-ink-secondary)" }}>
                {/* Map will be mounted here in Phase 2 */}
                <p>Map component loading (Phase 2)...</p>
              </div>
            </section>

            <section className="flow-section" id="section-detail">
              <div className="section-header">
                <div className="section-title-wrap">
                  <span className="section-number">02</span>
                  <h2 className="section-title">Point Detail & Diurnal Curve</h2>
                </div>
                <span className="section-desc">
                  Coarse baseline vs 1 km microclimate comparison for selected coordinate
                </span>
              </div>
              <div style={{ padding: "24px", color: "var(--c-ink-secondary)" }}>
                {/* Detail Panel and SVG Chart will be mounted here in Phase 3 */}
                <p>Point detail and diurnal temperature curve loading (Phase 3)...</p>
              </div>
            </section>

            <section className="flow-section" id="section-advisory">
              <div className="section-header">
                <div className="section-title-wrap">
                  <span className="section-number">03</span>
                  <h2 className="section-title">Farm Advisory</h2>
                </div>
                <span className="section-desc">
                  Agronomic rules and spray/irrigation guidance
                </span>
              </div>
              <div style={{ padding: "24px", color: "var(--c-ink-secondary)" }}>
                {/* Advisory result and audit trace will be mounted here in Phase 6 */}
                <p>Advisory result and tool reasoning log loading (Phase 6)...</p>
              </div>
            </section>
          </div>
        ) : (
          <div className="flow-section" id="section-accuracy">
            <div className="section-header">
              <div className="section-title-wrap">
                <span className="section-number">REF</span>
                <h2 className="section-title">Model Coverage & Benchmark Accuracy</h2>
              </div>
              <span className="section-desc">
                Empirical validation results across Zone XIV and Pan-India 15 ACZ fleet
              </span>
            </div>
            <div style={{ padding: "24px", color: "var(--c-ink-secondary)" }}>
              {/* Coverage & accuracy page content will be mounted here in Phase 7 */}
              <p>Coverage & accuracy reference view loading (Phase 7)...</p>
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <div className="footer-container">
          <div>
            <strong>Agromet Agentic Engine (A²E)</strong> • Portfolio demonstration for Rajasthan microclimate downscaling.
          </div>
          <div>
            Baseline: Leave-one-out lapse rate (-6.5 °C/km) • Target: ERA5-Land reanalysis.
          </div>
        </div>
      </footer>
    </div>
  );
};
