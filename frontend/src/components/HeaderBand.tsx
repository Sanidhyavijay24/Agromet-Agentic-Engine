import React from "react";

interface HeaderBandProps {
  activeView: "main" | "accuracy";
  onSelectView: (view: "main" | "accuracy") => void;
  lastUpdatedText?: string;
}

export const HeaderBand: React.FC<HeaderBandProps> = ({
  activeView,
  onSelectView,
  lastUpdatedText = "Demo snapshot • Today, 14:00 IST",
}) => {
  return (
    <header className="header-band" role="banner">
      <div className="header-container">
        <div className="header-titles">
          <div className="header-title-row">
            <span className="header-title">Agromet Agentic Engine (A²E)</span>
            <span className="header-badge">Zone XIV • Rajasthan</span>
          </div>
          <p className="header-subtitle">
            1 km microclimate temperature downscaling and farm advisory for semi-arid Rajasthan.
          </p>
        </div>

        <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "8px" }}>
          <nav className="header-nav" aria-label="Main Navigation">
            <button
              type="button"
              className={`nav-link-btn ${activeView === "main" ? "active" : ""}`}
              onClick={() => onSelectView("main")}
              aria-current={activeView === "main" ? "page" : undefined}
            >
              Map & Advisory
            </button>
            <button
              type="button"
              className={`nav-link-btn ${activeView === "accuracy" ? "active" : ""}`}
              onClick={() => onSelectView("accuracy")}
              aria-current={activeView === "accuracy" ? "page" : undefined}
            >
              Coverage & accuracy
            </button>
          </nav>
          <div className="header-meta">
            <span>{lastUpdatedText}</span>
          </div>
        </div>
      </div>
    </header>
  );
};
