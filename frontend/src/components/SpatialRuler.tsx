/**
 * @file SpatialRuler.tsx
 * @description Architectural spatial resolution ruler pinned to the viewport margin
 * @module frontend/components
 */

import React from "react";

interface ScaleTick {
  label: string;
  sublabel: string;
  isMajor: boolean;
}

const SCALE_TICKS: ScaleTick[] = [
  { label: "50 km", sublabel: "ERA5 Global Reanalysis", isMajor: true },
  { label: "30 km", sublabel: "Global Numerical Models", isMajor: false },
  { label: "25 km", sublabel: "GFS / IMD Regional Grid", isMajor: true },
  { label: "15 km", sublabel: "NCMRWF Meso-Scale", isMajor: false },
  { label: "10 km", sublabel: "High-Res WRF Physics", isMajor: true },
  { label: "5 km", sublabel: "Sub-Mesoscale Convection", isMajor: false },
  { label: "3 km", sublabel: "Topographic Interpolation", isMajor: false },
  { label: "1 km", sublabel: "A²E Physics-ML Downscaled", isMajor: true },
];

export const SpatialRuler: React.FC = () => {
  return (
    <aside className="spatial-ruler" aria-label="Spatial Resolution Scale" title="Spatial Resolution Scale">
      <div className="ruler-tick-container">
        {SCALE_TICKS.map((tick, idx) => (
          <div
            key={idx}
            className={`ruler-mark ${tick.isMajor ? "major" : ""}`}
            title={`${tick.label}: ${tick.sublabel}`}
          >
            {tick.isMajor && (
              <span className="ruler-label">
                {tick.label}
              </span>
            )}
          </div>
        ))}
      </div>
    </aside>
  );
};
