/**
 * @file DownscalerSection.tsx
 * @description Section 1: Interactive Microclimate Downscaler Studio on Drafting Paper Sheet
 * @module frontend/components/section_downscaler
 */

import React, { useMemo, useState } from "react";
import { COORDINATE_PRESETS, calculateTerrainFeatures, generateDiurnalCurve } from "../../services/api";
import { CoordinatePreset } from "../../types";
import { CoordinatePicker } from "./CoordinatePicker";
import { DiurnalCurveChart } from "./DiurnalCurveChart";
import { TerrainFeatureBar } from "./TerrainFeatureBar";

interface DownscalerSectionProps {
  lat: number;
  lon: number;
  onChangeLat: (lat: number) => void;
  onChangeLon: (lon: number) => void;
}

export const DownscalerSection: React.FC<DownscalerSectionProps> = ({
  lat,
  lon,
  onChangeLat,
  onChangeLon,
}) => {
  const [selectedPreset, setSelectedPreset] = useState<CoordinatePreset>(COORDINATE_PRESETS[0]);

  // Compute terrain features and diurnal series dynamically
  const terrainFeatures = useMemo(() => calculateTerrainFeatures(lat, lon), [lat, lon]);
  const diurnalData = useMemo(() => generateDiurnalCurve(lat, lon), [lat, lon]);

  return (
    <section className="paper-sheet" id="section-downscaler">
      <div className="sheet-header">
        <div>
          <div className="sheet-number">MODULE 01 • SPATIAL TERRAIN PHYSICS</div>
          <h2 className="sheet-title">1 km² Microclimate Downscaler</h2>
        </div>
        <div className="sheet-caption">
          Coarse numerical weather models (ERA5/GFS 25–50 km) miss localized cold-air pooling, slope aspect insolation, and elevation lapse rates. Our 28-feature model resolves microclimates down to 1 km² on 30m USGS SRTM DEM rasters.
        </div>
      </div>

      {/* Coordinate & Preset Picker */}
      <CoordinatePicker
        selectedPreset={selectedPreset}
        onSelectPreset={setSelectedPreset}
        lat={lat}
        lon={lon}
        onChangeLat={onChangeLat}
        onChangeLon={onChangeLon}
      />

      {/* 30m SRTM Terrain Features Bar */}
      <TerrainFeatureBar features={terrainFeatures} />

      {/* 24-Hour Diurnal Curve SVG */}
      <DiurnalCurveChart data={diurnalData} locationName={selectedPreset.name} />
    </section>
  );
};
