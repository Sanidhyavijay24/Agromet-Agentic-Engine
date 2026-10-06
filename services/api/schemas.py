"""
@file schemas.py
@description Shared Pydantic v2 data contracts for all SIH 26074 services.
             Every inter-service boundary MUST use these models to validate
             payloads. No service may accept or emit raw dicts at its public
             interface.
@module services/api
"""

from __future__ import annotations

from datetime import date as date_type, datetime, timezone
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
#  Primitives
# ─────────────────────────────────────────────────────────────────────────────


class GeoLocation(BaseModel):
    """
    Geographic reference point for a panchayat or observation station.

    Attributes:
        latitude:    WGS-84 decimal degrees, clamped to India's bounding box.
        longitude:   WGS-84 decimal degrees, clamped to India's bounding box.
        elevation_m: SRTM / DEM elevation in metres above sea level (optional).
        district:    Revenue district name (e.g., "Jaipur").
        panchayat:   Gram-panchayat name within the district (optional).
    """

    latitude: Annotated[float, Field(ge=-90.0, le=90.0, description="WGS-84 latitude")]
    longitude: Annotated[float, Field(ge=-180.0, le=180.0, description="WGS-84 longitude")]
    elevation_m: float | None = Field(
        default=None,
        ge=-500.0,
        le=9000.0,
        description="Elevation in metres (DEM-derived)",
    )
    district: str = Field(..., min_length=1, max_length=120, description="Revenue district name")
    panchayat: str | None = Field(
        default=None, max_length=120, description="Gram-panchayat name"
    )

    @field_validator("district", "panchayat", mode="before")
    @classmethod
    def strip_and_title(cls, v: str | None) -> str | None:
        """Normalise case and strip surrounding whitespace."""
        if v is None:
            return v
        return v.strip().title()


# ─────────────────────────────────────────────────────────────────────────────
#  Ingestion Layer
# ─────────────────────────────────────────────────────────────────────────────

SourceStatus = Literal["live", "mock", "cached"]


class RawBulletin(BaseModel):
    """
    Raw weather bulletin as downloaded from IMD or another authority.

    The ingestion layer emits this model; the LLM extractor consumes it.

    Attributes:
        source:        Data origin label (e.g., "IMD_NOWCAST", "IMD_DISTRICT_BULLETIN").
        timestamp:     UTC datetime of bulletin issuance or download.
        raw_text:      Full unprocessed bulletin text (PDF-extracted or HTML-scraped).
        metadata:      Arbitrary source-specific key-value metadata (station ID, URL, etc.).
        source_status: Fidelity status of origin data ('live', 'mock', or 'cached').
    """

    source: str = Field(..., min_length=1, max_length=100)
    timestamp: datetime = Field(..., description="UTC bulletin timestamp")
    raw_text: str = Field(..., min_length=0, max_length=200_000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_status: SourceStatus = Field(
        default="live",
        description="Fidelity status of origin bulletin ('live', 'mock', or 'cached')",
    )

    @field_validator("source", mode="before")
    @classmethod
    def upper_source(cls, v: str) -> str:
        return v.strip().upper()


class HourlyForecastPoint(BaseModel):
    """Single hourly observation / forecast row from Open-Meteo or equivalent."""

    time: datetime
    temperature_2m_c: float | None = None
    relative_humidity_2m_pct: float | None = None
    precipitation_mm: float | None = None
    surface_pressure_hpa: float | None = None
    wind_speed_10m_kmh: float | None = None
    # Microclimate & Agro-meteorological Covariates
    solar_radiation_w_m2: float | None = None
    shortwave_radiation_w_m2: float | None = None
    soil_temperature_0_to_7cm_c: float | None = None
    soil_moisture_0_to_7cm_m3m3: float | None = None
    et0_evapotranspiration_mm: float | None = None
    # Agromet operations -- used by the spray-window optimizer and advisory rules,
    # not by the ML downscaler.
    precipitation_probability_pct: float | None = None
    dew_point_2m_c: float | None = None
    wind_gusts_10m_kmh: float | None = None


class BaselineForecast(BaseModel):
    """
    7-day hourly + daily forecast for a single GeoLocation.

    Emitted by the ingestion layer and consumed by the ML downscaler.

    Attributes:
        location:        Resolved geographic context with elevation.
        hourly:          List of hourly forecast rows (up to 168 rows for 7 days).
        daily_max_temp:  Per-day maximum temperature (°C).
        daily_min_temp:  Per-day minimum temperature (°C).
        daily_rain_sum:  Per-day precipitation sum (mm).
        fetched_at:      UTC timestamp of API call.
        source:          Data provider identifier.
    """

    location: GeoLocation
    hourly: list[HourlyForecastPoint] = Field(default_factory=list)
    daily_dates: list[str] = Field(default_factory=list)
    daily_max_temp: list[float] = Field(default_factory=list)
    daily_min_temp: list[float] = Field(default_factory=list)
    daily_rain_sum: list[float] = Field(default_factory=list)
    daily_pop: list[int] = Field(default_factory=list)
    daily_weather_code: list[int] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = Field(default="OPEN_METEO")


# ───────────────────────────────────────────────────────────────────────────────
#  NASA POWER Layer
# ───────────────────────────────────────────────────────────────────────────────

_NASA_NODATA: float = -999.0


def _coerce_nasa_nodata(v: float | None) -> float | None:
    """Coerce NASA POWER fill value -999.0 to None (missing data sentinel)."""
    if v is not None and abs(v - _NASA_NODATA) < 0.01:
        return None
    return v


class NASAPowerDailyRecord(BaseModel):
    """
    Single-day NASA POWER Agroclimatology record for a point location.

    NASA POWER uses -999.0 as its no-data fill value. Any field receiving
    that value is coerced to None by the field validators below, preventing
    corrupted (negative) values from entering downstream models.

    NOTE: POWER has a 2–4 day ingestion latency. Callers must ensure the
          query window ends at least 4 days before today to avoid null rows.

    Attributes:
        date:                  ISO-8601 date string (YYYYMMDD from POWER API).
        t2m_c:                 Temperature at 2 m height (°C). None = missing.
        precip_corr_mm:        Bias-corrected precipitation (mm). None = missing.
        rh2m_pct:              Relative humidity at 2 m (%RH). None = missing.
        solar_radiation_mj_m2: All-sky surface solar radiation (MJ/m²). None = missing.
    """

    date: str = Field(..., pattern=r"^\d{8}$", description="YYYYMMDD date string from POWER")
    t2m_c: float | None = Field(default=None, description="2 m temperature (°C)")
    precip_corr_mm: float | None = Field(default=None, description="Bias-corrected precip (mm)")
    rh2m_pct: float | None = Field(default=None, description="2 m relative humidity (%RH)")
    solar_radiation_mj_m2: float | None = Field(
        default=None, description="All-sky surface downward solar radiation (MJ/m²)"
    )

    @field_validator("t2m_c", "precip_corr_mm", "rh2m_pct", "solar_radiation_mj_m2", mode="before")
    @classmethod
    def mask_nasa_nodata(cls, v: float | None) -> float | None:
        """Coerce NASA POWER -999.0 fill value to None."""
        return _coerce_nasa_nodata(v)


class NASAPowerResponse(BaseModel):
    """
    Container for a NASA POWER daily agroclimatology response.

    Attributes:
        latitude:      Query latitude.
        longitude:     Query longitude.
        start_date:    First date in window (YYYYMMDD).
        end_date:      Last date in window (YYYYMMDD, offset −4 days from today).
        records:       Ordered list of daily records.
        fetched_at:    UTC timestamp of API call.
    """

    latitude: float
    longitude: float
    start_date: str
    end_date: str
    records: list[NASAPowerDailyRecord] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ───────────────────────────────────────────────────────────────────────────────
#  ERA5 / CDS Layer
# ───────────────────────────────────────────────────────────────────────────────


class ERA5DownloadRequest(BaseModel):
    """
    Parameters for a single ERA5-Land historical reanalysis download job.

    Attributes:
        year:      4-digit year string (e.g., "2024").
        months:    List of zero-padded month strings (e.g., ["01", "02"]).
        days:      List of zero-padded day strings (e.g., ["01", ..., "31"]).
        variables: ERA5-Land variable shortnames.
        area:      Bounding box [north, west, south, east] in decimal degrees.
        output_path: Local filesystem path for the downloaded .nc file.
    """

    year: str = Field(..., pattern=r"^\d{4}$")
    months: list[str] = Field(..., min_length=1)
    days: list[str] = Field(..., min_length=1)
    variables: list[str] = Field(
        default=["2m_temperature", "total_precipitation", "surface_pressure"]
    )
    area: list[float] = Field(
        default=[32.0, 68.0, 8.0, 97.0],
        description="[north, west, south, east] bounding box",
    )
    output_path: str = Field(..., description="Absolute or relative path for .nc output")


# ─────────────────────────────────────────────────────────────────────────────
#  LLM Extractor Layer
# ─────────────────────────────────────────────────────────────────────────────


class ExtractedWeatherPayload(BaseModel):
    """
    Structured weather information extracted by the LLM from a RawBulletin.

    The LLM extractor MUST return this schema; it is validated before being
    forwarded downstream. Every parameter defaults to None or [] to ensure that
    unmentioned fields are never hallucinated.

    Attributes:
        district:            District name if mentioned in the bulletin.
        valid_from:          Forecast start date string (e.g. '04-09-2026').
        valid_to:            Forecast end date string (e.g. '08-09-2026').
        min_temp_c:          Minimum temperature forecast (°C); None if absent.
        max_temp_c:          Maximum temperature forecast (°C); None if absent.
        rainfall_min_mm:     Lower bound of numerical rainfall (mm); None if absent.
        rainfall_max_mm:     Upper bound of numerical rainfall (mm); None if absent.
        rainfall_qualifier:  Verbatim qualitative rainfall phrase (e.g. 'light to moderate')
                             when no numerical mm amounts are given.
        wind_speed_kmh:      Expected wind speed (km/h); None if absent.
        wind_direction:      Wind direction string (e.g., 'SW'); None if absent.
        weather_condition:   Human-readable condition label (e.g., 'Partly Cloudy'); None if absent.
        warnings:            List of active severe weather warnings (empty list [] if nil/none).
        advisory_text:       Full extracted advisory paragraph; None if absent.
        source_status:       Fidelity status propagated from RawBulletin ('live', 'mock', 'cached').
        source_hash:         SHA-256 digest hash of raw bulletin text for traceability.
    """

    district: str | None = Field(default=None, max_length=120)
    valid_from: str | None = Field(default=None, max_length=50)
    valid_to: str | None = Field(default=None, max_length=50)
    min_temp_c: float | None = Field(default=None, ge=-30.0, le=60.0)
    max_temp_c: float | None = Field(default=None, ge=-30.0, le=60.0)
    rainfall_min_mm: float | None = Field(default=None, ge=0.0, le=1000.0)
    rainfall_max_mm: float | None = Field(default=None, ge=0.0, le=1000.0)
    rainfall_qualifier: str | None = Field(default=None, max_length=200)
    wind_speed_kmh: float | None = Field(default=None, ge=0.0, le=400.0)
    wind_direction: str | None = Field(default=None, max_length=50)
    weather_condition: str | None = Field(default=None, max_length=200)
    warnings: list[str] = Field(default_factory=list)
    advisory_text: str | None = Field(default=None, max_length=10000)
    source_status: SourceStatus = Field(default="live")
    source_hash: str | None = Field(default=None, max_length=64)

    @field_validator("warnings", mode="before")
    @classmethod
    def sanitize_warnings(cls, v: Any) -> list[str]:
        """Strip whitespace and discard sentinel 'no warning' strings."""
        if not v:
            return []
        if isinstance(v, str):
            v = [v]
        cleaned: list[str] = []
        for w in v:
            if not isinstance(w, str):
                continue
            s = w.strip()
            if s.lower() in {
                "no warning issued",
                "nil",
                "none",
                "no warning",
                "no severe weather warning",
                "no severe weather warning issued",
                "nil warning",
            }:
                continue
            if s:
                cleaned.append(s)
        return cleaned

    @model_validator(mode="after")
    def check_temp_order(self) -> "ExtractedWeatherPayload":
        if (
            self.min_temp_c is not None
            and self.max_temp_c is not None
            and self.min_temp_c > self.max_temp_c
        ):
            raise ValueError(
                f"min_temp_c ({self.min_temp_c}) must be ≤ max_temp_c ({self.max_temp_c})"
            )
        return self

    @model_validator(mode="after")
    def check_rainfall_order(self) -> "ExtractedWeatherPayload":
        if (
            self.rainfall_min_mm is not None
            and self.rainfall_max_mm is not None
            and self.rainfall_min_mm > self.rainfall_max_mm
        ):
            raise ValueError(
                f"rainfall_min_mm ({self.rainfall_min_mm}) must be ≤ rainfall_max_mm ({self.rainfall_max_mm})"
            )
        return self


# ─────────────────────────────────────────────────────────────────────────────
#  ML Downscaler Layer
# ─────────────────────────────────────────────────────────────────────────────


class DownscaledForecast(BaseModel):
    """
    Single-variable downscaled forecast for a specific panchayat and time step.

    The ML downscaler emits one instance per (panchayat, variable, timestamp) triplet.

    Attributes:
        panchayat_id:     Unique panchayat identifier (e.g., LGD code).
        timestamp:        UTC datetime for this forecast step.
        target_variable:  Meteorological variable name (e.g., "temperature_2m_c").
        baseline_value:   Coarse NWP model value before downscaling.
        downscaled_value: Bias-corrected, high-resolution value after downscaling.
        confidence_score: Model confidence in [0.0, 1.0]; 1.0 = highest confidence.
        covariates_used:  Map of covariate name → value used in this prediction.
    """

    panchayat_id: str = Field(..., min_length=1, max_length=50)
    timestamp: datetime
    target_variable: str = Field(..., min_length=1, max_length=80)
    baseline_value: float
    downscaled_value: float
    confidence_score: Annotated[float, Field(ge=0.0, le=1.0)]
    covariates_used: dict[str, float] = Field(default_factory=dict)


class DownscaledForecastBatch(BaseModel):
    """Batch of downscaled forecasts for a single panchayat (all variables)."""

    panchayat_id: str
    location: GeoLocation
    forecasts: list[DownscaledForecast] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────────────────────────────────────────
#  Advisory Engine Layer
# ─────────────────────────────────────────────────────────────────────────────

RiskLevel = Literal["Low", "Moderate", "Severe"]


class TriggeredAdvisory(BaseModel):
    """
    One advisory rule that fired, with the evidence that made it fire.

    `evidence` holds plain-language reasons naming the value, the threshold, the day and
    the source, e.g. "daily minimum temperature <= 4 C within 72 h: 2026-12-18 (2.8 C)
    [hourly forecast]". It exists so a DAMU officer reviewing an automated advisory can
    see why it was issued without reading the rule file.
    """

    rule_id: str = Field(..., min_length=1, max_length=80)
    risk_level: RiskLevel
    recommended_action: str = Field(..., min_length=1, max_length=1000)
    vernacular_summary: dict[str, str] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    confidence_tag: Literal["SOURCED", "ASSUMPTION"]
    citation: str = Field(default="", max_length=1000)


SprayStatus = Literal["good", "marginal", "unsuitable"]


class SprayConcern(BaseModel):
    """One reason an hour is less than ideal for spraying."""

    code: str = Field(..., description="Stable machine code, e.g. WIND_STRONG, RAIN_SOON")
    severity: Literal["block", "caution"]
    message: str = Field(..., description="English, with the actual values")
    message_hi: str = Field(default="", description="Hindi")


class SprayHourAssessment(BaseModel):
    """Suitability of starting a spray at one forecast hour, and why."""

    time: datetime
    time_ist: str
    status: SprayStatus
    concerns: list[SprayConcern] = Field(default_factory=list)
    temperature_c: float | None = None
    relative_humidity_pct: float | None = None
    delta_t_c: float | None = None
    wind_kmh: float | None = None
    gust_kmh: float | None = None
    rain_in_rainfast_period_mm: float | None = None
    max_rain_probability_pct: float | None = None


class SprayWindow(BaseModel):
    """A run of consecutive usable hours long enough to spray a field."""

    start: datetime
    end: datetime
    label: str
    label_hi: str
    duration_h: int
    quality: Literal["good", "marginal"]
    good_hours: int
    marginal_hours: int
    cautions: list[str] = Field(default_factory=list)
    mean_wind_kmh: float | None = None
    max_rain_probability_pct: float | None = None
    mean_delta_t_c: float | None = None
    # The stretch inside this window with the fewest concerns, one spraying session long.
    best_slot_start: datetime | None = None
    best_slot_end: datetime | None = None
    best_slot_label: str = ""
    best_slot_label_hi: str = ""
    best_slot_cautions: list[str] = Field(default_factory=list)


class SprayWindowReport(BaseModel):
    """Ranked spray windows for one panchayat, with an hour-by-hour explanation."""

    profile: str
    profile_label: str
    profile_label_hi: str
    rainfast_hours: int
    horizon_h: int
    evaluated_from: datetime
    best_window: SprayWindow | None = Field(
        default=None, description="Earliest workable window: pests and disease rarely wait")
    cleanest_window: SprayWindow | None = Field(
        default=None, description="Earliest window whose best slot has no concerns at all, if any")
    windows: list[SprayWindow] = Field(default_factory=list)
    summary: str
    summary_hi: str
    blocking_hours_by_reason: dict[str, int] = Field(
        default_factory=dict,
        description="Working-hours blocked by each weather reason across the horizon",
    )
    hours: list[SprayHourAssessment] = Field(default_factory=list)
    data_gaps: list[str] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    provenance: str = ""


class DailyTemperatureRecord(BaseModel):
    """One day's temperature extremes, used for growing-degree-day accumulation."""

    date: date_type
    tmax_c: float | None = None
    tmin_c: float | None = None
    precipitation_mm: float | None = None
    source: Literal["archive", "forecast"] = "archive"


class PhenologyStageEvent(BaseModel):
    """When a crop stage begins, and how that date was obtained."""

    stage: str
    stage_label: str
    stage_label_hi: str
    gdd_threshold: float
    date: date_type | None = None
    basis: Literal["observed", "forecast", "extrapolated", "not_reached"]


class PhenologyReport(BaseModel):
    """Crop stage estimated from accumulated growing-degree days since sowing."""

    crop: str
    sowing_date: date_type
    as_of: date_type
    days_after_sowing: int
    base_temp_c: float
    upper_temp_c: float
    accumulated_gdd: float
    current_stage: str
    current_stage_label: str
    current_stage_label_hi: str
    stage_progress_pct: float | None = None
    next_stage: str | None = None
    next_stage_label: str | None = None
    gdd_to_next_stage: float | None = None
    next_stage_date: date_type | None = None
    next_stage_date_basis: Literal["forecast", "extrapolated"] | None = None
    timeline: list[PhenologyStageEvent] = Field(default_factory=list)
    registry_stage: str | None = Field(
        default=None, description="The static stage from the panchayat registry, for comparison")
    agrees_with_registry: bool | None = None
    observed_days: int = 0
    missing_days: int = 0
    method: str = ""
    confidence_tag: Literal["SOURCED", "ASSUMPTION"] = "ASSUMPTION"
    citation: str = ""
    note: str = ""
    summary: str = ""
    summary_hi: str = ""


RainfallCategory = Literal[
    "Large Excess", "Excess", "Normal", "Deficient", "Large Deficient", "No Rain", "Dry season"
]


class RainfallDeparture(BaseModel):
    """Rainfall over a period compared with its 1991-2020 normal, in IMD categories."""

    label: str
    start: date_type
    end: date_type
    days: int
    rainfall_mm: float
    normal_mm: float
    departure_pct: float | None = Field(
        default=None, description="None when the normal is too small for a percentage to mean anything")
    category: RainfallCategory
    category_hi: str
    summary: str
    summary_hi: str


class TemperatureAnomalyDay(BaseModel):
    """One day's forecast extremes against their normals, with IMD heat/cold-wave status."""

    date: date_type
    tmax_c: float | None = None
    tmax_normal_c: float | None = None
    tmax_departure_c: float | None = None
    tmin_c: float | None = None
    tmin_normal_c: float | None = None
    tmin_departure_c: float | None = None
    heat_wave: Literal["none", "heat wave", "severe heat wave"] = "none"
    cold_wave: Literal["none", "cold wave", "severe cold wave"] = "none"


class ClimatologyReport(BaseModel):
    """How the coming week and the season so far compare with the local climate."""

    source: str
    normal_period: str
    normal_years: int
    imd_season: str
    forecast_window: RainfallDeparture | None = None
    season_to_date: RainfallDeparture | None = None
    temperature: list[TemperatureAnomalyDay] = Field(default_factory=list)
    mean_tmax_departure_c: float | None = None
    mean_tmin_departure_c: float | None = None
    heat_wave_days: int = 0
    cold_wave_days: int = 0
    wave_criteria_applied: bool = True
    summary: str = ""
    summary_hi: str = ""
    provenance: str = ""


class AgrometAdvisoryResult(BaseModel):
    """
    Crop-specific agrometeorology advisory for a panchayat.

    Generated by the advisory_engine from DownscaledForecast + crop metadata.

    Attributes:
        panchayat_id:        Unique panchayat identifier.
        crop:                Crop name (e.g., "Wheat", "Paddy", "Cotton").
        crop_stage:          Phenological stage (e.g., "Tillering", "Grain Filling").
        risk_level:          Aggregated risk classification.
        recommended_action:  Primary recommended farm action in English.
        vernacular_summary:  Map of ISO 639-1 language code → translated summary.
                             Must include at least one entry.
        low_confidence:      Flag set to True if ML downscaling confidence < 0.80.
        source_status:       Origin data fidelity propagated from RawBulletin ('live', 'mock', 'cached').
        rainfall_qualifier:  Verbatim qualitative rainfall phrase from bulletin if present.
        matched_rules:       List of rule identifiers triggered during evaluation.
    """

    panchayat_id: str = Field(..., min_length=1, max_length=50)
    crop: str = Field(..., min_length=1, max_length=80)
    crop_stage: str = Field(..., min_length=1, max_length=80)
    risk_level: RiskLevel
    recommended_action: str = Field(..., min_length=1, max_length=1000)
    vernacular_summary: dict[str, str] = Field(
        ...,
        description="ISO 639-1 code → translated advisory (e.g., {'hi': '...', 'raj': '...'})",
    )
    low_confidence: bool = Field(
        default=False,
        description="Flag indicating if ML downscaled weather forecast has lower confidence (< 0.80)",
    )
    source_status: SourceStatus = Field(
        default="live",
        description="Fidelity status propagated from origin bulletin ('live', 'mock', 'cached')",
    )
    rainfall_qualifier: str | None = Field(
        default=None,
        max_length=200,
        description="Qualitative rainfall phrase extracted from bulletin if present",
    )
    matched_rules: list[str] = Field(
        default_factory=list,
        description="IDs of agronomic advisory rules triggered during evaluation",
    )
    triggered_advisories: list[TriggeredAdvisory] = Field(
        default_factory=list,
        description=(
            "Every triggered rule, most severe first, each with its own advice and the "
            "evidence behind it. risk_level / recommended_action above repeat the first entry."
        ),
    )
    weather_sources: list[str] = Field(
        default_factory=list,
        description="Weather evidence available for this evaluation: 'hourly forecast', 'IMD bulletin'",
    )

    @field_validator("vernacular_summary")
    @classmethod
    def at_least_one_language(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("vernacular_summary must contain at least one language entry")
        return v


# ─────────────────────────────────────────────────────────────────────────────
#  API Response Wrappers
# ─────────────────────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    """Specific error descriptor."""

    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    """Standardized error envelope."""

    success: bool = False
    error: ErrorDetail
    errors: list[ErrorDetail] = Field(default_factory=list)


class APIResponse(BaseModel):
    """Generic API response envelope used by the FastAPI gateway."""

    success: bool = True
    message: str = ""
    data: Any = None
    errors: list[ErrorDetail] = Field(default_factory=list)


class HealthCheckResponse(BaseModel):
    """Response model for GET /health and GET /api/v1/health."""

    status: Literal["ok", "degraded", "down"]
    version: str
    services: dict[str, Literal["ok", "degraded", "down"]] = Field(default_factory=dict)


class PanchayatSummary(BaseModel):
    """Summary record of a registered panchayat for directory listing."""

    panchayat_id: str
    name: str
    name_hi: str
    district: str
    block: str
    coordinates: GeoLocation
    crop: str
    crop_hi: str
    crop_stage: str


class DailyForecastPoint(BaseModel):
    """Single-day forecast record for 7-day outlook."""

    date: str
    temperature_max_c: float
    temperature_min_c: float
    precipitation_sum_mm: float = 0.0
    precipitation_probability_pct: int = 0
    weather_code: int = 0


class HyperlocalForecastData(BaseModel):
    """Unified payload returned by GET /api/v1/forecast/{panchayat_id}."""

    panchayat_id: str
    panchayat_name: str
    panchayat_name_hi: str
    district: str
    block: str
    coordinates: GeoLocation
    crop: str
    crop_stage: str
    downscaled_weather: DownscaledForecast
    baseline_weather: HourlyForecastPoint
    advisory: AgrometAdvisoryResult
    bhoonidhi: dict[str, Any] | None = None
    daily_forecast: list[DailyForecastPoint] = Field(default_factory=list)
    spray_windows: SprayWindowReport | None = None
    phenology: PhenologyReport | None = None
    climatology: ClimatologyReport | None = None
    agro_climatic_zone: dict[str, Any] | None = None
    cached: bool = False



# ─────────────────────────────────────────────────────────────────────────────
#  MOSDAC / ISRO Satellite Layer
# ─────────────────────────────────────────────────────────────────────────────


class MosdacGranule(BaseModel):
    """Metadata record for a satellite granule returned by MOSDAC search."""

    identifier: str
    granule_id: str
    dataset_id: str
    summary: str = ""
    updated: datetime | None = None
    enclosure_link: str = ""
    search_link: str = ""
    bound_box: dict[str, Any] = Field(default_factory=dict)


class MosdacTelemetry(BaseModel):
    """
    Visual atmospheric/cloud observation record derived from MOSDAC INSAT-3DR/3DS.

    Used strictly as a visual overlay in frontend interfaces.
    """

    panchayat_id: str
    satellite: str = "INSAT-3DR"
    sensor: str = "IMAGER"
    dataset_id: str = "3RIMG_L2B_CMK"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cloud_fraction_pct: float = Field(
        ..., ge=0.0, le=100.0, description="Cloud cover percentage (0-100%)"
    )
    cloud_top_temp_c: float | None = Field(
        default=None, description="Cloud top brightness temperature (°C)"
    )
    rainfall_rate_mmh: float = Field(
        default=0.0, ge=0.0, description="Hydro-Estimator precipitation rate (mm/h)"
    )
    source_status: Literal["live", "cached", "mock"] = "cached"
    attribution: str = Field(
        default="Data Source: MOSDAC/SAC/ISRO. https://mosdac.gov.in",
        description="Mandatory MOSDAC data citation",
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Bhoonidhi / NRSC / ISRO Satellite Layer
# ─────────────────────────────────────────────────────────────────────────────


class BhoonidhiSoilMoisture(BaseModel):
    """Bhoonidhi EOS-04 SAR 0-7cm surface soil moisture product."""

    soil_moisture_m3_m3: float = Field(
        ..., description="Volumetric soil moisture (m³/m³)"
    )
    layer: str = "0-7cm SAR Soil Moisture"
    satellite: str = "EOS-04 (RISAT-1A SAR MRS)"
    source: str = "Bhoonidhi / NRSC / ISRO"
    attribution: str = "Data Source: Bhoonidhi / NRSC / ISRO (Indian Space Policy 2023). https://bhoonidhi.nrsc.gov.in"
    matched_items: int = 1
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BhoonidhiCartoDEM(BaseModel):
    """Bhoonidhi CartoSat-1 30m Digital Elevation Model product."""

    collection: str = "CartoSat-1_PAN_CartoDEM_30m"
    resolution: str = "30m Indian CartoDEM"
    sensor: str = "CartoSat-1 PAN Stereo"
    attribution: str = "Data Source: Bhoonidhi / NRSC / ISRO (Indian Space Policy 2023). https://bhoonidhi.nrsc.gov.in"
    matched_items: int = 1
    status: str = "ready"


class BhoonidhiTelemetry(BaseModel):
    """
    Bhoonidhi (NRSC / ISRO) Earth Observation satellite telemetry payload.
    """

    panchayat_id: str
    panchayat_name: str
    district: str
    coordinates: GeoLocation
    soil_moisture: BhoonidhiSoilMoisture | dict[str, Any]
    cartodem: BhoonidhiCartoDEM | dict[str, Any]
    attribution: str = Field(
        default="Data Source: Bhoonidhi / NRSC / ISRO (Indian Space Policy 2023). https://bhoonidhi.nrsc.gov.in",
        description="Mandatory ISRO / Bhoonidhi data citation",
    )


