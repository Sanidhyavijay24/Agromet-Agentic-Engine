"""
@file __init__.py
@description Public API of the ingestion service package.

             Import directly from `services.ingestion` to get all key fetchers.
             This keeps downstream code decoupled from internal module names.
@module services/ingestion

Usage:
    from services.ingestion import (
        # Open-Meteo
        fetch_baseline_forecast,
        fetch_elevation,
        # IMD public (agromet + nowcast + current wx)
        fetch_agromet_bulletin,
        fetch_imd_nowcast,
        fetch_imd_current_weather,
        # NASA POWER
        fetch_nasa_power,
        # ERA5-Land / CDS
        verify_cds_connection,
        download_era5_land,
        build_download_request,
        # Offline utilities
        read_mock_bulletin,
    )
"""

from services.ingestion.open_meteo_client import (
    fetch_baseline_forecast,
    fetch_elevation,
)
from services.ingestion.imd_public_client import (
    fetch_agromet_bulletin,
    fetch_imd_current_weather,
    fetch_imd_nowcast,
    read_mock_bulletin,
)
from services.ingestion.mosdac_client import (
    MOSDAC_ATTRIBUTION,
    authenticate_mosdac,
    fetch_mosdac_telemetry,
    prefetch_mosdac_panchayats,
    search_mosdac_datasets,
)
from services.ingestion.bhoonidhi_client import (
    BHOONIDHI_ATTRIBUTION,
    authenticate_bhoonidhi,
    fetch_bhoonidhi_cartodem_elevation,
    fetch_bhoonidhi_soil_moisture,
    search_bhoonidhi_catalog,
)

__all__ = [
    # Open-Meteo
    "fetch_baseline_forecast",
    "fetch_elevation",
    # IMD public
    "fetch_agromet_bulletin",
    "fetch_imd_nowcast",
    "fetch_imd_current_weather",
    # MOSDAC / ISRO
    "search_mosdac_datasets",
    "authenticate_mosdac",
    "fetch_mosdac_telemetry",
    "prefetch_mosdac_panchayats",
    "MOSDAC_ATTRIBUTION",
    # Bhoonidhi / NRSC / ISRO
    "search_bhoonidhi_catalog",
    "authenticate_bhoonidhi",
    "fetch_bhoonidhi_soil_moisture",
    "fetch_bhoonidhi_cartodem_elevation",
    "BHOONIDHI_ATTRIBUTION",
    # Offline utilities
    "read_mock_bulletin",
]
