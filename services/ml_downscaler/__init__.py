"""
@file __init__.py
@description Public exports for the ML downscaler subsystem, resolved lazily.
@module services/ml_downscaler

WHY LAZY (PEP 562)
------------------
These exports used to be eager `from ... import ...` statements at module scope, which
meant that importing ANY submodule pulled in the whole subsystem. Importing
`services.ml_downscaler.dem_source` -- a small numpy-only DEM reader -- transitively
required tenacity (via dataset.py) and xgboost (via train.py), so a diagnostic script
whose entire job was to report missing dependencies could not start without them.

Module-level `__getattr__` keeps the public API byte-identical:
    from services.ml_downscaler import train_residual_model     # still works
    from services.ml_downscaler import FEATURES_LIST            # still works
while deferring each heavy import until the name is actually touched. It also removes
xgboost from the FastAPI gateway's import path until the first prediction.
"""

from __future__ import annotations

from typing import Any

# exported name -> submodule that defines it
_EXPORTS: dict[str, str] = {
    # Dataset & Features
    "DEFAULT_BOUNDING_BOX": "dataset",
    "generate_grid_coordinates": "dataset",
    "fetch_historical_point_archive": "dataset",
    "engineer_downscaling_features": "dataset",
    "extract_downscaling_dataset": "dataset",
    # Training
    "FEATURES_LIST": "train",
    "load_training_dataset": "train",
    "spatial_block_split": "train",
    "spatial_three_way_split": "train",
    "train_residual_model": "train",
    # Inference
    "load_downscaler_model": "inference",
    "downscale_point_forecast": "inference",
    "downscale_forecast_batch": "inference",
    # Terrain (numpy only, no heavy dependencies)
    "extract_terrain": "dem_source",
    "dem_availability": "dem_source",
    "discover_tiles": "dem_source",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public export on first access by importing only its own submodule."""
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{submodule}")
    value = getattr(module, name)
    globals()[name] = value  # cache so later lookups skip this path
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
