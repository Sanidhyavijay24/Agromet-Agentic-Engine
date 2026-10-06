"""
@file conftest.py
@description Shared test configuration and markers for Agromet Agentic Engine.
@module tests
"""

from __future__ import annotations

from pathlib import Path
import pytest

from services.ml_downscaler.inference import DEFAULT_MODEL_PATH

MODEL_AVAILABLE = DEFAULT_MODEL_PATH.exists()
requires_model = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason="Requires Zone XIV model artifact (residual_model_acz_14.joblib). Run 'python scripts/fetch_artifacts.py' to download.",
)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "requires_model: mark test as requiring trained model artifact")
    config.addinivalue_line("markers", "requires_network: mark test as requiring network access")
