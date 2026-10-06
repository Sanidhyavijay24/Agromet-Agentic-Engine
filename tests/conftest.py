"""
@file conftest.py
@description Shared test fixtures for Agromet Agentic Engine.
@module tests
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="session")
def _isolated_climatology_cache(tmp_path_factory: pytest.TempPathFactory):
    """
    Safely isolate climatology cache if module is present.
    """
    try:
        import services.advisory_engine.climatology as clim
        original = clim.CLIMATOLOGY_DIR
        clim.CLIMATOLOGY_DIR = tmp_path_factory.mktemp("climatology")
        yield
        clim.CLIMATOLOGY_DIR = original
    except ImportError:
        yield
