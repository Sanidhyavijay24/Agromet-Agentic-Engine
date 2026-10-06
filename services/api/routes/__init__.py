"""
@file __init__.py
@description Export API route sub-routers.
@module services/api/routes
"""

from services.api.routes.forecast import router as forecast_router
from services.api.routes.agent import router as agent_router
from services.api.routes.health import router as health_router

__all__ = ["forecast_router", "agent_router", "health_router"]
