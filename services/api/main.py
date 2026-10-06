"""
@file main.py
@description FastAPI application entrypoint for Agromet Agentic Engine (A²E).
@module services/api
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from services.api.routes.forecast import router as forecast_router
from services.api.routes.agent import router as agent_router
from services.api.routes.health import router as health_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle startup and shutdown hooks."""
    logger.info("Starting Agromet Agentic Engine (A²E) FastAPI Gateway...")
    logger.info("Active flagship downscaling zone: Zone XIV (Western Dry / Rajasthan)")
    yield
    logger.info("Shutting down Agromet Agentic Engine Gateway.")


app = FastAPI(
    title="Agromet Agentic Engine (A²E) API",
    description="Physics-Guided 1km Microclimate Downscaling & Autonomous Agronomic Intelligence",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
allowed_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Register API v1 routers
app.include_router(health_router, prefix="/api/v1")
app.include_router(forecast_router, prefix="/api/v1")
app.include_router(agent_router, prefix="/api/v1")


@app.get("/")
async def root_ping():
    """Root status ping."""
    return {
        "engine": "Agromet Agentic Engine (A²E)",
        "version": "2.0.0",
        "status": "operational",
        "active_live_zone": "Zone XIV (Western Dry / Rajasthan)",
        "macro_fleet": "15 ICAR Agro-Climatic Zones (+58.9% macro error reduction)",
        "docs_url": "/docs"
    }
