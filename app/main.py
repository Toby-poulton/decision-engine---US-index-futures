"""
main.py
Application factory. Run with:
    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import router

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown hooks)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup — nothing to warm up for now (engine is stateless per-request)
    yield
    # Shutdown — nothing to clean up


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="Signal Engine API",
        description=(
            "ICT-style US index futures signal engine for ES/NQ.\n\n"
            "Strategy: Liquidity Sweep → Displacement → BOS/CHOCH → IFVG → Retracement\n\n"
            "Includes regime detection (TRENDING_BULL / TRENDING_BEAR / RANGING) "
            "that suppresses low-efficiency IFVG signals during ranging markets "
            "and counter-trend setups."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    # Restrict in production — allow all origins here for dev/evaluation
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Global exception handler ─────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": f"Unhandled server error: {type(exc).__name__}: {exc}"},
        )

    # ── Routers ─────────────────────────────────────────────────────────────
    app.include_router(router)

    return app


app = create_app()
