"""
routes.py
All API endpoints.  Thin layer — validation is in schemas.py, logic in adapter.py.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status

from .adapter import run_regime_only, run_signal
from .schemas import (
    HealthResponse,
    RegimeOnlyRequest,
    RegimeOnlyResponse,
    SignalRequest,
    SignalResponse,
)

router = APIRouter()

API_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["meta"],
)
def health() -> HealthResponse:
    """Returns 200 OK when the service is running."""
    return HealthResponse(status="ok", version=API_VERSION)


# ---------------------------------------------------------------------------
# POST /signal
# ---------------------------------------------------------------------------

@router.post(
    "/signal",
    response_model=SignalResponse,
    summary="Generate a trade signal",
    tags=["signals"],
    responses={
        200: {"description": "Signal generated (may be no_trade — check setup_valid)"},
        422: {"description": "Validation error — malformed candles or missing fields"},
        500: {"description": "Internal engine error"},
    },
)
def generate_signal_endpoint(body: SignalRequest) -> SignalResponse:
    """
    Run the full ICT signal pipeline on the supplied candle data.

    ### Pipeline summary
    1. Session & news guard (NY 09:30–12:00 ET, ±30 min news blackout)
    2. ATR volatility filter (suppresses low-vol compression)
    3. 15m fractal swing high/low detection
    4. **Regime detection** (TRENDING_BULL / TRENDING_BEAR / RANGING)
       — RANGING → immediate `no_trade`
       — Counter-trend direction → immediate `no_trade`
    5. 5m liquidity sweep detection (penetration + close-back)
    6. Displacement (body/range vs ATR thresholds)
    7. BOS / CHOCH confirmation
    8. IFVG identification (3-candle imbalance in displacement leg)
    9. First retracement into IFVG
    10. Position sizing ($100–$300 risk, max 5 contracts)

    ### Data requirements
    - `candles_5m` and `candles_15m` must be sorted **oldest → newest**
    - Both must be UTC-aware timestamps
    - Minimum: 15 × 5m bars, 11 × 15m bars (recommend 30 and 40 respectively)
    - For reliable regime detection: 60+ × 15m bars (covers 2× ADX period + history)

    ### Response interpretation
    - `setup_valid: false` + `direction: "no_trade"` → engine blocked the signal
    - `reasons` list shows every gate: ✓ = passed, ✗ = blocked here
    - `confidence_score` is 0 for all `no_trade` responses
    - `regime` block is always populated (even on `no_trade`) when regime
      detection had enough data
    """
    try:
        return run_signal(
            candles_5m_in=body.candles_5m,
            candles_15m_in=body.candles_15m,
            instrument=body.instrument,
            news_times=body.news_times,
            trades_today=body.trades_today,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Engine error: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# POST /regime
# ---------------------------------------------------------------------------

@router.post(
    "/regime",
    response_model=RegimeOnlyResponse,
    summary="Detect market regime only",
    tags=["signals"],
    responses={
        200: {"description": "Regime classification with per-indicator votes"},
        422: {"description": "Validation error"},
        500: {"description": "Internal engine error"},
    },
)
def regime_endpoint(body: RegimeOnlyRequest) -> RegimeOnlyResponse:
    """
    Run regime detection without the full signal pipeline.

    Useful for:
    - Pre-session regime dashboard
    - Debugging why signals are being suppressed
    - External systems that want regime context without full pipeline cost

    Returns ADX, EMA slope, swing structure vote, ATR expansion vote, and
    final `regime` classification with per-indicator vote counts.
    """
    try:
        return run_regime_only(
            candles_5m_in=body.candles_5m,
            candles_15m_in=body.candles_15m,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Regime engine error: {exc}",
        ) from exc
