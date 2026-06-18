"""
adapter.py
Converts Pydantic request models → engine dataclasses and vice versa.

Keeping this in its own module means the engine (signal_engine.py) and the
API schemas (schemas.py) stay completely independent.  The router only imports
from here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .schemas import CandleIn, SignalResponse, RegimeOnlyResponse, TakeProfitOut, RegimeOut
from .signal_engine import (
    Candle as EngineCandle,
    generate_signal,
    detect_regime,
    detect_swings,
    calc_atr,
    SWING_LOOKBACK_15M,
    ATR_PERIOD,
)


# ---------------------------------------------------------------------------
# Candle conversion
# ---------------------------------------------------------------------------

def candles_from_request(bars: list[CandleIn]) -> list[EngineCandle]:
    """Convert a list of Pydantic CandleIn → engine Candle dataclasses."""
    return [
        EngineCandle(
            timestamp=b.timestamp,
            open=b.open,
            high=b.high,
            low=b.low,
            close=b.close,
            volume=b.volume,
        )
        for b in bars
    ]


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def run_signal(
    candles_5m_in:  list[CandleIn],
    candles_15m_in: list[CandleIn],
    instrument:     str,
    news_times:     list[datetime],
    trades_today:   int,
) -> SignalResponse:
    """
    Full pipeline: validate inputs, call generate_signal(), return typed response.
    """
    c5  = candles_from_request(candles_5m_in)
    c15 = candles_from_request(candles_15m_in)

    result = generate_signal(
        candles_5m=c5,
        candles_15m=c15,
        instrument=instrument,
        news_times=news_times,
        trades_today=trades_today,
    )

    raw = result.to_dict()

    # Build take_profit sub-model if present
    tp_model = None
    if raw.get("take_profit"):
        tp_model = TakeProfitOut(
            tp1=raw["take_profit"]["tp1"],
            tp2=raw["take_profit"]["tp2"],
        )

    # Build regime sub-model if present
    regime_model = None
    if raw.get("regime"):
        r = raw["regime"]
        regime_model = RegimeOut(
            regime=r["regime"],
            adx=r.get("adx"),
            ema_slope_pct=r.get("ema_slope_pct"),
            swing_vote=r.get("swing_vote"),
            atr_vote=r.get("atr_vote"),
            votes_bull=r.get("votes_bull", 0),
            votes_bear=r.get("votes_bear", 0),
            votes_range=r.get("votes_range", 0),
            notes=r.get("notes", []),
        )

    return SignalResponse(
        direction=raw["direction"],
        setup_valid=raw["setup_valid"],
        reasons=raw["reasons"],
        entry_zone=raw.get("entry_zone"),
        stop_loss=raw.get("stop_loss"),
        take_profit=tp_model,
        risk_per_trade_usd=raw.get("risk_per_trade_usd"),
        position_size_contracts=raw.get("position_size_contracts"),
        confidence_score=raw.get("confidence_score", 0),
        regime=regime_model,
        generated_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Regime-only endpoint
# ---------------------------------------------------------------------------

def run_regime_only(
    candles_5m_in:  list[CandleIn],
    candles_15m_in: list[CandleIn],
) -> RegimeOnlyResponse:
    """
    Run regime detection only — no sweep/displacement/IFVG pipeline.
    Useful for dashboard overlays, pre-session checks, or debugging.
    """
    c5  = candles_from_request(candles_5m_in)
    c15 = candles_from_request(candles_15m_in)

    swings = detect_swings(c15, SWING_LOOKBACK_15M)
    result = detect_regime(c15, swings, c5)
    raw    = result.to_dict()

    return RegimeOnlyResponse(
        regime=raw["regime"],
        adx=raw.get("adx"),
        ema_slope_pct=raw.get("ema_slope_pct"),
        swing_vote=raw.get("swing_vote"),
        atr_vote=raw.get("atr_vote"),
        votes_bull=raw.get("votes_bull", 0),
        votes_bear=raw.get("votes_bear", 0),
        votes_range=raw.get("votes_range", 0),
        notes=raw.get("notes", []),
        generated_at=datetime.now(tz=timezone.utc),
    )
