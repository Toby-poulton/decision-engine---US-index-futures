"""
Pydantic v2 schemas for the Signal Engine API.

Canonical timestamp format throughout: ISO-8601 with UTC offset,
e.g. "2024-03-19T13:30:00+00:00".  The engine works in UTC internally.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# REQUEST SCHEMAS
# ---------------------------------------------------------------------------

class CandleIn(BaseModel):
    """A single OHLCV bar supplied by the caller."""

    timestamp: datetime = Field(
        ...,
        description="Bar close time, UTC-aware ISO-8601. "
                    "E.g. '2024-03-19T14:30:00+00:00'",
    )
    open:   float = Field(..., gt=0)
    high:   float = Field(..., gt=0)
    low:    float = Field(..., gt=0)
    close:  float = Field(..., gt=0)
    volume: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def ohlc_consistency(self) -> "CandleIn":
        if self.high < self.open or self.high < self.close:
            raise ValueError("high must be ≥ open and close")
        if self.low > self.open or self.low > self.close:
            raise ValueError("low must be ≤ open and close")
        if self.high < self.low:
            raise ValueError("high must be ≥ low")
        return self

    @field_validator("timestamp")
    @classmethod
    def must_be_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError(
                "timestamp must be timezone-aware (include UTC offset or Z)"
            )
        return v


class SignalRequest(BaseModel):
    """
    Request body for POST /signal.

    Minimum recommended bar counts:
      candles_5m  : 30+ (engine needs ~14 for ATR seed, 20 for sweep scan)
      candles_15m : 25+ (engine needs 2×SWING_LOOKBACK+1 = 11 for pivots,
                         plus ADX needs 2×14+1 = 29 for convergence)

    Both lists must be sorted oldest → newest.
    """

    candles_5m: list[CandleIn] = Field(
        ...,
        min_length=15,
        description="5-minute OHLCV bars, oldest first. Minimum 15, recommend 30+.",
    )
    candles_15m: list[CandleIn] = Field(
        ...,
        min_length=11,
        description="15-minute OHLCV bars, oldest first. Minimum 11, recommend 40+.",
    )
    instrument: Literal["ES", "NQ"] = Field(
        default="NQ",
        description="Futures instrument. ES ($12.50/tick) or NQ ($5.00/tick).",
    )
    news_times: list[datetime] = Field(
        default_factory=list,
        description="UTC-aware datetimes of scheduled news events today. "
                    "Engine will block signals within ±30 min of each.",
    )
    trades_today: int = Field(
        default=0,
        ge=0,
        description="Number of trades already taken today. "
                    "Engine blocks new signals once this hits 2.",
    )

    @field_validator("news_times", mode="before")
    @classmethod
    def news_must_be_tz_aware(cls, v: list) -> list:
        for dt in v:
            if isinstance(dt, datetime) and dt.tzinfo is None:
                raise ValueError("All news_times must be timezone-aware")
        return v

    @model_validator(mode="after")
    def candles_sorted(self) -> "SignalRequest":
        for name, bars in [("candles_5m", self.candles_5m),
                           ("candles_15m", self.candles_15m)]:
            for i in range(1, len(bars)):
                if bars[i].timestamp <= bars[i - 1].timestamp:
                    raise ValueError(
                        f"{name}[{i}].timestamp ({bars[i].timestamp}) must be "
                        f"strictly after [{i-1}] ({bars[i-1].timestamp}). "
                        f"Supply bars oldest → newest."
                    )
        return self


# ---------------------------------------------------------------------------
# RESPONSE SCHEMAS
# ---------------------------------------------------------------------------

class TakeProfitOut(BaseModel):
    tp1: float
    tp2: float


class RegimeOut(BaseModel):
    regime:        str
    adx:           Optional[float]
    ema_slope_pct: Optional[float]
    swing_vote:    Optional[str]
    atr_vote:      Optional[str]
    votes_bull:    int
    votes_bear:    int
    votes_range:   int
    notes:         list[str]


class SignalResponse(BaseModel):
    """
    Response for POST /signal.

    `setup_valid` is the single authoritative flag — if False, all trade
    fields (entry_zone, stop_loss, take_profit, risk, contracts) are null.
    Check `reasons` for the exact gate that fired.
    """

    direction:               Literal["long", "short", "no_trade"]
    setup_valid:             bool
    reasons:                 list[str]
    entry_zone:              Optional[list[float]] = None
    stop_loss:               Optional[float]       = None
    take_profit:             Optional[TakeProfitOut] = None
    risk_per_trade_usd:      Optional[float]       = None
    position_size_contracts: Optional[int]         = None
    confidence_score:        int
    regime:                  Optional[RegimeOut]   = None
    generated_at:            datetime = Field(
        description="UTC timestamp when this signal was generated."
    )


class HealthResponse(BaseModel):
    status:  Literal["ok"]
    version: str


class RegimeOnlyRequest(BaseModel):
    """Request body for POST /regime — regime detection without full signal pipeline."""

    candles_5m: list[CandleIn] = Field(..., min_length=15)
    candles_15m: list[CandleIn] = Field(..., min_length=11)

    @model_validator(mode="after")
    def candles_sorted(self) -> "RegimeOnlyRequest":
        for name, bars in [("candles_5m", self.candles_5m),
                           ("candles_15m", self.candles_15m)]:
            for i in range(1, len(bars)):
                if bars[i].timestamp <= bars[i - 1].timestamp:
                    raise ValueError(
                        f"{name}[{i}].timestamp must be strictly after [{i-1}]"
                    )
        return self


class RegimeOnlyResponse(BaseModel):
    regime:        str
    adx:           Optional[float]
    ema_slope_pct: Optional[float]
    swing_vote:    Optional[str]
    atr_vote:      Optional[str]
    votes_bull:    int
    votes_bear:    int
    votes_range:   int
    notes:         list[str]
    generated_at:  datetime
