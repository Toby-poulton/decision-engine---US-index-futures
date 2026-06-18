"""
tests/test_api.py
Full pytest suite for the Signal Engine API.

Run: pytest tests/ -v
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

UTC = timezone.utc
# 09:30 ET on a Tuesday in March 2024 (EDT = UTC-4) → 13:30 UTC
SESSION_OPEN = datetime(2024, 3, 19, 13, 30, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ts(offset_min: int) -> str:
    """UTC ISO-8601 string offset by `offset_min` from session open."""
    return (SESSION_OPEN + timedelta(minutes=offset_min)).isoformat()


def bar(offset_min: int, o: float, h: float, l: float, c: float,
        vol: float = 1000.0) -> dict:
    return {
        "timestamp": ts(offset_min),
        "open": o, "high": h, "low": l, "close": c, "volume": vol,
    }


def flat_bars_5m(n: int = 30, base: float = 5202.0) -> list[dict]:
    """Flat 5m candles — no pattern, every timestamp strictly increasing."""
    return [bar(i * 5, base, base + 5, base - 2, base + 1) for i in range(n)]


def flat_bars_15m(n: int = 20, base: float = 5202.0) -> list[dict]:
    return [bar(i * 15, base, base + 6, base - 2, base + 2) for i in range(n)]


def _stepped_trend(direction: str = "bull", n_steps: int = 5) -> list[dict]:
    """
    Stepped trend bars with correct monotonic timestamps.

    Each step: 7 advance bars + 5 pullback bars = 12 bars/step → 60 bars total.
    Trend bars start at offset -(n_steps*12)*15 = -900 min before session open.
    Session-local swing-structure bars appended at +0..+285 min (20 × 15m bars),
    with price nudges applied while preserving each bar's own timestamp.
    """
    bars: list[dict] = []
    price = 5050.0 if direction == "bull" else 5400.0
    step_delta = +5.0 if direction == "bull" else -5.0

    total_bars = n_steps * 12  # 60
    bar_index = 0

    for _ in range(n_steps):
        for _ in range(7):   # advance phase
            offset = (bar_index - total_bars) * 15
            if direction == "bull":
                bars.append(bar(offset, price, price + 14, price - 2, price + 12))
            else:
                bars.append(bar(offset, price, price + 2, price - 14, price - 12))
            price += step_delta
            bar_index += 1

        for j in range(5):   # retracement phase
            offset = (bar_index - total_bars) * 15
            drift = (-4.0 if j < 3 else 0) * (1 if direction == "bull" else -1)
            if direction == "bull":
                bars.append(bar(offset, price, price + 3, price - 6, price + drift))
            else:
                bars.append(bar(offset, price, price + 6, price - 3, price + drift))
            price += drift
            bar_index += 1

    # Append 20 session-context 15m bars (offset 0..285 min, strictly after trend bars)
    base_price = 5200.0
    for k in range(20):
        bars.append(bar(k * 15, base_price, base_price + 6, base_price - 2, base_price + 2))

    # Plant swing levels by modifying price values of existing bars (timestamps unchanged)
    # bars[-15] = k=5  → offset +75  min  → swing HIGH 5212
    # bars[-13] = k=7  → offset +105 min  → swing LOW  5190
    b = bars[-15]
    bars[-15] = {**b, "open": 5204.0, "high": 5212.0, "low": 5201.0, "close": 5208.0}
    b = bars[-13]
    bars[-13] = {**b, "open": 5202.0, "high": 5204.0, "low": 5190.0, "close": 5192.0}

    return bars


def stepped_bull_15m() -> list[dict]:
    return _stepped_trend("bull")


def stepped_bear_15m() -> list[dict]:
    return _stepped_trend("bear")


def ranging_15m() -> list[dict]:
    """Sine-wave chop — regime should classify as RANGING."""
    bars: list[dict] = []
    mid = 5200.0
    total = 60
    for i in range(total):
        offset = (i - total) * 15       # -900 .. -15
        val = mid + 20 * math.sin(i * 0.3)
        bars.append(bar(offset, val, val + 5, val - 5, val + 1))
    for k in range(20):
        bars.append(bar(k * 15, 5200, 5206, 5198, 5203))
    b = bars[-15]
    bars[-15] = {**b, "open": 5204.0, "high": 5212.0, "low": 5201.0, "close": 5208.0}
    b = bars[-13]
    bars[-13] = {**b, "open": 5202.0, "high": 5204.0, "low": 5190.0, "close": 5192.0}
    return bars


def valid_long_5m() -> list[dict]:
    """5m bars with a valid LONG IFVG setup planted inside session hours."""
    bars = flat_bars_5m(30)
    bars[12] = bar(60,  5202, 5203, 5187, 5191)   # sweep LOW 5190
    bars[13] = bar(65,  5191, 5196, 5190, 5195)   # close back above
    bars[14] = bar(70,  5195, 5205, 5194, 5203)   # displacement
    bars[15] = bar(75,  5203, 5215, 5202, 5213)   # BOS above 5212
    bars[16] = bar(80,  5213, 5222, 5212, 5220)
    bars[17] = bar(85,  5220, 5221, 5207, 5210)   # retrace into IFVG [5205,5212]
    bars[18] = bar(90,  5210, 5213, 5206, 5208)
    bars[19] = bar(95,  5205, 5210, 5205, 5207)
    return bars


def valid_short_5m() -> list[dict]:
    """5m bars with a valid SHORT IFVG setup."""
    bars = flat_bars_5m(30)
    bars[12] = bar(60, 5207, 5220, 5204, 5206)    # sweep HIGH 5212
    bars[13] = bar(65, 5206, 5208, 5200, 5201)    # close back below
    bars[14] = bar(70, 5201, 5202, 5191, 5193)    # displacement
    bars[15] = bar(75, 5193, 5194, 5181, 5183)    # BOS below 5190
    bars[16] = bar(80, 5183, 5184, 5174, 5176)
    bars[17] = bar(85, 5176, 5188, 5174, 5186)    # retrace into IFVG [5184,5191]
    bars[18] = bar(90, 5186, 5190, 5183, 5185)
    bars[19] = bar(95, 5185, 5188, 5182, 5184)
    return bars


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self):
        r = client.get("/health")
        assert r.status_code == 200

    def test_body(self):
        r = client.get("/health")
        j = r.json()
        assert j["status"] == "ok"
        assert "version" in j


# ---------------------------------------------------------------------------
# POST /signal — input validation
# ---------------------------------------------------------------------------

class TestSignalValidation:
    EP = "/signal"

    def _post(self, body):
        return client.post(self.EP, json=body)

    def test_missing_candles_5m_returns_422(self):
        r = self._post({"candles_15m": flat_bars_15m(), "instrument": "NQ"})
        assert r.status_code == 422

    def test_missing_candles_15m_returns_422(self):
        r = self._post({"candles_5m": flat_bars_5m()})
        assert r.status_code == 422

    def test_too_few_5m_bars_returns_422(self):
        r = self._post({"candles_5m": flat_bars_5m(5), "candles_15m": stepped_bull_15m()})
        assert r.status_code == 422

    def test_too_few_15m_bars_returns_422(self):
        r = self._post({"candles_5m": flat_bars_5m(30), "candles_15m": flat_bars_15m(3)})
        assert r.status_code == 422

    def test_invalid_instrument_returns_422(self):
        r = self._post({
            "candles_5m": flat_bars_5m(), "candles_15m": stepped_bull_15m(),
            "instrument": "SPX",
        })
        assert r.status_code == 422

    def test_candles_unsorted_returns_422(self):
        bars = flat_bars_5m(20)
        bars[5], bars[6] = bars[6], bars[5]
        r = self._post({"candles_5m": bars, "candles_15m": stepped_bull_15m()})
        assert r.status_code == 422

    def test_high_below_close_returns_422(self):
        bad = {"timestamp": ts(0), "open": 100.0, "high": 95.0,
               "low": 90.0, "close": 98.0}
        bars = flat_bars_5m(20)
        bars[0] = bad
        r = self._post({"candles_5m": bars, "candles_15m": stepped_bull_15m()})
        assert r.status_code == 422

    def test_low_above_open_returns_422(self):
        bad = {"timestamp": ts(0), "open": 100.0, "high": 105.0,
               "low": 103.0, "close": 104.0}
        bars = flat_bars_5m(20)
        bars[0] = bad
        r = self._post({"candles_5m": bars, "candles_15m": stepped_bull_15m()})
        assert r.status_code == 422

    def test_naive_timestamp_returns_422(self):
        bad = {"timestamp": "2024-03-19T13:30:00",   # no tz offset
               "open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0}
        bars = flat_bars_5m(20)
        bars[0] = bad
        r = self._post({"candles_5m": bars, "candles_15m": stepped_bull_15m()})
        assert r.status_code == 422

    def test_negative_trades_today_returns_422(self):
        r = self._post({
            "candles_5m": flat_bars_5m(), "candles_15m": stepped_bull_15m(),
            "trades_today": -1,
        })
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# POST /signal — no_trade scenarios
# ---------------------------------------------------------------------------

class TestSignalNoTrade:
    EP = "/signal"

    def _post(self, body):
        return client.post(self.EP, json=body)

    def _assert_no_trade(self, r):
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text}"
        j = r.json()
        assert j["direction"] == "no_trade"
        assert j["setup_valid"] is False
        assert j["confidence_score"] == 0
        assert j["entry_zone"] is None
        assert j["stop_loss"] is None
        assert j["take_profit"] is None
        assert j["risk_per_trade_usd"] is None
        assert j["position_size_contracts"] is None
        assert len(j["reasons"]) >= 1
        return j

    def test_outside_session_returns_no_trade(self):
        """All timestamps 3h before session → outside_ny_session gate."""
        pre = datetime(2024, 3, 19, 10, 0, tzinfo=UTC)   # 06:00 ET

        def pb5(i):
            t = (pre + timedelta(minutes=i * 5)).isoformat()
            return {"timestamp": t, "open": 5202.0, "high": 5207.0,
                    "low": 5199.0, "close": 5204.0}

        def pb15(i):
            t = (pre + timedelta(minutes=i * 15)).isoformat()
            return {"timestamp": t, "open": 5202.0, "high": 5207.0,
                    "low": 5199.0, "close": 5204.0}

        r = self._post({
            "candles_5m":  [pb5(i)  for i in range(20)],
            "candles_15m": [pb15(i) for i in range(15)],
        })
        j = self._assert_no_trade(r)
        assert any("session" in reason.lower() for reason in j["reasons"])

    def test_daily_limit_reached_returns_no_trade(self):
        r = self._post({
            "candles_5m":   valid_long_5m(),
            "candles_15m":  stepped_bull_15m(),
            "instrument":   "NQ",
            "trades_today": 2,
        })
        j = self._assert_no_trade(r)
        assert any("limit" in reason.lower() for reason in j["reasons"])

    def test_ranging_regime_returns_no_trade(self):
        r = self._post({
            "candles_5m":  valid_long_5m(),
            "candles_15m": ranging_15m(),
            "instrument":  "NQ",
        })
        j = self._assert_no_trade(r)
        assert any("ranging" in reason.lower() for reason in j["reasons"])
        assert j["regime"] is not None
        assert j["regime"]["regime"] == "ranging"

    def test_no_setup_flat_candles_returns_no_trade(self):
        r = self._post({
            "candles_5m":  flat_bars_5m(30),
            "candles_15m": stepped_bull_15m(),
            "instrument":  "NQ",
        })
        self._assert_no_trade(r)

    def test_news_blackout_returns_no_trade(self):
        news_time = ts(130)   # 15 min before last bar (offset 145 = bar[29]) → within ±30 min
        r = self._post({
            "candles_5m":   valid_long_5m(),
            "candles_15m":  stepped_bull_15m(),
            "instrument":   "NQ",
            "news_times":   [news_time],
        })
        j = self._assert_no_trade(r)
        assert any("news" in reason.lower() for reason in j["reasons"])


# ---------------------------------------------------------------------------
# POST /signal — valid signal scenarios
# ---------------------------------------------------------------------------

class TestSignalValid:
    EP = "/signal"

    def _post(self, body):
        return client.post(self.EP, json=body)

    def _assert_valid(self, r, expected_dir: str):
        assert r.status_code == 200, f"Got {r.status_code}: {r.text}"
        j = r.json()
        assert j["direction"] == expected_dir, f"reasons: {j['reasons']}"
        assert j["setup_valid"] is True
        assert j["confidence_score"] > 0
        assert j["entry_zone"] is not None and len(j["entry_zone"]) == 2
        assert j["entry_zone"][0] < j["entry_zone"][1]
        assert j["stop_loss"] is not None
        assert j["take_profit"] is not None
        if expected_dir == "long":
            assert j["take_profit"]["tp2"] > j["take_profit"]["tp1"]
        else:
            assert j["take_profit"]["tp2"] < j["take_profit"]["tp1"]
        assert j["risk_per_trade_usd"] is not None
        assert 100 <= j["risk_per_trade_usd"] <= 300
        assert 1 <= j["position_size_contracts"] <= 5
        assert j["regime"] is not None
        assert "generated_at" in j
        return j

    def test_valid_long_signal_bull_regime(self):
        r = self._post({
            "candles_5m":  valid_long_5m(),
            "candles_15m": stepped_bull_15m(),
            "instrument":  "NQ",
        })
        j = self._assert_valid(r, "long")
        assert j["regime"]["regime"] == "trending_bull"

    def test_valid_short_signal_bear_regime(self):
        r = self._post({
            "candles_5m":  valid_short_5m(),
            "candles_15m": stepped_bear_15m(),
            "instrument":  "NQ",
        })
        j = self._assert_valid(r, "short")
        assert j["regime"]["regime"] == "trending_bear"

    def test_stop_below_entry_low_for_long(self):
        r = self._post({
            "candles_5m":  valid_long_5m(),
            "candles_15m": stepped_bull_15m(),
            "instrument":  "NQ",
        })
        j = r.json()
        if j["setup_valid"]:
            assert j["stop_loss"] < j["entry_zone"][0]

    def test_stop_above_entry_high_for_short(self):
        r = self._post({
            "candles_5m":  valid_short_5m(),
            "candles_15m": stepped_bear_15m(),
            "instrument":  "NQ",
        })
        j = r.json()
        if j["setup_valid"]:
            assert j["stop_loss"] > j["entry_zone"][1]

    def test_risk_within_bounds(self):
        r = self._post({
            "candles_5m":  valid_long_5m(),
            "candles_15m": stepped_bull_15m(),
            "instrument":  "NQ",
        })
        j = r.json()
        if j["setup_valid"]:
            assert 100 <= j["risk_per_trade_usd"] <= 300

    def test_confidence_between_1_and_100(self):
        r = self._post({
            "candles_5m":  valid_long_5m(),
            "candles_15m": stepped_bull_15m(),
            "instrument":  "NQ",
        })
        j = r.json()
        assert 0 <= j["confidence_score"] <= 100

    def test_reasons_always_present(self):
        r = self._post({
            "candles_5m":  valid_long_5m(),
            "candles_15m": stepped_bull_15m(),
            "instrument":  "NQ",
        })
        j = r.json()
        assert isinstance(j["reasons"], list) and len(j["reasons"]) > 0

    def test_es_instrument_accepted(self):
        """ES $12.50/tick — may hit risk bounds, should not crash."""
        r = self._post({
            "candles_5m":  valid_long_5m(),
            "candles_15m": stepped_bull_15m(),
            "instrument":  "ES",
        })
        assert r.status_code == 200
        assert r.json()["direction"] in ("long", "short", "no_trade")

    def test_counter_trend_long_in_bear_suppressed(self):
        r = self._post({
            "candles_5m":  valid_long_5m(),
            "candles_15m": stepped_bear_15m(),
            "instrument":  "NQ",
        })
        j = r.json()
        assert j["direction"] == "no_trade"
        assert any(
            "counter" in reason.lower() or "suppress" in reason.lower()
            for reason in j["reasons"]
        )

    def test_counter_trend_short_in_bull_suppressed(self):
        r = self._post({
            "candles_5m":  valid_short_5m(),
            "candles_15m": stepped_bull_15m(),
            "instrument":  "NQ",
        })
        j = r.json()
        assert j["direction"] == "no_trade"
        assert any(
            "counter" in reason.lower() or "suppress" in reason.lower()
            for reason in j["reasons"]
        )


# ---------------------------------------------------------------------------
# POST /regime
# ---------------------------------------------------------------------------

class TestRegimeEndpoint:
    EP = "/regime"

    def _post(self, body):
        return client.post(self.EP, json=body)

    def test_returns_200(self):
        r = self._post({
            "candles_5m":  flat_bars_5m(30),
            "candles_15m": stepped_bull_15m(),
        })
        assert r.status_code == 200, r.text

    def test_bull_regime_detected(self):
        r = self._post({
            "candles_5m":  flat_bars_5m(30),
            "candles_15m": stepped_bull_15m(),
        })
        j = r.json()
        assert j["regime"] == "trending_bull"
        assert j["votes_bull"] >= 2

    def test_bear_regime_detected(self):
        r = self._post({
            "candles_5m":  flat_bars_5m(30),
            "candles_15m": stepped_bear_15m(),
        })
        j = r.json()
        assert j["regime"] == "trending_bear"
        assert j["votes_bear"] >= 2

    def test_ranging_regime_detected(self):
        r = self._post({
            "candles_5m":  flat_bars_5m(30),
            "candles_15m": ranging_15m(),
        })
        assert r.json()["regime"] == "ranging"

    def test_response_has_all_fields(self):
        r = self._post({
            "candles_5m":  flat_bars_5m(30),
            "candles_15m": stepped_bull_15m(),
        })
        j = r.json()
        for field in ("regime", "adx", "ema_slope_pct", "swing_vote",
                      "atr_vote", "votes_bull", "votes_bear", "votes_range",
                      "notes", "generated_at"):
            assert field in j, f"missing field: {field}"

    def test_notes_is_list_of_strings(self):
        r = self._post({
            "candles_5m":  flat_bars_5m(30),
            "candles_15m": stepped_bull_15m(),
        })
        j = r.json()
        assert isinstance(j["notes"], list)
        assert all(isinstance(n, str) for n in j["notes"])

    def test_missing_candles_5m_returns_422(self):
        r = self._post({"candles_15m": flat_bars_15m()})
        assert r.status_code == 422

    def test_too_few_bars_returns_422(self):
        r = self._post({"candles_5m": flat_bars_5m(3), "candles_15m": flat_bars_15m(3)})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Response schema consistency
# ---------------------------------------------------------------------------

class TestResponseSchema:
    REQUIRED_KEYS = {
        "direction", "setup_valid", "reasons", "entry_zone",
        "stop_loss", "take_profit", "risk_per_trade_usd",
        "position_size_contracts", "confidence_score", "regime", "generated_at",
    }

    def test_all_keys_present_on_no_trade(self):
        r = client.post("/signal", json={
            "candles_5m":  flat_bars_5m(30),
            "candles_15m": stepped_bull_15m(),
            "instrument":  "NQ",
        })
        assert r.status_code == 200
        assert self.REQUIRED_KEYS <= set(r.json().keys())

    def test_all_keys_present_on_valid_signal(self):
        r = client.post("/signal", json={
            "candles_5m":  valid_long_5m(),
            "candles_15m": stepped_bull_15m(),
            "instrument":  "NQ",
        })
        assert r.status_code == 200
        assert self.REQUIRED_KEYS <= set(r.json().keys())

    def test_generated_at_is_utc_iso(self):
        r = client.post("/signal", json={
            "candles_5m":  flat_bars_5m(30),
            "candles_15m": stepped_bull_15m(),
        })
        assert r.status_code == 200
        dt = datetime.fromisoformat(r.json()["generated_at"])
        assert dt.tzinfo is not None
