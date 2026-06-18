#!/usr/bin/env python3
"""
test_client.py
Standalone script to smoke-test a running Signal Engine API instance.

Usage:
    # Against local dev server:
    python test_client.py

    # Against deployed instance:
    python test_client.py --base-url https://your-app.onrender.com

    # Verbose (print full JSON responses):
    python test_client.py --verbose
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timedelta, timezone

try:
    import requests
except ImportError:
    sys.exit("requests is not installed. Run: pip install requests")

UTC = timezone.utc
SESSION_OPEN = datetime(2024, 3, 19, 13, 30, tzinfo=UTC)   # 09:30 ET


# ---------------------------------------------------------------------------
# Candle builders
# ---------------------------------------------------------------------------

def ts(offset_min: int) -> str:
    return (SESSION_OPEN + timedelta(minutes=offset_min)).isoformat()


def bar(offset_min: int, o: float, h: float, l: float, c: float) -> dict:
    return {"timestamp": ts(offset_min), "open": o, "high": h, "low": l, "close": c}


def flat_bars_5m(n: int = 30) -> list[dict]:
    return [bar(i * 5, 5202, 5207, 5199, 5204) for i in range(n)]


def stepped_bull_15m() -> list[dict]:
    bars, price, i = [], 5050.0, 0
    for _ in range(5):
        for _ in range(7):
            bars.append(bar(i * 15 - 900, price, price + 14, price - 2, price + 12))
            price += 5.0;  i += 1
        for j in range(5):
            drop = 4.0 if j < 3 else 0
            bars.append(bar(i * 15 - 900, price, price + 3, price - 6, price - drop))
            price -= drop;  i += 1
    for k in range(20):
        bars.append(bar(k * 15, 5200, 5206, 5198, 5203))
    bars[-15] = bar(-15 * 15, 5204, 5212, 5201, 5208)
    bars[-13] = bar(-13 * 15, 5202, 5204, 5190, 5192)
    return bars


def valid_long_5m() -> list[dict]:
    bars = flat_bars_5m(30)
    bars[12] = bar(60,  5202, 5203, 5187, 5191)
    bars[13] = bar(65,  5191, 5196, 5190, 5195)
    bars[14] = bar(70,  5195, 5205, 5194, 5203)
    bars[15] = bar(75,  5203, 5215, 5202, 5213)
    bars[16] = bar(80,  5213, 5222, 5212, 5220)
    bars[17] = bar(85,  5220, 5221, 5207, 5210)
    bars[18] = bar(90,  5210, 5213, 5206, 5208)
    bars[19] = bar(95,  5205, 5210, 5205, 5207)
    return bars


def ranging_15m() -> list[dict]:
    bars = []
    mid = 5200.0
    for i in range(60):
        val = mid + 20 * math.sin(i * 0.3)
        bars.append(bar(i * 15 - 900, val, val + 5, val - 5, val + 1))
    for k in range(20):
        bars.append(bar(k * 15, 5200, 5206, 5198, 5203))
    bars[-15] = bar(-15 * 15, 5204, 5212, 5201, 5208)
    bars[-13] = bar(-13 * 15, 5202, 5204, 5190, 5192)
    return bars


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------

PASS = "\033[92m✓ PASS\033[0m"
FAIL = "\033[91m✗ FAIL\033[0m"
SKIP = "\033[93m~ SKIP\033[0m"


def run_test(
    label: str,
    session: requests.Session,
    method: str,
    url: str,
    body: dict,
    expect_status: int,
    assertions: list[tuple[str, callable]],
    verbose: bool,
) -> bool:
    try:
        resp = session.request(method, url, json=body, timeout=15)
    except requests.ConnectionError:
        print(f"{FAIL}  {label}")
        print(f"         → Connection refused. Is the server running?")
        return False

    j = {}
    try:
        j = resp.json()
    except Exception:
        pass

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"  Status: {resp.status_code}")
        print(json.dumps(j, indent=2, default=str))

    ok = resp.status_code == expect_status
    failures = []
    if not ok:
        failures.append(f"status {resp.status_code} ≠ {expect_status}")

    for desc, check in assertions:
        try:
            result = check(j)
            if not result:
                failures.append(desc)
        except Exception as e:
            failures.append(f"{desc}: {e}")

    if failures:
        print(f"{FAIL}  {label}")
        for f in failures:
            print(f"         → {f}")
        return False
    else:
        print(f"{PASS}  {label}")
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    session = requests.Session()
    results = []

    print(f"\nSignal Engine API — smoke test")
    print(f"Target: {base}\n")

    # ── Health ───────────────────────────────────────────────────────────────
    results.append(run_test(
        "GET /health → 200 ok",
        session, "GET", f"{base}/health", {},
        200,
        [
            ("status=ok", lambda j: j.get("status") == "ok"),
            ("version present", lambda j: "version" in j),
        ],
        args.verbose,
    ))

    # ── /signal: ranging regime → no_trade ──────────────────────────────────
    results.append(run_test(
        "POST /signal — ranging regime → no_trade",
        session, "POST", f"{base}/signal",
        {"candles_5m": valid_long_5m(), "candles_15m": ranging_15m(), "instrument": "NQ"},
        200,
        [
            ("direction=no_trade",  lambda j: j.get("direction") == "no_trade"),
            ("setup_valid=false",   lambda j: j.get("setup_valid") is False),
            ("regime present",      lambda j: j.get("regime") is not None),
            ("regime=ranging",      lambda j: j.get("regime", {}).get("regime") == "ranging"),
            ("reasons non-empty",   lambda j: len(j.get("reasons", [])) > 0),
        ],
        args.verbose,
    ))

    # ── /signal: valid LONG in bull regime ───────────────────────────────────
    results.append(run_test(
        "POST /signal — LONG in bull regime → valid signal",
        session, "POST", f"{base}/signal",
        {"candles_5m": valid_long_5m(), "candles_15m": stepped_bull_15m(), "instrument": "NQ"},
        200,
        [
            ("direction=long",         lambda j: j.get("direction") == "long"),
            ("setup_valid=true",       lambda j: j.get("setup_valid") is True),
            ("entry_zone present",     lambda j: j.get("entry_zone") is not None),
            ("entry_zone has 2 items", lambda j: len(j.get("entry_zone", [])) == 2),
            ("stop_loss present",      lambda j: j.get("stop_loss") is not None),
            ("tp1 < tp2 (long)",       lambda j: j["take_profit"]["tp1"] < j["take_profit"]["tp2"]),
            ("risk $100-$300",         lambda j: 100 <= j.get("risk_per_trade_usd", 0) <= 300),
            ("contracts 1-5",          lambda j: 1 <= j.get("position_size_contracts", 0) <= 5),
            ("confidence 1-100",       lambda j: 1 <= j.get("confidence_score", 0) <= 100),
            ("regime=trending_bull",   lambda j: j.get("regime", {}).get("regime") == "trending_bull"),
            ("generated_at present",   lambda j: "generated_at" in j),
        ],
        args.verbose,
    ))

    # ── /signal: daily limit ─────────────────────────────────────────────────
    results.append(run_test(
        "POST /signal — trades_today=2 → daily limit no_trade",
        session, "POST", f"{base}/signal",
        {"candles_5m": valid_long_5m(), "candles_15m": stepped_bull_15m(),
         "instrument": "NQ", "trades_today": 2},
        200,
        [
            ("direction=no_trade", lambda j: j.get("direction") == "no_trade"),
            ("limit in reasons",   lambda j: any("limit" in r.lower() for r in j.get("reasons", []))),
        ],
        args.verbose,
    ))

    # ── /signal: bad input ───────────────────────────────────────────────────
    results.append(run_test(
        "POST /signal — too few bars → 422",
        session, "POST", f"{base}/signal",
        {"candles_5m": flat_bars_5m(3), "candles_15m": stepped_bull_15m()},
        422,
        [],
        args.verbose,
    ))

    results.append(run_test(
        "POST /signal — invalid instrument → 422",
        session, "POST", f"{base}/signal",
        {"candles_5m": valid_long_5m(), "candles_15m": stepped_bull_15m(), "instrument": "SPY"},
        422,
        [],
        args.verbose,
    ))

    # ── /regime ──────────────────────────────────────────────────────────────
    results.append(run_test(
        "POST /regime — bull data → trending_bull",
        session, "POST", f"{base}/regime",
        {"candles_5m": flat_bars_5m(30), "candles_15m": stepped_bull_15m()},
        200,
        [
            ("regime=trending_bull", lambda j: j.get("regime") == "trending_bull"),
            ("votes_bull >= 2",      lambda j: j.get("votes_bull", 0) >= 2),
            ("notes is list",        lambda j: isinstance(j.get("notes"), list)),
        ],
        args.verbose,
    ))

    results.append(run_test(
        "POST /regime — ranging data → ranging",
        session, "POST", f"{base}/regime",
        {"candles_5m": flat_bars_5m(30), "candles_15m": ranging_15m()},
        200,
        [
            ("regime=ranging", lambda j: j.get("regime") == "ranging"),
        ],
        args.verbose,
    ))

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    colour = "\033[92m" if passed == total else "\033[91m"
    print(f"\n{colour}{'─'*40}")
    print(f"  {passed}/{total} tests passed")
    print(f"{'─'*40}\033[0m\n")

    sys.exit(0 if passed == total else 1)


# ---------------------------------------------------------------------------
# Equivalent curl commands (for reference)
# ---------------------------------------------------------------------------
CURL_EXAMPLES = """
# Health check
curl http://localhost:8000/health

# Regime detection only
curl -s -X POST http://localhost:8000/regime \\
  -H "Content-Type: application/json" \\
  -d '{"candles_5m": [...], "candles_15m": [...]}' | python3 -m json.tool

# Full signal (no_trade — flat candles)
curl -s -X POST http://localhost:8000/signal \\
  -H "Content-Type: application/json" \\
  -d '{
    "candles_5m":  [{"timestamp":"2024-03-19T14:00:00+00:00","open":5202,"high":5207,"low":5199,"close":5204}],
    "candles_15m": [{"timestamp":"2024-03-19T13:30:00+00:00","open":5202,"high":5207,"low":5199,"close":5204}],
    "instrument": "NQ",
    "trades_today": 0
  }' | python3 -m json.tool
"""

if __name__ == "__main__":
    main()
