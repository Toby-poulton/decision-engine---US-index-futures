"""
=============================================================================
US INDEX FUTURES SIGNAL ENGINE  —  ES / NQ  (ICT-Style Setups)
=============================================================================
Strategy : Liquidity Sweep → Displacement → BOS/CHOCH → IFVG → Retracement
Session  : NY 09:30–12:00 ET only
Timeframes: 15m (swing structure) · 5m (trigger / entry)
Risk     : $100–$300 per trade · $2,000 max account drawdown buffer
=============================================================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Optional

import pytz

# ---------------------------------------------------------------------------
# CONSTANTS & PARAMETERS  (tune here — do not scatter magic numbers)
# ---------------------------------------------------------------------------

# ── Instrument tick values ──────────────────────────────────────────────────
TICK_VALUE = {
    "ES": 12.50,   # 0.25 point = $12.50 per mini contract
    "NQ": 5.00,    # 0.25 point = $5.00 per mini contract
}
TICK_SIZE = {
    "ES": 0.25,
    "NQ": 0.25,
}

# ── Session ─────────────────────────────────────────────────────────────────
NY_TZ              = pytz.timezone("America/New_York")
SESSION_START      = time(9, 30)
SESSION_END        = time(12, 0)

# ── Swing detection (fractal pivots on 15m) ─────────────────────────────────
SWING_LOOKBACK_15M = 5          # bars left AND right for pivot confirmation
SWING_LOOKBACK_5M  = 10         # bars used when falling back to 5m structure

# ── Liquidity sweep ─────────────────────────────────────────────────────────
SWEEP_CLOSE_BACK_BARS = 3       # candle must close back inside within N bars
SWEEP_MIN_WICK_ATR    = 0.25    # wick beyond swing must be ≥ 0.25 × ATR

# ── Displacement thresholds ─────────────────────────────────────────────────
DISP_BODY_ATR_MULT    = 0.60    # candle body ≥ 0.60 × ATR (strong momentum)
DISP_RANGE_ATR_MULT   = 0.80    # candle high-low range ≥ 0.80 × ATR
DISP_MIN_CANDLES      = 1       # at minimum 1 qualifying candle in disp leg
DISP_MAX_CANDLES      = 5       # displacement leg capped at 5 candles

# ── Volatility / no-trade filter ────────────────────────────────────────────
ATR_PERIOD            = 14
ATR_LOW_MULT          = 0.40    # ATR < 0.40 × 20-period median → low vol
ATR_MEDIAN_PERIOD     = 20

# ── IFVG detection ──────────────────────────────────────────────────────────
IFVG_MIN_SIZE_ATR     = 0.20    # gap must be ≥ 0.20 × ATR to count
IFVG_MAX_FILL_PCT     = 0.50    # entry valid while ≤ 50 % of gap is filled

# ── Risk management ─────────────────────────────────────────────────────────
RISK_MIN_USD          = 100
RISK_MAX_USD          = 300
ACCOUNT_MAX_LOSS_USD  = 2_000
MAX_CONTRACTS         = 5
TRADES_PER_DAY_MAX    = 2

# ── News blackout ───────────────────────────────────────────────────────────
NEWS_BUFFER_MINUTES   = 30

# ── Take-profit ratios ──────────────────────────────────────────────────────
TP1_RR = 1.5   # 1.5 R
TP2_RR = 3.0   # 3.0 R


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

class Direction(str, Enum):
    LONG     = "long"
    SHORT    = "short"
    NO_TRADE = "no_trade"


@dataclass
class Candle:
    timestamp: datetime   # UTC-aware
    open:  float
    high:  float
    low:   float
    close: float
    volume: float = 0.0

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low


@dataclass
class SwingLevel:
    price: float
    timestamp: datetime
    is_high: bool          # True = swing high, False = swing low
    bar_index: int


@dataclass
class IFVG:
    low:  float
    high: float
    direction: Direction    # direction trade is expected to take after fill
    formed_at: datetime
    candle_indices: tuple   # (i-2, i-1, i) indices in the displacement leg


@dataclass
class TradeSignal:
    direction:              Direction
    setup_valid:            bool
    reasons:                list[str]
    entry_zone:             Optional[tuple[float, float]]
    stop_loss:              Optional[float]
    take_profit:            Optional[dict]
    risk_per_trade_usd:     Optional[float]
    position_size_contracts: Optional[int]
    confidence_score:       int            # 0–100
    regime:                 Optional["RegimeResult"] = None   # populated after regime module

    def to_dict(self) -> dict:
        return {
            "direction":               self.direction.value,
            "setup_valid":             self.setup_valid,
            "reasons":                 self.reasons,
            "entry_zone":              list(self.entry_zone) if self.entry_zone else None,
            "stop_loss":               self.stop_loss,
            "take_profit":             self.take_profit,
            "risk_per_trade_usd":      self.risk_per_trade_usd,
            "position_size_contracts": self.position_size_contracts,
            "confidence_score":        self.confidence_score,
            "regime":                  self.regime.to_dict() if self.regime else None,
        }


# ---------------------------------------------------------------------------
# UTILITY HELPERS
# ---------------------------------------------------------------------------

def to_ny(dt: datetime) -> datetime:
    """Convert UTC-aware datetime to NY local time."""
    return dt.astimezone(NY_TZ)


def in_session(dt: datetime) -> bool:
    """True if timestamp falls within NY session window."""
    ny = to_ny(dt)
    return SESSION_START <= ny.time() < SESSION_END


def near_news(dt: datetime, news_times: list[datetime]) -> bool:
    """True if dt is within ±NEWS_BUFFER_MINUTES of any news event."""
    buf = timedelta(minutes=NEWS_BUFFER_MINUTES)
    for event in news_times:
        if abs(dt - event) <= buf:
            return True
    return False


def calc_atr(candles: list[Candle], period: int = ATR_PERIOD) -> list[float]:
    """
    Wilder ATR — returns a list aligned with `candles`.
    Index 0..(period-1) are None; thereafter float values.
    """
    n = len(candles)
    atrs: list[Optional[float]] = [None] * n
    if n < period + 1:
        return atrs   # type: ignore

    # True ranges
    trs = []
    for i in range(1, n):
        c = candles[i]
        p = candles[i - 1]
        tr = max(c.high - c.low,
                 abs(c.high - p.close),
                 abs(c.low  - p.close))
        trs.append(tr)

    # Seed with simple average of first `period` TRs
    seed = sum(trs[:period]) / period
    atrs[period] = seed
    for i in range(period + 1, n):
        atrs[i] = (atrs[i - 1] * (period - 1) + trs[i - 1]) / period  # type: ignore

    return atrs   # type: ignore


def is_low_volatility(atrs: list[Optional[float]],
                      current_idx: int,
                      median_period: int = ATR_MEDIAN_PERIOD,
                      threshold_mult: float = ATR_LOW_MULT) -> bool:
    """
    Low vol = current ATR < threshold_mult × median of last `median_period` ATRs.
    Returns True (no-trade) if ATR is compressed.
    """
    cur = atrs[current_idx]
    if cur is None:
        return True   # not enough data → be conservative

    lookback = []
    for i in range(max(0, current_idx - median_period), current_idx + 1):
        if atrs[i] is not None:
            lookback.append(atrs[i])

    if len(lookback) < median_period // 2:
        return True   # not enough history

    sorted_lb = sorted(lookback)
    mid = len(sorted_lb) // 2
    median = (sorted_lb[mid - 1] + sorted_lb[mid]) / 2 if len(sorted_lb) % 2 == 0 \
             else sorted_lb[mid]

    return cur < threshold_mult * median


# ---------------------------------------------------------------------------
# STEP 1 — SWING HIGH / LOW DETECTION  (fractal pivot method)
# ---------------------------------------------------------------------------

def detect_swings(candles: list[Candle],
                  lookback: int = SWING_LOOKBACK_15M) -> list[SwingLevel]:
    """
    Fractal pivot: a bar at index i is a swing HIGH if its high is the
    highest among bars [i-lookback … i+lookback].  Similarly for lows.

    Returns list of SwingLevel objects sorted by bar_index.
    NOTE: requires len(candles) ≥ 2*lookback + 1.
          The last `lookback` bars cannot yet be confirmed swings.
    """
    swings: list[SwingLevel] = []
    n = len(candles)

    for i in range(lookback, n - lookback):
        window_h = [candles[j].high for j in range(i - lookback, i + lookback + 1)]
        window_l = [candles[j].low  for j in range(i - lookback, i + lookback + 1)]

        is_swing_h = candles[i].high == max(window_h)
        is_swing_l = candles[i].low  == min(window_l)

        # Exclude ambiguous bars where both fire simultaneously
        if is_swing_h and not is_swing_l:
            swings.append(SwingLevel(
                price=candles[i].high,
                timestamp=candles[i].timestamp,
                is_high=True,
                bar_index=i,
            ))
        elif is_swing_l and not is_swing_h:
            swings.append(SwingLevel(
                price=candles[i].low,
                timestamp=candles[i].timestamp,
                is_high=False,
                bar_index=i,
            ))

    return swings


def get_most_recent_swings(swings: list[SwingLevel],
                           as_of_bar: int
                           ) -> tuple[Optional[SwingLevel], Optional[SwingLevel]]:
    """
    Return (most_recent_swing_high, most_recent_swing_low) before `as_of_bar`.
    """
    visible = [s for s in swings if s.bar_index < as_of_bar]
    highs = [s for s in visible if s.is_high]
    lows  = [s for s in visible if not s.is_high]

    last_high = max(highs, key=lambda s: s.bar_index) if highs else None
    last_low  = max(lows,  key=lambda s: s.bar_index) if lows  else None

    return last_high, last_low


# ---------------------------------------------------------------------------
# STEP 2 — LIQUIDITY SWEEP DETECTION
# ---------------------------------------------------------------------------

def detect_sweep(candles: list[Candle],
                 swing: SwingLevel,
                 sweep_start_idx: int,
                 atr: float,
                 is_sweep_high: bool) -> Optional[int]:
    """
    Checks candles from `sweep_start_idx` onward for a sweep of `swing`.

    SWEEP DEFINITION:
      Bullish sweep (of swing LOW):
        - A candle's LOW prints BELOW swing.price  (takes liquidity)
        - That candle OR one of the next SWEEP_CLOSE_BACK_BARS candles
          CLOSES BACK ABOVE swing.price
        - The wick below swing must be ≥ SWEEP_MIN_WICK_ATR × ATR

      Bearish sweep (of swing HIGH):
        - A candle's HIGH prints ABOVE swing.price
        - Closes back BELOW swing.price within SWEEP_CLOSE_BACK_BARS
        - Wick above ≥ SWEEP_MIN_WICK_ATR × ATR

    Returns the index of the sweep candle if detected, else None.
    """
    min_wick = SWEEP_MIN_WICK_ATR * atr

    for i in range(sweep_start_idx, len(candles)):
        c = candles[i]

        if is_sweep_high:
            # Sweep of a prior HIGH — price spikes above then rejects
            if c.high <= swing.price:
                continue
            # Penetration = how far price pushed BEYOND the swing level
            penetration_above = c.high - swing.price
            if penetration_above < min_wick:
                continue
            # Check close-back within N bars
            for j in range(i, min(i + SWEEP_CLOSE_BACK_BARS + 1, len(candles))):
                if candles[j].close < swing.price:
                    return i
        else:
            # Sweep of a prior LOW — price dips below then recovers
            if c.low >= swing.price:
                continue
            # Penetration = how far price pushed BEYOND the swing level
            penetration_below = swing.price - c.low
            if penetration_below < min_wick:
                continue
            for j in range(i, min(i + SWEEP_CLOSE_BACK_BARS + 1, len(candles))):
                if candles[j].close > swing.price:
                    return i

    return None


# ---------------------------------------------------------------------------
# STEP 3 — DISPLACEMENT DETECTION  (follows immediately after sweep)
# ---------------------------------------------------------------------------

@dataclass
class DisplacementResult:
    valid: bool
    start_idx: int
    end_idx: int
    direction: Direction
    candle_indices: list[int]       # indices of qualifying displacement candles


def detect_displacement(candles: list[Candle],
                        sweep_idx: int,
                        atr: float,
                        expected_direction: Direction) -> DisplacementResult:
    """
    After the sweep, we expect a strong impulsive move AWAY from the swept level.

    DISPLACEMENT CRITERIA (every qualifying candle must satisfy):
      1. Candle body ≥ DISP_BODY_ATR_MULT × ATR
      2. Candle range ≥ DISP_RANGE_ATR_MULT × ATR
      3. Candle closes in the expected direction

    The displacement leg must contain ≥ DISP_MIN_CANDLES qualifying candles
    and span no more than DISP_MAX_CANDLES bars total.

    Additionally, the displacement must BREAK MARKET STRUCTURE (see BOS/CHOCH
    check in the main pipeline — this function flags the leg; structure break
    is verified separately via detect_bos_choch).
    """
    min_body  = DISP_BODY_ATR_MULT  * atr
    min_range = DISP_RANGE_ATR_MULT * atr

    qualifying: list[int] = []
    end_idx = sweep_idx

    for i in range(sweep_idx + 1,
                   min(sweep_idx + 1 + DISP_MAX_CANDLES, len(candles))):
        c = candles[i]
        body_ok  = c.body  >= min_body
        range_ok = c.range >= min_range
        dir_ok   = (c.is_bullish  if expected_direction == Direction.LONG
                    else not c.is_bullish)

        if body_ok and range_ok and dir_ok:
            qualifying.append(i)
            end_idx = i
        elif qualifying:
            # First non-qualifying candle ends the leg
            break

    valid = len(qualifying) >= DISP_MIN_CANDLES

    return DisplacementResult(
        valid=valid,
        start_idx=sweep_idx + 1,
        end_idx=end_idx,
        direction=expected_direction,
        candle_indices=qualifying,
    )


# ---------------------------------------------------------------------------
# STEP 4 — MARKET STRUCTURE BREAK  (BOS / CHOCH)
# ---------------------------------------------------------------------------

def detect_bos_choch(candles: list[Candle],
                     disp: DisplacementResult,
                     last_swing_high: Optional[SwingLevel],
                     last_swing_low:  Optional[SwingLevel]) -> tuple[bool, str]:
    """
    BOS  (Break of Structure) : price closes beyond the most recent
          swing HIGH (for bullish disp) or swing LOW (for bearish disp)
          in the direction of displacement.

    CHOCH (Change of Character): first BULLISH close above a prior
          swing HIGH after a downtrend (or first BEARISH close below
          prior swing LOW after an uptrend).

    For our system both BOS and CHOCH satisfy the structure-break condition.
    We simply verify that at least one displacement candle closed beyond
    the relevant prior swing level.

    Returns (confirmed: bool, label: str)
    """
    if not disp.valid:
        return False, "displacement_invalid"

    for idx in disp.candle_indices:
        c = candles[idx]
        if disp.direction == Direction.LONG and last_swing_high:
            if c.close > last_swing_high.price:
                return True, "BOS_bullish"
        elif disp.direction == Direction.SHORT and last_swing_low:
            if c.close < last_swing_low.price:
                return True, "BOS_bearish"

    # Looser CHOCH check — did price invalidate prior structure?
    if disp.direction == Direction.LONG and last_swing_high:
        end_candle = candles[disp.end_idx]
        if end_candle.high > last_swing_high.price:
            return True, "CHOCH_bullish"
    elif disp.direction == Direction.SHORT and last_swing_low:
        end_candle = candles[disp.end_idx]
        if end_candle.low < last_swing_low.price:
            return True, "CHOCH_bearish"

    return False, "no_structure_break"


# ---------------------------------------------------------------------------
# STEP 5 — IFVG DETECTION  (Inverse Fair Value Gap)
# ---------------------------------------------------------------------------

def detect_ifvg(candles: list[Candle],
                disp: DisplacementResult,
                atr: float) -> Optional[IFVG]:
    """
    IFVG (3-candle imbalance formed DURING the displacement leg):

      BULLISH IFVG:
        candles[i-2].high  <  candles[i].low
        → gap between top of candle i-2 and bottom of candle i
        (price left an imbalance that will be re-tested from below)

      BEARISH IFVG:
        candles[i-2].low   >  candles[i].high
        → gap between bottom of candle i-2 and top of candle i

    Additional filters:
      - Gap size ≥ IFVG_MIN_SIZE_ATR × ATR
      - All 3 candles must be within the displacement leg indices
      - We return the LAST (most recent / deepest) valid IFVG in the leg
        because that is closest to structure and highest quality.
    """
    min_gap = IFVG_MIN_SIZE_ATR * atr
    leg_set = set(disp.candle_indices)
    result:  Optional[IFVG] = None

    for i in range(disp.start_idx + 2, disp.end_idx + 1):
        if i     not in leg_set: continue
        if i - 1 not in leg_set: continue
        if i - 2 not in leg_set: continue

        c0, c1, c2 = candles[i - 2], candles[i - 1], candles[i]

        if disp.direction == Direction.LONG:
            gap_low  = c0.high
            gap_high = c2.low
            if gap_high > gap_low and (gap_high - gap_low) >= min_gap:
                result = IFVG(
                    low=gap_low,
                    high=gap_high,
                    direction=Direction.LONG,
                    formed_at=c2.timestamp,
                    candle_indices=(i - 2, i - 1, i),
                )

        elif disp.direction == Direction.SHORT:
            gap_low  = c2.high
            gap_high = c0.low
            if gap_high > gap_low and (gap_high - gap_low) >= min_gap:
                result = IFVG(
                    low=gap_low,
                    high=gap_high,
                    direction=Direction.SHORT,
                    formed_at=c2.timestamp,
                    candle_indices=(i - 2, i - 1, i),
                )

    return result   # None if no qualifying IFVG found


# ---------------------------------------------------------------------------
# STEP 6 — ENTRY ZONE  (first retracement into IFVG)
# ---------------------------------------------------------------------------

def check_retracement_into_ifvg(candles: list[Candle],
                                 ifvg: IFVG,
                                 after_idx: int) -> Optional[int]:
    """
    Scans candles after the displacement for the FIRST bar that retraces
    INTO the IFVG zone.

    Entry conditions:
      LONG : candle low ≤ ifvg.high AND candle low ≥ ifvg.low
             (price dips into the gap from above)
      SHORT: candle high ≥ ifvg.low AND candle high ≤ ifvg.high
             (price rallies into the gap from below)

    We also check that the IFVG is not more than 50% filled
    (IFVG_MAX_FILL_PCT) to ensure the imbalance is still valid.

    Returns bar index of the entry candle, or None if not yet reached.
    """
    # Entry zone is the full IFVG range.
    # Valid entry: price touches the zone (c.low <= ifvg.high for long).
    # Invalidated: price CLOSES entirely through the zone (beyond it).
    # IFVG_MAX_FILL_PCT controls the close-threshold for invalidation:
    #   if close < ifvg.low + (1-FILL_PCT)*gap → zone too far penetrated.
    gap_size   = ifvg.high - ifvg.low
    # Invalidation close level: price closed beyond (1-PCT) of the gap
    invalid_close_long  = ifvg.low  - IFVG_MAX_FILL_PCT * gap_size
    invalid_close_short = ifvg.high + IFVG_MAX_FILL_PCT * gap_size

    for i in range(after_idx, len(candles)):
        c = candles[i]

        if ifvg.direction == Direction.LONG:
            # Entry: price dips into IFVG zone from above
            if c.low <= ifvg.high and c.low >= ifvg.low - gap_size * 0.1:
                return i  # touched the zone — valid entry bar
            # Invalidated: closed well below the zone
            if c.close < invalid_close_long:
                return None

        elif ifvg.direction == Direction.SHORT:
            # Entry: price rallies into IFVG zone from below
            if c.high >= ifvg.low and c.high <= ifvg.high + gap_size * 0.1:
                return i  # touched the zone — valid entry bar
            # Invalidated: closed well above the zone
            if c.close > invalid_close_short:
                return None

    return None


# ---------------------------------------------------------------------------
# RISK & POSITION SIZING
# ---------------------------------------------------------------------------

def calculate_risk(entry_zone: tuple[float, float],
                   stop_loss: float,
                   direction: Direction,
                   instrument: str) -> tuple[Optional[float], Optional[int]]:
    """
    Calculates USD risk and position size.

    Returns (risk_usd, contracts) or (None, None) if sizing is invalid.

    Stop distance is measured from the WORST entry edge of the zone:
      LONG  → worst entry = entry_zone[1] (high of zone)
      SHORT → worst entry = entry_zone[0] (low of zone)
    """
    tick_val  = TICK_VALUE.get(instrument, 12.50)
    tick_size = TICK_SIZE.get(instrument, 0.25)

    worst_entry = entry_zone[1] if direction == Direction.LONG else entry_zone[0]

    if direction == Direction.LONG:
        stop_dist_points = worst_entry - stop_loss
    else:
        stop_dist_points = stop_loss - worst_entry

    if stop_dist_points <= 0:
        return None, None

    ticks_risk    = stop_dist_points / tick_size
    risk_per_cont = ticks_risk * tick_val

    # Find the contract count that keeps risk in [$100, $300]
    contracts = None
    for n in range(1, MAX_CONTRACTS + 1):
        total_risk = risk_per_cont * n
        if RISK_MIN_USD <= total_risk <= RISK_MAX_USD:
            contracts = n
            break
        if total_risk > RISK_MAX_USD:
            break

    if contracts is None:
        return None, None

    return round(risk_per_cont * contracts, 2), contracts


def calculate_take_profits(entry_price: float,
                           stop_loss: float,
                           direction: Direction) -> dict:
    """Calculates TP1 and TP2 based on R multiples."""
    risk = abs(entry_price - stop_loss)
    if direction == Direction.LONG:
        return {
            "tp1": round(entry_price + TP1_RR * risk, 2),
            "tp2": round(entry_price + TP2_RR * risk, 2),
        }
    else:
        return {
            "tp1": round(entry_price - TP1_RR * risk, 2),
            "tp2": round(entry_price - TP2_RR * risk, 2),
        }


# ---------------------------------------------------------------------------
# CONFIDENCE SCORING
# ---------------------------------------------------------------------------

def compute_confidence(reasons: list[str],
                       sweep_wick_atr_ratio: float,
                       disp_candle_count: int,
                       ifvg_size_atr_ratio: float,
                       structure_label: str,
                       regime: "Optional[RegimeResult]" = None,
                       signal_direction: "Optional[Direction]" = None) -> int:
    """
    Heuristic confidence score 0–100 based on signal quality.

    Scoring breakdown (max 100):
      Base (steps passed)     — up to 50
      Sweep quality           — up to 15  (wick penetration vs ATR)
      Displacement strength   — up to 15  (qualifying candle count)
      IFVG size               — up to 10  (gap size vs ATR)
      Structure break type    — up to 10  (BOS > CHOCH)
      Regime bonus            — up to 10  (strong trend alignment)
      Regime counter-trend    — -REGIME_COUNTER_TREND_PENALTY (if applied,
                                 signals should have been blocked earlier,
                                 but penalty is a safety net)
    """
    score = 0

    # Base: all steps passed (max 50)
    passed = sum(1 for r in reasons if "✓" in r)
    score += min(passed * 8, 50)

    # Sweep quality (max 15)
    score += min(int(sweep_wick_atr_ratio * 15), 15)

    # Displacement strength (max 15)
    score += min(disp_candle_count * 5, 15)

    # IFVG size (max 10)
    score += min(int(ifvg_size_atr_ratio * 10), 10)

    # Structure break type (max 10)
    if "BOS" in structure_label:
        score += 10
    elif "CHOCH" in structure_label:
        score += 6

    # Regime quality adjustment (max +10 / min -20)
    if regime is not None and signal_direction is not None:
        trend_strength = regime.votes_bull + regime.votes_bear  # max 4

        # Bonus: strong with-trend signal (ADX high, all votes aligned)
        is_with_trend = (
            (regime.regime == Regime.TRENDING_BULL and signal_direction == Direction.LONG) or
            (regime.regime == Regime.TRENDING_BEAR and signal_direction == Direction.SHORT)
        )
        if is_with_trend:
            # Scale bonus by strength of trend conviction
            bonus = min(int(trend_strength * 2.5), 10)
            score += bonus

        # Penalty: counter-trend signals that slipped through
        is_counter = (
            (regime.regime == Regime.TRENDING_BULL and signal_direction == Direction.SHORT) or
            (regime.regime == Regime.TRENDING_BEAR and signal_direction == Direction.LONG)
        )
        if is_counter:
            score -= REGIME_COUNTER_TREND_PENALTY

    return max(0, min(score, 100))


# ---------------------------------------------------------------------------
# MAIN SIGNAL ENGINE
# ---------------------------------------------------------------------------

def generate_signal(candles_5m:   list[Candle],
                    candles_15m:  list[Candle],
                    instrument:   str = "ES",
                    news_times:   list[datetime] = None,
                    trades_today: int = 0) -> TradeSignal:
    """
    Master pipeline.  Call this once per 5-minute bar close.

    Args:
        candles_5m   : Recent 5-minute candles (at least 50 recommended)
        candles_15m  : Recent 15-minute candles (at least 40 recommended)
        instrument   : "ES" or "NQ"
        news_times   : List of UTC-aware datetimes for news events today
        trades_today : Number of trades already taken today

    Returns:
        TradeSignal  : Complete signal object (call .to_dict() for JSON)
                       Includes 'regime' field with full RegimeResult breakdown.

    Regime suppression rules (applied after Step 2 swing detection):
        RANGING       → immediate no_trade for all IFVG signals
        TRENDING_BULL → SHORT signals suppressed; LONG signals proceed
        TRENDING_BEAR → LONG signals suppressed; SHORT signals proceed
        Counter-trend → confidence penalised by REGIME_COUNTER_TREND_PENALTY
    """
    if news_times is None:
        news_times = []

    reasons: list[str] = []

    def no_trade(msg: str) -> TradeSignal:
        reasons.append(f"✗ {msg}")
        return TradeSignal(
            direction=Direction.NO_TRADE,
            setup_valid=False,
            reasons=reasons,
            entry_zone=None,
            stop_loss=None,
            take_profit=None,
            risk_per_trade_usd=None,
            position_size_contracts=None,
            confidence_score=0,
        )

    # ── 0. Basic guards ─────────────────────────────────────────────────────
    if not candles_5m or not candles_15m:
        return no_trade("insufficient_data")

    current_bar  = candles_5m[-1]
    current_time = current_bar.timestamp

    if trades_today >= TRADES_PER_DAY_MAX:
        return no_trade(f"daily_trade_limit_reached ({trades_today}/{TRADES_PER_DAY_MAX})")

    if not in_session(current_time):
        return no_trade("outside_ny_session")
    reasons.append("✓ within_ny_session")

    if near_news(current_time, news_times):
        return no_trade("within_news_blackout_window")
    reasons.append("✓ no_news_proximity")

    # ── 1. ATR & volatility filter ──────────────────────────────────────────
    atrs_5m   = calc_atr(candles_5m,  ATR_PERIOD)
    atr_idx   = len(candles_5m) - 1
    current_atr = atrs_5m[atr_idx]

    if current_atr is None:
        return no_trade("atr_not_calculable_insufficient_data")

    if is_low_volatility(atrs_5m, atr_idx):
        return no_trade(f"low_volatility_compression (ATR={current_atr:.2f})")
    reasons.append(f"✓ volatility_ok (ATR={current_atr:.2f})")

    # ── 2. Swing detection on 15m ───────────────────────────────────────────
    swings_15m = detect_swings(candles_15m, SWING_LOOKBACK_15M)
    if len(swings_15m) < 2:
        return no_trade("insufficient_swing_structure_on_15m")

    # Map 15m swings to approximate 5m context (use most recent)
    last_high_15m, last_low_15m = get_most_recent_swings(swings_15m,
                                                          len(candles_15m))

    if last_high_15m is None or last_low_15m is None:
        return no_trade("missing_swing_high_or_low_on_15m")
    reasons.append(f"✓ swing_levels_identified "
                   f"(H={last_high_15m.price}, L={last_low_15m.price})")

    # ── 2b. REGIME DETECTION ────────────────────────────────────────────────
    # Run before any further signal logic so regime can gate ALL downstream steps.
    # Swings are already computed above and reused here.
    regime_result = detect_regime(candles_15m, swings_15m, candles_5m)

    if regime_result.regime == Regime.RANGING:
        # IFVG signals in a ranging market have poor expectancy.
        # Price chops through imbalance zones without follow-through.
        regime_sig = no_trade(
            f"regime=RANGING — IFVG signals suppressed "
            f"(votes: bull={regime_result.votes_bull} "
            f"bear={regime_result.votes_bear} "
            f"range={regime_result.votes_range})"
        )
        regime_sig.regime = regime_result
        return regime_sig
    reasons.append(
        f"✓ regime={regime_result.regime.value} "
        f"(bull={regime_result.votes_bull} "
        f"bear={regime_result.votes_bear} "
        f"range={regime_result.votes_range})"
    )

    # ── 3. Liquidity sweep detection ────────────────────────────────────────
    #  Check for sweep of EITHER swing high (bearish setup) or swing low (bullish)
    #  on 5m candles.  We scan the most recent 20 bars for a qualifying sweep.
    scan_start = max(0, len(candles_5m) - 20)

    sweep_idx_high = detect_sweep(
        candles_5m, last_high_15m, scan_start, current_atr, is_sweep_high=True
    )
    sweep_idx_low  = detect_sweep(
        candles_5m, last_low_15m,  scan_start, current_atr, is_sweep_high=False
    )

    # Ambiguity check — if BOTH fire within 2 bars of each other → no_trade
    if sweep_idx_high is not None and sweep_idx_low is not None:
        if abs(sweep_idx_high - sweep_idx_low) <= 2:
            return no_trade("conflicting_sweeps_detected_ambiguous")

    # When BOTH exist: prefer the EARLIER sweep (lower bar index).
    # Rationale: the later one may be the displacement leg itself
    # masquerading as a sweep (e.g., BOS candle wicking through prior high).
    if sweep_idx_high is not None and sweep_idx_low is not None:
        sweep_is_high = sweep_idx_high < sweep_idx_low
    elif sweep_idx_high is not None:
        sweep_is_high = True
    elif sweep_idx_low is not None:
        sweep_is_high = False
    else:
        return no_trade("no_liquidity_sweep_detected")

    sweep_idx  = sweep_idx_high if sweep_is_high else sweep_idx_low
    swept_swing = last_high_15m  if sweep_is_high else last_low_15m

    # Direction: sweep of HIGH → expect SHORT; sweep of LOW → expect LONG
    expected_dir = Direction.SHORT if sweep_is_high else Direction.LONG

    # ── 3b. REGIME-DIRECTION ALIGNMENT CHECK ────────────────────────────────
    # TRENDING regimes only allow with-trend signals.
    # A SHORT signal in a TRENDING_BULL regime (or vice versa) is suppressed.
    is_counter_trend = (
        (regime_result.regime == Regime.TRENDING_BULL and expected_dir == Direction.SHORT) or
        (regime_result.regime == Regime.TRENDING_BEAR and expected_dir == Direction.LONG)
    )
    if is_counter_trend:
        ct_sig = no_trade(
            f"regime={regime_result.regime.value} suppresses "
            f"{expected_dir.value.upper()} signal "
            f"(counter-trend — only with-trend IFVGs allowed)"
        )
        ct_sig.regime = regime_result
        return ct_sig
    reasons.append(f"✓ regime_direction_aligned ({expected_dir.value} with {regime_result.regime.value})")

    # Wick quality metric for confidence scoring
    sweep_candle = candles_5m[sweep_idx]

    penetration = (sweep_candle.high - swept_swing.price) if sweep_is_high else (swept_swing.price - sweep_candle.low)
    sweep_wick_atr_ratio = penetration / current_atr
    reasons.append(f"✓ liquidity_sweep_confirmed at={swept_swing.price} "
                   f"dir={expected_dir.value} wick_atr_ratio={sweep_wick_atr_ratio:.2f}")

    # ── 4. Displacement ─────────────────────────────────────────────────────
    disp = detect_displacement(candles_5m, sweep_idx, current_atr, expected_dir)

    if not disp.valid:
        return no_trade(f"displacement_failed_threshold "
                        f"(qualifying_candles={len(disp.candle_indices)})")
    reasons.append(f"✓ displacement_valid "
                   f"({len(disp.candle_indices)} candles, dir={disp.direction.value})")

    # ── 5. BOS / CHOCH ──────────────────────────────────────────────────────
    bos_ok, bos_label = detect_bos_choch(
        candles_5m, disp, last_high_15m, last_low_15m
    )
    if not bos_ok:
        return no_trade(f"market_structure_break_failed ({bos_label})")
    reasons.append(f"✓ market_structure_break ({bos_label})")

    # ── 6. IFVG ─────────────────────────────────────────────────────────────
    ifvg = detect_ifvg(candles_5m, disp, current_atr)
    if ifvg is None:
        return no_trade("no_qualifying_ifvg_in_displacement_leg")

    ifvg_size_atr_ratio = (ifvg.high - ifvg.low) / current_atr
    reasons.append(f"✓ ifvg_identified "
                   f"zone=[{ifvg.low}, {ifvg.high}] "
                   f"size_atr_ratio={ifvg_size_atr_ratio:.2f}")

    # ── 7. Retracement into IFVG ────────────────────────────────────────────
    entry_idx = check_retracement_into_ifvg(candles_5m, ifvg,
                                             after_idx=disp.end_idx + 1)
    if entry_idx is None:
        return no_trade("no_retracement_into_ifvg_yet_or_ifvg_invalidated")
    reasons.append(f"✓ price_retesting_ifvg at bar_idx={entry_idx}")

    # ── 8. Entry zone & stop loss ────────────────────────────────────────────
    entry_zone = (ifvg.low, ifvg.high)
    tick = TICK_SIZE.get(instrument, 0.25)

    if expected_dir == Direction.LONG:
        # Stop just below IFVG low (1 tick buffer).
        # Rationale: if price closes below the IFVG entirely, the
        # institutional imbalance is filled and thesis is invalidated.
        # The sweep low provides directional CONTEXT but not the stop
        # level — using sweep low creates too-wide risk for funded accounts.
        # Stop sits 1 tick below IFVG low. If price closes below the
        # imbalance zone, the setup is invalidated — no need for a wider stop.
        stop_loss = round(ifvg.low - tick, 2)
    else:
        # Stop sits 1 tick above IFVG high for shorts.
        stop_loss = round(ifvg.high + tick, 2)

    # ── 9. Risk & position sizing ────────────────────────────────────────────
    risk_usd, contracts = calculate_risk(entry_zone, stop_loss,
                                         expected_dir, instrument)
    if risk_usd is None:
        return no_trade("position_size_outside_risk_bounds — "
                        "stop_too_wide_or_too_tight")

    # Mid-zone price for TP calculation
    mid_entry = (entry_zone[0] + entry_zone[1]) / 2
    take_profit = calculate_take_profits(mid_entry, stop_loss, expected_dir)

    reasons.append(f"✓ risk_sizing_valid "
                   f"(${risk_usd}, {contracts} contract(s))")

    # ── 10. Confidence score ─────────────────────────────────────────────────
    confidence = compute_confidence(
        reasons,
        sweep_wick_atr_ratio,
        len(disp.candle_indices),
        ifvg_size_atr_ratio,
        bos_label,
        regime=regime_result,
        signal_direction=expected_dir,
    )

    return TradeSignal(
        direction=expected_dir,
        setup_valid=True,
        reasons=reasons,
        entry_zone=entry_zone,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_per_trade_usd=risk_usd,
        position_size_contracts=contracts,
        confidence_score=confidence,
        regime=regime_result,
    )


# ---------------------------------------------------------------------------
# PSEUDOCODE SUMMARY  (human-readable pipeline reference)
# ---------------------------------------------------------------------------
"""
PIPELINE (plain English, mirrors the code above):

  INPUT: 5m candles[], 15m candles[], instrument, news_times[], trades_today

  ─── GUARD RAILS ────────────────────────────────────────────────────────────
  IF trades_today >= 2           → NO_TRADE
  IF current_bar NOT in NY 9:30–12:00 → NO_TRADE
  IF current_bar within ±30min of news → NO_TRADE

  ─── VOLATILITY ─────────────────────────────────────────────────────────────
  Compute ATR(14) on 5m candles
  IF ATR < 0.40 × median(ATR, last 20 bars) → NO_TRADE (compression)

  ─── SWING DETECTION (15m) ───────────────────────────────────────────────────
  For each 15m bar i in range [lookback … n-lookback]:
    IF bar[i].high = max(window [i-5 … i+5]) → swing HIGH
    IF bar[i].low  = min(window [i-5 … i+5]) → swing LOW
  Get most_recent_swing_high and most_recent_swing_low

  ─── SWEEP DETECTION (5m) ────────────────────────────────────────────────────
  For recent 5m bars (last 20):
    Check if any bar WICKS BEYOND swing level AND closes back inside within 3 bars
    AND wick ≥ 0.25 × ATR
  Determine direction: sweep HIGH → SHORT; sweep LOW → LONG
  IF both sweep within 2 bars of each other → NO_TRADE (ambiguous)
  IF no sweep → NO_TRADE

  ─── DISPLACEMENT (5m) ───────────────────────────────────────────────────────
  Starting bar AFTER sweep, scan up to 5 bars:
    Qualifying candle: body ≥ 0.60×ATR AND range ≥ 0.80×ATR AND closes in direction
  IF qualifying_count < 1 → NO_TRADE

  ─── BOS / CHOCH ─────────────────────────────────────────────────────────────
  IF LONG:  any displacement candle closes ABOVE last swing HIGH → BOS/CHOCH ✓
  IF SHORT: any displacement candle closes BELOW last swing LOW  → BOS/CHOCH ✓
  IF neither → NO_TRADE

  ─── IFVG ────────────────────────────────────────────────────────────────────
  For each 3-candle group within displacement leg:
    LONG:  IF candles[i-2].high < candles[i].low AND gap ≥ 0.20×ATR → IFVG
    SHORT: IF candles[i-2].low  > candles[i].high AND gap ≥ 0.20×ATR → IFVG
  Take the LAST (deepest) valid IFVG in the leg
  IF none → NO_TRADE

  ─── RETRACEMENT ─────────────────────────────────────────────────────────────
  After displacement ends, scan forward:
    IF price enters IFVG zone AND fills ≤ 50% of gap → entry_bar confirmed
    IF price closes entirely through IFVG → IFVG invalidated → NO_TRADE
  IF never reaches IFVG (within current session) → NO_TRADE (yet)

  ─── SIZING ──────────────────────────────────────────────────────────────────
  stop_loss = swept_swing.price ± 0.10×ATR buffer
  entry_zone = (IFVG.low, IFVG.high)
  worst_entry_edge = entry_zone[1] (long) or entry_zone[0] (short)
  stop_distance = |worst_entry - stop_loss|
  risk_per_contract = (stop_distance / tick_size) × tick_value
  contracts = largest N in [1…5] where risk_per_contract×N ∈ [$100, $300]
  IF no valid N → NO_TRADE

  ─── OUTPUT ──────────────────────────────────────────────────────────────────
  Emit TradeSignal JSON with all fields populated
"""


# ---------------------------------------------------------------------------
# EDGE CASES & FAILURE MODES
# ---------------------------------------------------------------------------
"""
KNOWN EDGE CASES:

  1. GAP OPENS (ES/NQ gap at 9:30)
     ─ ATR inflated by opening gap candle → may incorrectly classify weak
       displacement as strong. MITIGATION: skip the first 2 bars (9:30, 9:35)
       as signal candidates; use them only for structure context.

  2. PRE-MARKET SWING LEVELS
     ─ 15m swings detected in pre-market may be invalid reference points
       for NY session liquidity. MITIGATION: filter swings to regular-hours
       only, or use a separate pre-market high/low as context but not signal.

  3. SWEEP THAT IS ALSO AN ENTRY CANDLE
     ─ Sometimes the sweep candle itself is large enough to qualify as
       displacement. This creates a same-bar ambiguity.
       MITIGATION: displacement must start on the bar AFTER sweep_idx.

  4. IFVG PARTIALLY OVERLAPPING PRIOR STRUCTURE
     ─ An IFVG sitting inside a prior consolidation zone weakens the signal.
       MITIGATION: add a consolidation-zone detector (Bollinger Band squeeze
       or low-range cluster) and invalidate IFVG if ≥ 50% overlaps a
       consolidation zone.

  5. DOUBLE SWEEP PATTERNS
     ─ Price sweeps a level, displaces, then sweeps again. The engine may
       fire twice. MITIGATION: after a signal is emitted, suppress further
       signals on the same swing level for the rest of the session.

  6. NEWS SPIKE FALSE SWEEPS
     ─ A news candle can mimic a sweep+displacement without genuine intent.
       MITIGATION: the ±30 min news blackout handles this; however, for
       unscheduled FOMC/Fed events, the trader must manually override.

  7. THIN PRE-HOLIDAY SESSIONS
     ─ ATR may be unusually low, compressing into the low-volatility filter.
       MITIGATION: add a calendar filter for half-days and US holidays.

  8. RAPID PRICE ACTION (multiple swings in <10 bars)
     ─ Fractal pivot detection requires lookback bars on both sides.
       Fast-moving markets may not confirm swings until the moment has passed.
       MITIGATION: use a shorter lookback (SWING_LOOKBACK_5M = 3) as a
       secondary confirmation layer for 5m structure.
"""


# ---------------------------------------------------------------------------
# PARAMETER RECOMMENDATIONS TABLE
# ---------------------------------------------------------------------------
"""
PARAMETER                   DEFAULT     RANGE TO TEST    RATIONALE
─────────────────────────── ─────────── ──────────────── ─────────────────────
SWING_LOOKBACK_15M          5           3–7              Lower = more swings,
                                                         noisier; higher = fewer
                                                         but more significant

ATR_PERIOD                  14          10–21            Standard Wilder ATR;
                                                         lower is more reactive

DISP_BODY_ATR_MULT          0.60        0.50–0.75        Core filter for
                                                         momentum strength

DISP_RANGE_ATR_MULT         0.80        0.65–1.00        Full-candle range check;
                                                         set higher to demand
                                                         stronger impulse

SWEEP_MIN_WICK_ATR          0.25        0.15–0.40        Larger = cleaner sweeps
                                                         only; smaller = more
                                                         signals, weaker quality

SWEEP_CLOSE_BACK_BARS       3           2–5              Tighter = demands faster
                                                         rejection (cleaner)

ATR_LOW_MULT                0.40        0.30–0.50        Volatility floor; raise
                                                         in trending markets

IFVG_MIN_SIZE_ATR           0.20        0.15–0.30        Minimum imbalance size;
                                                         smaller = more IFVGs,
                                                         noisier entries

IFVG_MAX_FILL_PCT           0.50        0.40–0.65        How far into the gap
                                                         entry is still valid

TP1_RR                      1.5         1.0–2.0          First target; scale half
TP2_RR                      3.0         2.5–4.0          Runner target

NEWS_BUFFER_MINUTES         30          20–45            Widen to 45 around
                                                         CPI/NFP/FOMC
"""


# ---------------------------------------------------------------------------
# ROBUSTNESS IMPROVEMENTS (ROADMAP)
# ---------------------------------------------------------------------------
"""
PHASE 1 — VALIDATE CORE LOGIC
  □ Backtest on 6 months ES/NQ 5m data (2023–2024)
  □ Log all NO_TRADE reasons to find most common failure point
  □ Target: 2–4 signals/week; >55% win rate at 1.5R

PHASE 2 — ADD SECONDARY FILTERS
  □ Volume spike confirmation on displacement candles
    (displacement candle volume > 1.5× 10-bar average)
  □ Session high/low context (is sweep near session open H/L?)
  □ Orderflow / delta confirmation (if data available)

PHASE 3 — ADAPTIVE PARAMETERS
  □ Scale ATR thresholds by rolling 5-day regime:
    trending → relax displacement thresholds
    ranging  → tighten IFVG size minimum
  □ Do NOT optimise parameters per-symbol to avoid overfitting

PHASE 4 — EXECUTION INTEGRATION
  □ Add live OHLCV ingestion (Polygon.io / Databento / Rithmic)
  □ Webhook to TradingView alert or broker API
  □ Enforce daily loss limit kill-switch:
    IF realised_loss_today >= ACCOUNT_MAX_LOSS_USD → halt all signals

PHASE 5 — REVIEW LOOP
  □ Weekly signal review: log confidence scores vs outcomes
  □ If confidence 80+ signals win < 50% → re-examine IFVG definition
  □ If low-confidence signals outperform → investigate what they capture
"""


# ---------------------------------------------------------------------------
# EXAMPLE USAGE
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from datetime import timezone as tz

    # Build a minimal fake dataset for smoke-test
    # In production: replace with live/historical OHLCV feed
    base_time = datetime(2024, 3, 15, 13, 30, tzinfo=tz.utc)  # = 9:30 ET

    def make_candle(offset_min, o, h, l, c):
        return Candle(
            timestamp=base_time + timedelta(minutes=offset_min),
            open=o, high=h, low=l, close=c,
        )

    # 15m swing structure (simplified: 3 swings)
    candles_15m = [
        make_candle(i * 15, 5200, 5220, 5190, 5210) for i in range(30)
    ]
    # Plant a swing low at index 10
    candles_15m[10] = make_candle(150, 5200, 5205, 5150, 5155)
    # Plant a swing high at index 20
    candles_15m[20] = make_candle(300, 5240, 5280, 5235, 5245)

    # 5m candles — range then sweep+displacement
    candles_5m = [
        make_candle(i * 5, 5200, 5215, 5195, 5208) for i in range(40)
    ]
    # Sweep of low at bar 30 (dips to 5148, closes back above 5150)
    candles_5m[30] = make_candle(150, 5195, 5196, 5148, 5155)
    candles_5m[31] = make_candle(155, 5155, 5158, 5153, 5156)   # closes back above
    # Displacement (3 strong bullish candles)
    candles_5m[32] = make_candle(160, 5156, 5190, 5154, 5188)
    candles_5m[33] = make_candle(165, 5188, 5215, 5185, 5212)
    candles_5m[34] = make_candle(170, 5212, 5240, 5210, 5238)
    # Retracement into IFVG
    candles_5m[35] = make_candle(175, 5238, 5238, 5188, 5195)
    candles_5m[36] = make_candle(180, 5195, 5200, 5187, 5192)

    signal = generate_signal(
        candles_5m=candles_5m,
        candles_15m=candles_15m,
        instrument="ES",
        news_times=[],
        trades_today=0,
    )

    print(json.dumps(signal.to_dict(), indent=2))

# =============================================================================
# REGIME DETECTION MODULE
# =============================================================================
# Classifies the current market into one of three states:
#   TRENDING_BULL  — directional uptrend, favour LONG IFVGs only
#   TRENDING_BEAR  — directional downtrend, favour SHORT IFVGs only
#   RANGING        — no clear trend, suppress ALL IFVG signals
#
# Four independent sub-indicators vote; a 3/4 supermajority is required to
# declare a trend. Anything below that threshold is classified as RANGING.
# This is deliberately conservative — when in doubt, ranging wins.
#
# Suppression rules applied in generate_signal():
#   RANGING        → no_trade (IFVGs are low-probability in choppy markets)
#   TRENDING_BULL  → suppress SHORT signals; LONG signals continue pipeline
#   TRENDING_BEAR  → suppress LONG signals; SHORT signals continue pipeline
#   Counter-trend  → confidence score penalised by REGIME_COUNTER_TREND_PENALTY
# =============================================================================

# ── Regime detection parameters ─────────────────────────────────────────────
REGIME_ADX_PERIOD            = 14     # ADX lookback (Wilder smoothing)
REGIME_ADX_TREND_THRESHOLD   = 25.0   # ADX > 25 → trending vote
REGIME_ADX_RANGE_THRESHOLD   = 20.0   # ADX < 20 → ranging vote
REGIME_EMA_PERIOD            = 21     # EMA period for slope measurement
REGIME_EMA_SLOPE_TREND_PCT   = 0.015  # slope > 0.015% of price/bar → trend vote (~0.8pts/bar on ES 5200)
REGIME_EMA_SLOPE_RANGE_PCT   = 0.01   # slope < 0.01% of price/bar → range vote
REGIME_SWING_LOOKBACK        = 4      # number of recent swings per type to analyse
REGIME_ATR_EXPAND_MULT       = 1.20   # ATR > 1.20 × median → expanding (trend vote)
REGIME_ATR_CONTRACT_MULT     = 0.85   # ATR < 0.85 × median → contracting (range vote)
REGIME_ATR_MEDIAN_PERIOD     = 20     # rolling median period for ATR expansion check
REGIME_VOTES_REQUIRED        = 2      # minimum directional votes out of 3 (excl ATR) to declare TREND
REGIME_COUNTER_TREND_PENALTY = 20     # confidence points deducted for counter-trend


class Regime(str, Enum):
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING       = "ranging"
    UNKNOWN       = "unknown"   # insufficient data


@dataclass
class RegimeResult:
    regime:        Regime
    adx:           Optional[float]         # final ADX value
    ema_slope_pct: Optional[float]         # EMA slope as % of price per bar
    swing_vote:    Optional[str]           # "bull", "bear", "mixed", or None
    atr_vote:      Optional[str]           # "expanding", "contracting", or "neutral"
    votes_bull:    int = 0                 # how many indicators voted BULL
    votes_bear:    int = 0                 # how many indicators voted BEAR
    votes_range:   int = 0                 # how many indicators voted RANGE
    notes:         list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "regime":        self.regime.value,
            "adx":           round(self.adx, 2) if self.adx else None,
            "ema_slope_pct": round(self.ema_slope_pct, 4) if self.ema_slope_pct else None,
            "swing_vote":    self.swing_vote,
            "atr_vote":      self.atr_vote,
            "votes_bull":    self.votes_bull,
            "votes_bear":    self.votes_bear,
            "votes_range":   self.votes_range,
            "notes":         self.notes,
        }


# ---------------------------------------------------------------------------
# INDICATOR 1 — ADX  (Average Directional Index, Wilder smoothing)
# ---------------------------------------------------------------------------

def calc_adx(candles: list[Candle],
             period: int = REGIME_ADX_PERIOD
             ) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Computes ADX, +DI, -DI using Wilder smoothing on the supplied candles.
    Returns (adx, plus_di, minus_di) for the LAST bar, or (None, None, None)
    if insufficient data.

    Algorithm:
      1. True Range (TR)         = max(H-L, |H-pC|, |L-pC|)
      2. +DM                     = max(H - pH, 0) if H-pH > pL-L else 0
         -DM                     = max(pL - L, 0) if pL-L > H-pH else 0
      3. Wilder smooth TR, +DM, -DM over `period` bars
      4. +DI = 100 × smooth_+DM / smooth_TR
         -DI = 100 × smooth_-DM / smooth_TR
      5. DX  = 100 × |+DI - -DI| / (+DI + -DI)
      6. ADX = Wilder smooth of DX over `period` bars
    """
    n = len(candles)
    min_bars = period * 2 + 1
    if n < min_bars:
        return None, None, None

    trs, plus_dms, minus_dms = [], [], []

    for i in range(1, n):
        c, p = candles[i], candles[i - 1]
        tr = max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
        up   = c.high - p.high
        down = p.low  - c.low

        pdm = up   if up > down and up > 0   else 0.0
        mdm = down if down > up and down > 0 else 0.0

        trs.append(tr)
        plus_dms.append(pdm)
        minus_dms.append(mdm)

    # Wilder seed (simple average of first `period` values)
    def wilder_smooth(values: list[float]) -> list[float]:
        if len(values) < period:
            return []
        smoothed = [None] * len(values)
        smoothed[period - 1] = sum(values[:period]) / period
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i - 1] * (period - 1) / period + values[i]
        return smoothed

    sm_tr  = wilder_smooth(trs)
    sm_pdm = wilder_smooth(plus_dms)
    sm_mdm = wilder_smooth(minus_dms)

    # Build DX series where all three smoothed values are available
    dx_series = []
    for i in range(period - 1, len(sm_tr)):
        if sm_tr[i] is None or sm_tr[i] == 0:
            dx_series.append(None)
            continue
        pdi = 100 * sm_pdm[i] / sm_tr[i]
        mdi = 100 * sm_mdm[i] / sm_tr[i]
        denom = pdi + mdi
        dx = 100 * abs(pdi - mdi) / denom if denom > 0 else 0.0
        dx_series.append((dx, pdi, mdi))

    # Filter valid entries
    valid_dx = [(dx, p, m) for x in dx_series if x is not None for dx, p, m in [x]]

    if len(valid_dx) < period:
        return None, None, None

    # Wilder smooth DX → ADX
    # DX is already normalised 0–100, so we use the full Wilder formula
    # that divides by N: ADX_new = (ADX_old * (N-1) + DX_new) / N
    # This ensures ADX converges to the DX value and stays bounded 0–100.
    adx_seed = sum(v[0] for v in valid_dx[:period]) / period
    adx = adx_seed
    for v in valid_dx[period:]:
        adx = (adx * (period - 1) + v[0]) / period   # ← divide by period

    last_pdi = valid_dx[-1][1]
    last_mdi = valid_dx[-1][2]

    return adx, last_pdi, last_mdi


def adx_vote(adx: Optional[float],
             plus_di: Optional[float],
             minus_di: Optional[float]
             ) -> tuple[str, str]:
    """
    Translates ADX/DI values into a directional vote.

    Returns (vote, note):
      vote  — "bull", "bear", "range", or "neutral"
      note  — human-readable explanation
    """
    if adx is None:
        return "neutral", "adx=insufficient_data"

    if adx > REGIME_ADX_TREND_THRESHOLD:
        if plus_di is not None and minus_di is not None:
            if plus_di > minus_di:
                return "bull", f"adx={adx:.1f}>25 +DI({plus_di:.1f})>-DI({minus_di:.1f})"
            else:
                return "bear", f"adx={adx:.1f}>25 -DI({minus_di:.1f})>+DI({plus_di:.1f})"
        return "neutral", f"adx={adx:.1f}>25 but DI unavailable"

    if adx < REGIME_ADX_RANGE_THRESHOLD:
        return "range", f"adx={adx:.1f}<20 (low_directional_strength)"

    return "neutral", f"adx={adx:.1f} in neutral zone 20-25"


# ---------------------------------------------------------------------------
# INDICATOR 2 — EMA SLOPE
# ---------------------------------------------------------------------------

def calc_ema(candles: list[Candle], period: int) -> list[Optional[float]]:
    """
    Exponential Moving Average.  Returns list aligned with `candles`.
    Values at indices < period-1 are None.
    """
    k = 2.0 / (period + 1)
    n = len(candles)
    emas: list[Optional[float]] = [None] * n
    if n < period:
        return emas

    emas[period - 1] = sum(c.close for c in candles[:period]) / period
    for i in range(period, n):
        emas[i] = candles[i].close * k + emas[i - 1] * (1 - k)
    return emas


def ema_slope_vote(candles: list[Candle],
                   period: int = REGIME_EMA_PERIOD
                   ) -> tuple[str, str, Optional[float]]:
    """
    Measures EMA slope as a percentage of current price per bar.
    Positive slope > threshold → bull vote.
    Negative slope < -threshold → bear vote.
    Flat (|slope| < range_threshold) → range vote.

    Returns (vote, note, slope_pct).
    """
    emas = calc_ema(candles, period)
    n = len(candles)

    # Need at least 3 recent EMA values to compute slope
    if n < period + 2:
        return "neutral", "ema=insufficient_data", None

    # Find last 3 valid EMA values
    valid = [(i, v) for i, v in enumerate(emas) if v is not None]
    if len(valid) < 3:
        return "neutral", "ema=insufficient_data", None

    # Slope = change over last 3 bars, normalised to current price
    i3, e3 = valid[-1]
    i1, e1 = valid[-3]
    bars_elapsed = max(i3 - i1, 1)
    current_price = candles[-1].close
    slope_pct = ((e3 - e1) / bars_elapsed) / current_price * 100

    if slope_pct > REGIME_EMA_SLOPE_TREND_PCT:
        return "bull", f"ema_slope={slope_pct:.4f}%/bar (rising)", slope_pct
    if slope_pct < -REGIME_EMA_SLOPE_TREND_PCT:
        return "bear", f"ema_slope={slope_pct:.4f}%/bar (falling)", slope_pct
    if abs(slope_pct) < REGIME_EMA_SLOPE_RANGE_PCT:
        return "range", f"ema_slope={slope_pct:.4f}%/bar (flat)", slope_pct
    return "neutral", f"ema_slope={slope_pct:.4f}%/bar (inconclusive)", slope_pct


# ---------------------------------------------------------------------------
# INDICATOR 3 — SWING STRUCTURE (HH/HL vs LH/LL)
# ---------------------------------------------------------------------------

def swing_structure_vote(swings: list[SwingLevel],
                         lookback: int = REGIME_SWING_LOOKBACK
                         ) -> tuple[str, str]:
    """
    Classifies swing structure using the last N swing highs and N swing lows.

    TRENDING BULL : sequence of Higher Highs AND Higher Lows
    TRENDING BEAR : sequence of Lower Highs  AND Lower Lows
    RANGING       : mixed or alternating structure (broken HH/LL sequence)

    Logic:
      - Extract last `lookback` confirmed swing highs (sorted oldest→newest)
      - Extract last `lookback` confirmed swing lows
      - Count how many consecutive pairs form HH (each high > previous high)
      - Count how many consecutive pairs form HL (each low > previous low)
      - If HH_score ≥ lookback-1 AND HL_score ≥ lookback-1 → BULL
      - If LH_score ≥ lookback-1 AND LL_score ≥ lookback-1 → BEAR
      - Otherwise → MIXED / RANGING
    """
    if not swings:
        return "neutral", "swing=insufficient_data"

    def dedup_consecutive(levels):
        """Remove consecutive swing levels at the same price (pivot artefacts)."""
        out = []
        for s in levels:
            if not out or abs(s.price - out[-1].price) > 0.01:
                out.append(s)
        return out

    highs_raw = sorted([s for s in swings if s.is_high],  key=lambda x: x.bar_index)
    lows_raw  = sorted([s for s in swings if not s.is_high], key=lambda x: x.bar_index)
    highs = dedup_consecutive(highs_raw)[-lookback:]
    lows  = dedup_consecutive(lows_raw)[-lookback:]

    if len(highs) < 2 or len(lows) < 2:
        return "neutral", "swing=insufficient_confirmed_swings"

    # Count consecutive Higher Highs
    hh = sum(1 for i in range(1, len(highs)) if highs[i].price > highs[i-1].price)
    lh = sum(1 for i in range(1, len(highs)) if highs[i].price < highs[i-1].price)
    hl = sum(1 for i in range(1, len(lows))  if lows[i].price  > lows[i-1].price)
    ll = sum(1 for i in range(1, len(lows))  if lows[i].price  < lows[i-1].price)

    pairs = len(highs) - 1   # number of consecutive pairs available
    threshold = max(1, pairs - 1)   # allow 1 failure in the sequence

    bull = hh >= threshold and hl >= threshold
    bear = lh >= threshold and ll >= threshold

    if bull and not bear:
        return "bull", f"swing=HH({hh}/{pairs})_HL({hl}/{pairs})"
    if bear and not bull:
        return "bear", f"swing=LH({lh}/{pairs})_LL({ll}/{pairs})"
    return "mixed", f"swing=mixed HH{hh}_LH{lh}_HL{hl}_LL{ll}"


# ---------------------------------------------------------------------------
# INDICATOR 4 — ATR EXPANSION RATIO
# ---------------------------------------------------------------------------

def atr_expansion_vote(atrs: list[Optional[float]],
                       current_idx: int,
                       median_period: int = REGIME_ATR_MEDIAN_PERIOD,
                       expand_mult:   float = REGIME_ATR_EXPAND_MULT,
                       contract_mult: float = REGIME_ATR_CONTRACT_MULT
                       ) -> tuple[str, str]:
    """
    Compares current ATR to rolling median:
      Expanding (> expand_mult × median)   → trending vote (directional undefined)
      Contracting (< contract_mult × median) → ranging vote
      Neutral otherwise.

    Note: ATR expansion alone doesn't tell direction — it merely confirms
    that volatility supports a trending move. Direction comes from other votes.
    """
    cur = atrs[current_idx]
    if cur is None:
        return "neutral", "atr_expansion=insufficient_data"

    history = [atrs[i] for i in range(max(0, current_idx - median_period), current_idx)
               if atrs[i] is not None]
    if len(history) < median_period // 2:
        return "neutral", "atr_expansion=insufficient_history"

    srt = sorted(history)
    mid = len(srt) // 2
    median = (srt[mid - 1] + srt[mid]) / 2 if len(srt) % 2 == 0 else srt[mid]

    ratio = cur / median if median > 0 else 1.0

    if ratio >= expand_mult:
        return "expanding", f"atr_ratio={ratio:.2f}>={expand_mult} (expanding)"
    if ratio <= contract_mult:
        return "contracting", f"atr_ratio={ratio:.2f}<={contract_mult} (contracting)"
    return "neutral", f"atr_ratio={ratio:.2f} (neutral)"


# ---------------------------------------------------------------------------
# REGIME AGGREGATOR
# ---------------------------------------------------------------------------

def detect_regime(candles_15m: list[Candle],
                  swings_15m:  list[SwingLevel],
                  candles_5m:  list[Candle]) -> RegimeResult:
    """
    Runs all four sub-indicators and aggregates their votes into a single
    Regime classification.

    Voting rules:
      - ADX/DI       → votes "bull", "bear", "range", or "neutral"
      - EMA slope    → votes "bull", "bear", "range", or "neutral"
      - Swing struct → votes "bull", "bear", "mixed" (= range), or "neutral"
      - ATR expansion→ votes "expanding" (amplifies trend votes by 1)
                       or "contracting" (amplifies range vote by 1)
                       or "neutral"

    Aggregation:
      bull_votes  = count of "bull" votes from ADX + EMA + swing
      bear_votes  = count of "bear" votes
      range_votes = count of "range"/"mixed" votes

      ATR modifier:
        expanding   → +1 to whichever of bull/bear has more votes (amplifies trend)
        contracting → +1 to range_votes (amplifies ranging signal)

      Final:
        IF bull_votes  >= REGIME_VOTES_REQUIRED → TRENDING_BULL
        IF bear_votes  >= REGIME_VOTES_REQUIRED → TRENDING_BEAR
        IF range_votes >= REGIME_VOTES_REQUIRED → RANGING
        ELSE                                    → RANGING  (conservative default)
    """
    notes = []

    # ── Sub-indicator 1: ADX ─────────────────────────────────────────────────
    adx_val, plus_di, minus_di = calc_adx(candles_15m, REGIME_ADX_PERIOD)
    adx_v, adx_note = adx_vote(adx_val, plus_di, minus_di)
    notes.append(f"[ADX] {adx_note}")

    # ── Sub-indicator 2: EMA slope ───────────────────────────────────────────
    ema_v, ema_note, slope_pct = ema_slope_vote(candles_15m, REGIME_EMA_PERIOD)
    notes.append(f"[EMA] {ema_note}")

    # ── Sub-indicator 3: Swing structure ─────────────────────────────────────
    sw_v, sw_note = swing_structure_vote(swings_15m, REGIME_SWING_LOOKBACK)
    notes.append(f"[SWING] {sw_note}")

    # ── Sub-indicator 4: ATR expansion (on 5m for intraday sensitivity) ──────
    atrs_5m  = calc_atr(candles_5m, ATR_PERIOD)
    atr_idx  = len(candles_5m) - 1
    atr_v, atr_note = atr_expansion_vote(atrs_5m, atr_idx)
    notes.append(f"[ATR_EXP] {atr_note}")

    # ── Tally votes ──────────────────────────────────────────────────────────
    bull_votes  = sum(1 for v in [adx_v, ema_v, sw_v] if v == "bull")
    bear_votes  = sum(1 for v in [adx_v, ema_v, sw_v] if v == "bear")
    range_votes = sum(1 for v in [adx_v, ema_v, sw_v] if v in ("range", "mixed"))

    # ATR modifier
    if atr_v == "expanding":
        # Amplify whichever directional side is winning; if tied → neutral
        if bull_votes > bear_votes:
            bull_votes += 1
            notes.append("[ATR_EXP] +1 bull (expansion confirms directional move)")
        elif bear_votes > bull_votes:
            bear_votes += 1
            notes.append("[ATR_EXP] +1 bear (expansion confirms directional move)")
        # If tied, expansion doesn't break the tie
    elif atr_v == "contracting":
        range_votes += 1
        notes.append("[ATR_EXP] +1 range (contraction confirms compression)")

    # ── Final classification ─────────────────────────────────────────────────
    if bull_votes >= REGIME_VOTES_REQUIRED:
        regime = Regime.TRENDING_BULL
    elif bear_votes >= REGIME_VOTES_REQUIRED:
        regime = Regime.TRENDING_BEAR
    elif range_votes >= REGIME_VOTES_REQUIRED:
        regime = Regime.RANGING
    else:
        # Ambiguous — default to RANGING (conservative: no trade over bad trade)
        regime = Regime.RANGING
        notes.append("regime=ambiguous_defaulting_to_ranging")

    return RegimeResult(
        regime=regime,
        adx=adx_val,
        ema_slope_pct=slope_pct,
        swing_vote=sw_v,
        atr_vote=atr_v,
        votes_bull=bull_votes,
        votes_bear=bear_votes,
        votes_range=range_votes,
        notes=notes,
    )
