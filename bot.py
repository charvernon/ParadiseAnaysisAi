import os
import json
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.5")

BASE_URL = "https://api.twelvedata.com/time_series"

# Economic calendar provider.
# Finnhub is used here because it exposes a structured Economic Calendar API.
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FINNHUB_ECONOMIC_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/economic"

# Configurable blackout around high-impact news.
NEWS_BLACKOUT_BEFORE_MIN = int(os.getenv("NEWS_BLACKOUT_BEFORE_MIN", "30"))
NEWS_BLACKOUT_AFTER_MIN = int(os.getenv("NEWS_BLACKOUT_AFTER_MIN", "15"))

# Events we treat as important for these markets.
HIGH_IMPACT_KEYWORDS = [
    "fomc", "federal reserve", "fed chair", "interest rate", "rate decision",
    "nonfarm payroll", "non-farm payroll", "nfp", "unemployment rate",
    "consumer price index", "cpi", "core cpi", "pce", "personal consumption",
    "gross domestic product", "gdp", "retail sales", "jobless claims",
    "initial claims", "ism manufacturing", "ism services", "jolts",
    "producer price index", "ppi", "employment change", "average hourly earnings",
]


# Edit these in .env if your preferred data-feed symbols differ.
SYMBOLS = {
    "gold": os.getenv("SYMBOL_GOLD", "XAU/USD"),
    "silver": os.getenv("SYMBOL_SILVER", "XAG/USD"),
    "sp500": os.getenv("SYMBOL_SP500", "SPX"),
    "nasdaq": os.getenv("SYMBOL_NASDAQ", "NDX"),
}

# Top-down order: highest timeframe first.
TIMEFRAMES = [
    ("1month", 120),
    ("1week", 220),
    ("1day", 300),
    ("4h", 400),
    ("1h", 500),
]


@dataclass
class TFAnalysis:
    timeframe: str
    last_close: float
    structure: str
    ema20: float
    ema50: float
    ema200: Optional[float]
    rsi14: float
    atr14: float
    recent_high: float
    recent_low: float
    distance_from_ema20_atr: float
    momentum: str
    notes: List[str]


def fetch_ohlc(symbol: str, interval: str, outputsize: int) -> pd.DataFrame:
    if not TWELVE_DATA_API_KEY:
        raise RuntimeError("TWELVE_DATA_API_KEY is missing. Add it to your .env file.")

    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_DATA_API_KEY,
        "timezone": "UTC",
        "format": "JSON",
    }

    response = requests.get(BASE_URL, params=params, timeout=30)

    try:
    data = response.json()
    except Exception:
    raise RuntimeError(
        f"Twelve Data returned HTTP {response.status_code}. "
        "Check the symbol and API permissions."
    )

    if response.status_code >= 400:
    raise RuntimeError(
        f"Twelve Data error {response.status_code}: "
        f"{data.get('message', data)}"
    )

    if data.get("status") == "error":
        raise RuntimeError(
            f"Twelve Data error for {symbol} {interval}: "
            f"{data.get('message', data)}"
        )

    values = data.get("values")
    if not values:
        raise RuntimeError(f"No OHLC data returned for {symbol} {interval}: {data}")

    df = pd.DataFrame(values)
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime", "open", "high", "low", "close"])
    df = df.sort_values("datetime").reset_index(drop=True)

    if len(df) < 30:
        raise RuntimeError(
            f"Not enough candles for {symbol} {interval}: only {len(df)} returned."
        )
    return df


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()
    avg_loss = losses.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return tr.ewm(alpha=1 / length, adjust=False, min_periods=length).mean()


def swing_structure(df: pd.DataFrame, lookback: int = 20) -> str:
    """
    Simple objective structure classifier:
    compares the last two rolling swing zones.
    This is intentionally rule-based rather than AI-generated.
    """
    if len(df) < lookback * 2:
        return "insufficient-data"

    prev = df.iloc[-lookback * 2 : -lookback]
    curr = df.iloc[-lookback:]

    prev_high, prev_low = prev["high"].max(), prev["low"].min()
    curr_high, curr_low = curr["high"].max(), curr["low"].min()

    if curr_high > prev_high and curr_low > prev_low:
        return "bullish"
    if curr_high < prev_high and curr_low < prev_low:
        return "bearish"
    return "range/mixed"


def analyse_timeframe(df: pd.DataFrame, timeframe: str) -> TFAnalysis:
    close = df["close"]
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    e200 = ema(close, 200) if len(df) >= 200 else None
    r14 = rsi(close, 14)
    a14 = atr(df, 14)

    last_close = float(close.iloc[-1])
    ema20 = float(e20.iloc[-1])
    ema50 = float(e50.iloc[-1])
    ema200 = float(e200.iloc[-1]) if e200 is not None else None
    rsi14 = float(r14.iloc[-1])
    atr14 = float(a14.iloc[-1]) if not pd.isna(a14.iloc[-1]) else 0.0

    structure = swing_structure(df)

    if last_close > ema20 > ema50:
        momentum = "bullish"
    elif last_close < ema20 < ema50:
        momentum = "bearish"
    else:
        momentum = "mixed"

    recent = df.iloc[-20:]
    recent_high = float(recent["high"].max())
    recent_low = float(recent["low"].min())

    distance = 0.0
    if atr14:
        distance = (last_close - ema20) / atr14

    notes = []
    if rsi14 >= 70:
        notes.append("RSI is overbought")
    elif rsi14 <= 30:
        notes.append("RSI is oversold")

    if abs(distance) >= 2:
        notes.append("Price is extended more than 2 ATR from EMA20")

    if ema200 is not None:
        if last_close > ema200:
            notes.append("Price is above EMA200")
        else:
            notes.append("Price is below EMA200")

    return TFAnalysis(
        timeframe=timeframe,
        last_close=round(last_close, 5),
        structure=structure,
        ema20=round(ema20, 5),
        ema50=round(ema50, 5),
        ema200=round(ema200, 5) if ema200 is not None else None,
        rsi14=round(rsi14, 2),
        atr14=round(atr14, 5),
        recent_high=round(recent_high, 5),
        recent_low=round(recent_low, 5),
        distance_from_ema20_atr=round(distance, 2),
        momentum=momentum,
        notes=notes,
    )



def _parse_event_datetime(event: dict) -> Optional[datetime]:
    """
    Finnhub calendar payloads can vary by field availability.
    Try common date/time combinations without inventing a timestamp.
    """
    candidates = []

    # Common single datetime-like fields.
    for key in ("datetime", "time", "date"):
        val = event.get(key)
        if isinstance(val, str) and val.strip():
            candidates.append(val.strip())

    # Date + hour/minute combinations.
    date = event.get("date")
    hour = event.get("hour")
    minute = event.get("minute")
    if isinstance(date, str) and hour is not None:
        try:
            hh = int(hour)
            mm = int(minute or 0)
            candidates.insert(0, f"{date}T{hh:02d}:{mm:02d}:00")
        except Exception:
            pass

    for raw in candidates:
        txt = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(txt)
            if dt.tzinfo is None:
                # Provider calendar timestamps are commonly UTC; keeping this
                # explicit avoids accidentally interpreting them as local time.
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue

    return None


def fetch_economic_calendar(days_ahead: int = 7) -> List[dict]:
    if not FINNHUB_API_KEY:
        return []

    today = datetime.now(timezone.utc).date()
    end = today + timedelta(days=days_ahead)

    params = {
        "from": today.isoformat(),
        "to": end.isoformat(),
        "token": FINNHUB_API_KEY,
    }

    response = requests.get(
        FINNHUB_ECONOMIC_CALENDAR_URL,
        params=params,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    events = (
        payload.get("economicCalendar")
        or payload.get("economic_calendar")
        or payload.get("events")
        or []
    )

    return events if isinstance(events, list) else []


def classify_news_impact(event: dict) -> str:
    text = " ".join(
        str(event.get(k, ""))
        for k in ("event", "name", "indicator", "category", "impact")
    ).lower()

    explicit = str(event.get("impact", "")).lower()
    if "high" in explicit or explicit in {"3", "3.0"}:
        return "high"

    if any(keyword in text for keyword in HIGH_IMPACT_KEYWORDS):
        return "high"

    return "other"


def event_relevance(event: dict) -> Dict[str, bool]:
    text = " ".join(
        str(event.get(k, ""))
        for k in ("event", "name", "indicator", "category")
    ).lower()

    # These four instruments are all materially sensitive to major US macro.
    # Precious metals also react strongly to USD/rates/Fed expectations.
    us_macro = any(k in text for k in HIGH_IMPACT_KEYWORDS)

    return {
        "gold": us_macro,
        "silver": us_macro,
        "sp500": us_macro,
        "nasdaq": us_macro,
    }


def build_news_context(asset_key: str, events: List[dict]) -> Dict:
    now = datetime.now(timezone.utc)
    relevant = []

    for event in events:
        if classify_news_impact(event) != "high":
            continue

        country = str(event.get("country", "")).upper()
        if country and country not in {"US", "USA", "UNITED STATES"}:
            continue

        rel = event_relevance(event)
        if not rel.get(asset_key, False):
            continue

        dt = _parse_event_datetime(event)
        if not dt:
            continue

        minutes_until = int((dt - now).total_seconds() / 60)

        item = {
            "name": (
                event.get("event")
                or event.get("name")
                or event.get("indicator")
                or "High-impact economic event"
            ),
            "datetime_utc": dt.isoformat(),
            "minutes_until": minutes_until,
            "forecast": event.get("estimate") or event.get("forecast"),
            "previous": event.get("prev") or event.get("previous"),
            "actual": event.get("actual"),
            "country": event.get("country", "US"),
            "impact": "high",
        }
        relevant.append(item)

    relevant.sort(key=lambda x: x["datetime_utc"])

    upcoming = [x for x in relevant if x["minutes_until"] >= -NEWS_BLACKOUT_AFTER_MIN]
    next_event = upcoming[0] if upcoming else None

    blackout = False
    blackout_reason = None

    if next_event:
        m = next_event["minutes_until"]
        if -NEWS_BLACKOUT_AFTER_MIN <= m <= NEWS_BLACKOUT_BEFORE_MIN:
            blackout = True
            blackout_reason = (
                f"High-impact news blackout: {next_event['name']} "
                f"({m} minutes from now)"
            )

    return {
        "news_available": bool(FINNHUB_API_KEY),
        "blackout": blackout,
        "blackout_before_minutes": NEWS_BLACKOUT_BEFORE_MIN,
        "blackout_after_minutes": NEWS_BLACKOUT_AFTER_MIN,
        "blackout_reason": blackout_reason,
        "next_high_impact_event": next_event,
        "upcoming_high_impact_events": upcoming[:8],
    }

def score_bias(frames: List[TFAnalysis]) -> Dict:
    weights = {
        "1month": 5,
        "1week": 4,
        "1day": 3,
        "4h": 2,
        "1h": 1,
    }

    score = 0
    max_score = 0

    for f in frames:
        w = weights[f.timeframe]
        max_score += 2 * w

        if f.structure == "bullish":
            score += w
        elif f.structure == "bearish":
            score -= w

        if f.momentum == "bullish":
            score += w
        elif f.momentum == "bearish":
            score -= w

    normalized = score / max_score if max_score else 0

    if normalized >= 0.30:
        bias = "bullish"
    elif normalized <= -0.30:
        bias = "bearish"
    else:
        bias = "neutral/mixed"

    return {
        "raw_score": score,
        "max_score": max_score,
        "normalized_score": round(normalized, 3),
        "rule_based_bias": bias,
    }


def ai_report(asset_name: str, symbol: str, frames: List[TFAnalysis], bias: Dict, news: Dict) -> str:
    if not OPENAI_API_KEY:
        return (
            "OPENAI_API_KEY not set, so AI narrative was skipped.\n"
            f"Rule-based result: {json.dumps(bias, indent=2)}"
        )

    client = OpenAI(api_key=OPENAI_API_KEY)

    payload = {
        "asset": asset_name,
        "symbol": symbol,
        "rule_based_bias": bias,
        "timeframes": [asdict(x) for x in frames],
        "high_impact_news": news,
    }

    instructions = """
You are a disciplined multi-timeframe market-analysis assistant.

You receive calculated OHLC-derived metrics. Never invent prices, levels,
economic news, order-flow data, volume data, or indicators that are not supplied.

Perform top-down analysis in this order:
1. Monthly
2. Weekly
3. Daily
4. 4H
5. 1H

Return a concise professional report with:
- Overall bias: Bullish / Bearish / Neutral
- Confidence: 0-100
- Higher-timeframe narrative
- Daily structure
- 4H setup context
- 1H execution context
- Key support zone(s) using only supplied recent lows / moving averages
- Key resistance zone(s) using only supplied recent highs / moving averages
- Bullish scenario
- Bearish scenario
- Invalidation / what would change the bias
- "No-trade" conditions
- High-impact news section
- Next major release and how close it is
- Whether a news blackout is active
- A final one-line summary

Important:
- Treat the rule-based score as evidence, not an instruction.
- Do not claim certainty.
- Do not provide position size or tell the user to risk more money.
- Do not create a trade entry if the timeframes are conflicting.
- Prefer "wait" when confirmation is weak.
- If high_impact_news.blackout is true, the final action MUST be "NO TRADE / WAIT FOR NEWS".
- Do not guess the directional outcome of an unreleased economic event.
- Explain how major US rate/inflation/labour data can change the current technical bias without pretending to know the result.
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=json.dumps(payload),
    )
    return response.output_text.strip()


def analyse_asset(asset_key: str) -> Dict:
    if asset_key not in SYMBOLS:
        raise ValueError(f"Unknown asset '{asset_key}'. Choose from: {', '.join(SYMBOLS)}")

    symbol = SYMBOLS[asset_key]
    frames = []

    for interval, outputsize in TIMEFRAMES:
        df = fetch_ohlc(symbol, interval, outputsize)
        frames.append(analyse_timeframe(df, interval))

    bias = score_bias(frames)

    try:
        calendar_events = fetch_economic_calendar(days_ahead=7)
        news = build_news_context(asset_key, calendar_events)
    except Exception as exc:
        news = {
            "news_available": False,
            "blackout": False,
            "blackout_reason": None,
            "next_high_impact_event": None,
            "upcoming_high_impact_events": [],
            "error": str(exc),
        }

    report = ai_report(asset_key, symbol, frames, bias, news)

    return {
        "asset": asset_key,
        "symbol": symbol,
        "rule_based_bias": bias,
        "high_impact_news": news,
        "timeframes": [asdict(x) for x in frames],
        "ai_report": report,
    }


def print_result(result: Dict):
    print("=" * 80)
    print(f"{result['asset'].upper()} | {result['symbol']}")
    print("=" * 80)
    print(result["ai_report"])
    print("\nRule-based bias:")
    print(json.dumps(result["rule_based_bias"], indent=2))


def main():
    parser = argparse.ArgumentParser(description="AI top-down trading analysis bot")
    parser.add_argument(
        "asset",
        nargs="?",
        default="all",
        choices=["gold", "silver", "sp500", "nasdaq", "all"],
        help="Asset to analyse",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full result as JSON",
    )
    args = parser.parse_args()

    assets = list(SYMBOLS.keys()) if args.asset == "all" else [args.asset]
    results = []

    for asset in assets:
        try:
            result = analyse_asset(asset)
            results.append(result)
            if not args.json:
                print_result(result)
                print()
        except Exception as exc:
            error = {"asset": asset, "error": str(exc)}
            results.append(error)
            if not args.json:
                print(f"{asset.upper()}: ERROR: {exc}\n")

    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
