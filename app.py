import os
import time
from datetime import datetime, timezone

import streamlit as st

from bot import analyse_asset, SYMBOLS

st.set_page_config(
    page_title="Top-Down Trading AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    .market-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .bias-bullish {
        border-left: 6px solid #22c55e;
    }
    .bias-bearish {
        border-left: 6px solid #ef4444;
    }
    .bias-neutral {
        border-left: 6px solid #f59e0b;
    }
    .news-alert {
        border: 1px solid #ef4444;
        border-radius: 14px;
        padding: 14px;
        background: rgba(239,68,68,0.08);
    }
    .news-ok {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 14px;
    }
    .small-muted {
        opacity: 0.7;
        font-size: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

MARKETS = {
    "Gold": "gold",
    "Silver": "silver",
    "S&P 500": "sp500",
    "Nasdaq-100": "nasdaq",
}

def inject_streamlit_secrets_into_env():
    """
    Allows the same bot.py to work locally with .env or on Streamlit Cloud
    using Settings > Secrets.
    """
    secret_keys = [
        "TWELVE_DATA_API_KEY",
        "OPENAI_API_KEY",
        "FINNHUB_API_KEY",
        "OPENAI_MODEL",
        "NEWS_BLACKOUT_BEFORE_MIN",
        "NEWS_BLACKOUT_AFTER_MIN",
        "SYMBOL_GOLD",
        "SYMBOL_SILVER",
        "SYMBOL_SP500",
        "SYMBOL_NASDAQ",
    ]
    for key in secret_keys:
        try:
            value = st.secrets.get(key)
        except Exception:
            value = None
        if value is not None and str(value).strip():
            os.environ[key] = str(value)

inject_streamlit_secrets_into_env()

def utc_to_display(iso_string):
    if not iso_string:
        return "Unknown"
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%d %b %Y %H:%M UTC")
    except Exception:
        return str(iso_string)

def bias_class(text):
    txt = (text or "").lower()
    if "bull" in txt:
        return "bias-bullish"
    if "bear" in txt:
        return "bias-bearish"
    return "bias-neutral"

def render_market(result):
    asset = result.get("asset", "")
    symbol = result.get("symbol", "")
    bias = result.get("rule_based_bias", {})
    rule_bias = bias.get("rule_based_bias", "unknown")
    score = bias.get("normalized_score", 0)
    news = result.get("high_impact_news", {})

    st.markdown(
        f"""
        <div class="market-card {bias_class(rule_bias)}">
            <h3 style="margin:0">{asset.upper()} — {symbol}</h3>
            <p style="margin:6px 0 0 0">
                <b>Technical bias:</b> {rule_bias.upper()}
                &nbsp; | &nbsp;
                <b>Score:</b> {score}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    next_event = news.get("next_high_impact_event")
    if news.get("blackout"):
        st.markdown(
            f"""
            <div class="news-alert">
                <b>🚨 HIGH-IMPACT NEWS BLACKOUT</b><br>
                {news.get("blackout_reason") or "Major news is close. Wait for the event to pass."}
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif next_event:
        st.markdown(
            f"""
            <div class="news-ok">
                <b>📰 Next high-impact event</b><br>
                {next_event.get("name", "Event")}<br>
                <span class="small-muted">
                    {utc_to_display(next_event.get("datetime_utc"))}
                    · {next_event.get("minutes_until")} minutes away
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("No upcoming high-impact US event was returned by the news feed.")

    st.subheader("AI top-down analysis")
    st.markdown(result.get("ai_report", "No AI report returned."))

    frames = result.get("timeframes", [])
    if frames:
        st.subheader("Timeframe breakdown")
        for frame in frames:
            title = frame.get("timeframe", "").upper()
            with st.expander(
                f"{title} · {frame.get('structure', 'n/a')} structure · "
                f"{frame.get('momentum', 'n/a')} momentum"
            ):
                c1, c2, c3 = st.columns(3)
                c1.metric("Last close", frame.get("last_close"))
                c2.metric("RSI 14", frame.get("rsi14"))
                c3.metric("ATR 14", frame.get("atr14"))

                c1, c2, c3 = st.columns(3)
                c1.metric("EMA 20", frame.get("ema20"))
                c2.metric("EMA 50", frame.get("ema50"))
                c3.metric("EMA 200", frame.get("ema200"))

                c1, c2 = st.columns(2)
                c1.metric("Recent 20-bar high", frame.get("recent_high"))
                c2.metric("Recent 20-bar low", frame.get("recent_low"))

                notes = frame.get("notes") or []
                if notes:
                    st.write("Notes:", " • ".join(notes))

    upcoming = news.get("upcoming_high_impact_events") or []
    if upcoming:
        st.subheader("Upcoming high-impact news")
        for e in upcoming[:6]:
            forecast = e.get("forecast")
            previous = e.get("previous")
            actual = e.get("actual")
            detail = []
            if forecast is not None:
                detail.append(f"Forecast: {forecast}")
            if previous is not None:
                detail.append(f"Previous: {previous}")
            if actual is not None:
                detail.append(f"Actual: {actual}")
            suffix = " | ".join(detail)
            st.write(
                f"**{e.get('name', 'Event')}** — "
                f"{utc_to_display(e.get('datetime_utc'))}"
                + (f" — {suffix}" if suffix else "")
            )

st.title("📈 Top-Down Trading AI")
st.caption(
    "Gold · Silver · S&P 500 · Nasdaq-100 | "
    "Monthly → Weekly → Daily → 4H → 1H | High-impact US news filter"
)

with st.sidebar:
    st.header("Settings")
    st.caption("API keys should be stored in Streamlit Secrets when deployed.")
    st.write("Configured markets:")
    for name, key in MARKETS.items():
        st.write(f"• {name}: `{SYMBOLS.get(key)}`")
    st.write("")
    st.caption(
        "This app provides market analysis only. It does not place trades."
    )

col1, col2 = st.columns([2, 1])
with col1:
    selected_name = st.selectbox(
        "Market",
        list(MARKETS.keys()),
        index=3,
    )
with col2:
    refresh = st.button("🔄 Refresh analysis", use_container_width=True)

asset_key = MARKETS[selected_name]

if refresh:
    st.cache_data.clear()

@st.cache_data(ttl=900, show_spinner=False)
def cached_analysis(asset):
    return analyse_asset(asset)

with st.spinner(f"Analysing {selected_name}..."):
    result = cached_analysis(asset_key)

if result.get("error"):
    st.error(result["error"])
else:
    render_market(result)

st.divider()
st.caption(
    "Analysis is cached for 15 minutes to reduce unnecessary API usage. "
    "Use Refresh analysis to force a fresh run."
)
