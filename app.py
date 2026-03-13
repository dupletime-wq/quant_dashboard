from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import re
from typing import Any

import FinanceDataReader as fdr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL


PLOT_FONT = {
    "family": '"Segoe UI", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif',
    "size": 14,
    "color": "#102a43",
}
GRID_COLOR = "rgba(71, 85, 105, 0.18)"
KOREAN_SUFFIX_PATTERN = re.compile(r"^(?P<code>\d{6})\.(KS|KQ)$", re.IGNORECASE)
CORE_DASHBOARD_VIEWS = [
    "Elder Impulse",
    "TD Sequential",
    "Robust STL",
    "SMC",
    "SuperTrend",
    "Williams Vix Fix",
    "Squeeze Momentum",
    "Nadaraya-Watson",
    "Lorentzian Classification",
    "CVD Divergence",
]
SPECIAL_ACTION_VIEWS = ["Market Pulse", "Options Flow"]
CACHE_DIR = Path(__file__).resolve().parent / ".cache"


@dataclass
class DashboardSummary:
    ticker: str
    last_close: float
    daily_change: float
    monthly_change: float
    quarter_change: float
    elder_label: str
    td_label: str
    stl_label: str
    market_label: str
    options_label: str


def configure_page() -> None:
    st.set_page_config(
        page_title="Quant Fusion Dashboard",
        page_icon="Q",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --page-ink: #102a43;
            --muted-ink: #52606d;
            --panel-border: rgba(148, 163, 184, 0.22);
            --bull: #0f766e;
            --bear: #b42318;
            --accent: #dd6b20;
            --neutral: #64748b;
        }
        .stApp {
            color: var(--page-ink);
            background:
                radial-gradient(circle at top left, rgba(255, 239, 213, 0.96), transparent 30%),
                radial-gradient(circle at top right, rgba(186, 230, 253, 0.55), transparent 24%),
                linear-gradient(180deg, #fffdf8 0%, #f6efe2 46%, #fffaf2 100%);
        }
        .block-container {
            max-width: 1520px;
            padding-top: 1.2rem;
            padding-bottom: 2.4rem;
        }
        .hero {
            background:
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.18), transparent 20%),
                linear-gradient(130deg, #0f172a 0%, #134e4a 42%, #f59e0b 100%);
            border-radius: 30px;
            padding: 2rem 2.2rem;
            color: #fffdf8;
            margin-bottom: 1rem;
            box-shadow: 0 28px 78px rgba(15, 23, 42, 0.18);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.3rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0.65rem 0 0;
            color: rgba(255, 250, 242, 0.94);
            max-width: 64rem;
            line-height: 1.6;
            font-size: 1rem;
        }
        .metric-card {
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(255,255,255,0.82));
            border: 1px solid var(--panel-border);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            min-height: 8rem;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(12px);
        }
        .metric-card.bull {
            border-top: 4px solid var(--bull);
        }
        .metric-card.bear {
            border-top: 4px solid var(--bear);
        }
        .metric-card.neutral {
            border-top: 4px solid var(--neutral);
        }
        .metric-card.accent {
            border-top: 4px solid var(--accent);
        }
        .metric-label {
            color: var(--muted-ink);
            font-size: 0.82rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: var(--page-ink);
            font-size: 1.72rem;
            font-weight: 800;
            line-height: 1.12;
            margin-top: 0.5rem;
        }
        .metric-subtitle {
            color: var(--muted-ink);
            font-size: 0.93rem;
            margin-top: 0.42rem;
            line-height: 1.48;
        }
        .section-note {
            color: var(--muted-ink);
            font-size: 0.98rem;
            line-height: 1.55;
            margin-top: 0.15rem;
            margin-bottom: 0.8rem;
        }
        div[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(255,255,255,0.55), transparent 24%),
                linear-gradient(180deg, rgba(255,250,240,0.94), rgba(255,245,233,0.92));
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.66);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 999px;
            padding: 0.55rem 1rem;
            color: var(--page-ink);
            font-weight: 700;
        }
        div[data-testid="stTabs"] [aria-selected="true"] {
            background: linear-gradient(120deg, #102a43 0%, #0f766e 100%);
            color: #fffdf8;
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.14);
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 20px;
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.05);
        }
        div[data-testid="stPlotlyChart"] {
            background: rgba(255,255,255,0.54);
            border-radius: 24px;
            padding: 0.25rem;
            border: 1px solid rgba(148, 163, 184, 0.14);
        }
        .signal-panel {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 24px;
            padding: 1rem 1rem 0.6rem;
            box-shadow: 0 18px 34px rgba(15, 23, 42, 0.06);
        }
        .data-pill {
            display: inline-block;
            margin: 0.15rem 0.4rem 0.15rem 0;
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            background: rgba(15, 23, 42, 0.06);
            color: var(--page-ink);
            font-size: 0.9rem;
            border: 1px solid rgba(148, 163, 184, 0.16);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(title: str, value: str, subtitle: str, tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def normalize_datetime_index(index: Any) -> pd.DatetimeIndex:
    dt_index = pd.to_datetime(index)
    if isinstance(dt_index, pd.DatetimeIndex) and dt_index.tz is not None:
        dt_index = dt_index.tz_localize(None)
    return dt_index


def normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    frame = flatten_columns(df.copy())
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame.index = normalize_datetime_index(frame.index)
    required = [col for col in ["Open", "High", "Low", "Close"] if col in frame.columns]
    if required:
        frame = frame.dropna(subset=required)
    else:
        frame = frame.dropna()
    return frame.sort_index().copy()


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _daily_cache_path(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{name}.pkl"


def load_daily_cached_payload(name: str) -> tuple[str | None, Any]:
    cache_path = _daily_cache_path(name)
    if not cache_path.exists():
        return None, None
    try:
        with cache_path.open("rb") as cache_file:
            payload = pickle.load(cache_file)
        return payload.get("cache_date"), payload.get("data")
    except Exception:
        return None, None


def save_daily_cached_payload(name: str, data: Any) -> None:
    cache_path = _daily_cache_path(name)
    with cache_path.open("wb") as cache_file:
        pickle.dump({"cache_date": datetime.now().strftime("%Y-%m-%d"), "data": data}, cache_file)


def get_or_refresh_daily_payload(name: str, fetcher: Any) -> Any:
    today = datetime.now().strftime("%Y-%m-%d")
    cache_date, cached_data = load_daily_cached_payload(name)
    if cache_date == today and cached_data is not None:
        return cached_data

    fresh_data = fetcher()
    if fresh_data is not None:
        save_daily_cached_payload(name, fresh_data)
        return fresh_data
    return cached_data


def clear_daily_payload_cache(*names: str) -> None:
    target_names = names or ("market_pulse", "spx_options_payload")
    for name in target_names:
        cache_path = _daily_cache_path(name)
        if cache_path.exists():
            cache_path.unlink()


def get_yfinance_candidates(ticker: str) -> list[str]:
    normalized = ticker.strip().upper()
    match = KOREAN_SUFFIX_PATTERN.match(normalized)
    if match:
        code = match.group("code")
        return dedupe_preserve_order([normalized, f"{code}.KS", f"{code}.KQ", code])
    if re.fullmatch(r"\d{6}", normalized):
        return [f"{normalized}.KS", f"{normalized}.KQ", normalized]
    return [normalized]


def get_fdr_candidates(ticker: str) -> list[str]:
    normalized = ticker.strip().upper()
    match = KOREAN_SUFFIX_PATTERN.match(normalized)
    if match:
        return [match.group("code"), normalized]
    if re.fullmatch(r"\d{6}", normalized):
        return [normalized]
    return [normalized]


@st.cache_data(ttl=3600, show_spinner=False)
def download_price_data(ticker: str, period: str = "3y") -> tuple[pd.DataFrame, str, str]:
    for candidate in get_yfinance_candidates(ticker):
        try:
            df = yf.download(
                candidate,
                period=period,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception:
            continue

        frame = normalize_ohlcv_frame(df)
        if not frame.empty:
            return frame, "Yahoo Finance", candidate

    for candidate in get_fdr_candidates(ticker):
        try:
            df = fdr.DataReader(candidate, "2020-01-01")
        except Exception:
            continue

        frame = normalize_ohlcv_frame(df)
        if not frame.empty:
            return frame, "FinanceDataReader", candidate

    return pd.DataFrame(), "Yahoo Finance", ticker.strip().upper()


@st.cache_data(ttl=3600, show_spinner=False)
def download_stl_data(ticker: str) -> tuple[pd.DataFrame, str, str]:
    for candidate in get_fdr_candidates(ticker):
        try:
            df = fdr.DataReader(candidate, "2020-01-01")
        except Exception:
            continue

        frame = normalize_ohlcv_frame(df)
        if not frame.empty and "Close" in frame.columns:
            return frame, "FinanceDataReader", candidate

    for candidate in get_yfinance_candidates(ticker):
        try:
            df = yf.download(
                candidate,
                period="6y",
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
        except Exception:
            continue

        frame = normalize_ohlcv_frame(df)
        if not frame.empty and "Close" in frame.columns:
            return frame, "Yahoo Finance fallback", candidate

    return pd.DataFrame(), "Unavailable", ticker.strip().upper()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = df["Close"].shift(1)
    true_range = np.maximum(
        df["High"] - df["Low"],
        np.maximum((df["High"] - prev_close).abs(), (df["Low"] - prev_close).abs()),
    )
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def pct_change_from_index(series: pd.Series, periods: int) -> float:
    if len(series) <= periods:
        return np.nan
    base = float(series.iloc[-periods - 1])
    if base == 0:
        return np.nan
    return float(series.iloc[-1] / base - 1)


def classify_market_score(score: float) -> tuple[str, str]:
    if score >= 0.8:
        return "Extreme Greed", "bear"
    if score >= 0.6:
        return "Greed", "accent"
    if score <= 0.2:
        return "Extreme Fear", "bull"
    if score <= 0.4:
        return "Fear", "bull"
    return "Neutral", "neutral"


def apply_figure_style(
    fig: go.Figure,
    *,
    title: str,
    height: int,
    showlegend: bool = True,
    legend_y: float = 1.06,
    xaxis_rangeslider_visible: bool = False,
) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0.02, y=0.97, xanchor="left", yanchor="top"),
        height=height,
        margin=dict(l=22, r=22, t=92, b=18),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        font=PLOT_FONT,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.96)", font_size=13, font_family=PLOT_FONT["family"]),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            y=legend_y,
            x=0,
            bgcolor="rgba(255,255,255,0.72)",
            bordercolor="rgba(148, 163, 184, 0.18)",
            borderwidth=1,
        ),
        xaxis_rangeslider_visible=xaxis_rangeslider_visible,
    )
    fig.update_xaxes(
        showgrid=False,
        showspikes=True,
        spikemode="across",
        spikecolor="rgba(15, 23, 42, 0.18)",
        spikesnap="cursor",
    )
    fig.update_yaxes(gridcolor=GRID_COLOR, zeroline=False)
    return fig


def compute_overview_figure(df: pd.DataFrame) -> go.Figure:
    view = df.tail(220).copy()
    view["MA21"] = view["Close"].rolling(21).mean()
    view["MA50"] = view["Close"].rolling(50).mean()
    view["MA200"] = view["Close"].rolling(200).mean()
    view["Volume_Color"] = np.where(view["Close"] >= view["Open"], "#0f766e", "#b42318")

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
    )
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            increasing_line_color="#0f766e",
            decreasing_line_color="#b42318",
            increasing_fillcolor="#0f766e",
            decreasing_fillcolor="#b42318",
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view.index, y=view["MA21"], name="MA 21", line=dict(color="#dd6b20", width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MA50"], name="MA 50", line=dict(color="#2563eb", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MA200"], name="MA 200", line=dict(color="#1f2937", width=1.5, dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["Volume"], marker_color=view["Volume_Color"], name="Volume", opacity=0.58), row=2, col=1)
    return apply_figure_style(fig, title="Price Structure Snapshot", height=690)


def compute_elder_impulse(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["EMA13"] = work["Close"].ewm(span=13, adjust=False).mean()
    work["EMA65"] = work["Close"].ewm(span=65, adjust=False).mean()

    ema12 = work["Close"].ewm(span=12, adjust=False).mean()
    ema26 = work["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    work["MACD_Line"] = macd_line
    work["Signal_Line"] = signal_line
    work["MACD_Hist"] = macd_line - signal_line

    ema_rising = work["EMA13"].diff() > 0
    ema_falling = work["EMA13"].diff() < 0
    hist_rising = work["MACD_Hist"].diff() > 0
    hist_falling = work["MACD_Hist"].diff() < 0
    work["Impulse_State"] = np.select([ema_rising & hist_rising, ema_falling & hist_falling], [1, -1], default=0)
    work["Impulse_Color"] = work["Impulse_State"].map({1: "#0f766e", -1: "#b42318", 0: "#2563eb"})
    work["Long_Term_Up"] = work["Close"] > work["EMA65"]

    buy_condition = (work["Impulse_State"] == 1) & work["Long_Term_Up"]
    sell_condition = (work["Impulse_State"] == -1) & (~work["Long_Term_Up"])
    work["Buy_Signal"] = np.where(buy_condition & ~buy_condition.shift(fill_value=False), work["Low"] * 0.985, np.nan)
    work["Sell_Signal"] = np.where(sell_condition & ~sell_condition.shift(fill_value=False), work["High"] * 1.015, np.nan)
    return work


def elder_signal_label(state: int, long_term_up: bool) -> tuple[str, str]:
    if state == 1 and long_term_up:
        return "Bullish impulse", "bull"
    if state == -1 and not long_term_up:
        return "Bearish impulse", "bear"
    if state == 1:
        return "Bullish, below trend filter", "accent"
    if state == -1:
        return "Bearish, above trend filter", "accent"
    return "Neutral", "neutral"


def build_elder_figure(df: pd.DataFrame) -> go.Figure:
    view = df.tail(220)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            increasing_line_color="#0f766e",
            decreasing_line_color="#b42318",
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view.index, y=view["EMA13"], name="EMA 13", line=dict(color="#64748b", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["EMA65"], name="EMA 65", line=dict(color="#dd6b20", width=1.8, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["Buy_Signal"], mode="markers", marker=dict(symbol="triangle-up", size=11, color="#0f766e"), name="Buy trigger"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["Sell_Signal"], mode="markers", marker=dict(symbol="triangle-down", size=11, color="#b42318"), name="Sell trigger"), row=1, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["MACD_Hist"], marker_color=view["Impulse_Color"].tolist(), name="MACD histogram"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#64748b", row=2, col=1)
    return apply_figure_style(fig, title="Elder Impulse and Trend Filter", height=730)


def compute_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["Buy_Setup"] = 0
    work["Sell_Setup"] = 0
    work["Buy_Countdown"] = 0
    work["Sell_Countdown"] = 0

    buy_setup_loc = work.columns.get_loc("Buy_Setup")
    sell_setup_loc = work.columns.get_loc("Sell_Setup")
    buy_countdown_loc = work.columns.get_loc("Buy_Countdown")
    sell_countdown_loc = work.columns.get_loc("Sell_Countdown")

    active_buy_countdown = False
    active_sell_countdown = False
    buy_count = 0
    sell_count = 0
    close_values = work["Close"].to_numpy(dtype=float)

    for idx in range(4, len(work)):
        work.iat[idx, buy_setup_loc] = int(work.iat[idx - 1, buy_setup_loc]) + 1 if close_values[idx] < close_values[idx - 4] else 0
        work.iat[idx, sell_setup_loc] = int(work.iat[idx - 1, sell_setup_loc]) + 1 if close_values[idx] > close_values[idx - 4] else 0

        if work["Buy_Setup"].iat[idx] == 9:
            active_buy_countdown = True
            buy_count = 0
        if work["Sell_Setup"].iat[idx] == 9:
            active_sell_countdown = True
            sell_count = 0

        if active_buy_countdown and close_values[idx] <= close_values[idx - 2]:
            buy_count += 1
            work.iat[idx, buy_countdown_loc] = buy_count
            if buy_count == 13:
                active_buy_countdown = False

        if active_sell_countdown and close_values[idx] >= close_values[idx - 2]:
            sell_count += 1
            work.iat[idx, sell_countdown_loc] = sell_count
            if sell_count == 13:
                active_sell_countdown = False

    work["MA21"] = work["Close"].rolling(21).mean()
    work["MA50"] = work["Close"].rolling(50).mean()
    work["MA200"] = work["Close"].rolling(200).mean()
    work["RSI"] = calc_rsi(work["Close"])
    work["Tenkan_Sen"] = (work["High"].rolling(9).max() + work["Low"].rolling(9).min()) / 2
    work["Kijun_Sen"] = (work["High"].rolling(26).max() + work["Low"].rolling(26).min()) / 2
    work["Senkou_Span_A"] = ((work["Tenkan_Sen"] + work["Kijun_Sen"]) / 2).shift(26)
    work["Senkou_Span_B"] = ((work["High"].rolling(52).max() + work["Low"].rolling(52).min()) / 2).shift(26)
    return work


def td_signal_label(df: pd.DataFrame) -> tuple[str, str]:
    buy_cd = int(df["Buy_Countdown"].iloc[-1])
    sell_cd = int(df["Sell_Countdown"].iloc[-1])
    buy_setup = int(df["Buy_Setup"].iloc[-1])
    sell_setup = int(df["Sell_Setup"].iloc[-1])
    if buy_cd == 13:
        return "Buy countdown 13", "bull"
    if sell_cd == 13:
        return "Sell countdown 13", "bear"
    if buy_setup >= 7:
        return f"Buy setup {buy_setup}", "bull"
    if sell_setup >= 7:
        return f"Sell setup {sell_setup}", "bear"
    return "No exhaustion signal", "neutral"


def build_td_figure(df: pd.DataFrame) -> go.Figure:
    view = df.tail(180).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.05)
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            increasing_line_color="#0f766e",
            decreasing_line_color="#b42318",
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view.index, y=view["MA21"], name="MA 21", line=dict(color="#dd6b20", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MA50"], name="MA 50", line=dict(color="#2563eb", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["MA200"], name="MA 200", line=dict(color="#111827", width=1.2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["Senkou_Span_A"], mode="lines", line=dict(color="rgba(15,118,110,0.1)"), name="Cloud A"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["Senkou_Span_B"],
            mode="lines",
            line=dict(color="rgba(180,35,24,0.1)"),
            fill="tonexty",
            fillcolor="rgba(245,158,11,0.10)",
            name="Ichimoku cloud",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view.index, y=view["RSI"], name="RSI", line=dict(color="#7c3aed", width=1.6)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#b42318", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#0f766e", row=2, col=1)

    for idx, row in view.iterrows():
        if row["Buy_Setup"] in {7, 8, 9}:
            fig.add_annotation(x=idx, y=row["Low"] * 0.992, text=f"B{int(row['Buy_Setup'])}", showarrow=False, font=dict(color="#0f766e", size=10), row=1, col=1)
        if row["Sell_Setup"] in {7, 8, 9}:
            fig.add_annotation(x=idx, y=row["High"] * 1.008, text=f"S{int(row['Sell_Setup'])}", showarrow=False, font=dict(color="#b42318", size=10), row=1, col=1)
        if row["Buy_Countdown"] == 13:
            fig.add_annotation(x=idx, y=row["Low"] * 0.975, text="BUY 13", showarrow=False, font=dict(color="#0f766e", size=11), row=1, col=1)
        if row["Sell_Countdown"] == 13:
            fig.add_annotation(x=idx, y=row["High"] * 1.02, text="SELL 13", showarrow=False, font=dict(color="#b42318", size=11), row=1, col=1)

    return apply_figure_style(fig, title="TD Sequential with Trend Context", height=770)


def calc_rsi_numpy(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    ma_down = down.ewm(com=period - 1, adjust=True, min_periods=period).mean()
    rsi = ma_up / (ma_up + ma_down) * 100
    return rsi.fillna(50)


def get_slope_numpy(values: np.ndarray, window: int) -> float:
    if len(values) < window:
        return 0.0
    sample = values[-window:]
    x_axis = np.arange(len(sample))
    slope, _ = np.polyfit(x_axis, sample, 1)
    return float(slope)


def get_smart_extended_data(
    series_values: np.ndarray,
    horizon: int = 20,
    short_win: int = 20,
    long_win: int = 100,
    alpha: float = 0.6,
    decay: float = 0.96,
) -> np.ndarray:
    current_len = len(series_values)
    slope_short = get_slope_numpy(series_values, short_win)
    slope_long = get_slope_numpy(series_values, min(current_len, long_win))
    effective_alpha = max(alpha, 0.9) if slope_short < slope_long else alpha
    hybrid_slope = (slope_short * effective_alpha) + (slope_long * (1 - effective_alpha))
    decay_factors = decay ** np.arange(1, horizon + 1)
    future_steps = np.cumsum(hybrid_slope * decay_factors)
    return np.concatenate([series_values, series_values[-1] + future_steps])


def calc_rolling_stl_enhanced(
    series: pd.Series,
    trend_window: int = 31,
    horizon: int = 20,
    alpha: float = 0.6,
    decay: float = 0.96,
    min_history: int = 150,
) -> tuple[pd.Series, pd.Series]:
    prices = series.to_numpy(dtype=float)
    dates = series.index
    count = len(prices)
    rolling_trend = np.full(count, np.nan)
    rolling_resid = np.full(count, np.nan)
    if trend_window % 2 == 0:
        trend_window += 1

    valid_prices = np.where(prices > 0, prices, np.nan)
    log_prices = np.log(valid_prices)
    for cursor in range(min_history, count):
        current_log_data = log_prices[: cursor + 1]
        if np.isnan(current_log_data).any():
            continue
        extended_data = get_smart_extended_data(
            current_log_data,
            horizon=horizon,
            short_win=20,
            long_win=100,
            alpha=alpha,
            decay=decay,
        )
        result = STL(extended_data, period=5, trend=trend_window, robust=False).fit()
        rolling_trend[cursor] = result.trend[cursor]
        rolling_resid[cursor] = result.resid[cursor]

    return pd.Series(np.exp(rolling_trend), index=dates), pd.Series(rolling_resid, index=dates)


def calculate_hybrid_score(
    resid_series: pd.Series,
    rsi_series: pd.Series,
    w_resid: float = 0.6,
    w_rsi: float = 0.4,
    smooth_win: int = 3,
) -> pd.Series:
    resid_vol = resid_series.rolling(20).std().bfill()
    z_score = resid_series / resid_vol.replace(0, np.nan)
    score_resid = ((z_score.clip(-3, 3) + 3) / 6) * 100
    raw_score = (score_resid * w_resid) + (rsi_series * w_rsi)
    return raw_score.rolling(smooth_win).mean()


def compute_stl_cycle(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["RSI"] = calc_rsi_numpy(work["Close"])
    trend, resid = calc_rolling_stl_enhanced(
        work["Close"],
        trend_window=31,
        horizon=20,
        alpha=0.6,
        decay=0.96,
        min_history=150,
    )
    work["Trend"] = trend
    work["Residual"] = resid
    work["Cycle_Score"] = calculate_hybrid_score(work["Residual"], work["RSI"], w_resid=0.6, w_rsi=0.4, smooth_win=3).clip(0, 100)
    return work


def stl_signal_label(score: float) -> tuple[str, str]:
    if score >= 85:
        return "Overheated", "bear"
    if score >= 65:
        return "Extended", "accent"
    if score <= 15:
        return "Deep pullback", "bull"
    if score <= 35:
        return "Accumulation zone", "bull"
    return "Balanced cycle", "neutral"


def build_stl_figure(df: pd.DataFrame) -> go.Figure:
    view = df.tail(260).copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["Close"],
            mode="markers+lines",
            marker=dict(
                size=7,
                color=view["Cycle_Score"],
                colorscale=[[0.0, "#1d4ed8"], [0.5, "#f59e0b"], [1.0, "#b42318"]],
                cmin=0,
                cmax=100,
                line=dict(width=0),
                colorbar=dict(title="Cycle", tickfont=dict(size=12)),
            ),
            line=dict(color="rgba(15,23,42,0.18)", width=1.2),
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view.index, y=view["Trend"], name="STL trend", line=dict(color="#0f766e", width=2.1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["Cycle_Score"], name="Cycle score", line=dict(color="#f59e0b", width=1.8)), row=2, col=1)
    fig.add_hrect(y0=0, y1=10, fillcolor="rgba(37,99,235,0.12)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=90, y1=100, fillcolor="rgba(180,35,24,0.12)", line_width=0, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#64748b", row=2, col=1)
    return apply_figure_style(fig, title="Robust STL Cycle Dashboard", height=730)


def get_vol_label(volume: float, vol_ma: float) -> str:
    if np.isnan(vol_ma) or vol_ma == 0:
        return ""
    ratio = volume / vol_ma
    if ratio >= 2.5:
        return "Ultra"
    if ratio >= 2.0:
        return "Super"
    if ratio >= 1.5:
        return "High"
    if ratio >= 1.0:
        return "Mid"
    return "Low"


def calculate_volume_profile(df: pd.DataFrame, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if df.empty:
        return np.array([]), np.array([])
    price_bins = np.linspace(df["Low"].min(), df["High"].max(), bins)
    vol_profile, bin_edges = np.histogram((df["High"] + df["Low"]) / 2, bins=price_bins, weights=df["Volume"])
    return (bin_edges[:-1] + bin_edges[1:]) / 2, vol_profile


def get_order_blocks(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bull: list[dict[str, Any]] = []
    bear: list[dict[str, Any]] = []
    body = (df["Close"] - df["Open"]).abs()
    candle_range = df["High"] - df["Low"]
    body_mean = body.rolling(10).mean()

    for i in range(20, len(df)):
        atr = df["ATR"].iat[i]
        vol_ma = df["VOL_MA"].iat[i]
        if np.isnan(atr):
            continue
        is_strong_move = (candle_range.iat[i] > atr * 1.2) or (body.iat[i] > body_mean.iat[i] * 1.5)
        if not is_strong_move:
            continue

        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        label = get_vol_label(curr["Volume"], vol_ma)
        strength = float(candle_range.iat[i])
        if curr["Close"] < curr["Open"] and prev["Close"] >= prev["Open"]:
            bear.append({"date": df.index[i - 1], "top": float(prev["High"]), "bottom": float(prev["Low"]), "type": "bear", "strength": strength, "label": f"Bear OB {label}".strip()})
        elif curr["Close"] > curr["Open"] and prev["Close"] <= prev["Open"]:
            bull.append({"date": df.index[i - 1], "top": float(prev["High"]), "bottom": float(prev["Low"]), "type": "bull", "strength": strength, "label": f"Bull OB {label}".strip()})
    return bull, bear


def get_fair_value_gaps(df: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bull: list[dict[str, Any]] = []
    bear: list[dict[str, Any]] = []
    for i in range(20, len(df)):
        atr = df["ATR"].iat[i]
        vol_ma = df["VOL_MA"].iat[i]
        if np.isnan(atr):
            continue
        min_gap = atr * 0.3
        label = get_vol_label(df["Volume"].iat[i - 1], vol_ma)
        if df["Low"].iat[i] > df["High"].iat[i - 2]:
            gap = float(df["Low"].iat[i] - df["High"].iat[i - 2])
            if gap > min_gap:
                bull.append({"date": df.index[i - 1], "top": float(df["Low"].iat[i]), "bottom": float(df["High"].iat[i - 2]), "type": "bull", "size": gap, "label": f"Bull FVG {label}".strip()})
        if df["High"].iat[i] < df["Low"].iat[i - 2]:
            gap = float(df["Low"].iat[i - 2] - df["High"].iat[i])
            if gap > min_gap:
                bear.append({"date": df.index[i - 1], "top": float(df["Low"].iat[i - 2]), "bottom": float(df["High"].iat[i]), "type": "bear", "size": gap, "label": f"Bear FVG {label}".strip()})
    return bull, bear


def filter_active_zones(zones: list[dict[str, Any]], df: pd.DataFrame, limit: int = 3, sort_key: str = "strength") -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for zone in zones:
        try:
            start_idx = df.index.get_loc(zone["date"])
        except KeyError:
            continue
        future = df.iloc[start_idx + 1 :]
        if future.empty:
            active.append(zone)
            continue
        broken = (future["Close"] < zone["bottom"]).any() if zone["type"] == "bull" else (future["Close"] > zone["top"]).any()
        if not broken:
            active.append(zone)
    active.sort(key=lambda item: item.get(sort_key, 0), reverse=True)
    return active[:limit]


def compute_smc(df: pd.DataFrame) -> dict[str, Any]:
    work = df.copy()
    work["ATR"] = calc_atr(work, period=14)
    work["VOL_MA"] = work["Volume"].rolling(20).mean()
    work["EMA21"] = work["Close"].ewm(span=21, adjust=False).mean()
    work["SMA200"] = work["Close"].rolling(200).mean()
    work["RSI"] = calc_rsi(work["Close"])

    bull_obs, bear_obs = get_order_blocks(work)
    bull_fvgs, bear_fvgs = get_fair_value_gaps(work)
    active_bull_ob = filter_active_zones(bull_obs, work, limit=3, sort_key="strength")
    active_bear_ob = filter_active_zones(bear_obs, work, limit=3, sort_key="strength")
    active_bull_fvg = filter_active_zones(bull_fvgs, work, limit=3, sort_key="size")
    active_bear_fvg = filter_active_zones(bear_fvgs, work, limit=3, sort_key="size")

    view = work.tail(170).copy()
    centers, profile = calculate_volume_profile(view)
    poc_price = float(centers[profile.argmax()]) if len(centers) else float("nan")

    high_idx = argrelextrema(view["High"].to_numpy(), np.greater, order=5)[0]
    low_idx = argrelextrema(view["Low"].to_numpy(), np.less, order=5)[0]
    swings: list[dict[str, Any]] = []
    for idx in high_idx:
        swings.append({"date": view.index[idx], "price": float(view["High"].iat[idx]), "type": "H"})
    for idx in low_idx:
        swings.append({"date": view.index[idx], "price": float(view["Low"].iat[idx]), "type": "L"})
    swings.sort(key=lambda item: item["date"])
    eq_price = float("nan")
    if swings:
        recent_swings = [item for item in swings if item["date"] > (view.index[-1] - timedelta(days=200))]
        if recent_swings:
            range_high = max(item["price"] for item in recent_swings)
            range_low = min(item["price"] for item in recent_swings)
            eq_price = (range_high + range_low) / 2

    return {
        "view": view,
        "poc_price": poc_price,
        "eq_price": eq_price,
        "future_end": view.index[-1] + timedelta(days=35),
        "levels": sorted(swings, key=lambda item: item["date"], reverse=True)[:4],
        "active_bull_ob": active_bull_ob,
        "active_bear_ob": active_bear_ob,
        "active_bull_fvg": active_bull_fvg,
        "active_bear_fvg": active_bear_fvg,
    }


def smc_signal_label(smc_data: dict[str, Any]) -> tuple[str, str]:
    view = smc_data["view"]
    close = float(view["Close"].iloc[-1])
    ema21 = float(view["EMA21"].iloc[-1])
    poc = smc_data["poc_price"]
    if np.isnan(poc):
        poc = close
    if close >= ema21 and close >= poc:
        return "Premium above value area", "bear"
    if close <= ema21 and close <= poc:
        return "Discount inside demand zone", "bull"
    return "Mixed structure", "neutral"


def build_smc_figure(smc_data: dict[str, Any]) -> go.Figure:
    view = smc_data["view"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.82, 0.18], vertical_spacing=0.04)
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            increasing_line_color="#0f766e",
            decreasing_line_color="#b42318",
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view.index, y=view["EMA21"], name="EMA 21", line=dict(color="#dd6b20", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["SMA200"], name="SMA 200", line=dict(color="#111827", width=1.4, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["RSI"], name="RSI", line=dict(color="#7c3aed", width=1.6)), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="#b42318", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#0f766e", row=2, col=1)

    x0 = view.index[0]
    x1 = smc_data["future_end"]
    label_x = view.index[-1] + timedelta(days=7)
    zone_configs = [
        (smc_data["active_bull_ob"], "rgba(15,118,110,0.22)", "#0f766e"),
        (smc_data["active_bear_ob"], "rgba(180,35,24,0.22)", "#b42318"),
        (smc_data["active_bull_fvg"], "rgba(37,99,235,0.18)", "#2563eb"),
        (smc_data["active_bear_fvg"], "rgba(245,158,11,0.20)", "#dd6b20"),
    ]
    for zones, fill_color, line_color in zone_configs:
        for zone in zones:
            zone_start = max(zone["date"], x0)
            fig.add_shape(
                type="rect",
                x0=zone_start,
                x1=x1,
                y0=zone["bottom"],
                y1=zone["top"],
                fillcolor=fill_color,
                line=dict(color=line_color, width=1),
                row=1,
                col=1,
            )
            fig.add_annotation(
                x=label_x,
                y=(zone["top"] + zone["bottom"]) / 2,
                text=zone["label"],
                showarrow=False,
                font=dict(color=line_color, size=10),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor=line_color,
                borderwidth=1,
                xanchor="left",
                row=1,
                col=1,
            )

    if not np.isnan(smc_data["poc_price"]):
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[smc_data["poc_price"], smc_data["poc_price"]],
                mode="lines",
                line=dict(color="#334155", width=1.5, dash="dash"),
                name="POC",
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=label_x,
            y=smc_data["poc_price"],
            text="POC",
            showarrow=False,
            font=dict(color="#334155", size=10),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#334155",
            borderwidth=1,
            xanchor="left",
            row=1,
            col=1,
        )

    if not np.isnan(smc_data["eq_price"]):
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[smc_data["eq_price"], smc_data["eq_price"]],
                mode="lines",
                line=dict(color="#f59e0b", width=1.5),
                name="EQ (50%)",
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=view.index[-20] if len(view) > 20 else view.index[-1],
            y=smc_data["eq_price"],
            text="EQ (50%)",
            showarrow=False,
            font=dict(color="#111827", size=10),
            bgcolor="rgba(245,158,11,0.85)",
            row=1,
            col=1,
        )

    for level in smc_data["levels"]:
        color = "#0f766e" if level["type"] == "L" else "#b42318"
        fig.add_trace(
            go.Scatter(
                x=[level["date"], x1],
                y=[level["price"], level["price"]],
                mode="lines",
                line=dict(color=color, width=1.2, dash="dot"),
                opacity=0.55,
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=label_x,
            y=level["price"],
            text="S/R",
            showarrow=False,
            font=dict(color=color, size=10),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor=color,
            borderwidth=1,
            xanchor="left",
            row=1,
            col=1,
        )

    fig.update_xaxes(range=[x0, x1], row=1, col=1)
    return apply_figure_style(fig, title="Smart Money Concepts", height=860)


def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    work = df.copy()
    work["ATR"] = calc_atr(work, period=period)
    hl2 = (work["High"] + work["Low"]) / 2
    basic_upper = hl2 + (multiplier * work["ATR"])
    basic_lower = hl2 - (multiplier * work["ATR"])

    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    supertrend = pd.Series(np.nan, index=work.index, dtype=float)
    direction = pd.Series(1, index=work.index, dtype=int)

    for i in range(1, len(work)):
        if pd.isna(work["ATR"].iat[i]):
            continue

        prev_close = work["Close"].iat[i - 1]
        if pd.isna(final_upper.iat[i - 1]):
            final_upper.iat[i] = basic_upper.iat[i]
        elif basic_upper.iat[i] < final_upper.iat[i - 1] or prev_close > final_upper.iat[i - 1]:
            final_upper.iat[i] = basic_upper.iat[i]
        else:
            final_upper.iat[i] = final_upper.iat[i - 1]

        if pd.isna(final_lower.iat[i - 1]):
            final_lower.iat[i] = basic_lower.iat[i]
        elif basic_lower.iat[i] > final_lower.iat[i - 1] or prev_close < final_lower.iat[i - 1]:
            final_lower.iat[i] = basic_lower.iat[i]
        else:
            final_lower.iat[i] = final_lower.iat[i - 1]

        if i == 1 or pd.isna(supertrend.iat[i - 1]):
            direction.iat[i] = 1 if work["Close"].iat[i] >= hl2.iat[i] else -1
        elif supertrend.iat[i - 1] == final_upper.iat[i - 1]:
            direction.iat[i] = 1 if work["Close"].iat[i] > final_upper.iat[i] else -1
        else:
            direction.iat[i] = -1 if work["Close"].iat[i] < final_lower.iat[i] else 1

        supertrend.iat[i] = final_lower.iat[i] if direction.iat[i] == 1 else final_upper.iat[i]

    work["SuperTrend"] = supertrend
    work["Direction"] = direction
    work["LongFlip"] = (work["Direction"] == 1) & (work["Direction"].shift(1) == -1)
    work["ShortFlip"] = (work["Direction"] == -1) & (work["Direction"].shift(1) == 1)
    return work.tail(220).copy()


def supertrend_signal_label(supertrend_df: pd.DataFrame | None) -> tuple[str, str]:
    if supertrend_df is None or supertrend_df.empty:
        return "Not loaded", "neutral"
    latest = int(supertrend_df["Direction"].iloc[-1])
    if latest > 0:
        return "Trend support active", "bull"
    return "Trend resistance active", "bear"


def build_supertrend_figure(supertrend_df: pd.DataFrame) -> go.Figure:
    view = supertrend_df.copy()
    bull_mask = view["Direction"] > 0
    bear_mask = view["Direction"] < 0
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.76, 0.24], vertical_spacing=0.05)
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            increasing_line_color="#0f766e",
            decreasing_line_color="#b42318",
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view.index, y=view["SuperTrend"].where(bull_mask), mode="lines", name="Bull ST", line=dict(color="#0f766e", width=2.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["SuperTrend"].where(bear_mask), mode="lines", name="Bear ST", line=dict(color="#b42318", width=2.6)), row=1, col=1)
    long_flips = view[view["LongFlip"]]
    short_flips = view[view["ShortFlip"]]
    if not long_flips.empty:
        fig.add_trace(go.Scatter(x=long_flips.index, y=long_flips["Low"] * 0.995, mode="markers", name="Long flip", marker=dict(color="#0f766e", symbol="triangle-up", size=12)), row=1, col=1)
    if not short_flips.empty:
        fig.add_trace(go.Scatter(x=short_flips.index, y=short_flips["High"] * 1.005, mode="markers", name="Short flip", marker=dict(color="#b42318", symbol="triangle-down", size=12)), row=1, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["ATR"], name="ATR", marker_color="#64748b"), row=2, col=1)
    return apply_figure_style(fig, title="SuperTrend Regime", height=820)


def compute_williams_vix_fix(
    df: pd.DataFrame,
    pd_window: int = 22,
    bbl: int = 20,
    mult: float = 2.0,
    lb: int = 50,
    ph: float = 0.85,
) -> pd.DataFrame:
    work = df.copy()
    highest_close = work["Close"].rolling(pd_window).max()
    lowest_close = work["Close"].rolling(pd_window).min()
    work["WVF"] = ((highest_close - work["Low"]) / highest_close.replace(0, np.nan)) * 100
    work["WVF_Mid"] = work["WVF"].rolling(bbl).mean()
    work["WVF_Upper"] = work["WVF_Mid"] + (mult * work["WVF"].rolling(bbl).std())
    work["WVF_RangeHigh"] = work["WVF"].rolling(lb).max() * ph
    work["Oversold"] = (work["WVF"] >= work["WVF_Upper"]) | (work["WVF"] >= work["WVF_RangeHigh"])
    work["OversoldExit"] = (work["Oversold"].shift(1).rolling(4).sum() == 4) & (~work["Oversold"])

    work["WVF_Inverse"] = ((work["High"] - lowest_close) / lowest_close.replace(0, np.nan)) * 100
    work["WVF_Inv_Mid"] = work["WVF_Inverse"].rolling(bbl).mean()
    work["WVF_Inv_Upper"] = work["WVF_Inv_Mid"] + (mult * work["WVF_Inverse"].rolling(bbl).std())
    work["WVF_Inv_RangeHigh"] = work["WVF_Inverse"].rolling(lb).max() * ph
    work["Overbought"] = (work["WVF_Inverse"] >= work["WVF_Inv_Upper"]) | (work["WVF_Inverse"] >= work["WVF_Inv_RangeHigh"])
    work["OverboughtExit"] = (work["Overbought"].shift(1).rolling(4).sum() == 4) & (~work["Overbought"])
    return work.tail(220).copy()


def vix_fix_signal_label(vix_fix_df: pd.DataFrame | None) -> tuple[str, str]:
    if vix_fix_df is None or vix_fix_df.empty:
        return "Not loaded", "neutral"
    if bool(vix_fix_df["Oversold"].iloc[-1]):
        return "Panic spike / oversold", "bull"
    if bool(vix_fix_df["Overbought"].iloc[-1]):
        return "Complacency spike / overbought", "bear"
    return "Inside normal volatility range", "neutral"


def build_vix_fix_figure(vix_fix_df: pd.DataFrame) -> go.Figure:
    view = vix_fix_df.copy()
    colors = np.where(view["Oversold"], "#0f766e", np.where(view["OversoldExit"], "#b42318", "#94a3b8"))
    inverse_colors = np.where(view["Overbought"], "#b42318", np.where(view["OverboughtExit"], "#0f766e", "#cbd5e1"))
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.42, 0.29, 0.29],
        vertical_spacing=0.05,
        subplot_titles=("Price", "Williams Vix Fix", "Inverse Williams Vix Fix"),
    )
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"], increasing_line_color="#0f766e", decreasing_line_color="#b42318", name="Price"), row=1, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["WVF"], name="WVF", marker_color=colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["WVF_Upper"], name="WVF band", line=dict(color="#111827", width=1.6, dash="dot")), row=2, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["WVF_RangeHigh"], name="Range high", line=dict(color="#dd6b20", width=1.4)), row=2, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["WVF_Inverse"], name="Inverse", marker_color=inverse_colors), row=3, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["WVF_Inv_Upper"], name="Inverse band", line=dict(color="#334155", width=1.6, dash="dot")), row=3, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["WVF_Inv_RangeHigh"], name="Inverse range high", line=dict(color="#7c3aed", width=1.4)), row=3, col=1)

    oversold_exit = view[view["OversoldExit"]]
    overbought_exit = view[view["OverboughtExit"]]
    if not oversold_exit.empty:
        fig.add_trace(go.Scatter(x=oversold_exit.index, y=oversold_exit["Low"] * 0.99, mode="markers", name="Oversold exit", marker=dict(color="#0f766e", symbol="diamond", size=10)), row=1, col=1)
    if not overbought_exit.empty:
        fig.add_trace(go.Scatter(x=overbought_exit.index, y=overbought_exit["High"] * 1.01, mode="markers", name="Overbought exit", marker=dict(color="#b42318", symbol="diamond", size=10)), row=1, col=1)
    return apply_figure_style(fig, title="Williams Vix Fix / Inverse", height=920)


def compute_squeeze_momentum(
    df: pd.DataFrame,
    bb_length: int = 20,
    kc_length: int = 20,
    bb_mult: float = 1.5,
    kc_mult: float = 1.5,
) -> pd.DataFrame:
    work = df.copy()
    work["Basis"] = work["Close"].rolling(bb_length).mean()
    dev = work["Close"].rolling(bb_length).std() * bb_mult
    work["UpperBB"] = work["Basis"] + dev
    work["LowerBB"] = work["Basis"] - dev

    work["KC_Mid"] = work["Close"].rolling(kc_length).mean()
    work["RangeMA"] = (work["High"] - work["Low"]).rolling(kc_length).mean()
    work["UpperKC"] = work["KC_Mid"] + (work["RangeMA"] * kc_mult)
    work["LowerKC"] = work["KC_Mid"] - (work["RangeMA"] * kc_mult)

    high_roll = work["High"].rolling(kc_length).max()
    low_roll = work["Low"].rolling(kc_length).min()
    work["Momentum"] = (work["Close"] - (((high_roll + low_roll) / 2) + work["KC_Mid"]) / 2).rolling(5).mean()
    work["SqueezeOn"] = (work["LowerBB"] > work["LowerKC"]) & (work["UpperBB"] < work["UpperKC"])
    work["SqueezeOff"] = (work["LowerBB"] < work["LowerKC"]) & (work["UpperBB"] > work["UpperKC"])
    return work.tail(220).copy()


def squeeze_signal_label(squeeze_df: pd.DataFrame | None) -> tuple[str, str]:
    if squeeze_df is None or squeeze_df.empty:
        return "Not loaded", "neutral"
    momentum = float(squeeze_df["Momentum"].iloc[-1])
    in_squeeze = bool(squeeze_df["SqueezeOn"].iloc[-1])
    if in_squeeze and momentum >= 0:
        return "Compression with positive bias", "accent"
    if in_squeeze and momentum < 0:
        return "Compression with negative bias", "bear"
    if momentum >= 0:
        return "Expansion to upside", "bull"
    return "Expansion to downside", "bear"


def build_squeeze_figure(squeeze_df: pd.DataFrame) -> go.Figure:
    view = squeeze_df.copy()
    hist_colors = np.where(view["Momentum"] >= 0, "#0f766e", "#b42318")
    squeeze_markers = np.where(view["SqueezeOn"], "#111827", np.where(view["SqueezeOff"], "#f59e0b", "#cbd5e1"))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.05)
    fig.add_trace(go.Candlestick(x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"], increasing_line_color="#0f766e", decreasing_line_color="#b42318", name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["UpperBB"], name="Upper BB", line=dict(color="#b42318", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["LowerBB"], name="Lower BB", line=dict(color="#b42318", width=1.4)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["UpperKC"], name="Upper KC", line=dict(color="#0f766e", width=1.4, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["LowerKC"], name="Lower KC", line=dict(color="#0f766e", width=1.4, dash="dot")), row=1, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["Momentum"], name="Momentum", marker_color=hist_colors), row=2, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=np.zeros(len(view)), mode="markers", name="Squeeze state", marker=dict(color=squeeze_markers, size=8)), row=2, col=1)
    fig.add_hline(y=0, line_color="#64748b", line_width=1, line_dash="dot", row=2, col=1)
    return apply_figure_style(fig, title="Squeeze Momentum", height=840)


def compute_nadaraya_watson(df: pd.DataFrame, window: int = 80, bandwidth: float = 12.0) -> pd.DataFrame:
    work = df.copy()
    closes = work["Close"].to_numpy(dtype=float)
    estimate = np.full(len(work), np.nan)
    for i in range(len(work)):
        start = max(0, i - window + 1)
        idx = np.arange(start, i + 1)
        distances = (idx - i) / bandwidth
        weights = np.exp(-0.5 * distances**2)
        estimate[i] = np.dot(weights, closes[start : i + 1]) / weights.sum()

    work["NW_Estimate"] = estimate
    residual = work["Close"] - work["NW_Estimate"]
    band = residual.rolling(20).std().fillna(0) * 1.5
    work["NW_Upper"] = work["NW_Estimate"] + band
    work["NW_Lower"] = work["NW_Estimate"] - band
    work["NW_Slope"] = pd.Series(estimate, index=work.index).diff()
    work["NW_Trend"] = np.where(work["NW_Slope"] > 0, 1, np.where(work["NW_Slope"] < 0, -1, 0))
    return work.tail(240).copy()


def nadaraya_signal_label(nw_df: pd.DataFrame | None) -> tuple[str, str]:
    if nw_df is None or nw_df.empty:
        return "Not loaded", "neutral"
    latest = int(nw_df["NW_Trend"].iloc[-1])
    if latest > 0:
        return "Kernel trend rising", "bull"
    if latest < 0:
        return "Kernel trend falling", "bear"
    return "Kernel trend flat", "neutral"


def build_nadaraya_figure(nw_df: pd.DataFrame) -> go.Figure:
    view = nw_df.copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22], vertical_spacing=0.05)
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            increasing_line_color="#0f766e",
            decreasing_line_color="#b42318",
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view.index, y=view["NW_Estimate"], name="NW estimate", line=dict(color="#111827", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["NW_Upper"], name="Upper band", line=dict(color="#2563eb", width=1.4, dash="dot")), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=view.index,
            y=view["NW_Lower"],
            name="Lower band",
            line=dict(color="#2563eb", width=1.4, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(37,99,235,0.08)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Bar(x=view.index, y=view["NW_Slope"], name="Slope", marker_color=np.where(view["NW_Slope"] >= 0, "#0f766e", "#b42318")), row=2, col=1)
    fig.add_hline(y=0, line_color="#64748b", line_width=1, line_dash="dot", row=2, col=1)
    return apply_figure_style(fig, title="Nadaraya-Watson Estimator", height=840)


def compute_lorentzian_classification(df: pd.DataFrame, lookback: int = 180, neighbors: int = 8, horizon: int = 4) -> pd.DataFrame:
    work = df.copy()
    work["RSI"] = calc_rsi(work["Close"])
    work["Ret5"] = work["Close"].pct_change(5).mul(100)
    work["Dist20"] = ((work["Close"] / work["Close"].rolling(20).mean()) - 1).mul(100)
    work["ATR_Norm"] = (calc_atr(work, 14) / work["Close"]).mul(100)
    feature_cols = ["RSI", "Ret5", "Dist20", "ATR_Norm"]
    features = work[feature_cols].fillna(0).to_numpy(dtype=float)
    future_label = np.sign(work["Close"].shift(-horizon) - work["Close"]).to_numpy(dtype=float)

    score = np.full(len(work), np.nan)
    confidence = np.full(len(work), np.nan)
    regime = np.zeros(len(work), dtype=int)

    for i in range(max(lookback, 30), len(work) - horizon):
        start = max(20, i - lookback)
        distances: list[tuple[float, float]] = []
        for j in range(start, i):
            if np.isnan(future_label[j]) or future_label[j] == 0:
                continue
            dist = float(np.log1p(np.abs(features[i] - features[j])).sum())
            distances.append((dist, future_label[j]))
        if not distances:
            continue
        nearest = sorted(distances, key=lambda item: item[0])[:neighbors]
        raw_score = sum(item[1] for item in nearest)
        score[i] = raw_score
        confidence[i] = abs(raw_score) / neighbors
        regime[i] = 1 if raw_score > 0 else -1 if raw_score < 0 else 0

    work["LC_Score"] = score
    work["LC_Confidence"] = confidence
    work["LC_Regime"] = regime
    work["LC_LongFlip"] = (work["LC_Regime"] == 1) & (work["LC_Regime"].shift(1) <= 0)
    work["LC_ShortFlip"] = (work["LC_Regime"] == -1) & (work["LC_Regime"].shift(1) >= 0)
    return work.tail(240).copy()


def lorentzian_signal_label(lc_df: pd.DataFrame | None) -> tuple[str, str]:
    if lc_df is None or lc_df.empty:
        return "Not loaded", "neutral"
    latest = int(lc_df["LC_Regime"].iloc[-1])
    confidence = float(lc_df["LC_Confidence"].fillna(0).iloc[-1])
    if latest > 0:
        return f"Bullish class ({confidence:.0%})", "bull"
    if latest < 0:
        return f"Bearish class ({confidence:.0%})", "bear"
    return "Classification neutral", "neutral"


def build_lorentzian_figure(lc_df: pd.DataFrame) -> go.Figure:
    view = lc_df.copy()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.05)
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            increasing_line_color="#0f766e",
            decreasing_line_color="#b42318",
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view.index, y=view["Close"].rolling(20).mean(), name="MA 20", line=dict(color="#111827", width=1.5)), row=1, col=1)
    long_flips = view[view["LC_LongFlip"]]
    short_flips = view[view["LC_ShortFlip"]]
    if not long_flips.empty:
        fig.add_trace(go.Scatter(x=long_flips.index, y=long_flips["Low"] * 0.995, mode="markers", name="Bull flip", marker=dict(color="#0f766e", size=11, symbol="triangle-up")), row=1, col=1)
    if not short_flips.empty:
        fig.add_trace(go.Scatter(x=short_flips.index, y=short_flips["High"] * 1.005, mode="markers", name="Bear flip", marker=dict(color="#b42318", size=11, symbol="triangle-down")), row=1, col=1)
    fig.add_trace(go.Bar(x=view.index, y=view["LC_Score"], name="Lorentzian score", marker_color=np.where(view["LC_Score"] >= 0, "#0f766e", "#b42318")), row=2, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["LC_Confidence"], name="Confidence", line=dict(color="#2563eb", width=1.8)), row=2, col=1)
    fig.add_hline(y=0, line_color="#64748b", line_width=1, line_dash="dot", row=2, col=1)
    return apply_figure_style(fig, title="Lorentzian Classification", height=840)


def compute_cvd_divergence(df: pd.DataFrame) -> dict[str, Any]:
    work = df.copy()
    candle_range = (work["High"] - work["Low"]).replace(0, np.nan)
    body_pressure = ((work["Close"] - work["Open"]) / candle_range).clip(-1, 1).fillna(0)
    work["Delta"] = work["Volume"] * body_pressure
    work["CVD"] = work["Delta"].cumsum()
    view = work.tail(240).copy()

    price_high_idx = argrelextrema(view["High"].to_numpy(), np.greater_equal, order=5)[0]
    price_low_idx = argrelextrema(view["Low"].to_numpy(), np.less_equal, order=5)[0]
    divergence: dict[str, Any] = {"type": "neutral", "points": []}

    if len(price_low_idx) >= 2:
        p1, p2 = price_low_idx[-2], price_low_idx[-1]
        if view["Low"].iat[p2] < view["Low"].iat[p1] and view["CVD"].iat[p2] > view["CVD"].iat[p1]:
            divergence = {
                "type": "bullish",
                "points": [(view.index[p1], float(view["Low"].iat[p1]), float(view["CVD"].iat[p1])), (view.index[p2], float(view["Low"].iat[p2]), float(view["CVD"].iat[p2]))],
            }
    if divergence["type"] == "neutral" and len(price_high_idx) >= 2:
        p1, p2 = price_high_idx[-2], price_high_idx[-1]
        if view["High"].iat[p2] > view["High"].iat[p1] and view["CVD"].iat[p2] < view["CVD"].iat[p1]:
            divergence = {
                "type": "bearish",
                "points": [(view.index[p1], float(view["High"].iat[p1]), float(view["CVD"].iat[p1])), (view.index[p2], float(view["High"].iat[p2]), float(view["CVD"].iat[p2]))],
            }
    return {"view": view, "divergence": divergence}


def cvd_signal_label(cvd_data: dict[str, Any] | None) -> tuple[str, str]:
    if cvd_data is None:
        return "Not loaded", "neutral"
    divergence_type = cvd_data["divergence"]["type"]
    if divergence_type == "bullish":
        return "Bullish CVD divergence", "bull"
    if divergence_type == "bearish":
        return "Bearish CVD divergence", "bear"
    return "No active divergence", "neutral"


def build_cvd_figure(cvd_data: dict[str, Any]) -> go.Figure:
    view = cvd_data["view"]
    divergence = cvd_data["divergence"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.05)
    fig.add_trace(
        go.Candlestick(
            x=view.index,
            open=view["Open"],
            high=view["High"],
            low=view["Low"],
            close=view["Close"],
            increasing_line_color="#0f766e",
            decreasing_line_color="#b42318",
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Bar(x=view.index, y=view["Delta"], name="Delta", marker_color=np.where(view["Delta"] >= 0, "#0f766e", "#b42318"), opacity=0.35), row=2, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["CVD"], name="CVD", line=dict(color="#111827", width=2.0)), row=2, col=1)
    if divergence["type"] != "neutral":
        p1, p2 = divergence["points"]
        price_color = "#0f766e" if divergence["type"] == "bullish" else "#b42318"
        fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[1], p2[1]], mode="lines+markers", name="Price divergence", line=dict(color=price_color, width=2.0)), row=1, col=1)
        fig.add_trace(go.Scatter(x=[p1[0], p2[0]], y=[p1[2], p2[2]], mode="lines+markers", name="CVD divergence", line=dict(color=price_color, width=2.0, dash="dot")), row=2, col=1)
    fig.add_hline(y=0, line_color="#64748b", line_width=1, line_dash="dot", row=2, col=1)
    return apply_figure_style(fig, title="Cumulative Volume Delta Divergence", height=840)


def get_probability(series: pd.Series, window: int, inverse: bool = False) -> pd.Series:
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    z_score = (series - roll_mean) / roll_std.replace(0, np.nan)
    prob = z_score.apply(lambda value: norm.cdf(value) if not np.isnan(value) else np.nan)
    return 1 - prob if inverse else prob


def _fetch_market_fear_greed() -> dict[str, Any] | None:
    tickers = ["SPY", "^VIX", "HYG", "IEF", "RSP", "XLY", "XLP", "UUP"]
    try:
        raw = yf.download(tickers, period="6y", progress=False, auto_adjust=False, threads=False)
        close_df = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        close_df = close_df.ffill().dropna()
    except Exception:
        return None

    factors = pd.DataFrame(index=close_df.index)
    ma20 = close_df["SPY"].rolling(20).mean()
    std20 = close_df["SPY"].rolling(20).std()
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)
    ma125 = close_df["SPY"].rolling(125).mean()

    factors["BB_Pos"] = get_probability((close_df["SPY"] - lower) / (upper - lower), 252)
    factors["RSI_Mom"] = get_probability(calc_rsi(close_df["SPY"]), 252)
    factors["MA125_Div"] = get_probability((close_df["SPY"] - ma125) / ma125, 252)
    factors["Breadth"] = get_probability(close_df["RSP"] / close_df["SPY"], 252)
    factors["Sector"] = get_probability(close_df["XLY"] / close_df["XLP"], 252)
    factors["Credit"] = get_probability(close_df["HYG"] / close_df["IEF"], 252)
    factors["VIX_Inv"] = get_probability(close_df["^VIX"], 252, inverse=True)
    factors["Dollar_Inv"] = get_probability(close_df["UUP"], 252, inverse=True)

    score = (
        0.10 * factors["BB_Pos"]
        + 0.10 * factors["RSI_Mom"]
        + 0.10 * factors["MA125_Div"]
        + 0.15 * factors["Breadth"]
        + 0.15 * factors["Sector"]
        + 0.10 * factors["Credit"]
        + 0.15 * factors["VIX_Inv"]
        + 0.15 * factors["Dollar_Inv"]
    )

    latest_score = float(score.dropna().iloc[-1])
    latest_factors = factors.dropna().iloc[-1].sort_values(ascending=False)
    plot_df = pd.DataFrame({
        "Score": score,
        "SPY": close_df["SPY"],
        "Breadth": factors["Breadth"],
        "Sector": factors["Sector"],
        "Credit": factors["Credit"],
    }).dropna()

    return {
        "latest_score": latest_score,
        "latest_factors": latest_factors,
        "plot_df": plot_df.tail(260),
        "status": classify_market_score(latest_score),
    }


def compute_market_fear_greed() -> dict[str, Any] | None:
    return get_or_refresh_daily_payload("market_pulse", _fetch_market_fear_greed)


def build_market_figure(market_data: dict[str, Any]) -> go.Figure:
    plot_df = market_data["plot_df"]
    latest_factors = market_data["latest_factors"]
    normalized_spy = (plot_df["SPY"] / plot_df["SPY"].iloc[0]) - 1

    fig = make_subplots(
        rows=2,
        cols=2,
        specs=[[{"type": "indicator"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}]],
        subplot_titles=("Pulse Gauge", "Score vs SPY", "Factor Breadth", "Internal Health"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=market_data["latest_score"] * 100,
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#102a43"},
                "steps": [
                    {"range": [0, 20], "color": "#dbeafe"},
                    {"range": [20, 40], "color": "#dcfce7"},
                    {"range": [40, 60], "color": "#f8fafc"},
                    {"range": [60, 80], "color": "#fef3c7"},
                    {"range": [80, 100], "color": "#fee2e2"},
                ],
            },
            title={"text": "Fear <-> Greed"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Score"], name="Smart score", line=dict(color="#0f766e", width=2.2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=plot_df.index, y=normalized_spy, name="SPY return", line=dict(color="#dd6b20", width=1.6, dash="dash")), row=1, col=2)
    fig.add_trace(go.Bar(x=latest_factors.index, y=latest_factors.values, marker_color="#2563eb", name="Latest factors"), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Breadth"], name="Breadth", line=dict(color="#dd6b20", width=1.8)), row=2, col=2)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Sector"], name="Sector", line=dict(color="#2563eb", width=1.8)), row=2, col=2)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Credit"], name="Credit", line=dict(color="#0f766e", width=1.8)), row=2, col=2)
    fig.add_trace(go.Scatter(x=plot_df.index, y=[0.8] * len(plot_df), mode="lines", line=dict(color="#b42318", width=1, dash="dot"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=plot_df.index, y=[0.2] * len(plot_df), mode="lines", line=dict(color="#0f766e", width=1, dash="dot"), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=plot_df.index, y=[0.5] * len(plot_df), mode="lines", line=dict(color="#64748b", width=1, dash="dot"), showlegend=False), row=2, col=2)
    return apply_figure_style(fig, title="Macro Fear and Greed Dashboard", height=840, legend_y=1.08)


def bs_greeks(S: float, K: float, T: float, r: float, q: float, sigma: float, option_type: str) -> dict[str, float]:
    if T <= 0.0001 or sigma <= 0.001 or S <= 0 or K <= 0:
        return {"gamma": 0.0, "vanna": 0.0, "charm": 0.0}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    vanna = -np.exp(-q * T) * norm.pdf(d1) * (d2 / sigma)
    term1 = q * np.exp(-q * T) * norm.cdf(d1 if option_type == "call" else -d1)
    term2 = np.exp(-q * T) * norm.pdf(d1) * ((r - q) / (sigma * np.sqrt(T)) - d2 / (2 * T))
    charm = (term1 - term2) if option_type == "call" else (-term1 - term2)
    return {"gamma": float(gamma), "vanna": float(vanna), "charm": float(charm)}


@st.cache_data(ttl=3600, show_spinner=False)
def get_risk_free_rate() -> float:
    try:
        irx = normalize_ohlcv_frame(yf.download("^IRX", period="5d", progress=False, auto_adjust=False, threads=False))
        return float(irx["Close"].iloc[-1] / 100) if not irx.empty else 0.04
    except Exception:
        return 0.04


def _fetch_spx_options_payload() -> dict[str, Any] | None:
    try:
        response = requests.get(
            "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if "data" not in payload:
            return None
        return payload
    except Exception:
        return None


def fetch_spx_options_payload() -> dict[str, Any] | None:
    return get_or_refresh_daily_payload("spx_options_payload", _fetch_spx_options_payload)


def extract_spx_expiries(payload: dict[str, Any] | None) -> list[str]:
    if not payload:
        return []
    options = payload.get("data", {}).get("options", [])
    expiries = set()
    for item in options:
        parsed = parse_spx_option_symbol(item.get("option", ""))
        if parsed:
            expiries.add(parsed[0])
    return sorted(expiries)


def parse_spx_option_symbol(symbol: str) -> tuple[str, str, float] | None:
    match = re.search(r"(\d{6})([CP])(\d{8})$", symbol)
    if not match:
        return None
    raw_date, cp_flag, raw_strike = match.groups()
    expiry = f"20{raw_date[:2]}-{raw_date[2:4]}-{raw_date[4:6]}"
    strike = int(raw_strike) / 1000.0
    return expiry, cp_flag, strike


@st.cache_data(ttl=900, show_spinner=False)
def get_spx_expiries() -> list[str]:
    return extract_spx_expiries(fetch_spx_options_payload())


@st.cache_data(ttl=1800, show_spinner=False)
def get_vix_term_structure() -> dict[str, float]:
    try:
        vix9d = normalize_ohlcv_frame(yf.download("^VIX9D", period="5d", progress=False, auto_adjust=False, threads=False))
        vix30d = normalize_ohlcv_frame(yf.download("^VIX", period="5d", progress=False, auto_adjust=False, threads=False))
        vix3m = normalize_ohlcv_frame(yf.download("^VIX3M", period="5d", progress=False, auto_adjust=False, threads=False))
        return {
            "vix9d": float(vix9d["Close"].iloc[-1]),
            "vix30d": float(vix30d["Close"].iloc[-1]),
            "vix3m": float(vix3m["Close"].iloc[-1]),
        }
    except Exception:
        return {"vix9d": 18.0, "vix30d": 19.0, "vix3m": 20.0}


def compute_options_analytics(
    expiry: str | None = None,
    spot_range_pct: float = 0.15,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    payload = payload or fetch_spx_options_payload()
    if not payload:
        return None

    data = payload.get("data", {})
    spot = float(data.get("current_price") or data.get("close") or 0)
    option_rows = data.get("options", [])
    expiries = extract_spx_expiries(payload)
    if not spot or not expiries:
        return None

    selected_expiry = expiry if expiry in expiries else expiries[0]
    target_token = pd.Timestamp(selected_expiry).strftime("%y%m%d")
    risk_free = get_risk_free_rate()
    vix_data = get_vix_term_structure()
    fallback_sigma = max(vix_data["vix30d"] / 100.0, 0.05)
    time_to_expiry = max(
        (pd.Timestamp(selected_expiry) - pd.Timestamp.now().normalize()).total_seconds() / (365.25 * 24 * 3600),
        0.001,
    )

    strike_map: dict[float, dict[str, float]] = {}
    total_call_volume = 0.0
    total_put_volume = 0.0

    for item in option_rows:
        symbol = item.get("option", "")
        if target_token not in symbol:
            continue

        parsed = parse_spx_option_symbol(symbol)
        if not parsed:
            continue

        _, cp_flag, strike = parsed
        open_interest = float(item.get("open_interest") or 0)
        volume = float(item.get("volume") or 0)
        implied_vol = item.get("implied_volatility") or item.get("volatility") or fallback_sigma
        sigma = float(implied_vol) if float(implied_vol) > 0 else fallback_sigma

        if cp_flag == "C":
            total_call_volume += volume
        else:
            total_put_volume += volume

        if open_interest <= 0:
            continue

        if strike not in strike_map:
            strike_map[strike] = {"call_oi": 0.0, "put_oi": 0.0, "gex": 0.0, "vanna": 0.0, "charm": 0.0}

        greeks = bs_greeks(spot, strike, time_to_expiry, risk_free, 0.014, sigma, "call" if cp_flag == "C" else "put")
        multiplier = spot * spot * 0.01 * 100
        gex = greeks["gamma"] * open_interest * multiplier
        vanna = greeks["vanna"] * open_interest * spot * 0.01 * 100
        charm = greeks["charm"] * open_interest * 100

        if cp_flag == "C":
            strike_map[strike]["call_oi"] += open_interest
            strike_map[strike]["gex"] += gex
            strike_map[strike]["vanna"] += vanna
            strike_map[strike]["charm"] += charm
        else:
            strike_map[strike]["put_oi"] += open_interest
            strike_map[strike]["gex"] -= gex
            strike_map[strike]["vanna"] -= vanna
            strike_map[strike]["charm"] -= charm

    if not strike_map:
        return None

    strike_view = pd.DataFrame.from_dict(strike_map, orient="index").reset_index().rename(columns={"index": "strike"})
    strike_view = strike_view.sort_values("strike")
    strike_view = strike_view[
        (strike_view["strike"] >= spot * (1 - spot_range_pct))
        & (strike_view["strike"] <= spot * (1 + spot_range_pct))
    ].copy()
    if strike_view.empty:
        return None

    strike_view["pain"] = [
        ((strike - strike_view["strike"]).clip(lower=0) * strike_view["call_oi"]).sum()
        + ((strike_view["strike"] - strike).clip(lower=0) * strike_view["put_oi"]).sum()
        for strike in strike_view["strike"]
    ]

    put_call_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 1.0
    max_pain = float(strike_view.loc[strike_view["pain"].idxmin(), "strike"])

    return {
        "underlying": "SPX",
        "spot": spot,
        "expiry": selected_expiry,
        "expiries": expiries,
        "strike_view": strike_view,
        "put_call_ratio": put_call_ratio,
        "max_pain": max_pain,
        "net_gex": float(strike_view["gex"].sum()),
        "net_vanna": float(strike_view["vanna"].sum()),
        "net_charm": float(strike_view["charm"].sum()),
        "vix9d": vix_data["vix9d"],
        "vix30d": vix_data["vix30d"],
        "vix3m": vix_data["vix3m"],
        "lower_bound": float(spot * (1 - spot_range_pct)),
        "upper_bound": float(spot * (1 + spot_range_pct)),
    }


def options_signal_label(options_data: dict[str, Any] | None) -> tuple[str, str]:
    if not options_data:
        return "SPX data unavailable", "neutral"
    pcr = 1.0 if np.isnan(options_data["put_call_ratio"]) else options_data["put_call_ratio"]
    net_gex = options_data["net_gex"]
    if pcr > 1.3 and net_gex < 0:
        return "Defensive flow", "bear"
    if pcr < 0.8 and net_gex > 0:
        return "Supportive dealer flow", "bull"
    return "Mixed positioning", "neutral"


def build_options_figure(options_data: dict[str, Any], spot: float) -> go.Figure:
    view = options_data["strike_view"]
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "indicator"}]],
        subplot_titles=("Dealer Net GEX", "Max Pain Profile", "Net Vanna", "Net Charm", "VIX Term Structure", "Put/Call Volume"),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    fig.add_trace(go.Bar(x=view["strike"], y=view["gex"] / 1e9, marker_color="#2563eb"), row=1, col=1)
    fig.add_trace(go.Scatter(x=view["strike"], y=view["pain"], fill="tozeroy", line=dict(color="#b42318", width=2)), row=1, col=2)
    fig.add_trace(go.Bar(x=view["strike"], y=view["vanna"] / 1e9, marker_color="#0f766e"), row=2, col=1)
    fig.add_trace(go.Bar(x=view["strike"], y=view["charm"] / 1e6, marker_color="#dd6b20"), row=2, col=2)
    fig.add_trace(go.Scatter(x=["9D", "30D", "3M"], y=[options_data["vix9d"], options_data["vix30d"], options_data["vix3m"]], mode="lines+markers", line=dict(width=3, color="#7c3aed"), fill="tozeroy"), row=3, col=1)
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=options_data["put_call_ratio"],
            gauge={
                "axis": {"range": [0, 2]},
                "bar": {"color": "#102a43"},
                "steps": [
                    {"range": [0, 0.7], "color": "#dcfce7"},
                    {"range": [0.7, 1.3], "color": "#f8fafc"},
                    {"range": [1.3, 2], "color": "#fee2e2"},
                ],
            },
            title={"text": "PCR"},
        ),
        row=3,
        col=2,
    )

    for axis_id in [1, 2, 3, 4]:
        xref = "x" if axis_id == 1 else f"x{axis_id}"
        yref = "y domain" if axis_id == 1 else f"y{axis_id} domain"
        fig.add_shape(
            type="line",
            x0=spot,
            x1=spot,
            y0=0,
            y1=1,
            xref=xref,
            yref=yref,
            line=dict(color="#64748b", width=1, dash="dash"),
        )
    for row, col in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        fig.update_xaxes(range=[options_data["lower_bound"], options_data["upper_bound"]], row=row, col=col)

    fig.add_shape(
        type="line",
        x0=options_data["max_pain"],
        x1=options_data["max_pain"],
        y0=0,
        y1=1,
        xref="x2",
        yref="y2 domain",
        line=dict(color="#b42318", width=1, dash="dot"),
    )
    return apply_figure_style(fig, title=f"SPX Options Positioning ({options_data['expiry']})", height=970, showlegend=False)


def build_summary(
    ticker: str,
    price_df: pd.DataFrame,
    elder_df: pd.DataFrame | None,
    td_df: pd.DataFrame | None,
    stl_df: pd.DataFrame | None,
    market_data: dict[str, Any] | None,
    options_data: dict[str, Any] | None,
) -> DashboardSummary:
    elder_label = "Not loaded"
    if elder_df is not None and not elder_df.empty:
        elder_label, _ = elder_signal_label(int(elder_df["Impulse_State"].iloc[-1]), bool(elder_df["Long_Term_Up"].iloc[-1]))
    td_label = "Not loaded"
    if td_df is not None and not td_df.empty:
        td_label, _ = td_signal_label(td_df)
    stl_label = "Not loaded"
    if stl_df is not None and not stl_df.dropna(subset=["Cycle_Score"]).empty:
        stl_label, _ = stl_signal_label(float(stl_df.dropna(subset=["Cycle_Score"])["Cycle_Score"].iloc[-1]))
    market_label = "Not loaded"
    if market_data is not None:
        market_label, _ = market_data["status"]
    options_label, _ = options_signal_label(options_data)
    return DashboardSummary(
        ticker=ticker,
        last_close=float(price_df["Close"].iloc[-1]),
        daily_change=float(price_df["Close"].pct_change().iloc[-1]),
        monthly_change=pct_change_from_index(price_df["Close"], 21),
        quarter_change=pct_change_from_index(price_df["Close"], 63),
        elder_label=elder_label,
        td_label=td_label,
        stl_label=stl_label,
        market_label=market_label,
        options_label=options_label,
    )


def tone_from_return(value: float) -> str:
    if np.isnan(value):
        return "neutral"
    return "bull" if value >= 0 else "bear"


def format_pct(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value * 100:+.2f}%"


def render_header(
    summary: DashboardSummary,
    active_view: str,
    market_data: dict[str, Any] | None,
    options_data: dict[str, Any] | None,
    supertrend_data: pd.DataFrame | None,
    vix_fix_data: pd.DataFrame | None,
    squeeze_data: pd.DataFrame | None,
    nadaraya_data: pd.DataFrame | None,
    lorentzian_data: pd.DataFrame | None,
    cvd_data: dict[str, Any] | None,
) -> None:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    view_status_map = {
        "Elder Impulse": (summary.elder_label, "Momentum and trend filter status."),
        "TD Sequential": (summary.td_label, "Setup and countdown exhaustion context."),
        "Robust STL": (summary.stl_label, "Cycle stretch versus smoothed trend."),
        "SMC": ("Loaded", "Smart money zones and active structures."),
        "SuperTrend": (supertrend_signal_label(supertrend_data)[0], "ATR band flips and live regime line."),
        "Williams Vix Fix": (vix_fix_signal_label(vix_fix_data)[0], "Panic/complacency spikes versus recent extremes."),
        "Squeeze Momentum": (squeeze_signal_label(squeeze_data)[0], "Bollinger versus Keltner compression state."),
        "Nadaraya-Watson": (nadaraya_signal_label(nadaraya_data)[0], "Kernel regression trend and adaptive envelope."),
        "Lorentzian Classification": (lorentzian_signal_label(lorentzian_data)[0], "Nearest-neighbor style regime classification."),
        "CVD Divergence": (cvd_signal_label(cvd_data)[0], "Price swing versus cumulative delta disagreement."),
        "Market Pulse": (market_data["status"][0] if market_data else "Not loaded", "Cross-asset risk appetite backdrop."),
        "Options Flow": (options_signal_label(options_data)[0], f"SPX max pain {options_data['max_pain']:.2f}" if options_data else "SPX options unavailable"),
    }
    active_status, active_subtitle = view_status_map.get(active_view, ("Loaded", ""))
    active_tone = "neutral"
    if active_view == "Elder Impulse":
        active_tone = "bull" if "Bullish" in summary.elder_label else "bear" if "Bearish" in summary.elder_label else "neutral"
    elif active_view == "TD Sequential":
        active_tone = "bull" if "Buy" in summary.td_label else "bear" if "Sell" in summary.td_label else "neutral"
    elif active_view == "Robust STL":
        active_tone = "bull" if "Bull" in summary.stl_label else "bear" if "Bear" in summary.stl_label else "neutral"
    elif active_view == "SuperTrend":
        active_tone = supertrend_signal_label(supertrend_data)[1]
    elif active_view == "Williams Vix Fix":
        active_tone = vix_fix_signal_label(vix_fix_data)[1]
    elif active_view == "Squeeze Momentum":
        active_tone = squeeze_signal_label(squeeze_data)[1]
    elif active_view == "Nadaraya-Watson":
        active_tone = nadaraya_signal_label(nadaraya_data)[1]
    elif active_view == "Lorentzian Classification":
        active_tone = lorentzian_signal_label(lorentzian_data)[1]
    elif active_view == "CVD Divergence":
        active_tone = cvd_signal_label(cvd_data)[1]
    elif active_view == "Market Pulse" and market_data:
        active_tone = market_data["status"][1]
    elif active_view == "Options Flow":
        active_tone = options_signal_label(options_data)[1]

    st.markdown(
        f"""
        <div class="hero">
            <h1>{summary.ticker} · {active_view}</h1>
            <p>
                Selected module only. Controls apply immediately when changed.
                Last refresh: {current_time}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(4)
    with cards[0]:
        render_metric_card("Last Close", f"{summary.last_close:,.2f}", f"1D move {format_pct(summary.daily_change)}", tone_from_return(summary.daily_change))
    with cards[1]:
        render_metric_card("1M Return", format_pct(summary.monthly_change), "21 trading days", tone_from_return(summary.monthly_change))
    with cards[2]:
        render_metric_card("1Q Return", format_pct(summary.quarter_change), "63 trading days", tone_from_return(summary.quarter_change))
    with cards[3]:
        render_metric_card(active_view, active_status, active_subtitle, active_tone)

def render_sidebar(default_ticker: str) -> tuple[str, str, str, str | None, dict[str, Any] | None]:
    st.sidebar.subheader("Dashboard Controls")

    ticker = st.sidebar.text_input(
        "Ticker",
        value=st.session_state.get("ticker", default_ticker),
        key="ticker_input",
    ).strip().upper()
    period_options = ["1y", "2y", "3y", "5y"]
    default_period = st.session_state.get("period", "3y")
    period = st.sidebar.selectbox(
        "History window",
        options=period_options,
        index=period_options.index(default_period) if default_period in period_options else period_options.index("3y"),
        key="period_select",
    )
    last_core_view = st.session_state.get("last_core_view", CORE_DASHBOARD_VIEWS[0])
    core_view = st.sidebar.radio(
        "Chart View",
        options=CORE_DASHBOARD_VIEWS,
        index=CORE_DASHBOARD_VIEWS.index(last_core_view) if last_core_view in CORE_DASHBOARD_VIEWS else 0,
        key="dashboard_view_select",
    )
    st.sidebar.caption("Macro / options quick actions")
    pulse_col, options_col = st.sidebar.columns(2)
    market_clicked = pulse_col.button(SPECIAL_ACTION_VIEWS[0], use_container_width=True)
    options_clicked = options_col.button(SPECIAL_ACTION_VIEWS[1], use_container_width=True)
    force_refresh = st.sidebar.checkbox("Force refresh cached data", value=False, key="force_refresh_toggle")
    if force_refresh:
        st.cache_data.clear()
        clear_daily_payload_cache()
        st.session_state["force_refresh_toggle"] = False
    st.sidebar.caption("Examples: NVDA, QQQ, SPY, TSLA, 005930.KS, 035420.KQ, BTC-USD")
    st.sidebar.caption("Korean equities accept both Yahoo suffixes and plain 6-digit codes.")

    active_ticker = ticker or default_ticker
    active_period = period
    st.session_state["ticker"] = active_ticker
    st.session_state["period"] = active_period
    active_view = st.session_state.get("dashboard_view", core_view)
    if core_view != last_core_view:
        active_view = core_view
    if market_clicked:
        active_view = SPECIAL_ACTION_VIEWS[0]
    elif options_clicked:
        active_view = SPECIAL_ACTION_VIEWS[1]
    st.session_state["last_core_view"] = core_view
    st.session_state["dashboard_view"] = active_view

    selected_expiry = None
    spx_payload = fetch_spx_options_payload()
    spx_expiries = extract_spx_expiries(spx_payload)
    if spx_expiries:
        saved_expiry = st.session_state.get("selected_spx_expiry")
        default_index = spx_expiries.index(saved_expiry) if saved_expiry in spx_expiries else 0
        selected_expiry = st.sidebar.selectbox(
            "SPX option expiry",
            options=spx_expiries,
            index=default_index,
            key="selected_spx_expiry_select",
        )
        st.session_state["selected_spx_expiry"] = selected_expiry
    else:
        st.sidebar.caption("SPX option chain is temporarily unavailable.")
    return active_ticker, active_period, active_view, selected_expiry, spx_payload

def render_data_status(price_df: pd.DataFrame, price_source: str, price_symbol: str, stl_df: pd.DataFrame | None, stl_source: str, stl_symbol: str) -> None:
    with st.sidebar.expander("Data diagnostics", expanded=False):
        price_date = price_df.index[-1].strftime("%Y-%m-%d") if not price_df.empty else "n/a"
        st.markdown(f'<span class="data-pill">Price: {price_source}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="data-pill">Resolved: {price_symbol}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="data-pill">Latest bar: {price_date}</span>', unsafe_allow_html=True)
        if stl_df is not None and not stl_df.empty:
            stl_date = stl_df.index[-1].strftime("%Y-%m-%d")
            st.markdown(f'<span class="data-pill">STL: {stl_source}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="data-pill">STL symbol: {stl_symbol}</span>', unsafe_allow_html=True)
            st.markdown(f'<span class="data-pill">STL latest: {stl_date}</span>', unsafe_allow_html=True)
        elif stl_source == "Not loaded":
            st.caption("STL diagnostics load only when Robust STL is selected.")
        else:
            st.warning("STL source could not return usable close data for this ticker.")


def build_signal_stack(
    elder_df: pd.DataFrame | None,
    td_df: pd.DataFrame | None,
    stl_df: pd.DataFrame | None,
    smc_data: dict[str, Any] | None,
    market_data: dict[str, Any] | None,
    options_data: dict[str, Any] | None,
) -> pd.DataFrame:
    elder_text = "Not loaded"
    if elder_df is not None and not elder_df.empty:
        elder_text, _ = elder_signal_label(int(elder_df["Impulse_State"].iloc[-1]), bool(elder_df["Long_Term_Up"].iloc[-1]))
    td_text = "Not loaded"
    if td_df is not None and not td_df.empty:
        td_text, _ = td_signal_label(td_df)
    if stl_df is not None and not stl_df.dropna(subset=["Cycle_Score"]).empty:
        stl_text, _ = stl_signal_label(float(stl_df.dropna(subset=["Cycle_Score"])["Cycle_Score"].iloc[-1]))
    else:
        stl_text = "Not loaded" if stl_df is None else "Unavailable"
    smc_text = "Not loaded"
    if smc_data is not None:
        smc_text, _ = smc_signal_label(smc_data)
    market_text = "Not loaded"
    if market_data is not None:
        market_text, _ = market_data["status"]
    options_text, _ = options_signal_label(options_data)
    return pd.DataFrame(
        [
            {"Model": "Elder Impulse", "Signal": elder_text, "What it means": "Short-term momentum aligned with trend filter."},
            {"Model": "TD Sequential", "Signal": td_text, "What it means": "Setup and countdown exhaustion context."},
            {"Model": "Robust STL", "Signal": stl_text, "What it means": "Cycle stretch versus smoothed trend."},
            {"Model": "SMC Bias", "Signal": smc_text, "What it means": "Location relative to value area and active zones."},
            {"Model": "Macro Pulse", "Signal": market_text, "What it means": "Risk appetite backdrop from cross-asset internals."},
            {"Model": "Options", "Signal": options_text, "What it means": "Dealer positioning and put/call flow read."},
        ]
    )


def main() -> None:
    configure_page()
    apply_custom_style()
    ticker, period, active_view, selected_expiry, spx_payload = render_sidebar(default_ticker="NVDA")

    with st.spinner(f"Loading data for {ticker}..."):
        price_df, price_source, price_symbol = download_price_data(ticker, period=period)
    if price_df.empty:
        st.error("No price history was returned. Check the ticker format and try again.")
        st.stop()

    need_elder = active_view == "Elder Impulse"
    need_td = active_view == "TD Sequential"
    need_stl = active_view == "Robust STL"
    need_smc = active_view == "SMC"
    need_supertrend = active_view == "SuperTrend"
    need_vix_fix = active_view == "Williams Vix Fix"
    need_squeeze = active_view == "Squeeze Momentum"
    need_nadaraya = active_view == "Nadaraya-Watson"
    need_lorentzian = active_view == "Lorentzian Classification"
    need_cvd = active_view == "CVD Divergence"
    need_market = active_view == "Market Pulse"
    need_options = active_view == "Options Flow"

    with st.spinner("Computing dashboards..."):
        elder_df = compute_elder_impulse(price_df) if need_elder else None
        td_df = compute_td_sequential(price_df) if need_td else None
        stl_source_df = pd.DataFrame()
        stl_source = "Not loaded"
        stl_symbol = "n/a"
        if need_stl:
            stl_source_df, stl_source, stl_symbol = download_stl_data(ticker)
        stl_df = compute_stl_cycle(stl_source_df) if need_stl and not stl_source_df.empty else None
        smc_data = compute_smc(price_df) if need_smc else None
        supertrend_data = compute_supertrend(price_df) if need_supertrend else None
        vix_fix_data = compute_williams_vix_fix(price_df) if need_vix_fix else None
        squeeze_data = compute_squeeze_momentum(price_df) if need_squeeze else None
        nadaraya_data = compute_nadaraya_watson(price_df) if need_nadaraya else None
        lorentzian_data = compute_lorentzian_classification(price_df) if need_lorentzian else None
        cvd_data = compute_cvd_divergence(price_df) if need_cvd else None
        market_data = compute_market_fear_greed() if need_market else None
        options_data = compute_options_analytics(expiry=selected_expiry, payload=spx_payload) if need_options else None

    summary = build_summary(ticker, price_df, elder_df, td_df, stl_df, market_data, options_data)
    render_header(summary, active_view, market_data, options_data, supertrend_data, vix_fix_data, squeeze_data, nadaraya_data, lorentzian_data, cvd_data)
    render_data_status(price_df, price_source, price_symbol, stl_df, stl_source, stl_symbol)
    st.markdown(
        f'<div class="section-note">Active view: <strong>{active_view}</strong>. Core indicators are selected from the chart picker, while Market Pulse and Options Flow run from their own quick-action buttons.</div>',
        unsafe_allow_html=True,
    )

    if active_view == "Elder Impulse":
        st.plotly_chart(build_elder_figure(elder_df), use_container_width=True)
    elif active_view == "TD Sequential":
        st.plotly_chart(build_td_figure(td_df), use_container_width=True)
    elif active_view == "Robust STL":
        if stl_df is None or stl_df.dropna(subset=["Trend", "Cycle_Score"]).empty:
            st.info("Robust STL could not build a valid cycle series for this ticker after trying both FinanceDataReader and Yahoo Finance.")
        else:
            st.plotly_chart(build_stl_figure(stl_df), use_container_width=True)
            st.caption(f"STL source: {stl_source} via `{stl_symbol}`")
    elif active_view == "SMC":
        st.plotly_chart(build_smc_figure(smc_data), use_container_width=True)
    elif active_view == "SuperTrend":
        st.plotly_chart(build_supertrend_figure(supertrend_data), use_container_width=True)
    elif active_view == "Williams Vix Fix":
        st.plotly_chart(build_vix_fix_figure(vix_fix_data), use_container_width=True)
    elif active_view == "Squeeze Momentum":
        st.plotly_chart(build_squeeze_figure(squeeze_data), use_container_width=True)
    elif active_view == "Nadaraya-Watson":
        st.plotly_chart(build_nadaraya_figure(nadaraya_data), use_container_width=True)
    elif active_view == "Lorentzian Classification":
        st.plotly_chart(build_lorentzian_figure(lorentzian_data), use_container_width=True)
    elif active_view == "CVD Divergence":
        st.plotly_chart(build_cvd_figure(cvd_data), use_container_width=True)
    elif active_view == "Market Pulse":
        st.plotly_chart(build_market_figure(market_data), use_container_width=True)
    elif active_view == "Options Flow":
        if not options_data:
            st.info("SPX option data could not be loaded from the CBOE delayed quotes feed.")
        else:
            option_cards = st.columns(4)
            with option_cards[0]:
                render_metric_card("Underlying", f"{options_data['underlying']} {options_data['spot']:,.1f}", options_data["expiry"], "neutral")
            with option_cards[1]:
                render_metric_card("Put/Call Volume", f"{options_data['put_call_ratio']:.2f}", "Above 1 implies defensive demand", "accent")
            with option_cards[2]:
                render_metric_card("Max Pain", f"{options_data['max_pain']:.2f}", "Strike with minimum aggregate pain", "neutral")
            with option_cards[3]:
                render_metric_card("Net GEX", f"{options_data['net_gex'] / 1e9:,.2f}B", "Positive often dampens volatility", "bull" if options_data["net_gex"] >= 0 else "bear")
            st.plotly_chart(build_options_figure(options_data, options_data["spot"]), use_container_width=True)

    st.caption("Data sources: Yahoo Finance, FinanceDataReader, CBOE delayed quotes. Options analytics depend on listed option availability and delayed data quality.")


if __name__ == "__main__":
    main()
