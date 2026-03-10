from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from scipy.stats import norm
from statsmodels.tsa.seasonal import STL


st.set_page_config(
    page_title="Quant Fusion Dashboard",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 239, 213, 0.95), transparent 30%),
                radial-gradient(circle at top right, rgba(186, 230, 253, 0.55), transparent 24%),
                linear-gradient(180deg, #fffdf8 0%, #f6efe2 45%, #fffaf2 100%);
        }
        .block-container {
            max-width: 1480px;
            padding-top: 1.4rem;
            padding-bottom: 2.2rem;
        }
        .hero {
            background: linear-gradient(125deg, #0f172a 0%, #134e4a 45%, #f59e0b 100%);
            border-radius: 28px;
            padding: 2rem 2.2rem;
            color: #fffdf8;
            margin-bottom: 1.1rem;
            box-shadow: 0 26px 70px rgba(15, 23, 42, 0.18);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            letter-spacing: -0.03em;
        }
        .hero p {
            margin: 0.55rem 0 0;
            color: rgba(255, 250, 242, 0.9);
            max-width: 58rem;
            line-height: 1.55;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.16);
            border-radius: 22px;
            padding: 1rem 1.05rem;
            min-height: 7.8rem;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.06);
            backdrop-filter: blur(12px);
        }
        .metric-card.bull {
            border-top: 4px solid #0f766e;
        }
        .metric-card.bear {
            border-top: 4px solid #b42318;
        }
        .metric-card.neutral {
            border-top: 4px solid #64748b;
        }
        .metric-card.accent {
            border-top: 4px solid #dd6b20;
        }
        .metric-label {
            color: #52606d;
            font-size: 0.86rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: #102a43;
            font-size: 1.72rem;
            font-weight: 800;
            line-height: 1.1;
            margin-top: 0.5rem;
        }
        .metric-subtitle {
            color: #52606d;
            font-size: 0.92rem;
            margin-top: 0.4rem;
            line-height: 1.45;
        }
        .section-note {
            color: #52606d;
            font-size: 0.96rem;
            margin-top: -0.2rem;
            margin-bottom: 0.7rem;
        }
        div[data-testid="stTabs"] button {
            font-weight: 700;
        }
        div[data-testid="stSidebar"] {
            background: rgba(255, 250, 240, 0.9);
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


@st.cache_data(ttl=3600, show_spinner=False)
def download_price_data(ticker: str, period: str = "3y") -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    df = flatten_columns(df)
    if df.empty:
        return df

    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.index = pd.to_datetime(df.index)
    return df.dropna(subset=["Open", "High", "Low", "Close"]).copy()


def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


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
    fig.add_trace(go.Scatter(x=view.index, y=view["MA200"], name="MA 200", line=dict(color="#1f2937", width=1.4, dash="dot")), row=1, col=1)
    fig.add_trace(
        go.Bar(x=view.index, y=view["Volume"], marker_color=view["Volume_Color"], name="Volume", opacity=0.55),
        row=2,
        col=1,
    )
    fig.update_layout(
        height=680,
        margin=dict(l=24, r=24, t=48, b=16),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        legend=dict(orientation="h", y=1.05, x=0),
        xaxis_rangeslider_visible=False,
        title="Price Structure Snapshot",
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)")
    fig.update_xaxes(showgrid=False)
    return fig


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
        return "Bullish Impulse", "bull"
    if state == -1 and not long_term_up:
        return "Bearish Impulse", "bear"
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
    fig.add_trace(
        go.Scatter(x=view.index, y=view["Buy_Signal"], mode="markers", marker=dict(symbol="triangle-up", size=11, color="#0f766e"), name="Buy trigger"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=view.index, y=view["Sell_Signal"], mode="markers", marker=dict(symbol="triangle-down", size=11, color="#b42318"), name="Sell trigger"),
        row=1,
        col=1,
    )
    fig.add_trace(go.Bar(x=view.index, y=view["MACD_Hist"], marker_color=view["Impulse_Color"].tolist(), name="MACD histogram"), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#64748b", row=2, col=1)
    fig.update_layout(
        height=720,
        margin=dict(l=24, r=24, t=48, b=16),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        title="Elder Impulse and Trend Filter",
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.22)")
    return fig


def compute_td_sequential(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["Buy_Setup"] = 0
    work["Sell_Setup"] = 0
    work["Buy_Countdown"] = 0
    work["Sell_Countdown"] = 0

    active_buy_countdown = False
    active_sell_countdown = False
    buy_count = 0
    sell_count = 0
    close_values = work["Close"].to_numpy(dtype=float)

    for idx in range(4, len(work)):
        work.iat[idx, work.columns.get_loc("Buy_Setup")] = int(work.iat[idx - 1, work.columns.get_loc("Buy_Setup")]) + 1 if close_values[idx] < close_values[idx - 4] else 0
        work.iat[idx, work.columns.get_loc("Sell_Setup")] = int(work.iat[idx - 1, work.columns.get_loc("Sell_Setup")]) + 1 if close_values[idx] > close_values[idx - 4] else 0

        if work["Buy_Setup"].iat[idx] == 9:
            active_buy_countdown = True
            buy_count = 0
        if work["Sell_Setup"].iat[idx] == 9:
            active_sell_countdown = True
            sell_count = 0

        if active_buy_countdown and close_values[idx] <= close_values[idx - 2]:
            buy_count += 1
            work.iat[idx, work.columns.get_loc("Buy_Countdown")] = buy_count
            if buy_count == 13:
                active_buy_countdown = False

        if active_sell_countdown and close_values[idx] >= close_values[idx - 2]:
            sell_count += 1
            work.iat[idx, work.columns.get_loc("Sell_Countdown")] = sell_count
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
        return "Buy Countdown 13", "bull"
    if sell_cd == 13:
        return "Sell Countdown 13", "bear"
    if buy_setup >= 7:
        return f"Buy Setup {buy_setup}", "bull"
    if sell_setup >= 7:
        return f"Sell Setup {sell_setup}", "bear"
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
        go.Scatter(x=view.index, y=view["Senkou_Span_B"], mode="lines", line=dict(color="rgba(180,35,24,0.1)"), fill="tonexty", fillcolor="rgba(245,158,11,0.10)", name="Ichimoku cloud"),
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

    fig.update_layout(
        height=760,
        margin=dict(l=24, r=24, t=48, b=16),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        title="TD Sequential with Trend Context",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.05, x=0),
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.22)")
    return fig

def compute_stl_cycle(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    log_close = np.log(work["Close"])
    stl = STL(log_close, period=5, trend=31, robust=True).fit()
    work["Trend"] = np.exp(stl.trend)
    work["Residual"] = stl.resid
    work["RSI"] = calc_rsi(work["Close"])

    resid_vol = work["Residual"].rolling(20).std().replace(0, np.nan).bfill()
    z_score = (work["Residual"] / resid_vol).clip(-3, 3)
    resid_score = ((z_score + 3) / 6) * 100
    work["Cycle_Score"] = ((0.6 * resid_score) + (0.4 * work["RSI"])).rolling(3).mean().clip(0, 100)
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
                colorbar=dict(title="Cycle"),
            ),
            line=dict(color="rgba(15,23,42,0.18)", width=1.2),
            name="Price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=view.index, y=view["Trend"], name="STL trend", line=dict(color="#0f766e", width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=view.index, y=view["Cycle_Score"], name="Cycle score", line=dict(color="#f59e0b", width=1.8)), row=2, col=1)
    fig.add_hrect(y0=0, y1=10, fillcolor="rgba(37,99,235,0.12)", line_width=0, row=2, col=1)
    fig.add_hrect(y0=90, y1=100, fillcolor="rgba(180,35,24,0.12)", line_width=0, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#64748b", row=2, col=1)
    fig.update_layout(
        height=720,
        margin=dict(l=24, r=24, t=48, b=16),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        title="Robust STL Cycle Dashboard",
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.22)")
    return fig


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
    work["TR"] = np.maximum(work["High"] - work["Low"], np.maximum((work["High"] - work["Close"].shift(1)).abs(), (work["Low"] - work["Close"].shift(1)).abs()))
    work["ATR"] = work["TR"].rolling(14).mean()
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

    current_price = float(view["Close"].iloc[-1])
    zones_table = []
    for item in active_bull_ob + active_bear_ob + active_bull_fvg + active_bear_fvg:
        midpoint = (item["top"] + item["bottom"]) / 2
        zones_table.append({"Zone": item["label"], "Bias": item["type"].upper(), "Range": f"{item['bottom']:.2f} - {item['top']:.2f}", "Distance %": round((current_price / midpoint - 1) * 100, 2)})

    return {
        "view": view,
        "poc_price": poc_price,
        "levels": sorted(swings, key=lambda item: item["date"], reverse=True)[:4],
        "zones_table": pd.DataFrame(zones_table),
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
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25], vertical_spacing=0.05)
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
    x1 = view.index[-1] + timedelta(days=14)
    for zones, color in [
        (smc_data["active_bull_ob"], "rgba(15,118,110,0.18)"),
        (smc_data["active_bear_ob"], "rgba(180,35,24,0.18)"),
        (smc_data["active_bull_fvg"], "rgba(37,99,235,0.14)"),
        (smc_data["active_bear_fvg"], "rgba(245,158,11,0.16)"),
    ]:
        for zone in zones:
            fig.add_hrect(y0=zone["bottom"], y1=zone["top"], x0=max(zone["date"], x0), x1=x1, fillcolor=color, line_width=0, row=1, col=1)

    if not np.isnan(smc_data["poc_price"]):
        fig.add_hline(y=smc_data["poc_price"], line_dash="dash", line_color="#334155", row=1, col=1, annotation_text="POC", annotation_position="top right")

    for level in smc_data["levels"]:
        color = "#0f766e" if level["type"] == "L" else "#b42318"
        fig.add_hline(y=level["price"], line_dash="dot", line_color=color, opacity=0.35, row=1, col=1)

    fig.update_layout(
        height=760,
        margin=dict(l=24, r=24, t=48, b=16),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        title="Smart Money Concepts",
        xaxis_rangeslider_visible=False,
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.22)")
    return fig

def get_probability(series: pd.Series, window: int, inverse: bool = False) -> pd.Series:
    roll_mean = series.rolling(window=window).mean()
    roll_std = series.rolling(window=window).std()
    z_score = (series - roll_mean) / roll_std.replace(0, np.nan)
    prob = z_score.apply(lambda value: norm.cdf(value) if not np.isnan(value) else np.nan)
    return 1 - prob if inverse else prob


@st.cache_data(ttl=7200, show_spinner=False)
def compute_market_fear_greed() -> dict[str, Any]:
    tickers = ["SPY", "^VIX", "HYG", "IEF", "RSP", "XLY", "XLP", "UUP"]
    raw = yf.download(tickers, period="6y", progress=False, auto_adjust=False)
    close_df = raw["Close"].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
    close_df = close_df.ffill().dropna()

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
    fig.add_hline(y=0.8, line_dash="dot", line_color="#b42318", row=1, col=2)
    fig.add_hline(y=0.2, line_dash="dot", line_color="#0f766e", row=1, col=2)
    fig.add_hline(y=0.5, line_dash="dot", line_color="#64748b", row=2, col=2)
    fig.update_layout(
        height=820,
        margin=dict(l=24, r=24, t=64, b=16),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        legend=dict(orientation="h", y=1.08, x=0),
        title="Macro Fear and Greed Dashboard",
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.22)")
    return fig


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
        irx = flatten_columns(yf.download("^IRX", period="5d", progress=False, auto_adjust=False))
        return float(irx["Close"].iloc[-1] / 100) if not irx.empty else 0.04
    except Exception:
        return 0.04


@st.cache_data(ttl=3600, show_spinner=False)
def get_option_expiries(ticker: str) -> list[str]:
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def get_options_chain(ticker: str, expiry: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    chain = yf.Ticker(ticker).option_chain(expiry)
    return chain.calls.copy(), chain.puts.copy()


def compute_options_analytics(ticker: str, spot: float, expiry: str | None = None) -> dict[str, Any] | None:
    expiries = get_option_expiries(ticker)
    if not expiries:
        return None
    selected_expiry = expiry if expiry in expiries else expiries[0]
    try:
        calls, puts = get_options_chain(ticker, selected_expiry)
    except Exception:
        return None
    if calls.empty and puts.empty:
        return None

    time_to_expiry = max((pd.Timestamp(selected_expiry) - pd.Timestamp.now().normalize()).days / 365.25, 0.0027)
    risk_free = get_risk_free_rate()
    frames = []
    for option_type, chain in [("call", calls), ("put", puts)]:
        if chain.empty:
            continue
        chain = chain.copy()
        chain["openInterest"] = pd.to_numeric(chain["openInterest"], errors="coerce").fillna(0)
        chain["volume"] = pd.to_numeric(chain["volume"], errors="coerce").fillna(0)
        chain["impliedVolatility"] = pd.to_numeric(chain["impliedVolatility"], errors="coerce")
        fallback_iv = chain["impliedVolatility"].replace(0, np.nan).dropna().median()
        chain["impliedVolatility"] = chain["impliedVolatility"].replace(0, np.nan).fillna(0.35 if np.isnan(fallback_iv) else fallback_iv)
        chain["option_type"] = option_type
        frames.append(chain[["strike", "openInterest", "volume", "impliedVolatility", "option_type"]])

    options_df = pd.concat(frames, ignore_index=True)
    options_df = options_df[(options_df["strike"] >= spot * 0.85) & (options_df["strike"] <= spot * 1.15)].copy()
    if options_df.empty:
        return None

    multiplier = spot * spot * 0.01 * 100
    rows = []
    for _, row in options_df.iterrows():
        greeks = bs_greeks(spot, float(row["strike"]), time_to_expiry, risk_free, 0.0, float(row["impliedVolatility"]), str(row["option_type"]))
        sign = 1 if row["option_type"] == "call" else -1
        rows.append({
            "strike": float(row["strike"]),
            "type": row["option_type"],
            "openInterest": float(row["openInterest"]),
            "volume": float(row["volume"]),
            "gex": sign * greeks["gamma"] * row["openInterest"] * multiplier,
            "vanna": sign * greeks["vanna"] * row["openInterest"] * spot * 100,
            "charm": sign * greeks["charm"] * row["openInterest"] * 100,
        })

    expo_df = pd.DataFrame(rows)
    expo_df["call_oi"] = np.where(expo_df["type"] == "call", expo_df["openInterest"], 0.0)
    expo_df["put_oi"] = np.where(expo_df["type"] == "put", expo_df["openInterest"], 0.0)
    strike_view = expo_df.groupby("strike", as_index=False).agg(gex=("gex", "sum"), vanna=("vanna", "sum"), charm=("charm", "sum"), call_oi=("call_oi", "sum"), put_oi=("put_oi", "sum")).sort_values("strike")

    total_call_volume = float(expo_df.loc[expo_df["type"] == "call", "volume"].sum())
    total_put_volume = float(expo_df.loc[expo_df["type"] == "put", "volume"].sum())
    put_call_ratio = total_put_volume / total_call_volume if total_call_volume else np.nan
    strike_view["pain"] = [((strike - strike_view["strike"]).clip(lower=0) * strike_view["call_oi"]).sum() + ((strike_view["strike"] - strike).clip(lower=0) * strike_view["put_oi"]).sum() for strike in strike_view["strike"]]
    max_pain = float(strike_view.loc[strike_view["pain"].idxmin(), "strike"])

    return {
        "expiry": selected_expiry,
        "expiries": expiries,
        "strike_view": strike_view,
        "put_call_ratio": put_call_ratio,
        "max_pain": max_pain,
        "net_gex": float(strike_view["gex"].sum()),
    }


def options_signal_label(options_data: dict[str, Any] | None) -> tuple[str, str]:
    if not options_data:
        return "No listed options", "neutral"
    pcr = 1.0 if np.isnan(options_data["put_call_ratio"]) else options_data["put_call_ratio"]
    net_gex = options_data["net_gex"]
    if pcr > 1.3 and net_gex < 0:
        return "Defensive flow", "bear"
    if pcr < 0.8 and net_gex > 0:
        return "Supportive dealer flow", "bull"
    return "Mixed positioning", "neutral"


def build_options_figure(options_data: dict[str, Any], spot: float) -> go.Figure:
    view = options_data["strike_view"]
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Dealer GEX", "Open Interest", "Pain Profile", "Vanna and Charm"), vertical_spacing=0.12, horizontal_spacing=0.08)
    fig.add_trace(go.Bar(x=view["strike"], y=view["gex"] / 1e6, marker_color="#2563eb", name="GEX (M)"), row=1, col=1)
    fig.add_trace(go.Bar(x=view["strike"], y=view["call_oi"], marker_color="#0f766e", name="Call OI"), row=1, col=2)
    fig.add_trace(go.Bar(x=view["strike"], y=-view["put_oi"], marker_color="#b42318", name="Put OI"), row=1, col=2)
    fig.add_trace(go.Scatter(x=view["strike"], y=view["pain"], fill="tozeroy", line=dict(color="#dd6b20", width=2), name="Pain"), row=2, col=1)
    fig.add_trace(go.Bar(x=view["strike"], y=view["vanna"] / 1e6, marker_color="#0f766e", name="Vanna (M)"), row=2, col=2)
    fig.add_trace(go.Scatter(x=view["strike"], y=view["charm"] / 1e4, line=dict(color="#111827", width=2), name="Charm (x10k)"), row=2, col=2)
    for row in [1, 2]:
        for col in [1, 2]:
            fig.add_vline(x=spot, line_dash="dash", line_color="#64748b", opacity=0.7, row=row, col=col)
    fig.add_vline(x=options_data["max_pain"], line_dash="dot", line_color="#b42318", row=2, col=1)
    fig.update_layout(
        height=820,
        margin=dict(l=24, r=24, t=64, b=16),
        paper_bgcolor="rgba(255,255,255,0.0)",
        plot_bgcolor="rgba(255,255,255,0.0)",
        title=f"Options Positioning ({options_data['expiry']})",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.22)")
    return fig

def build_summary(
    ticker: str,
    price_df: pd.DataFrame,
    elder_df: pd.DataFrame,
    td_df: pd.DataFrame,
    stl_df: pd.DataFrame,
    market_data: dict[str, Any],
    options_data: dict[str, Any] | None,
) -> DashboardSummary:
    elder_label, _ = elder_signal_label(int(elder_df["Impulse_State"].iloc[-1]), bool(elder_df["Long_Term_Up"].iloc[-1]))
    td_label, _ = td_signal_label(td_df)
    stl_label, _ = stl_signal_label(float(stl_df["Cycle_Score"].iloc[-1]))
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


def render_header(summary: DashboardSummary, market_data: dict[str, Any], options_data: dict[str, Any] | None) -> None:
    market_status, market_tone = market_data["status"]
    options_status, options_tone = options_signal_label(options_data)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    st.markdown(
        f"""
        <div class="hero">
            <h1>{summary.ticker} Quant Fusion Dashboard</h1>
            <p>
                One-click research board for price structure, Elder Impulse, TD Sequential,
                STL cycle scoring, smart money zones, macro sentiment, and options flow.
                Last refresh: {current_time}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(6)
    with cards[0]:
        render_metric_card("Last Close", f"{summary.last_close:,.2f}", f"1D move {format_pct(summary.daily_change)}", tone_from_return(summary.daily_change))
    with cards[1]:
        render_metric_card("1M Return", format_pct(summary.monthly_change), "21 trading days", tone_from_return(summary.monthly_change))
    with cards[2]:
        render_metric_card("1Q Return", format_pct(summary.quarter_change), "63 trading days", tone_from_return(summary.quarter_change))
    with cards[3]:
        elder_tone = "bull" if "Bullish" in summary.elder_label else "bear" if "Bearish" in summary.elder_label else "neutral"
        render_metric_card("Elder", summary.elder_label, summary.td_label, elder_tone)
    with cards[4]:
        render_metric_card("STL Cycle", summary.stl_label, market_status, market_tone)
    with cards[5]:
        subtitle = f"Max pain {options_data['max_pain']:.2f}" if options_data else "No option chain"
        render_metric_card("Options Flow", options_status, subtitle, options_tone)


def render_sidebar(default_ticker: str) -> tuple[str, str, str | None]:
    with st.sidebar.form("query_form"):
        st.subheader("Dashboard Controls")
        ticker = st.text_input("Ticker", value=st.session_state.get("ticker", default_ticker)).strip().upper()
        period_options = ["1y", "2y", "3y", "5y"]
        default_period = st.session_state.get("period", "3y")
        period = st.selectbox("History window", options=period_options, index=period_options.index(default_period) if default_period in period_options else 2)
        st.caption("Examples: NVDA, QQQ, SPY, TSLA, 005930.KS, BTC-USD")
        submitted = st.form_submit_button("Run Dashboard", use_container_width=True)

    if submitted:
        st.session_state["ticker"] = ticker or default_ticker
        st.session_state["period"] = period

    active_ticker = st.session_state.get("ticker", default_ticker)
    active_period = st.session_state.get("period", "3y")
    expiry = None
    expiries = st.session_state.get("option_expiries")
    if expiries:
        saved_expiry = st.session_state.get("selected_expiry")
        default_index = expiries.index(saved_expiry) if saved_expiry in expiries else 0
        expiry = st.sidebar.selectbox("Option expiry", options=expiries, index=default_index)
        st.session_state["selected_expiry"] = expiry
    return active_ticker, active_period, expiry


def main() -> None:
    apply_custom_style()
    ticker, period, selected_expiry = render_sidebar(default_ticker="NVDA")

    with st.spinner(f"Loading data for {ticker}..."):
        price_df = download_price_data(ticker, period=period)
    if price_df.empty:
        st.error("No price history was returned. Check the ticker format and try again.")
        st.stop()

    with st.spinner("Computing dashboards..."):
        elder_df = compute_elder_impulse(price_df)
        td_df = compute_td_sequential(price_df)
        stl_df = compute_stl_cycle(price_df)
        smc_data = compute_smc(price_df)
        market_data = compute_market_fear_greed()
        options_data = compute_options_analytics(ticker=ticker, spot=float(price_df["Close"].iloc[-1]), expiry=selected_expiry)

    st.session_state["option_expiries"] = options_data["expiries"] if options_data else get_option_expiries(ticker)

    summary = build_summary(ticker, price_df, elder_df, td_df, stl_df, market_data, options_data)
    render_header(summary, market_data, options_data)
    st.markdown('<div class="section-note">The overview tab gives a fast read. The remaining tabs keep each model isolated so you can validate the signal instead of trusting a single composite number.</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "Elder Impulse", "TD Sequential", "Robust STL", "SMC", "Market Pulse", "Options Flow"])

    with tabs[0]:
        overview_cols = st.columns([1.6, 1])
        with overview_cols[0]:
            st.plotly_chart(compute_overview_figure(price_df), use_container_width=True)
        with overview_cols[1]:
            elder_text, _ = elder_signal_label(int(elder_df["Impulse_State"].iloc[-1]), bool(elder_df["Long_Term_Up"].iloc[-1]))
            td_text, _ = td_signal_label(td_df)
            stl_text, _ = stl_signal_label(float(stl_df["Cycle_Score"].iloc[-1]))
            smc_text, _ = smc_signal_label(smc_data)
            market_text, _ = market_data["status"]
            options_text, _ = options_signal_label(options_data)
            st.subheader("Signal Stack")
            st.write(f"**Elder Impulse:** {elder_text}")
            st.write(f"**TD Sequential:** {td_text}")
            st.write(f"**STL Cycle:** {stl_text}")
            st.write(f"**SMC Bias:** {smc_text}")
            st.write(f"**Macro Pulse:** {market_text}")
            st.write(f"**Options Positioning:** {options_text}")

            latest = price_df.tail(10).copy()
            latest["Return %"] = latest["Close"].pct_change().mul(100)
            st.subheader("Recent Tape")
            st.dataframe(latest[["Open", "High", "Low", "Close", "Volume", "Return %"]].round({"Open": 2, "High": 2, "Low": 2, "Close": 2, "Return %": 2}), use_container_width=True, hide_index=False)

    with tabs[1]:
        st.plotly_chart(build_elder_figure(elder_df), use_container_width=True)

    with tabs[2]:
        st.plotly_chart(build_td_figure(td_df), use_container_width=True)

    with tabs[3]:
        st.plotly_chart(build_stl_figure(stl_df), use_container_width=True)

    with tabs[4]:
        smc_cols = st.columns([1.8, 1])
        with smc_cols[0]:
            st.plotly_chart(build_smc_figure(smc_data), use_container_width=True)
        with smc_cols[1]:
            st.subheader("Active Zones")
            if smc_data["zones_table"].empty:
                st.info("No active order block or FVG zone was detected in the current window.")
            else:
                st.dataframe(smc_data["zones_table"], use_container_width=True, hide_index=True)

            st.subheader("Value Area")
            poc_price = smc_data["poc_price"]
            if np.isnan(poc_price):
                st.write("POC could not be estimated for the current sample.")
            else:
                st.write(f"Point of control: **{poc_price:,.2f}**")
                st.write(f"Last close vs POC: **{(price_df['Close'].iloc[-1] / poc_price - 1) * 100:+.2f}%**")

    with tabs[5]:
        st.plotly_chart(build_market_figure(market_data), use_container_width=True)

    with tabs[6]:
        if not options_data:
            st.info("No option chain is available for this ticker on Yahoo Finance.")
        else:
            option_cards = st.columns(4)
            with option_cards[0]:
                render_metric_card("Expiry", options_data["expiry"], "Current selection", "neutral")
            with option_cards[1]:
                render_metric_card("Put/Call Volume", f"{options_data['put_call_ratio']:.2f}", "Above 1 implies defensive demand", "accent")
            with option_cards[2]:
                render_metric_card("Max Pain", f"{options_data['max_pain']:.2f}", "Strike with minimum aggregate pain", "neutral")
            with option_cards[3]:
                render_metric_card("Net GEX", f"{options_data['net_gex'] / 1e6:,.1f}M", "Positive often dampens volatility", "bull" if options_data["net_gex"] >= 0 else "bear")
            st.plotly_chart(build_options_figure(options_data, float(price_df["Close"].iloc[-1])), use_container_width=True)

    st.caption("Data source: Yahoo Finance. Options analytics depend on listed option availability and delayed data.")


if __name__ == "__main__":
    main()
