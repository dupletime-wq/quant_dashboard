from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
from pathlib import Path
import re
import sys
from typing import Any

import FinanceDataReader as fdr
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from matplotlib.patches import Rectangle, Wedge
from matplotlib.ticker import MaxNLocator
try:
    from setuptools import _distutils as setuptools_distutils
    sys.modules.setdefault("distutils", setuptools_distutils)
    if hasattr(setuptools_distutils, "version"):
        sys.modules.setdefault("distutils.version", setuptools_distutils.version)
except Exception:
    pass
try:
    import pandas_datareader.data as pdr_web
    PDR_IMPORT_ERROR: str | None = None
except Exception as exc:
    pdr_web = None
    PDR_IMPORT_ERROR = str(exc)
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
CHART_SURFACE_COLOR = "#fffaf4"
PLOTLY_CONFIG = {"responsive": True, "displaylogo": False}
KOREAN_SUFFIX_PATTERN = re.compile(r"^(?P<code>\d{6})\.(KS|KQ)$", re.IGNORECASE)
CORE_DASHBOARD_VIEWS = [
    "Elder Impulse",
    "TD Sequential",
    "Robust STL",
    "SMC",
    "SuperTrend",
    "Williams Vix Fix",
    "Squeeze Momentum",
]
SPECIAL_ACTION_BUTTONS = [
    {"label": "Fear & Greed", "view": "Market Pulse", "caption": "Cross-asset risk appetite"},
    {"label": "Canary", "view": "Canary Momentum", "caption": "Risk-on/off rotation"},
    {"label": "Option Gamma", "view": "Options Flow", "caption": "SPX dealer positioning"},
    {"label": "ETF Sortino", "view": "ETF Sortino Leadership", "caption": "ETF leadership and sector risk"},
    {"label": "Fed Watch", "view": "Fed Watch", "caption": "Fed/Treasury liquidity plumbing"},
]
SPECIAL_ACTION_LABELS = {item["view"]: item["label"] for item in SPECIAL_ACTION_BUTTONS}
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
FED_WATCH_PERIOD_OFFSETS = {
    "1y": pd.DateOffset(years=1),
    "2y": pd.DateOffset(years=2),
    "3y": pd.DateOffset(years=3),
    "5y": pd.DateOffset(years=5),
}
FED_WATCH_SERIES_SPECS = {
    "SOFR": {"fred": "SOFR", "scale": 1.0, "unit": "pct"},
    "IORB": {"fred": "IORB", "scale": 1.0, "unit": "pct"},
    "IOER": {"fred": "IOER", "scale": 1.0, "unit": "pct"},
    "US10Y": {"fred": "DGS10", "scale": 1.0, "unit": "pct"},
    "US3M": {"fred": "DGS3MO", "scale": 1.0, "unit": "pct"},
    "FED_Treasuries": {"fred": "WSHOTSL", "scale": 0.001, "unit": "billions"},
    "Bank_Treasuries": {"fred": "USGSEC", "scale": 1.0, "unit": "billions"},
    "SOFR_Vol": {"fred": "SOFRVOL", "scale": 1.0, "unit": "billions"},
    "WALCL": {"fred": "WALCL", "scale": 0.001, "unit": "billions"},
    "TGA": {"fred": "WTREGEN", "scale": 0.001, "unit": "billions"},
    "Reserves": {"fred": "WRESBAL", "scale": 0.001, "unit": "billions"},
    "ON_RRP": {"fred": "RRPONTSYD", "scale": 1.0, "unit": "billions"},
}
FED_WATCH_CACHE_KEYS = tuple(f"fed_watch_{period}" for period in FED_WATCH_PERIOD_OFFSETS)


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


LOOKBACK_PERIOD = "2y"
REALTIME_LAGS = [21, 63, 126, 252]
MONTHLY_LAGS = [1, 3, 6, 12]
CANARY_TICKERS = ["TIP", "QQQ", "SPY", "VEA", "VWO", "BND"]
ATTACK_TICKERS = ["QQQ", "SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT"]
SAFE_ASSET = "BIL"
CANARY_RULE = "all_positive"
CANARY_MIN_POSITIVE = 4
CANARY_ANALYZER_TICKERS = sorted(set(CANARY_TICKERS + ATTACK_TICKERS + [SAFE_ASSET]))
DEFAULT_SORTINO_TOP_N = 30
SORTINO_LOOKBACK_DAYS = 126
SORTINO_ANNUALIZATION = 252
SORTINO_SECTOR_TOP_COUNT = 20
ETF_UNIVERSE_SEEDS = [
    "SPY", "IVV", "VOO", "VTI", "ITOT", "SCHB", "QQQ", "QQQM", "DIA", "IWM", "IJH", "IJR",
    "VUG", "SCHG", "VTV", "SCHV", "IWF", "IWD", "VBR", "VBK", "MTUM", "QUAL", "USMV", "VLUE",
    "RSP", "NOBL", "SCHD", "DGRO", "VIG", "VO", "VB", "VEA", "IEFA", "VWO", "IEMG", "VXUS",
    "XLF", "VFH", "KBE", "KRE", "XLY", "VCR", "XLP", "VDC", "XLE", "VDE", "XLI", "VIS", "XLK",
    "VGT", "SMH", "SOXX", "IGV", "XLC", "VOX", "XLB", "VAW", "XLV", "VHT", "IBB", "XBI", "XLU",
    "VPU", "XLRE", "VNQ", "REET", "ITA", "PAVE", "XME", "COPX", "URA", "LIT", "BOTZ", "ARKK",
    "ARKW", "ARKQ", "ARKF", "ARKG", "ROBO", "SKYY", "CIBR", "HACK", "CLOU", "TAN", "ICLN",
    "PBW", "QCLN", "KWEB", "FXI", "ASHR", "EWJ", "EWY", "EWT", "INDA", "EWH", "EWZ", "EWW",
    "EWA", "EWC", "EWG", "EWU", "EWL", "EWS", "EIDO", "EPI", "MCHI", "FDN", "XRT", "XHB",
]
ETF_THEME_LABEL_MAP = {
    "SPY": "Broad Market", "IVV": "Broad Market", "VOO": "Broad Market", "VTI": "Broad Market",
    "ITOT": "Broad Market", "SCHB": "Broad Market", "QQQ": "Large Cap Growth", "QQQM": "Large Cap Growth",
    "DIA": "Large Cap Value", "IWM": "Small Cap", "IJH": "Mid Cap", "IJR": "Small Cap",
    "VUG": "Growth", "SCHG": "Growth", "VTV": "Value", "SCHV": "Value", "IWF": "Growth", "IWD": "Value",
    "VBR": "Small Cap Value", "VBK": "Small Cap Growth", "MTUM": "Momentum", "QUAL": "Quality", "USMV": "Low Volatility",
    "VLUE": "Value", "RSP": "Equal Weight", "NOBL": "Dividend", "SCHD": "Dividend", "DGRO": "Dividend", "VIG": "Dividend",
    "VEA": "Developed Markets", "IEFA": "Developed Markets", "VWO": "Emerging Markets", "IEMG": "Emerging Markets", "VXUS": "International",
    "XLF": "Financials", "VFH": "Financials", "KBE": "Banks", "KRE": "Regional Banks", "XLY": "Consumer Discretionary",
    "VCR": "Consumer Discretionary", "XLP": "Consumer Staples", "VDC": "Consumer Staples", "XLE": "Energy", "VDE": "Energy",
    "XLI": "Industrials", "VIS": "Industrials", "XLK": "Technology", "VGT": "Technology", "SMH": "Semiconductors", "SOXX": "Semiconductors",
    "IGV": "Software", "XLC": "Communication Services", "VOX": "Communication Services", "XLB": "Materials", "VAW": "Materials",
    "XLV": "Health Care", "VHT": "Health Care", "IBB": "Biotech", "XBI": "Biotech", "XLU": "Utilities", "VPU": "Utilities",
    "XLRE": "Real Estate", "VNQ": "Real Estate", "REET": "Global Real Estate", "ITA": "Aerospace & Defense", "PAVE": "Infrastructure",
    "XME": "Metals & Mining", "COPX": "Copper Miners", "URA": "Uranium", "LIT": "Battery Tech", "BOTZ": "Robotics & AI",
    "ARKK": "Disruptive Innovation", "ARKW": "Internet Innovation", "ARKQ": "Autonomous Tech", "ARKF": "Fintech", "ARKG": "Genomics",
    "ROBO": "Robotics", "SKYY": "Cloud", "CIBR": "Cybersecurity", "HACK": "Cybersecurity", "CLOU": "Cloud",
    "TAN": "Solar", "ICLN": "Clean Energy", "PBW": "Clean Energy", "QCLN": "Clean Energy",
    "KWEB": "China Internet", "FXI": "China Large Cap", "ASHR": "China A Shares", "EWJ": "Japan", "EWY": "Korea", "EWT": "Taiwan",
    "INDA": "India", "EWH": "Hong Kong", "EWZ": "Brazil", "EWW": "Mexico", "EWA": "Australia", "EWC": "Canada",
    "EWG": "Germany", "EWU": "United Kingdom", "EWL": "Switzerland", "EWS": "Singapore", "EIDO": "Indonesia", "EPI": "India", "MCHI": "China",
    "FDN": "Internet", "XRT": "Retail", "XHB": "Homebuilders",
}


def configure_page() -> None:
    st.set_page_config(
        page_title="Quant Fusion Dashboard",
        page_icon="Q",
        layout="wide",
        initial_sidebar_state="auto",
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
            background: #fffaf4;
            border-radius: 24px;
            padding: 0.35rem;
            border: 1px solid rgba(148, 163, 184, 0.18);
            box-shadow: 0 16px 30px rgba(15, 23, 42, 0.05);
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
        .sidebar-section-card {
            margin: 0.3rem 0 0.7rem;
            padding: 0.85rem 0.95rem;
            border-radius: 18px;
            background: linear-gradient(160deg, rgba(255, 255, 255, 0.92), rgba(240, 249, 255, 0.92));
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 12px 24px rgba(15, 23, 42, 0.08);
            color: var(--page-ink);
        }
        .sidebar-section-eyebrow {
            font-size: 0.72rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            font-weight: 800;
            color: #0f766e;
        }
        .sidebar-section-title {
            margin-top: 0.26rem;
            font-size: 0.98rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }
        .sidebar-section-copy {
            margin-top: 0.24rem;
            font-size: 0.84rem;
            line-height: 1.45;
            color: var(--muted-ink);
        }
        div[data-testid="stSidebar"] button[kind="secondary"] {
            min-height: 3.25rem;
            border-radius: 18px;
            border: 1px solid rgba(15, 23, 42, 0.1);
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(255,249,240,0.92));
            color: var(--page-ink);
            font-weight: 800;
            letter-spacing: -0.01em;
            box-shadow: 0 12px 22px rgba(15, 23, 42, 0.07);
            transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
        }
        div[data-testid="stSidebar"] button[kind="secondary"]:hover {
            border-color: rgba(15, 118, 110, 0.38);
            transform: translateY(-1px);
            box-shadow: 0 16px 28px rgba(15, 23, 42, 0.11);
        }
        div[data-testid="stSidebar"] button[kind="secondary"]:focus {
            border-color: rgba(15, 118, 110, 0.52);
            box-shadow: 0 0 0 0.18rem rgba(15, 118, 110, 0.12);
        }
        .quick-action-row-copy {
            min-height: 3.25rem;
            display: flex;
            align-items: center;
            padding: 0 0.15rem 0 0.55rem;
            color: var(--muted-ink);
            font-size: 0.8rem;
            line-height: 1.4;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 0.8rem;
                padding-right: 0.9rem;
                padding-left: 0.9rem;
                padding-bottom: 1.6rem;
            }
            .hero {
                border-radius: 22px;
                padding: 1.35rem 1.15rem;
            }
            .hero h1 {
                font-size: 1.55rem;
            }
            .hero p {
                font-size: 0.92rem;
            }
            .metric-card {
                min-height: 6.4rem;
                padding: 0.85rem 0.9rem;
                border-radius: 18px;
            }
            .metric-value {
                font-size: 1.3rem;
            }
            .metric-subtitle {
                font-size: 0.84rem;
            }
            .quick-action-row-copy {
                min-height: 2.8rem;
                padding-left: 0.3rem;
                font-size: 0.76rem;
            }
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap;
                gap: 0.7rem;
            }
            div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
                min-width: calc(50% - 0.4rem);
                flex: 1 1 calc(50% - 0.4rem);
            }
            div[data-testid="stPlotlyChart"] {
                border-radius: 18px;
                padding: 0.3rem;
                background: #fffaf4;
            }
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


def display_view_name(view_name: str) -> str:
    return SPECIAL_ACTION_LABELS.get(view_name, view_name)


def render_plotly_chart(fig: go.Figure) -> None:
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG)


def mobile_charts_enabled() -> bool:
    return bool(st.session_state.get("mobile_chart_mode", False))


def render_chart(title: str, plotly_fig: go.Figure, mobile_builder: Any | None = None) -> None:
    if mobile_charts_enabled() and callable(mobile_builder):
        st.markdown(f"#### {title}")
        mobile_fig = mobile_builder()
        if mobile_fig is not None:
            st.pyplot(mobile_fig, use_container_width=True)
            plt.close(mobile_fig)
        return
    render_plotly_chart(plotly_fig)


def _mobile_view_slice(data: pd.DataFrame | pd.Series, *, pad_days: int = 0) -> pd.DataFrame | pd.Series:
    return trim_to_history_window(data, pad_days=pad_days)


def _mobile_finalize_figure(fig: Any) -> Any:
    get_layout_engine = getattr(fig, "get_layout_engine", None)
    if callable(get_layout_engine) and get_layout_engine() is not None:
        return fig
    try:
        fig.tight_layout(pad=1.2)
    except RuntimeError:
        pass
    return fig


def _mobile_style_axis(axis: Any, ylabel: str | None = None, x_axis_type: str = "date") -> None:
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(True, axis="y", alpha=0.22, color="#64748b")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="x", labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    if hasattr(axis, "xaxis") and x_axis_type == "date":
        axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    elif hasattr(axis, "xaxis") and x_axis_type == "numeric":
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        try:
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
        except Exception:
            pass


def _mobile_add_candlesticks(axis: Any, df: pd.DataFrame, width_ratio: float = 0.65) -> None:
    if df.empty:
        return
    dates = mdates.date2num(pd.to_datetime(df.index).to_pydatetime())
    step = float(np.median(np.diff(dates))) if len(dates) > 1 else 1.0
    width = max(step * width_ratio, 0.35)
    for idx, row in enumerate(df.itertuples()):
        open_price = float(row.Open)
        high_price = float(row.High)
        low_price = float(row.Low)
        close_price = float(row.Close)
        color = "#0f766e" if close_price >= open_price else "#b42318"
        axis.vlines(dates[idx], low_price, high_price, color=color, linewidth=0.8, alpha=0.9)
        body_low = min(open_price, close_price)
        body_height = max(abs(close_price - open_price), 1e-6)
        axis.add_patch(
            Rectangle(
                (dates[idx] - width / 2, body_low),
                width,
                body_height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.75,
            )
        )
    axis.xaxis_date()


def _draw_mobile_gauge(
    axis: Any,
    score_pct: float,
    subtitle: str,
    *,
    value_y: float = 0.18,
    subtitle_y: float = -0.14,
    needle_length: float = 0.74,
    needle_width: float = 4.0,
    value_size: float = 22,
) -> None:
    segments = [
        (0, 20, "#dbeafe"),
        (20, 40, "#dcfce7"),
        (40, 60, "#f8fafc"),
        (60, 80, "#fef3c7"),
        (80, 100, "#fee2e2"),
    ]
    for start, end, color in segments:
        theta1 = 180 - (end * 1.8)
        theta2 = 180 - (start * 1.8)
        axis.add_patch(Wedge((0, 0), 1.0, theta1, theta2, width=0.24, facecolor=color, edgecolor="none"))

    score_pct = max(0.0, min(100.0, score_pct))
    theta = np.deg2rad(180 - (score_pct * 1.8))
    axis.plot([0, np.cos(theta) * needle_length], [0, np.sin(theta) * needle_length], color="#102a43", linewidth=needle_width, solid_capstyle="round")
    axis.scatter([0], [0], color="#102a43", s=40, zorder=3)
    axis.text(
        0,
        value_y,
        f"{score_pct:.1f}",
        ha="center",
        va="center",
        fontsize=value_size,
        color="#102a43",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.14", facecolor=CHART_SURFACE_COLOR, edgecolor="none", alpha=0.98),
        zorder=5,
    )
    axis.text(
        0,
        subtitle_y,
        subtitle,
        ha="center",
        va="center",
        fontsize=10,
        color="#52606d",
        bbox=dict(boxstyle="round,pad=0.08", facecolor=CHART_SURFACE_COLOR, edgecolor="none", alpha=0.96),
        zorder=5,
    )
    axis.set_xlim(-1.05, 1.05)
    axis.set_ylim(-0.08, 1.05)
    axis.set_aspect("equal")
    axis.axis("off")


def _mobile_shorten_labels(values: list[str], max_len: int = 20) -> list[str]:
    labels: list[str] = []
    for value in values:
        text = str(value)
        labels.append(text if len(text) <= max_len else f"{text[:max_len - 1]}...")
    return labels


def _build_mobile_gauge_figure(score_pct: float) -> Any:
    fig, ax = plt.subplots(figsize=(6.4, 2.6), constrained_layout=True)
    _draw_mobile_gauge(ax, score_pct, "Fear <-> Greed")
    return fig


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
    target_names = names or ("market_pulse", "spx_options_payload", *FED_WATCH_CACHE_KEYS)
    for name in target_names:
        cache_path = _daily_cache_path(name)
        if cache_path.exists():
            cache_path.unlink()


def history_window_start(period: str) -> pd.Timestamp:
    offset = FED_WATCH_PERIOD_OFFSETS.get(period, FED_WATCH_PERIOD_OFFSETS[LOOKBACK_PERIOD])
    return (pd.Timestamp.now().normalize() - offset).normalize()


def active_history_window() -> str:
    period = str(st.session_state.get("period", LOOKBACK_PERIOD))
    return period if period in FED_WATCH_PERIOD_OFFSETS else LOOKBACK_PERIOD


def trim_to_history_window(
    data: pd.DataFrame | pd.Series,
    *,
    period: str | None = None,
    pad_days: int = 0,
) -> pd.DataFrame | pd.Series:
    if data is None or getattr(data, "empty", True):
        return data
    if not isinstance(getattr(data, "index", None), pd.DatetimeIndex):
        return data
    start = history_window_start(period or active_history_window()) - pd.Timedelta(days=pad_days)
    return data.loc[data.index >= start].copy()


def _fred_series_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/series/{series_id}"


def _latest_series_date(series: pd.Series) -> pd.Timestamp | None:
    clean = series.dropna()
    if clean.empty:
        return None
    return pd.Timestamp(clean.index[-1])


def _series_delta(series: pd.Series, periods: int = 20) -> float:
    clean = series.dropna()
    if len(clean) <= periods:
        return np.nan
    return float(clean.iloc[-1] - clean.iloc[-periods - 1])


def _latest_frame_value(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    clean = frame[column].dropna()
    if clean.empty:
        return np.nan
    return float(clean.iloc[-1])


def _format_asof_date(timestamp: pd.Timestamp | None) -> str:
    return "n/a" if timestamp is None or pd.isna(timestamp) else pd.Timestamp(timestamp).strftime("%Y-%m-%d")


def _composite_latest_date(latest_dates: dict[str, pd.Timestamp], keys: list[str]) -> pd.Timestamp | None:
    dates = [latest_dates[key] for key in keys if key in latest_dates and latest_dates[key] is not None]
    if not dates:
        return None
    return min(dates)


def _normalize_fed_watch_series(
    alias: str,
    series: pd.Series,
    daily_index: pd.DatetimeIndex,
) -> tuple[pd.Series, pd.Series, pd.Timestamp | None]:
    clean = pd.to_numeric(series, errors="coerce").dropna().sort_index()
    if clean.empty:
        return pd.Series(dtype=float, name=alias), pd.Series(index=daily_index, dtype=float, name=alias), None
    clean.index = normalize_datetime_index(clean.index)
    clean.name = alias
    latest_date = _latest_series_date(clean)
    aligned = clean.reindex(daily_index).ffill()
    if alias == "IOER" and latest_date is not None:
        aligned.loc[aligned.index > latest_date] = np.nan
    return clean, aligned, latest_date


def _download_fred_batch_series(start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict[str, pd.Series], str | None]:
    if pdr_web is None:
        detail = f": {PDR_IMPORT_ERROR}" if PDR_IMPORT_ERROR else ""
        return {}, f"pandas_datareader import failed{detail}"

    fred_codes = [str(spec["fred"]) for spec in FED_WATCH_SERIES_SPECS.values()]
    code_to_alias = {str(spec["fred"]): alias for alias, spec in FED_WATCH_SERIES_SPECS.items()}

    try:
        frame = pdr_web.DataReader(fred_codes, "fred", start, end)
    except Exception as exc:
        return {}, str(exc)

    if frame is None or frame.empty:
        return {}, "batch FRED response was empty"

    frame.index = normalize_datetime_index(frame.index)
    frame = frame.sort_index()
    batch_series: dict[str, pd.Series] = {}
    for fred_code, alias in code_to_alias.items():
        if fred_code not in frame.columns:
            continue
        scaled = pd.to_numeric(frame[fred_code], errors="coerce").dropna() * float(FED_WATCH_SERIES_SPECS[alias]["scale"])
        scaled.name = alias
        if not scaled.empty:
            batch_series[alias] = scaled
    return batch_series, None


def _load_cached_fed_watch_series(cached_payload: dict[str, Any] | None, alias: str) -> pd.Series:
    if not cached_payload:
        return pd.Series(dtype=float, name=alias)

    raw_series_map = cached_payload.get("raw_series", {})
    cached_series = raw_series_map.get(alias) if isinstance(raw_series_map, dict) else None
    if isinstance(cached_series, pd.Series):
        cached_series = cached_series.copy()
    else:
        frame = cached_payload.get("frame")
        if not isinstance(frame, pd.DataFrame) or alias not in frame.columns:
            return pd.Series(dtype=float, name=alias)
        cached_series = pd.to_numeric(frame[alias], errors="coerce")
        latest_date = cached_payload.get("latest_dates", {}).get(alias) if isinstance(cached_payload.get("latest_dates"), dict) else None
        if latest_date is not None:
            cached_series = cached_series.loc[:latest_date]

    cached_series = pd.to_numeric(cached_series, errors="coerce").dropna()
    if cached_series.empty:
        return pd.Series(dtype=float, name=alias)
    cached_series.index = normalize_datetime_index(cached_series.index)
    cached_series.name = alias
    return cached_series.sort_index()


def _build_fed_watch_payload(
    period: str,
    *,
    cached_payload: dict[str, Any] | None = None,
    cached_date: str | None = None,
) -> dict[str, Any] | None:
    start = history_window_start(period)
    end = pd.Timestamp.now().normalize()
    daily_index = pd.date_range(start=start, end=end, freq="D")

    frame = pd.DataFrame(index=daily_index)
    source_status: list[dict[str, Any]] = []
    warnings_list: list[str] = []
    latest_dates: dict[str, pd.Timestamp] = {}
    raw_series: dict[str, pd.Series] = {}
    stale_fallback_count = 0
    live_series: dict[str, pd.Series] = {}
    batch_series, batch_error = _download_fred_batch_series(start, end)
    live_series.update(batch_series)
    if batch_error:
        warnings_list.append(f"Batch FRED request failed: {batch_error}")

    for alias, spec in FED_WATCH_SERIES_SPECS.items():
        series_id = str(spec["fred"])
        source_label = "Live"
        status_label = "OK"
        warning_text = ""

        preferred_series = live_series.get(alias)
        clean, aligned, latest_date = _normalize_fed_watch_series(
            alias,
            preferred_series if isinstance(preferred_series, pd.Series) else pd.Series(dtype=float, name=alias),
            daily_index,
        )

        if clean.empty:
            cached_series = _load_cached_fed_watch_series(cached_payload, alias)
            if not cached_series.empty:
                clean, aligned, latest_date = _normalize_fed_watch_series(alias, cached_series, daily_index)
                source_label = f"Cache {cached_date}" if cached_date else "Stale cache"
                status_label = "Stale fallback"
                stale_fallback_count += 1
                if batch_error:
                    warning_text = f"{alias} ({series_id}) was unavailable from batch FRED download. Using cached series from {cached_date or 'a previous run'}."
                else:
                    warning_text = f"{alias} ({series_id}) returned no usable live rows. Using cached series from {cached_date or 'a previous run'}."
            else:
                frame[alias] = np.nan
                status_label = "Failed" if batch_error else "Empty"
                warning_text = (
                    f"{alias} ({series_id}) failed to load from batch FRED download: {batch_error}"
                    if batch_error
                    else f"{alias} ({series_id}) returned no usable rows."
                )
                source_status.append(
                    {
                        "Series": alias,
                        "FRED Code": series_id,
                        "Rows": 0,
                        "Latest": "n/a",
                        "Unit": str(spec["unit"]),
                        "Source": "Unavailable",
                        "Status": status_label,
                        "URL": _fred_series_url(series_id),
                    }
                )
                warnings_list.append(warning_text)
                continue

        raw_series[alias] = clean
        if latest_date is not None:
            latest_dates[alias] = latest_date
        frame[alias] = aligned
        source_status.append(
            {
                "Series": alias,
                "FRED Code": series_id,
                "Rows": int(clean.shape[0]),
                "Latest": latest_date.strftime("%Y-%m-%d") if latest_date is not None else "n/a",
                "Unit": str(spec["unit"]),
                "Source": source_label,
                "Status": status_label,
                "URL": _fred_series_url(series_id),
            }
        )
        if warning_text:
            warnings_list.append(warning_text)

    frame["IORB_Combined"] = frame["IORB"].combine_first(frame["IOER"])
    frame["Stress_SOFR_IORB"] = frame["SOFR"] - frame["IORB_Combined"]
    frame["Spread_Carry"] = frame["US10Y"] - frame["SOFR"]
    frame["Spread_Curve"] = frame["US10Y"] - frame["US3M"]
    frame["Net_Liquidity"] = frame["WALCL"] - frame["TGA"] - frame["ON_RRP"]
    frame = frame.sort_index()
    if frame.dropna(how="all").empty:
        warnings_list.append("No usable live FRED rows were returned for this request window.")
    policy_rate_date = max(
        [latest_dates[key] for key in ["IORB", "IOER"] if key in latest_dates and latest_dates[key] is not None],
        default=None,
    )
    spread_date_candidates = [value for value in [latest_dates.get("SOFR"), policy_rate_date] if value is not None]

    card_dates = {
        "Net_Liquidity": _composite_latest_date(latest_dates, ["WALCL", "TGA", "ON_RRP"]),
        "Spread": min(spread_date_candidates) if spread_date_candidates else None,
        "TGA": latest_dates.get("TGA"),
        "ON_RRP": latest_dates.get("ON_RRP"),
    }

    return {
        "period": period,
        "frame": frame,
        "raw_series": raw_series,
        "latest_dates": latest_dates,
        "card_dates": card_dates,
        "source_status": pd.DataFrame(source_status),
        "warnings": warnings_list,
        "stale_fallback_count": stale_fallback_count,
        "stale_cache_date": cached_date,
    }


def fetch_fed_watch_data(period: str) -> dict[str, Any] | None:
    cache_period = period if period in FED_WATCH_PERIOD_OFFSETS else LOOKBACK_PERIOD
    cache_name = f"fed_watch_{cache_period}"
    today = datetime.now().strftime("%Y-%m-%d")
    cache_date, cached_data = load_daily_cached_payload(cache_name)
    cache_is_current_format = (
        isinstance(cached_data, dict)
        and "stale_fallback_count" in cached_data
        and isinstance(cached_data.get("source_status"), pd.DataFrame)
    )
    if cache_date == today and cached_data is not None and cache_is_current_format:
        return cached_data

    fresh_data = _build_fed_watch_payload(cache_period, cached_payload=cached_data, cached_date=cache_date)
    if fresh_data is not None:
        has_usable_rows = not fresh_data["frame"].dropna(how="all").empty
        if has_usable_rows:
            save_daily_cached_payload(cache_name, fresh_data)
            return fresh_data
        if cached_data is not None:
            return cached_data
        return fresh_data
    return cached_data


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
def download_price_data(ticker: str, period: str = LOOKBACK_PERIOD) -> tuple[pd.DataFrame, str, str]:
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


@st.cache_data(ttl=3600, show_spinner=False)
def download_multi_close_data(tickers: list[str], period: str = LOOKBACK_PERIOD) -> pd.DataFrame:
    raw = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if raw.empty:
        return pd.DataFrame(columns=tickers)

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"].copy()
        else:
            return pd.DataFrame(columns=tickers)
    else:
        column = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
        if column is None:
            return pd.DataFrame(columns=tickers)
        close = raw[[column]].copy()
        close.columns = [tickers[0]]

    close.index = normalize_datetime_index(close.index)
    close = close.sort_index().ffill()
    return close.reindex(columns=tickers)


def resample_month_end(series: pd.Series) -> pd.Series:
    try:
        return series.resample("ME").last()
    except ValueError:
        return series.resample("M").last()


def calc_avg_momentum(series: pd.Series, lags: list[int]) -> tuple[list[float], float]:
    clean = series.dropna()
    if clean.empty or len(clean) <= max(lags):
        return [np.nan] * len(lags), np.nan
    latest = float(clean.iloc[-1])
    moments = [((latest / float(clean.iloc[-(lag + 1)])) - 1) * 100 for lag in lags]
    return moments, float(np.mean(moments))


def get_completed_monthly(series_daily: pd.Series, asof: pd.Timestamp | None = None) -> pd.Series:
    anchor = pd.Timestamp.today().normalize() if asof is None else pd.Timestamp(asof).normalize()
    monthly = resample_month_end(series_daily.dropna())
    if len(monthly) and monthly.index[-1].to_period("M") == anchor.to_period("M"):
        monthly = monthly.iloc[:-1]
    return monthly


def classify_momentum(avg_momentum: float, positive_text: str, negative_text: str) -> str:
    if pd.isna(avg_momentum):
        return "⚪ 데이터 부족"
    return positive_text if avg_momentum > 0 else negative_text


def decide_canary_regime(signal_map: pd.Series, rule: str = CANARY_RULE, min_positive: int = CANARY_MIN_POSITIVE) -> tuple[float, int, int, str]:
    valid = signal_map.dropna()
    total = int(valid.shape[0])
    positive_count = int((valid > 0).sum())
    if total == 0:
        return np.nan, 0, 0, "⚪ 데이터 부족"
    if rule == "tip_only":
        if "TIP" not in valid.index:
            return np.nan, positive_count, total, "⚪ 데이터 부족"
        score = float(valid.loc["TIP"])
        return score, positive_count, total, "🟢 공격" if score > 0 else "🔴 대피"
    if rule == "majority":
        threshold = int(np.ceil(total / 2))
    elif rule == "at_least_n":
        threshold = min_positive
    else:
        threshold = total
    score = positive_count / total
    return score, positive_count, total, "🟢 공격" if positive_count >= threshold else "🔴 대피"


def pick_top_assets(df_assets: pd.DataFrame, score_column: str, top_n: int = 4) -> list[str]:
    valid = df_assets.dropna(subset=[score_column]).sort_values(score_column, ascending=False)
    return valid["자산"].head(top_n).tolist()


def build_canary_report(data_daily: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    records: list[list[Any]] = []
    for ticker in CANARY_TICKERS:
        daily_series = data_daily[ticker]
        monthly_series = get_completed_monthly(daily_series, asof=asof)
        rt_moms, rt_avg = calc_avg_momentum(daily_series, REALTIME_LAGS)
        eom_moms, eom_avg = calc_avg_momentum(monthly_series, MONTHLY_LAGS)
        records.append([
            ticker,
            round(eom_moms[0], 2) if not pd.isna(eom_moms[0]) else np.nan,
            round(eom_moms[1], 2) if not pd.isna(eom_moms[1]) else np.nan,
            round(eom_moms[2], 2) if not pd.isna(eom_moms[2]) else np.nan,
            round(eom_moms[3], 2) if not pd.isna(eom_moms[3]) else np.nan,
            round(eom_avg, 2) if not pd.isna(eom_avg) else np.nan,
            classify_momentum(eom_avg, "🟢 유지", "🔴 대피"),
            round(rt_moms[0], 2) if not pd.isna(rt_moms[0]) else np.nan,
            round(rt_moms[1], 2) if not pd.isna(rt_moms[1]) else np.nan,
            round(rt_moms[2], 2) if not pd.isna(rt_moms[2]) else np.nan,
            round(rt_moms[3], 2) if not pd.isna(rt_moms[3]) else np.nan,
            round(rt_avg, 2) if not pd.isna(rt_avg) else np.nan,
            classify_momentum(rt_avg, "🟢 안전", "🔴 위험"),
        ])
    return pd.DataFrame(records, columns=[
        "자산", "전월말 1M(%)", "전월말 3M(%)", "전월말 6M(%)", "전월말 12M 수익률(%)",
        "전월말 평균 모멘텀(%)", "전월말 상태", "실시간 1M(%)", "실시간 3M(%)",
        "실시간 6M(%)", "실시간 12M 수익률(%)", "실시간 평균 모멘텀(%)", "실시간 상태",
    ])


def build_attack_report(data_daily: pd.DataFrame, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    records: list[list[Any]] = []
    for ticker in ATTACK_TICKERS:
        daily_series = data_daily[ticker]
        monthly_series = get_completed_monthly(daily_series, asof=asof)
        rt_moms, rt_avg = calc_avg_momentum(daily_series, REALTIME_LAGS)
        _, eom_avg = calc_avg_momentum(monthly_series, MONTHLY_LAGS)
        records.append([
            ticker,
            round(eom_avg, 2) if not pd.isna(eom_avg) else np.nan,
            round(rt_moms[0], 2) if not pd.isna(rt_moms[0]) else np.nan,
            round(rt_moms[1], 2) if not pd.isna(rt_moms[1]) else np.nan,
            round(rt_moms[2], 2) if not pd.isna(rt_moms[2]) else np.nan,
            round(rt_moms[3], 2) if not pd.isna(rt_moms[3]) else np.nan,
            round(rt_avg, 2) if not pd.isna(rt_avg) else np.nan,
        ])
    return pd.DataFrame(records, columns=[
        "자산", "전월말 평균 모멘텀(%)", "실시간 1M(%)", "실시간 3M(%)",
        "실시간 6M(%)", "실시간 12M 수익률(%)", "실시간 평균 모멘텀(%)",
    ]).sort_values(by="실시간 평균 모멘텀(%)", ascending=False, na_position="last").reset_index(drop=True)


def compute_canary_momentum_dashboard(period: str = LOOKBACK_PERIOD) -> dict[str, Any] | None:
    data_daily = download_multi_close_data(CANARY_ANALYZER_TICKERS, period=period)
    if data_daily.empty:
        return None
    asof = pd.Timestamp(data_daily.dropna(how="all").index[-1])
    df_canary = build_canary_report(data_daily, asof=asof)
    df_assets = build_attack_report(data_daily, asof=asof)
    eom_signal = df_canary.set_index("자산")["전월말 평균 모멘텀(%)"]
    rt_signal = df_canary.set_index("자산")["실시간 평균 모멘텀(%)"]
    eom_score, eom_pos, eom_total, eom_status = decide_canary_regime(eom_signal)
    rt_score, rt_pos, rt_total, rt_status = decide_canary_regime(rt_signal)
    current_top4 = pick_top_assets(df_assets, "전월말 평균 모멘텀(%)")
    predicted_top4 = pick_top_assets(df_assets, "실시간 평균 모멘텀(%)")
    return {
        "asof": asof,
        "data_daily": data_daily,
        "canary_report": df_canary,
        "attack_report": df_assets,
        "eom": {"score": eom_score, "positive": eom_pos, "total": eom_total, "status": eom_status, "top_assets": current_top4},
        "realtime": {"score": rt_score, "positive": rt_pos, "total": rt_total, "status": rt_status, "top_assets": predicted_top4},
    }


def canary_signal_label(canary_data: dict[str, Any] | None) -> tuple[str, str]:
    if not canary_data:
        return "Not loaded", "neutral"
    realtime = canary_data["realtime"]["status"]
    eom = canary_data["eom"]["status"]
    if realtime == "🟢 공격":
        return f"Attack regime · {canary_data['realtime']['positive']}/{canary_data['realtime']['total']} positive", "bull"
    if realtime == "🔴 대피":
        return f"Defense regime · {canary_data['realtime']['positive']}/{canary_data['realtime']['total']} positive", "bear"
    if eom == "🟢 공격":
        return "Completed month remains in attack", "accent"
    return "Insufficient canary data", "neutral"


def build_canary_attack_figure(df_assets: pd.DataFrame) -> go.Figure:
    view = df_assets.copy().sort_values("실시간 평균 모멘텀(%)", ascending=True)
    colors = ["#0f766e" if (not pd.isna(v) and v >= 0) else "#b42318" for v in view["실시간 평균 모멘텀(%)"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=view["실시간 평균 모멘텀(%)"],
        y=view["자산"],
        orientation="h",
        marker_color=colors,
        text=view["실시간 평균 모멘텀(%)"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}%"),
        textposition="outside",
        name="실시간 평균 모멘텀",
    ))
    fig.add_vline(x=0, line_color="#64748b", line_dash="dot")
    return apply_figure_style(fig, title="Attack Universe Real-time Momentum Ranking", height=560, showlegend=False)


def render_canary_dashboard(canary_data: dict[str, Any]) -> None:
    canary_label, canary_tone = canary_signal_label(canary_data)
    cards = st.columns(4)
    with cards[0]:
        render_metric_card("Canary Rule", CANARY_RULE, f"Threshold setting {CANARY_MIN_POSITIVE}", "neutral")
    with cards[1]:
        render_metric_card("Completed Month", canary_data["eom"]["status"], f"{canary_data['eom']['positive']}/{canary_data['eom']['total']} canaries positive", "bull" if canary_data["eom"]["status"] == "🟢 공격" else "bear")
    with cards[2]:
        render_metric_card("Today", canary_data["realtime"]["status"], f"{canary_data['realtime']['positive']}/{canary_data['realtime']['total']} canaries positive", canary_tone)
    with cards[3]:
        top_assets = ", ".join(canary_data["realtime"]["top_assets"]) if canary_data["realtime"]["top_assets"] else SAFE_ASSET
        render_metric_card("Expected Rotation", top_assets, f"Fallback safe asset {SAFE_ASSET}", "accent")

    st.markdown('<div class="section-note">카나리아 자산(TIP, QQQ, SPY, VEA, VWO, BND)의 전월말 확정 모멘텀과 오늘 기준 실시간 모멘텀을 함께 비교해, 현재 포트폴리오와 다음 월말 리밸런싱 가능성을 한 번에 점검합니다.</div>', unsafe_allow_html=True)
    render_chart(
        "Attack Universe Real-time Momentum Ranking",
        build_canary_attack_figure(canary_data["attack_report"]),
        mobile_builder=lambda: build_mobile_canary_attack_figure(canary_data["attack_report"]),
    )

    table_left, table_right = st.columns([1.15, 1.0])
    with table_left:
        st.markdown("#### 카나리아 상태 비교")
        st.dataframe(canary_data["canary_report"], use_container_width=True, hide_index=True)
    with table_right:
        st.markdown("#### 공격 자산 모멘텀 순위")
        st.dataframe(canary_data["attack_report"], use_container_width=True, hide_index=True)

    st.markdown("#### 리밸런싱 액션 가이드")
    eom = canary_data["eom"]
    rt = canary_data["realtime"]
    current_line = f"현재 포트폴리오(전월말 확정): {'공격 모드 유지 → ' + ', '.join(eom['top_assets']) if eom['status'] == '🟢 공격' else '안전 자산 대기 → ' + SAFE_ASSET if eom['status'] == '🔴 대피' else '데이터 부족'}"
    future_line = f"월말 예측(오늘 종가 기준): {'공격 모드 유지 가능성 → ' + ', '.join(rt['top_assets']) if rt['status'] == '🟢 공격' else '안전 자산 이동 가능성 → ' + SAFE_ASSET if rt['status'] == '🔴 대피' else '데이터 부족'}"
    st.markdown(f"- 기준 영업일: **{canary_data['asof'].strftime('%Y-%m-%d')}**\n- 실시간 요약: **{canary_label}**\n- {current_line}\n- {future_line}")


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
        paper_bgcolor=CHART_SURFACE_COLOR,
        plot_bgcolor=CHART_SURFACE_COLOR,
        font=PLOT_FONT,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#fffdf9", font_size=13, font_family=PLOT_FONT["family"]),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            y=legend_y,
            x=0,
            bgcolor="rgba(255,253,249,0.94)",
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


def build_mobile_elder_figure(df: pd.DataFrame) -> Any:
    view = _mobile_view_slice(df).copy()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.0), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [3.2, 1.1]})
    _mobile_add_candlesticks(axes[0], view)
    axes[0].plot(view.index, view["EMA13"], color="#64748b", linewidth=1.2, label="EMA13")
    axes[0].plot(view.index, view["EMA65"], color="#dd6b20", linewidth=1.3, linestyle="--", label="EMA65")
    if view["Buy_Signal"].notna().any():
        buys = view.loc[view["Buy_Signal"].notna()]
        axes[0].scatter(buys.index, buys["Buy_Signal"], color="#0f766e", marker="^", s=26, label="Buy")
    if view["Sell_Signal"].notna().any():
        sells = view.loc[view["Sell_Signal"].notna()]
        axes[0].scatter(sells.index, sells["Sell_Signal"], color="#b42318", marker="v", s=26, label="Sell")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    colors = np.where(view["MACD_Hist"] >= 0, "#0f766e", "#b42318")
    axes[1].bar(view.index, view["MACD_Hist"], color=colors, width=3)
    axes[1].axhline(0, color="#64748b", linestyle=":", linewidth=1)
    _mobile_style_axis(axes[0], "Price")
    _mobile_style_axis(axes[1], "MACD")
    return _mobile_finalize_figure(fig)


def build_mobile_td_figure(df: pd.DataFrame) -> Any:
    view = _mobile_view_slice(df).copy()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.1), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [3.2, 1.1]})
    _mobile_add_candlesticks(axes[0], view)
    for column, color in [("MA21", "#dd6b20"), ("MA50", "#2563eb"), ("MA200", "#111827")]:
        axes[0].plot(view.index, view[column], color=color, linewidth=1.1, label=column)
    cloud_mask = view["Senkou_Span_A"].notna() & view["Senkou_Span_B"].notna()
    if cloud_mask.any():
        xvals = view.index[cloud_mask]
        span_a = view.loc[cloud_mask, "Senkou_Span_A"]
        span_b = view.loc[cloud_mask, "Senkou_Span_B"]
        axes[0].fill_between(xvals, span_a, span_b, color="#f59e0b", alpha=0.08)
    recent_view = view.tail(80)
    for idx, row in recent_view.iterrows():
        if row["Buy_Setup"] in {7, 8, 9}:
            axes[0].annotate(f"B{int(row['Buy_Setup'])}", (idx, row["Low"] * 0.992), fontsize=7, color="#0f766e")
        if row["Sell_Setup"] in {7, 8, 9}:
            axes[0].annotate(f"S{int(row['Sell_Setup'])}", (idx, row["High"] * 1.008), fontsize=7, color="#b42318")
        if row["Buy_Countdown"] == 13:
            axes[0].annotate("BUY 13", (idx, row["Low"] * 0.975), fontsize=7, color="#0f766e")
        if row["Sell_Countdown"] == 13:
            axes[0].annotate("SELL 13", (idx, row["High"] * 1.02), fontsize=7, color="#b42318")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    axes[1].plot(view.index, view["RSI"], color="#7c3aed", linewidth=1.4)
    axes[1].axhline(70, color="#b42318", linestyle=":", linewidth=1)
    axes[1].axhline(30, color="#0f766e", linestyle=":", linewidth=1)
    _mobile_style_axis(axes[0], "Price")
    _mobile_style_axis(axes[1], "RSI")
    return _mobile_finalize_figure(fig)


def build_mobile_stl_figure(df: pd.DataFrame) -> Any:
    view = _mobile_view_slice(df).copy()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.4), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [3, 1.1]})
    scatter = axes[0].scatter(
        view.index,
        view["Close"],
        c=view["Cycle_Score"],
        cmap="RdYlBu_r",
        vmin=0,
        vmax=100,
        s=12,
        zorder=3,
    )
    axes[0].plot(view.index, view["Close"], color="#102a43", linewidth=0.7, alpha=0.18)
    axes[0].plot(view.index, view["Trend"], color="#0f766e", linewidth=1.3, label="Trend")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    axes[1].plot(view.index, view["Cycle_Score"], color="#f59e0b", linewidth=1.5)
    axes[1].axhline(50, color="#64748b", linestyle=":", linewidth=1)
    axes[1].fill_between(view.index, 0, 10, color="#dbeafe", alpha=0.5)
    axes[1].fill_between(view.index, 90, 100, color="#fee2e2", alpha=0.5)
    axes[1].set_ylim(0, 100)
    _mobile_style_axis(axes[0], "Price")
    _mobile_style_axis(axes[1], "Cycle")
    fig.colorbar(scatter, ax=axes[0], pad=0.01, shrink=0.6)
    return _mobile_finalize_figure(fig)


def build_mobile_smc_figure(smc_data: dict[str, Any]) -> Any:
    view = smc_data["view"].copy()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 7.2), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [3.2, 1.0]})
    _mobile_add_candlesticks(axes[0], view)
    axes[0].plot(view.index, view["EMA21"], color="#dd6b20", linewidth=1.2, label="EMA21")
    axes[0].plot(view.index, view["SMA200"], color="#111827", linewidth=1.2, linestyle="--", label="SMA200")
    x0 = view.index[0]
    x1 = view.index[-1]
    for zone in smc_data["active_bull_ob"][:2]:
        axes[0].fill_between([x0, x1], zone["bottom"], zone["top"], color="#0f766e", alpha=0.12)
    for zone in smc_data["active_bear_ob"][:2]:
        axes[0].fill_between([x0, x1], zone["bottom"], zone["top"], color="#b42318", alpha=0.12)
    for zone in smc_data["active_bull_fvg"][:2]:
        axes[0].fill_between([x0, x1], zone["bottom"], zone["top"], color="#2563eb", alpha=0.10)
    for zone in smc_data["active_bear_fvg"][:2]:
        axes[0].fill_between([x0, x1], zone["bottom"], zone["top"], color="#dd6b20", alpha=0.10)
    if not np.isnan(smc_data["poc_price"]):
        axes[0].axhline(smc_data["poc_price"], color="#334155", linewidth=1.1, linestyle="--")
    if not np.isnan(smc_data["eq_price"]):
        axes[0].axhline(smc_data["eq_price"], color="#f59e0b", linewidth=1.1)
    for level in smc_data["levels"][:4]:
        axes[0].axhline(level["price"], color="#0f766e" if level["type"] == "L" else "#b42318", linewidth=0.8, linestyle=":")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    axes[1].plot(view.index, view["RSI"], color="#7c3aed", linewidth=1.4)
    axes[1].axhline(70, color="#b42318", linestyle=":", linewidth=1)
    axes[1].axhline(30, color="#0f766e", linestyle=":", linewidth=1)
    _mobile_style_axis(axes[0], "Price")
    _mobile_style_axis(axes[1], "RSI")
    return _mobile_finalize_figure(fig)


def build_mobile_supertrend_figure(df: pd.DataFrame) -> Any:
    view = df.copy()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.7), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [3.1, 1.0]})
    _mobile_add_candlesticks(axes[0], view)
    axes[0].plot(view.index, view["SuperTrend"].where(view["Direction"] > 0), color="#0f766e", linewidth=1.5, label="Bull ST")
    axes[0].plot(view.index, view["SuperTrend"].where(view["Direction"] < 0), color="#b42318", linewidth=1.5, label="Bear ST")
    long_flips = view[view["LongFlip"]]
    short_flips = view[view["ShortFlip"]]
    if not long_flips.empty:
        axes[0].scatter(long_flips.index, long_flips["Low"] * 0.995, color="#0f766e", marker="^", s=28)
    if not short_flips.empty:
        axes[0].scatter(short_flips.index, short_flips["High"] * 1.005, color="#b42318", marker="v", s=28)
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)
    axes[1].bar(view.index, view["ATR"], color="#64748b", width=2)
    _mobile_style_axis(axes[0], "Price")
    _mobile_style_axis(axes[1], "ATR")
    return _mobile_finalize_figure(fig)


def build_mobile_vix_fix_figure(df: pd.DataFrame) -> Any:
    view = df.copy()
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 8.0), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [2.3, 1.2, 1.2]})
    _mobile_add_candlesticks(axes[0], view)
    oversold_exit = view[view["OversoldExit"]]
    overbought_exit = view[view["OverboughtExit"]]
    if not oversold_exit.empty:
        axes[0].scatter(oversold_exit.index, oversold_exit["Low"] * 0.99, color="#0f766e", marker="D", s=22)
    if not overbought_exit.empty:
        axes[0].scatter(overbought_exit.index, overbought_exit["High"] * 1.01, color="#b42318", marker="D", s=22)
    axes[1].bar(view.index, view["WVF"], color=np.where(view["Oversold"], "#0f766e", "#94a3b8"), width=3)
    axes[1].plot(view.index, view["WVF_Upper"], color="#111827", linewidth=1.0, linestyle=":")
    axes[1].plot(view.index, view["WVF_RangeHigh"], color="#dd6b20", linewidth=1.0)
    axes[2].bar(view.index, view["WVF_Inverse"], color=np.where(view["Overbought"], "#b42318", "#cbd5e1"), width=2)
    axes[2].plot(view.index, view["WVF_Inv_Upper"], color="#334155", linewidth=1.0, linestyle=":")
    axes[2].plot(view.index, view["WVF_Inv_RangeHigh"], color="#7c3aed", linewidth=1.0)
    _mobile_style_axis(axes[0], "Price")
    _mobile_style_axis(axes[1], "WVF")
    _mobile_style_axis(axes[2], "Inverse")
    return _mobile_finalize_figure(fig)


def build_mobile_squeeze_figure(df: pd.DataFrame) -> Any:
    view = df.copy()
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 6.9), sharex=True, constrained_layout=True, gridspec_kw={"height_ratios": [3.1, 1.0]})
    _mobile_add_candlesticks(axes[0], view)
    for column, color, style in [
        ("UpperBB", "#b42318", "-"),
        ("LowerBB", "#b42318", "-"),
        ("UpperKC", "#0f766e", "--"),
        ("LowerKC", "#0f766e", "--"),
    ]:
        axes[0].plot(view.index, view[column], color=color, linewidth=1.0, linestyle=style)
    axes[1].bar(view.index, view["Momentum"], color=np.where(view["Momentum"] >= 0, "#0f766e", "#b42318"), width=3)
    marker_colors = np.where(view["SqueezeOn"], "#111827", np.where(view["SqueezeOff"], "#f59e0b", "#cbd5e1"))
    axes[1].scatter(view.index, np.zeros(len(view)), color=marker_colors, s=10)
    axes[1].axhline(0, color="#64748b", linestyle=":", linewidth=1)
    _mobile_style_axis(axes[0], "Price")
    _mobile_style_axis(axes[1], "Momentum")
    return _mobile_finalize_figure(fig)


def build_mobile_market_figure(market_data: dict[str, Any]) -> Any:
    plot_df = _mobile_view_slice(market_data["plot_df"]).copy()
    latest_factors = market_data["latest_factors"]
    normalized_spy = (plot_df["SPY"] / plot_df["SPY"].iloc[0]) - 1
    fig, axes = plt.subplots(4, 1, figsize=(6.4, 10.0), constrained_layout=True, gridspec_kw={"height_ratios": [1.6, 1.3, 1.0, 1.3]})
    _draw_mobile_gauge(axes[0], market_data["latest_score"] * 100, "Fear <-> Greed")
    axes[1].plot(plot_df.index, plot_df["Score"], color="#0f766e", linewidth=1.8, label="Smart score")
    axes[1].plot(plot_df.index, normalized_spy, color="#dd6b20", linewidth=1.2, linestyle="--", label="SPY return")
    axes[1].axhline(0.8, color="#b42318", linestyle=":", linewidth=1)
    axes[1].axhline(0.2, color="#0f766e", linestyle=":", linewidth=1)
    axes[1].legend(loc="upper left", fontsize=8, frameon=False)
    axes[2].bar(latest_factors.index.astype(str), latest_factors.values, color="#2563eb")
    axes[2].tick_params(axis="x", rotation=25, labelsize=8)
    axes[3].plot(plot_df.index, plot_df["Breadth"], color="#dd6b20", linewidth=1.2, label="Breadth")
    axes[3].plot(plot_df.index, plot_df["Sector"], color="#2563eb", linewidth=1.2, label="Sector")
    axes[3].plot(plot_df.index, plot_df["Credit"], color="#0f766e", linewidth=1.2, label="Credit")
    axes[3].axhline(0.5, color="#64748b", linestyle=":", linewidth=1)
    axes[3].legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    _mobile_style_axis(axes[1], "Score")
    _mobile_style_axis(axes[2], "Factor", x_axis_type="category")
    _mobile_style_axis(axes[3], "Health")
    return _mobile_finalize_figure(fig)


def estimate_zero_gamma_level(strike_view: pd.DataFrame, spot: float | None = None) -> float:
    if strike_view is None or strike_view.empty:
        return np.nan
    if "strike" not in strike_view.columns or "gex" not in strike_view.columns:
        return np.nan

    view = strike_view[["strike", "gex"]].dropna().sort_values("strike")
    if view.shape[0] < 2:
        return np.nan

    strikes = view["strike"].to_numpy(dtype=float)
    gex_values = view["gex"].to_numpy(dtype=float)
    candidates: list[float] = [float(level) for level in strikes[np.isclose(gex_values, 0.0)]]

    for idx in range(1, len(strikes)):
        prev_gex = gex_values[idx - 1]
        curr_gex = gex_values[idx]
        if np.isnan(prev_gex) or np.isnan(curr_gex) or prev_gex == 0 or curr_gex == 0:
            continue
        if np.sign(prev_gex) != np.sign(curr_gex):
            level = strikes[idx - 1] + ((0.0 - prev_gex) * (strikes[idx] - strikes[idx - 1]) / (curr_gex - prev_gex))
            candidates.append(float(level))

    if not candidates:
        return np.nan

    anchor = float(spot) if spot is not None and not np.isnan(spot) else float(np.median(strikes))
    return float(min(candidates, key=lambda value: abs(value - anchor)))


def build_mobile_options_figure(options_data: dict[str, Any]) -> Any:
    view = options_data["strike_view"].copy()
    strike_step = float(view["strike"].diff().dropna().median()) if view.shape[0] > 1 else 5.0
    bar_width = max(strike_step * 0.8, 1.0)
    zero_gamma_level = float(options_data.get("zero_gamma_level", np.nan))
    fig, axes = plt.subplots(6, 1, figsize=(6.4, 12.0), constrained_layout=True, gridspec_kw={"height_ratios": [1.15, 1.15, 1.0, 1.0, 0.9, 0.9]})
    axes[0].bar(view["strike"], view["gex"] / 1e9, color="#2563eb", width=bar_width, label="Net GEX")
    axes[0].axhline(0, color="#64748b", linestyle=":", linewidth=1)
    axes[0].axvline(options_data["spot"], color="#64748b", linestyle=":", linewidth=1.1, label="Spot")
    if not np.isnan(zero_gamma_level):
        axes[0].axvline(zero_gamma_level, color="#7c3aed", linestyle="--", linewidth=1.2, label="Zero Gamma")
        axes[1].axvline(zero_gamma_level, color="#7c3aed", linestyle="--", linewidth=1.1)
        axes[2].axvline(zero_gamma_level, color="#7c3aed", linestyle="--", linewidth=1.1)
        axes[3].axvline(zero_gamma_level, color="#7c3aed", linestyle="--", linewidth=1.1)
        axes[0].text(0.98, 0.94, f"Zero Gamma {zero_gamma_level:,.1f}", transform=axes[0].transAxes, ha="right", va="top", fontsize=8, color="#7c3aed")
    axes[1].plot(view["strike"], view["pain"], color="#b42318", linewidth=1.8)
    axes[1].fill_between(view["strike"], view["pain"], color="#fee2e2", alpha=0.5)
    axes[1].axvline(options_data["max_pain"], color="#b42318", linestyle=":", linewidth=1)
    axes[2].bar(view["strike"], view["vanna"] / 1e9, color="#0f766e", width=bar_width)
    axes[2].axhline(0, color="#64748b", linestyle=":", linewidth=1)
    axes[2].axvline(options_data["spot"], color="#64748b", linestyle=":", linewidth=1)
    axes[3].bar(view["strike"], view["charm"] / 1e6, color="#dd6b20", width=bar_width)
    axes[3].axhline(0, color="#64748b", linestyle=":", linewidth=1)
    axes[3].axvline(options_data["spot"], color="#64748b", linestyle=":", linewidth=1)
    axes[4].plot(["9D", "30D", "3M"], [options_data["vix9d"], options_data["vix30d"], options_data["vix3m"]], color="#7c3aed", linewidth=1.8, marker="o")
    score = max(0.0, min(100.0, float(options_data["put_call_ratio"]) * 50.0))
    _draw_mobile_gauge(axes[5], score, f"PCR {options_data['put_call_ratio']:.2f}", value_y=0.14, subtitle_y=-0.18, needle_length=0.64, needle_width=3.2, value_size=20)
    for axis in axes[:4]:
        axis.set_xlim(options_data["lower_bound"], options_data["upper_bound"])
    axes[0].legend(loc="upper left", fontsize=8, frameon=False, ncol=3)
    _mobile_style_axis(axes[0], "GEX (B)", x_axis_type="numeric")
    _mobile_style_axis(axes[1], "Pain", x_axis_type="numeric")
    _mobile_style_axis(axes[2], "Vanna (B)", x_axis_type="numeric")
    _mobile_style_axis(axes[3], "Charm (M)", x_axis_type="numeric")
    _mobile_style_axis(axes[4], "VIX", x_axis_type="category")
    return _mobile_finalize_figure(fig)


def build_mobile_fed_watch_figure(fed_watch_data: dict[str, Any]) -> Any:
    frame = fed_watch_data["frame"].copy()
    subset = [column for column in ["SOFR", "IORB_Combined", "WALCL", "TGA", "ON_RRP"] if column in frame.columns]
    if subset:
        frame = frame.dropna(how="all", subset=subset)
    fig, axes = plt.subplots(6, 1, figsize=(6.4, 13.0), constrained_layout=True)
    if _frame_has_data(frame, "FED_Treasuries"):
        axes[0].plot(frame.index, frame["FED_Treasuries"], color="#b42318", linewidth=1.5, linestyle="--", label="Fed Treasuries")
    if _frame_has_data(frame, "Bank_Treasuries"):
        axes[0].plot(frame.index, frame["Bank_Treasuries"], color="#2563eb", linewidth=1.5, label="Bank Treasuries")
    axes[0].legend(loc="upper left", fontsize=8, frameon=False)

    if _frame_has_data(frame, "Spread_Carry"):
        carry = frame["Spread_Carry"]
        axes[1].plot(frame.index, carry, color="#7f1d1d", linewidth=1.7)
        axes[1].fill_between(frame.index, 0, carry, where=carry >= 0, color="#0f766e", alpha=0.18)
        axes[1].fill_between(frame.index, 0, carry, where=carry < 0, color="#b42318", alpha=0.14)
        axes[1].axhline(0, color="#64748b", linestyle=":", linewidth=1)

    if _frame_has_data(frame, "SOFR"):
        axes[2].plot(frame.index, frame["SOFR"], color="#111827", linewidth=1.4, label="SOFR")
    if _frame_has_data(frame, "IORB_Combined"):
        axes[2].plot(frame.index, frame["IORB_Combined"], color="#b42318", linewidth=1.4, linestyle="--", label="IORB / IOER")
    if _frame_has_data(frame, "SOFR") and _frame_has_data(frame, "IORB_Combined"):
        stress_mask = frame["SOFR"] > frame["IORB_Combined"]
        if bool(stress_mask.fillna(False).any()):
            axes[2].fill_between(
                frame.index,
                frame["IORB_Combined"],
                frame["SOFR"],
                where=stress_mask,
                color="#b42318",
                alpha=0.14,
            )
    axes[2].legend(loc="upper left", fontsize=8, frameon=False)

    if _frame_has_data(frame, "SOFR_Vol"):
        axes[3].plot(frame.index, frame["SOFR_Vol"], color="#7c3aed", linewidth=1.7)

    if _frame_has_data(frame, "TGA"):
        axes[4].plot(frame.index, frame["TGA"], color="#dd6b20", linewidth=1.4, label="TGA")
    if _frame_has_data(frame, "ON_RRP"):
        axes[4].plot(frame.index, frame["ON_RRP"], color="#2563eb", linewidth=1.4, label="ON RRP")
    if _frame_has_data(frame, "Reserves"):
        axes[4].plot(frame.index, frame["Reserves"], color="#0f766e", linewidth=1.4, linestyle="--", label="Reserves")
    axes[4].legend(loc="upper left", fontsize=8, frameon=False, ncol=2)

    if _frame_has_data(frame, "Spread_Curve"):
        curve = frame["Spread_Curve"]
        axes[5].plot(frame.index, curve, color="#102a43", linewidth=1.7)
        axes[5].fill_between(frame.index, 0, curve, where=curve < 0, color="#64748b", alpha=0.16)
        axes[5].axhline(0, color="#b42318", linestyle=":", linewidth=1)

    for axis, label in zip(
        axes,
        ["Billions USD", "Spread (%)", "Rate (%)", "Billions USD", "Billions USD", "Spread (%)"],
    ):
        _mobile_style_axis(axis, label)
    return _mobile_finalize_figure(fig)


def build_mobile_canary_attack_figure(df_assets: pd.DataFrame, top_n: int = 6) -> Any:
    view = df_assets.head(top_n).copy().sort_values("실시간 평균 모멘텀(%)", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.2), constrained_layout=True)
    colors = ["#0f766e" if (not pd.isna(v) and v >= 0) else "#b42318" for v in view["실시간 평균 모멘텀(%)"]]
    ax.barh(view["자산"], view["실시간 평균 모멘텀(%)"], color=colors)
    ax.axvline(0, color="#64748b", linestyle=":", linewidth=1)
    _mobile_style_axis(ax, "Momentum (%)", x_axis_type="numeric")
    return _mobile_finalize_figure(fig)


def build_mobile_etf_sortino_figure(leaderboard: pd.DataFrame, top_n: int = 10) -> Any:
    view = leaderboard.head(top_n).copy().sort_values("Sortino", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    labels = _mobile_shorten_labels(view["Ticker"].astype(str).tolist(), max_len=10)
    colors = ["#0f766e" if not np.isnan(value) and value >= 0 else "#b42318" for value in view["Sortino"]]
    ax.barh(labels, view["Sortino"], color=colors)
    ax.axvline(0, color="#64748b", linestyle=":", linewidth=1)
    _mobile_style_axis(ax, "Sortino", x_axis_type="numeric")
    return _mobile_finalize_figure(fig)


def build_mobile_etf_sector_share_figure(sector_df: pd.DataFrame, top_n: int = 8) -> Any:
    view = sector_df.head(top_n).copy().sort_values("Share", ascending=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    ax.barh(_mobile_shorten_labels(view["Sector"].astype(str).tolist()), view["Share"], color="#2563eb")
    _mobile_style_axis(ax, "Share", x_axis_type="numeric")
    return _mobile_finalize_figure(fig)


def compute_overview_figure(df: pd.DataFrame) -> go.Figure:
    view = trim_to_history_window(df).copy()
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
    view = trim_to_history_window(df)
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
    view = trim_to_history_window(df).copy()
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
    view = trim_to_history_window(df).copy()
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

    view = trim_to_history_window(work).copy()
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
    return trim_to_history_window(work).copy()


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
    return trim_to_history_window(work).copy()


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
    return trim_to_history_window(work).copy()


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
        "plot_df": plot_df,
        "status": classify_market_score(latest_score),
    }


def compute_market_fear_greed() -> dict[str, Any] | None:
    return get_or_refresh_daily_payload("market_pulse", _fetch_market_fear_greed)


def build_market_figure(market_data: dict[str, Any]) -> go.Figure:
    plot_df = trim_to_history_window(market_data["plot_df"]).copy()
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
            title={"text": ""},
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
    fig = apply_figure_style(fig, title="", height=840, legend_y=1.10)
    fig.update_layout(
        margin=dict(l=22, r=22, t=118, b=18),
        legend=dict(
            orientation="h",
            y=1.10,
            x=0,
            font=dict(size=11),
            itemwidth=72,
            tracegroupgap=8,
        ),
    )
    fig.update_annotations(font=dict(size=12))
    return fig


def _frame_has_data(frame: pd.DataFrame, column: str) -> bool:
    return column in frame.columns and not frame[column].dropna().empty


def render_fed_watch_header(fed_watch_data: dict[str, Any]) -> None:
    frame = fed_watch_data["frame"]
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    period_label = str(fed_watch_data["period"]).upper()
    net_liquidity_value = _latest_frame_value(frame, "Net_Liquidity")
    tga_value = _latest_frame_value(frame, "TGA")
    on_rrp_value = _latest_frame_value(frame, "ON_RRP")
    stress_value = _latest_frame_value(frame, "Stress_SOFR_IORB")

    net_liquidity_delta = _series_delta(frame["Net_Liquidity"]) if _frame_has_data(frame, "Net_Liquidity") else np.nan
    tga_delta = _series_delta(frame["TGA"]) if _frame_has_data(frame, "TGA") else np.nan
    on_rrp_delta = _series_delta(frame["ON_RRP"]) if _frame_has_data(frame, "ON_RRP") else np.nan

    st.markdown(
        f"""
        <div class="hero">
            <h1>Fed Watch · Liquidity Plumbing</h1>
            <p>
                FRED-based macro quick view across the current {period_label} history window.
                Net liquidity follows WALCL - TGA - ON RRP. Last refresh: {current_time}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(4)
    with cards[0]:
        render_metric_card(
            "Net Liquidity",
            format_billions(net_liquidity_value),
            f"WALCL - TGA - ON RRP, as of {_format_asof_date(fed_watch_data['card_dates'].get('Net_Liquidity'))}",
            "bull" if not np.isnan(net_liquidity_delta) and net_liquidity_delta >= 0 else "bear" if not np.isnan(net_liquidity_delta) else "neutral",
        )
    with cards[1]:
        render_metric_card(
            "TGA",
            format_billions(tga_value),
            f"20D change {format_billions_change(tga_delta)} · as of {_format_asof_date(fed_watch_data['card_dates'].get('TGA'))}",
            "bear" if not np.isnan(tga_delta) and tga_delta > 0 else "bull" if not np.isnan(tga_delta) else "neutral",
        )
    with cards[2]:
        render_metric_card(
            "ON RRP",
            format_billions(on_rrp_value),
            f"20D change {format_billions_change(on_rrp_delta)} · as of {_format_asof_date(fed_watch_data['card_dates'].get('ON_RRP'))}",
            "bull" if not np.isnan(on_rrp_delta) and on_rrp_delta <= 0 else "accent" if not np.isnan(on_rrp_delta) else "neutral",
        )
    with cards[3]:
        render_metric_card(
            "SOFR - IORB",
            format_bps(stress_value),
            f"Plumbing stress spread, as of {_format_asof_date(fed_watch_data['card_dates'].get('Spread'))}",
            "bear" if not np.isnan(stress_value) and stress_value > 0 else "bull" if not np.isnan(stress_value) else "neutral",
        )


def build_fed_watch_figure(fed_watch_data: dict[str, Any]) -> go.Figure:
    frame = fed_watch_data["frame"].copy()
    subset = [column for column in ["SOFR", "IORB_Combined", "WALCL", "TGA", "ON_RRP"] if column in frame.columns]
    if subset:
        frame = frame.dropna(how="all", subset=subset)
    if frame.empty:
        frame = fed_watch_data["frame"].copy()

    fig = make_subplots(
        rows=3,
        cols=2,
        specs=[[{}, {}], [{}, {}], [{"secondary_y": True}, {}]],
        subplot_titles=(
            "Collateral Supply: Fed vs Banks",
            "Dealer Incentive: 10Y - SOFR",
            "Plumbing Stress: SOFR vs IORB",
            "Funding Volume: SOFR Volume",
            "Treasury and Fed Drains",
            "Traditional Curve: 10Y - 3M",
        ),
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
    )

    if _frame_has_data(frame, "FED_Treasuries"):
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["FED_Treasuries"], name="Fed Treasuries", line=dict(color="#b42318", width=2, dash="dash")),
            row=1,
            col=1,
        )
    if _frame_has_data(frame, "Bank_Treasuries"):
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["Bank_Treasuries"], name="Bank Treasuries", line=dict(color="#2563eb", width=2)),
            row=1,
            col=1,
        )

    if _frame_has_data(frame, "Spread_Carry"):
        carry = frame["Spread_Carry"]
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=carry.where(carry >= 0),
                mode="lines",
                line=dict(color="rgba(15, 118, 110, 0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=np.zeros(len(frame)),
                mode="lines",
                line=dict(color="rgba(15, 118, 110, 0)"),
                fill="tonexty",
                fillcolor="rgba(15, 118, 110, 0.20)",
                name="Positive carry",
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=carry.where(carry < 0),
                mode="lines",
                line=dict(color="rgba(180, 35, 24, 0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=np.zeros(len(frame)),
                mode="lines",
                line=dict(color="rgba(180, 35, 24, 0)"),
                fill="tonexty",
                fillcolor="rgba(180, 35, 24, 0.18)",
                name="Negative carry",
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(x=frame.index, y=carry, name="10Y - SOFR", line=dict(color="#7f1d1d", width=2)),
            row=1,
            col=2,
        )
        fig.add_hline(y=0, line_color="#64748b", line_dash="dot", row=1, col=2)

    if _frame_has_data(frame, "SOFR") and _frame_has_data(frame, "IORB_Combined"):
        stress_mask = frame["SOFR"] > frame["IORB_Combined"]
        if bool(stress_mask.fillna(False).any()):
            fig.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame["IORB_Combined"].where(stress_mask),
                    mode="lines",
                    line=dict(color="rgba(180, 35, 24, 0)"),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=frame.index,
                    y=frame["SOFR"].where(stress_mask),
                    mode="lines",
                    line=dict(color="rgba(180, 35, 24, 0)"),
                    fill="tonexty",
                    fillcolor="rgba(180, 35, 24, 0.22)",
                    name="Stress zone",
                    hoverinfo="skip",
                ),
                row=2,
                col=1,
            )
        fig.add_trace(go.Scatter(x=frame.index, y=frame["SOFR"], name="SOFR", line=dict(color="#111827", width=1.9)), row=2, col=1)
        fig.add_trace(go.Scatter(x=frame.index, y=frame["IORB_Combined"], name="IORB / IOER", line=dict(color="#b42318", width=2, dash="dash")), row=2, col=1)

    if _frame_has_data(frame, "SOFR_Vol"):
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["SOFR_Vol"], name="SOFR Volume", line=dict(color="#7c3aed", width=2)),
            row=2,
            col=2,
        )

    if _frame_has_data(frame, "TGA"):
        fig.add_trace(go.Scatter(x=frame.index, y=frame["TGA"], name="TGA", line=dict(color="#dd6b20", width=2)), row=3, col=1, secondary_y=False)
    if _frame_has_data(frame, "ON_RRP"):
        fig.add_trace(go.Scatter(x=frame.index, y=frame["ON_RRP"], name="ON RRP", line=dict(color="#2563eb", width=2)), row=3, col=1, secondary_y=False)
    if _frame_has_data(frame, "Reserves"):
        fig.add_trace(
            go.Scatter(x=frame.index, y=frame["Reserves"], name="Reserve Balances", line=dict(color="#0f766e", width=2, dash="dash")),
            row=3,
            col=1,
            secondary_y=True,
        )

    if _frame_has_data(frame, "Spread_Curve"):
        curve = frame["Spread_Curve"]
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=curve.where(curve < 0),
                mode="lines",
                line=dict(color="rgba(100, 116, 139, 0)"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=3,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=np.zeros(len(frame)),
                mode="lines",
                line=dict(color="rgba(100, 116, 139, 0)"),
                fill="tonexty",
                fillcolor="rgba(100, 116, 139, 0.18)",
                name="Inverted curve",
                hoverinfo="skip",
            ),
            row=3,
            col=2,
        )
        fig.add_trace(go.Scatter(x=frame.index, y=curve, name="10Y - 3M", line=dict(color="#102a43", width=2)), row=3, col=2)
        fig.add_hline(y=0, line_color="#b42318", line_dash="dot", row=3, col=2)

    fig.update_yaxes(title_text="Billions USD", row=1, col=1)
    fig.update_yaxes(title_text="Spread (%)", row=1, col=2)
    fig.update_yaxes(title_text="Rate (%)", row=2, col=1)
    fig.update_yaxes(title_text="Billions USD", row=2, col=2)
    fig.update_yaxes(title_text="TGA / ON RRP (B)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Reserves (B)", row=3, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Spread (%)", row=3, col=2)
    fig.update_xaxes(tickformat="%Y-%m")

    fig = apply_figure_style(fig, title="", height=1180, legend_y=1.12)
    fig.update_layout(
        margin=dict(l=22, r=22, t=118, b=18),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0,
            font=dict(size=11),
            itemwidth=72,
            tracegroupgap=8,
        ),
    )
    fig.update_annotations(font=dict(size=12))
    return fig


def _fed_watch_display_warnings(fed_watch_data: dict[str, Any]) -> list[str]:
    warnings_list = list(fed_watch_data.get("warnings", []))
    frame = fed_watch_data.get("frame")
    iorb_available = isinstance(frame, pd.DataFrame) and _frame_has_data(frame, "IORB")
    filtered: list[str] = []
    for warning in warnings_list:
        if iorb_available and "IOER (IOER) returned no usable rows." in warning:
            continue
        filtered.append(warning)
    return filtered


def render_fed_watch_dashboard(fed_watch_data: dict[str, Any]) -> None:
    if fed_watch_data["frame"].dropna(how="all").empty:
        st.warning("FRED data is temporarily unavailable for this window. Diagnostics below show which series failed or timed out.")

    stale_fallback_count = int(fed_watch_data.get("stale_fallback_count", 0) or 0)
    if stale_fallback_count > 0:
        cache_date = fed_watch_data.get("stale_cache_date")
        cache_text = f" from {cache_date}" if cache_date else ""
        st.info(f"Using stale cache for {stale_fallback_count} slow FRED series{cache_text}.")

    warnings_list = _fed_watch_display_warnings(fed_watch_data)
    if warnings_list:
        st.info("Partial FRED coverage: " + " | ".join(warnings_list[:4]))

    frame = fed_watch_data.get("frame")
    iorb_available = isinstance(frame, pd.DataFrame) and _frame_has_data(frame, "IORB")
    ioer_available = isinstance(frame, pd.DataFrame) and _frame_has_data(frame, "IOER")
    if iorb_available and not ioer_available:
        st.caption("IOER is a legacy series. Current plumbing spread and dashboard cards continue to use IORB when it is available.")

    st.markdown(
        '<div class="section-note">This view tracks collateral supply, funding stress, treasury cash drains, and a simple net-liquidity proxy using official FRED series. Values are normalized to billions of dollars where applicable.</div>',
        unsafe_allow_html=True,
    )
    if mobile_charts_enabled():
        st.markdown("#### Fed Watch Liquidity Dashboard")
        fig = build_mobile_fed_watch_figure(fed_watch_data)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        render_plotly_chart(build_fed_watch_figure(fed_watch_data))

    with st.expander("Fed Watch diagnostics", expanded=False):
        status_frame = fed_watch_data["source_status"].copy()
        st.dataframe(status_frame, use_container_width=True, hide_index=True)


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


def is_equity_etf(symbol: str, name: str = "", category: str = "", quote_type: str = "") -> bool:
    symbol_upper = (symbol or "").upper()
    text = " ".join(part for part in [name, category, quote_type] if part).lower()
    excluded_terms = [
        "bond", "treasury", "income", "ultra", "2x", "3x", "leveraged", "inverse", "short", "bear",
        "volatility", "vix", "currency", "fx", "commodity", "gold", "silver", "oil", "gas", "bitcoin",
        "crypto", "futures", "maturity", "single-stock", "single stock",
    ]
    if any(term in text for term in excluded_terms):
        return False
    excluded_symbols = {"SQQQ", "TQQQ", "SOXL", "SOXS", "SPXU", "UPRO", "TECL", "TECS", "UVXY", "SVXY"}
    if symbol_upper in excluded_symbols:
        return False
    return True


def extract_screen_quotes(screen_payload: Any) -> list[dict[str, Any]]:
    if isinstance(screen_payload, list):
        return [item for item in screen_payload if isinstance(item, dict)]
    if not isinstance(screen_payload, dict):
        return []
    for key in ["quotes", "data", "items"]:
        value = screen_payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            nested = value.get("quotes")
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
    finance_result = screen_payload.get("finance", {}).get("result", [])
    if finance_result and isinstance(finance_result[0], dict):
        quotes = finance_result[0].get("quotes")
        if isinstance(quotes, list):
            return [item for item in quotes if isinstance(item, dict)]
    return []


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_us_equity_etf_universe() -> pd.DataFrame:
    candidates: dict[str, dict[str, Any]] = {symbol: {"Ticker": symbol, "ETF Name": symbol} for symbol in ETF_UNIVERSE_SEEDS}
    screen_fn = getattr(yf, "screen", None)
    if callable(screen_fn):
        for query_name in ["most_actives", "day_gainers", "day_losers", "undervalued_growth_stocks", "aggressive_small_caps"]:
            try:
                payload = screen_fn(query_name, count=250)
            except Exception:
                continue
            for item in extract_screen_quotes(payload):
                symbol = str(item.get("symbol") or "").upper()
                quote_type = str(item.get("quoteType") or item.get("typeDisp") or "")
                name = str(item.get("shortName") or item.get("longName") or symbol)
                if not symbol or ("ETF" not in quote_type.upper() and symbol not in ETF_THEME_LABEL_MAP):
                    continue
                if not is_equity_etf(symbol, name=name, quote_type=quote_type):
                    continue
                candidates[symbol] = {"Ticker": symbol, "ETF Name": name}
    universe = pd.DataFrame(candidates.values()).drop_duplicates(subset=["Ticker"]).sort_values("Ticker").reset_index(drop=True)
    return universe


def normalize_sector_weights(raw_weights: Any) -> dict[str, float]:
    if raw_weights is None:
        return {}

    parsed: dict[str, float] = {}
    if isinstance(raw_weights, dict):
        iterator = raw_weights.items()
    elif isinstance(raw_weights, pd.Series):
        iterator = raw_weights.to_dict().items()
    elif isinstance(raw_weights, list):
        temp: dict[str, float] = {}
        for item in raw_weights:
            if isinstance(item, dict):
                sector_name = item.get("name") or item.get("sector") or item.get("label")
                weight = item.get("value") or item.get("weight") or item.get("percentage")
                if sector_name is not None and weight is not None:
                    temp[str(sector_name)] = float(weight)
        iterator = temp.items()
    else:
        return {}

    for key, value in iterator:
        try:
            numeric = float(str(value).replace("%", "").replace(",", "").strip())
        except Exception:
            continue
        if np.isnan(numeric):
            continue
        cleaned_key = str(key).replace("_", " ").replace("-", " ").title().strip()
        parsed[cleaned_key] = numeric

    if not parsed:
        return {}
    total = sum(abs(value) for value in parsed.values())
    if total <= 0:
        return {}
    scale = 100.0 if total > 1.5 else 1.0
    normalized = {key: value / scale for key, value in parsed.items()}
    total_normalized = sum(abs(value) for value in normalized.values())
    if total_normalized <= 0:
        return {}
    return {key: value / total_normalized for key, value in normalized.items() if value > 0}


def extract_sector_weights_from_funds_data(funds_data: Any) -> dict[str, float]:
    if funds_data is None:
        return {}
    if isinstance(funds_data, dict):
        for key in ["sector_weightings", "sectorWeightings", "sectorWeights"]:
            weights = normalize_sector_weights(funds_data.get(key))
            if weights:
                return weights
        return {}
    for attr in ["sector_weightings", "sectorWeightings", "sector_weights"]:
        if hasattr(funds_data, attr):
            weights = normalize_sector_weights(getattr(funds_data, attr))
            if weights:
                return weights
    return {}


def infer_etf_theme_label(ticker: str, etf_name: str, info: dict[str, Any] | None, sector_weights: dict[str, float]) -> str:
    if sector_weights:
        return max(sector_weights.items(), key=lambda item: item[1])[0]
    if ticker in ETF_THEME_LABEL_MAP:
        return ETF_THEME_LABEL_MAP[ticker]
    if info:
        for key in ["category", "fundFamily", "sector", "quoteType"]:
            value = info.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().title()
    name_lower = (etf_name or "").lower()
    keyword_map = {
        "semiconductor": "Semiconductors",
        "software": "Software",
        "technology": "Technology",
        "financial": "Financials",
        "health": "Health Care",
        "energy": "Energy",
        "industrial": "Industrials",
        "real estate": "Real Estate",
        "consumer": "Consumer",
        "internet": "Internet",
        "cloud": "Cloud",
        "cyber": "Cybersecurity",
        "dividend": "Dividend",
        "momentum": "Momentum",
        "quality": "Quality",
    }
    for keyword, label in keyword_map.items():
        if keyword in name_lower:
            return label
    return "Broad Market"


def safe_get_etf_metadata(ticker: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    sector_weights: dict[str, float] = {}
    name = ticker
    aum = np.nan
    try:
        instrument = yf.Ticker(ticker)
        try:
            info = instrument.get_info() or {}
        except Exception:
            info = getattr(instrument, "info", {}) or {}
        name = str(info.get("longName") or info.get("shortName") or ticker)
        aum_value = info.get("totalAssets") or info.get("netAssets")
        aum = float(aum_value) if aum_value not in (None, "") else np.nan
        try:
            funds_data = instrument.get_funds_data() if hasattr(instrument, "get_funds_data") else getattr(instrument, "funds_data", None)
        except Exception:
            funds_data = getattr(instrument, "funds_data", None)
        sector_weights = extract_sector_weights_from_funds_data(funds_data)
    except Exception:
        pass

    label = infer_etf_theme_label(ticker, name, info, sector_weights)
    return {
        "Ticker": ticker,
        "ETF Name": name,
        "Theme/Sector Label": label,
        "AUM": aum,
        "Sector Weights": sector_weights,
        "Sector Weight Coverage": float(sum(sector_weights.values())) if sector_weights else 0.0,
    }


def compute_sortino_ratio(series: pd.Series, annual_risk_free: float) -> tuple[float, float, float]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.shape[0] < SORTINO_LOOKBACK_DAYS:
        return np.nan, np.nan, np.nan
    daily_returns = clean.pct_change().dropna()
    if daily_returns.empty:
        return np.nan, np.nan, np.nan
    daily_target = (1 + annual_risk_free) ** (1 / SORTINO_ANNUALIZATION) - 1
    excess_returns = daily_returns - daily_target
    downside_returns = np.minimum(excess_returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside_returns))) * np.sqrt(SORTINO_ANNUALIZATION))
    annualized_excess = float(excess_returns.mean() * SORTINO_ANNUALIZATION)
    total_return = float(clean.iloc[-1] / clean.iloc[0] - 1)
    if downside_deviation <= 0 or np.isnan(downside_deviation):
        return np.nan, total_return, np.nan
    return annualized_excess / downside_deviation, total_return, downside_deviation


@st.cache_data(ttl=21600, show_spinner=False)
def compute_etf_sortino_leadership(top_n: int = DEFAULT_SORTINO_TOP_N) -> dict[str, Any] | None:
    universe = fetch_us_equity_etf_universe()
    if universe.empty:
        return None

    tickers = universe["Ticker"].tolist()
    try:
        raw = yf.download(
            tickers=tickers,
            period="1y",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
        )
    except Exception:
        return None
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return None

    try:
        close_df = raw["Close"].copy()
        volume_df = raw["Volume"].copy() if "Volume" in raw.columns.get_level_values(0) else pd.DataFrame(index=close_df.index, columns=close_df.columns)
    except Exception:
        return None

    close_df.index = normalize_datetime_index(close_df.index)
    volume_df.index = normalize_datetime_index(volume_df.index)
    close_df = close_df.sort_index().ffill()
    volume_df = volume_df.sort_index().fillna(0)

    annual_rf = get_risk_free_rate()
    records: list[dict[str, Any]] = []
    for ticker in tickers:
        if ticker not in close_df.columns:
            continue
        close_series = pd.to_numeric(close_df[ticker], errors="coerce").dropna()
        if close_series.shape[0] < SORTINO_LOOKBACK_DAYS:
            continue
        sortino, six_month_return, downside_vol = compute_sortino_ratio(close_series.tail(SORTINO_LOOKBACK_DAYS + 1), annual_rf)
        avg_dollar_volume = np.nan
        if ticker in volume_df.columns:
            vol_series = pd.to_numeric(volume_df[ticker], errors="coerce").reindex(close_series.index).fillna(0)
            avg_dollar_volume = float((close_series * vol_series).tail(63).mean()) if not close_series.empty else np.nan
        if np.isnan(avg_dollar_volume) or avg_dollar_volume < 2_000_000:
            continue
        records.append(
            {
                "Ticker": ticker,
                "Sortino": sortino,
                "6M Return": six_month_return,
                "Downside Vol": downside_vol,
                "Avg Dollar Volume": avg_dollar_volume,
            }
        )

    if not records:
        return None

    ranking = pd.DataFrame(records)
    ranking = ranking.sort_values(["Sortino", "Avg Dollar Volume"], ascending=[False, False], na_position="last").head(260).reset_index(drop=True)
    ranking["Percentile"] = ranking["Sortino"].rank(pct=True, ascending=False).fillna(0)

    enrich_count = max(top_n, SORTINO_SECTOR_TOP_COUNT)
    metadata_rows = [safe_get_etf_metadata(ticker) for ticker in ranking["Ticker"].head(enrich_count)]
    metadata_df = pd.DataFrame(metadata_rows)
    ranking = ranking.merge(metadata_df, on="Ticker", how="left")
    ranking["ETF Name"] = ranking["ETF Name"].fillna(ranking["Ticker"])
    ranking["Theme/Sector Label"] = ranking["Theme/Sector Label"].fillna("Broad Market")
    ranking["AUM or Avg Dollar Volume"] = np.where(
        ranking["AUM"].notna(),
        ranking["AUM"],
        ranking["Avg Dollar Volume"],
    )

    top_ranked = ranking.head(top_n).copy()
    sector_inputs = ranking.head(SORTINO_SECTOR_TOP_COUNT).copy()
    sector_weights_agg: dict[str, float] = {}
    covered_weight = 0.0
    total_weight = 0.0
    for _, row in sector_inputs.iterrows():
        base_weight = float(row["AUM"]) if not np.isnan(row["AUM"]) and float(row["AUM"]) > 0 else 1.0
        total_weight += base_weight
        weights = row.get("Sector Weights") if isinstance(row.get("Sector Weights"), dict) else {}
        if weights:
            covered_weight += base_weight
            for sector, weight in weights.items():
                sector_weights_agg[sector] = sector_weights_agg.get(sector, 0.0) + base_weight * float(weight)

    sector_df = pd.DataFrame(
        [{"Sector": sector, "Share": value} for sector, value in sector_weights_agg.items()]
    )
    if not sector_df.empty:
        sector_df["Share"] = sector_df["Share"] / sector_df["Share"].sum()
        sector_df = sector_df.sort_values("Share", ascending=False).reset_index(drop=True)
    coverage_ratio = covered_weight / total_weight if total_weight > 0 else 0.0
    top_sector = sector_df.iloc[0]["Sector"] if not sector_df.empty else "Unavailable"
    top_sector_share = float(sector_df.iloc[0]["Share"]) if not sector_df.empty else np.nan

    sector_stats_rows = []
    for _, row in sector_df.head(10).iterrows():
        sector_name = row["Sector"]
        contributor_count = int(
            sum(
                1
                for weights in sector_inputs["Sector Weights"]
                if isinstance(weights, dict) and sector_name in weights
            )
        )
        sector_stats_rows.append(
            {
                "Sector": sector_name,
                "Sector Share": row["Share"],
                "Contributor ETFs": contributor_count,
            }
        )
    sector_stats_df = pd.DataFrame(sector_stats_rows)

    leaderboard_df = top_ranked[
        [
            "Ticker",
            "ETF Name",
            "Theme/Sector Label",
            "Sortino",
            "6M Return",
            "Downside Vol",
            "AUM or Avg Dollar Volume",
            "Sector Weight Coverage",
            "Percentile",
        ]
    ].copy()
    return {
        "universe_size": int(ranking.shape[0]),
        "valid_sortino_count": int(ranking["Sortino"].notna().sum()),
        "median_sortino": float(ranking["Sortino"].dropna().median()) if not ranking["Sortino"].dropna().empty else np.nan,
        "top_etf": str(leaderboard_df.iloc[0]["Ticker"]) if not leaderboard_df.empty else "n/a",
        "top_etf_name": str(leaderboard_df.iloc[0]["ETF Name"]) if not leaderboard_df.empty else "n/a",
        "coverage_ratio": coverage_ratio,
        "top_sector": top_sector,
        "top_sector_share": top_sector_share,
        "leaderboard": leaderboard_df,
        "sector_share": sector_df,
        "sector_stats": sector_stats_df,
    }


def etf_sortino_signal_label(sortino_data: dict[str, Any] | None) -> tuple[str, str]:
    if not sortino_data:
        return "ETF sortino unavailable", "neutral"
    median_sortino = sortino_data.get("median_sortino", np.nan)
    top_sector = sortino_data.get("top_sector", "Unavailable")
    if np.isnan(median_sortino):
        return "ETF sortino unavailable", "neutral"
    if median_sortino >= 1.0:
        return f"Risk-adjusted leadership strong · {top_sector}", "bull"
    if median_sortino >= 0.5:
        return f"Leadership constructive · {top_sector}", "accent"
    return f"Leadership defensive · {top_sector}", "neutral"


def build_etf_sortino_figure(leaderboard: pd.DataFrame) -> go.Figure:
    view = leaderboard.head(20).copy().sort_values("Sortino", ascending=True)
    colors = ["#0f766e" if not np.isnan(value) and value >= 0 else "#b42318" for value in view["Sortino"]]
    labels = [f"{ticker} · {label}" for ticker, label in zip(view["Ticker"], view["Theme/Sector Label"])]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=view["Sortino"],
            y=labels,
            orientation="h",
            marker_color=colors,
            text=view["Sortino"].map(lambda value: "n/a" if np.isnan(value) else f"{value:.2f}"),
            textposition="outside",
            name="Sortino",
        )
    )
    fig.add_vline(x=0, line_color="#64748b", line_dash="dot")
    return apply_figure_style(fig, title="ETF Sortino Leadership Ranking", height=760, showlegend=False)


def build_etf_sector_share_figure(sector_df: pd.DataFrame) -> go.Figure:
    view = sector_df.head(10).copy().sort_values("Share", ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=view["Share"],
            y=view["Sector"],
            orientation="h",
            marker_color="#2563eb",
            text=view["Share"].map(lambda value: f"{value:.1%}"),
            textposition="outside",
            name="Sector share",
        )
    )
    return apply_figure_style(fig, title="Top ETF Sector Share", height=620, showlegend=False)


def render_etf_sortino_dashboard(sortino_data: dict[str, Any]) -> None:
    summary_cards = st.columns(4)
    with summary_cards[0]:
        render_metric_card("Universe Size", f"{sortino_data['universe_size']}", "Liquid US equity ETFs screened", "neutral")
    with summary_cards[1]:
        render_metric_card("Valid Sortino", f"{sortino_data['valid_sortino_count']}", "Recent 6M sortino coverage", "neutral")
    with summary_cards[2]:
        render_metric_card("Median Sortino", "n/a" if np.isnan(sortino_data["median_sortino"]) else f"{sortino_data['median_sortino']:.2f}", "Cross-sectional median", "accent")
    with summary_cards[3]:
        render_metric_card("Top ETF", sortino_data["top_etf"], sortino_data["top_etf_name"], "bull")

    render_chart(
        "ETF Sortino Leadership Ranking",
        build_etf_sortino_figure(sortino_data["leaderboard"]),
        mobile_builder=lambda: build_mobile_etf_sortino_figure(sortino_data["leaderboard"]),
    )

    table_left, table_right = st.columns([1.1, 0.9])
    with table_left:
        st.markdown("#### ETF Sortino Ranking")
        ranking_table = sortino_data["leaderboard"].copy()
        ranking_table["Sortino"] = ranking_table["Sortino"].map(lambda value: "n/a" if np.isnan(value) else f"{value:.2f}")
        ranking_table["6M Return"] = ranking_table["6M Return"].map(format_pct)
        ranking_table["Downside Vol"] = ranking_table["Downside Vol"].map(format_pct)
        ranking_table["AUM or Avg Dollar Volume"] = ranking_table["AUM or Avg Dollar Volume"].map(lambda value: "n/a" if np.isnan(value) else f"{value / 1e9:,.2f}B")
        ranking_table["Sector Weight Coverage"] = ranking_table["Sector Weight Coverage"].map(lambda value: "n/a" if np.isnan(value) else f"{value:.0%}")
        ranking_table["Percentile"] = ranking_table["Percentile"].map(lambda value: "n/a" if np.isnan(value) else f"{value:.0%}")
        st.dataframe(ranking_table, use_container_width=True, hide_index=True)
    with table_right:
        st.markdown("#### Sector Share")
        if sortino_data["sector_share"].empty:
            st.info("Sector holdings coverage is limited for the current ETF leadership set.")
        else:
            render_chart(
                "Top ETF Sector Share",
                build_etf_sector_share_figure(sortino_data["sector_share"]),
                mobile_builder=lambda: build_mobile_etf_sector_share_figure(sortino_data["sector_share"]),
            )
            sector_stats = sortino_data["sector_stats"].copy()
            sector_stats["Sector Share"] = sector_stats["Sector Share"].map(lambda value: f"{value:.1%}")
            st.dataframe(sector_stats, use_container_width=True, hide_index=True)

    st.markdown(
        f'<div class="section-note">Top ETF leadership is ranked by 6M Sortino ratio using a daily risk-free hurdle. Sector share aggregates the top {SORTINO_SECTOR_TOP_COUNT} ETFs using holdings-based sector weights when available. Current leading sector: <strong>{sortino_data["top_sector"]}</strong>. Coverage ratio: <strong>{sortino_data["coverage_ratio"]:.0%}</strong>.</div>',
        unsafe_allow_html=True,
    )


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


def nearest_spx_expiry(expiries: list[str]) -> str | None:
    if not expiries:
        return None
    today = pd.Timestamp.now().normalize()
    future_expiries = [expiry for expiry in expiries if pd.Timestamp(expiry) >= today]
    return future_expiries[0] if future_expiries else expiries[0]


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

    default_expiry = nearest_spx_expiry(expiries)
    selected_expiry = expiry if expiry in expiries else default_expiry
    if selected_expiry is None:
        return None
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
    zero_gamma_level = estimate_zero_gamma_level(strike_view, spot)

    lower_bound = float(spot * (1 - spot_range_pct))
    upper_bound = float(spot * (1 + spot_range_pct))
    if not np.isnan(zero_gamma_level) and spot > 0 and abs((zero_gamma_level / spot) - 1) <= 0.25:
        lower_bound = min(lower_bound, float(zero_gamma_level * 0.995))
        upper_bound = max(upper_bound, float(zero_gamma_level * 1.005))

    strike_view = strike_view[
        (strike_view["strike"] >= lower_bound)
        & (strike_view["strike"] <= upper_bound)
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
        "zero_gamma_level": zero_gamma_level,
        "net_gex": float(strike_view["gex"].sum()),
        "net_vanna": float(strike_view["vanna"].sum()),
        "net_charm": float(strike_view["charm"].sum()),
        "vix9d": vix_data["vix9d"],
        "vix30d": vix_data["vix30d"],
        "vix3m": vix_data["vix3m"],
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
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
    zero_gamma_level = float(options_data.get("zero_gamma_level", np.nan))
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
        if not np.isnan(zero_gamma_level):
            fig.add_shape(
                type="line",
                x0=zero_gamma_level,
                x1=zero_gamma_level,
                y0=0,
                y1=1,
                xref=xref,
                yref=yref,
                line=dict(color="#7c3aed", width=1.2, dash="dot"),
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
    if not np.isnan(zero_gamma_level):
        fig.add_annotation(
            x=zero_gamma_level,
            y=1.02,
            xref="x",
            yref="y domain",
            text="Zero Gamma",
            showarrow=False,
            font=dict(color="#7c3aed", size=10),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#7c3aed",
            borderwidth=1,
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


def format_billions(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:,.0f}B"


def format_billions_change(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value:+,.0f}B"


def format_bps(value: float) -> str:
    return "n/a" if np.isnan(value) else f"{value * 100:+.1f} bps"


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
    etf_sortino_data: dict[str, Any] | None,
    canary_data: dict[str, Any] | None,
) -> None:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    active_view_label = display_view_name(active_view)
    view_status_map = {
        "Elder Impulse": (summary.elder_label, "Momentum and trend filter status."),
        "TD Sequential": (summary.td_label, "Setup and countdown exhaustion context."),
        "Robust STL": (summary.stl_label, "Cycle stretch versus smoothed trend."),
        "SMC": ("Loaded", "Smart money zones and active structures."),
        "SuperTrend": (supertrend_signal_label(supertrend_data)[0], "ATR band flips and live regime line."),
        "Williams Vix Fix": (vix_fix_signal_label(vix_fix_data)[0], "Panic/complacency spikes versus recent extremes."),
        "Squeeze Momentum": (squeeze_signal_label(squeeze_data)[0], "Bollinger versus Keltner compression state."),
        "Canary Momentum": (canary_signal_label(canary_data)[0], "Dual-speed canary regime and rotation ranking."),
        "Market Pulse": (market_data["status"][0] if market_data else "Not loaded", "Cross-asset risk appetite backdrop."),
        "Options Flow": (options_signal_label(options_data)[0], f"SPX max pain {options_data['max_pain']:.2f}" if options_data else "SPX options unavailable"),
        "ETF Sortino Leadership": (etf_sortino_signal_label(etf_sortino_data)[0], "Large US equity ETF leadership ranked by 6M Sortino ratio."),
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
    elif active_view == "Canary Momentum":
        active_tone = canary_signal_label(canary_data)[1]
    elif active_view == "Market Pulse" and market_data:
        active_tone = market_data["status"][1]
    elif active_view == "Options Flow":
        active_tone = options_signal_label(options_data)[1]
    elif active_view == "ETF Sortino Leadership":
        active_tone = etf_sortino_signal_label(etf_sortino_data)[1]

    st.markdown(
        f"""
        <div class="hero">
            <h1>{summary.ticker} · {active_view_label}</h1>
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
        render_metric_card(active_view_label, active_status, active_subtitle, active_tone)

def render_sidebar(default_ticker: str) -> tuple[str, str, str, str | None, dict[str, Any] | None, int]:
    st.sidebar.subheader("Dashboard Controls")

    ticker = st.sidebar.text_input(
        "Ticker",
        value=st.session_state.get("ticker", default_ticker),
        key="ticker_input",
    ).strip().upper()
    period_options = ["1y", "2y", "3y", "5y"]
    default_period = st.session_state.get("period", LOOKBACK_PERIOD)
    period = st.sidebar.selectbox(
        "History window",
        options=period_options,
        index=period_options.index(default_period) if default_period in period_options else period_options.index(LOOKBACK_PERIOD),
        key="period_select",
    )
    last_core_view = st.session_state.get("last_core_view", CORE_DASHBOARD_VIEWS[0])
    core_view = st.sidebar.radio(
        "Chart View",
        options=CORE_DASHBOARD_VIEWS,
        index=CORE_DASHBOARD_VIEWS.index(last_core_view) if last_core_view in CORE_DASHBOARD_VIEWS else 0,
        key="dashboard_view_select",
    )
    st.sidebar.markdown(
        """
        <div class="sidebar-section-card">
            <div class="sidebar-section-eyebrow">Quick Access</div>
            <div class="sidebar-section-title">Special dashboards</div>
            <div class="sidebar-section-copy">Open cross-asset pulse, canary regime, SPX dealer gamma, ETF leadership, or Fed/Treasury liquidity plumbing without touching the main chart picker.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    clicked_special_view = None
    for action in SPECIAL_ACTION_BUTTONS:
        button_col, text_col = st.sidebar.columns([1.15, 1.0], gap="small")
        with button_col:
            if st.button(action["label"], key=f"special_action_{action['view']}", use_container_width=True):
                clicked_special_view = action["view"]
        with text_col:
            st.markdown(f'<div class="quick-action-row-copy">{action["caption"]}</div>', unsafe_allow_html=True)
    force_refresh = st.sidebar.checkbox("Force refresh cached data", value=False, key="force_refresh_toggle")
    if force_refresh:
        st.cache_data.clear()
        clear_daily_payload_cache()
        st.session_state["force_refresh_toggle"] = False
    mobile_chart_mode = st.sidebar.checkbox(
        "Mobile-friendly charts",
        value=bool(st.session_state.get("mobile_chart_mode", False)),
        key="mobile_chart_mode_toggle",
        help="Use simplified matplotlib charts designed for narrow screens.",
    )
    st.session_state["mobile_chart_mode"] = mobile_chart_mode
    st.sidebar.caption("Examples: NVDA, QQQ, SPY, TSLA, 005930.KS, 035420.KQ, BTC-USD")
    st.sidebar.caption("Korean equities accept both Yahoo suffixes and plain 6-digit codes.")

    active_ticker = ticker or default_ticker
    active_period = period
    st.session_state["ticker"] = active_ticker
    st.session_state["period"] = active_period
    active_view = st.session_state.get("dashboard_view", core_view)
    if core_view != last_core_view:
        active_view = core_view
    if clicked_special_view:
        active_view = clicked_special_view
    st.session_state["last_core_view"] = core_view
    st.session_state["dashboard_view"] = active_view
    if active_view in SPECIAL_ACTION_LABELS:
        st.sidebar.caption(f"Active quick view: {display_view_name(active_view)}")

    sortino_top_n = DEFAULT_SORTINO_TOP_N
    if active_view == "ETF Sortino Leadership":
        sortino_top_n = st.sidebar.selectbox(
            "ETF Sortino Top N",
            options=[20, 30, 50, 100],
            index=[20, 30, 50, 100].index(DEFAULT_SORTINO_TOP_N),
            key="etf_sortino_top_n_select",
        )

    selected_expiry = None
    spx_payload = None
    if active_view == "Options Flow":
        spx_payload = fetch_spx_options_payload()
        spx_expiries = extract_spx_expiries(spx_payload)
        if spx_expiries:
            default_expiry = nearest_spx_expiry(spx_expiries) or spx_expiries[0]
            prior_auto_expiry = st.session_state.get("selected_spx_expiry_auto")
            current_selected_expiry = st.session_state.get("selected_spx_expiry_select")
            should_auto_select = (
                current_selected_expiry not in spx_expiries
                or current_selected_expiry is None
                or (prior_auto_expiry != default_expiry and current_selected_expiry == prior_auto_expiry)
            )
            if should_auto_select:
                st.session_state["selected_spx_expiry_select"] = default_expiry
            st.session_state["selected_spx_expiry_auto"] = default_expiry
            default_index = spx_expiries.index(st.session_state.get("selected_spx_expiry_select", default_expiry))
            selected_expiry = st.sidebar.selectbox(
                "SPX option expiry",
                options=spx_expiries,
                index=default_index,
                key="selected_spx_expiry_select",
            )
            st.session_state["selected_spx_expiry"] = selected_expiry
        else:
            st.sidebar.caption("SPX option chain is temporarily unavailable.")
    return active_ticker, active_period, active_view, selected_expiry, spx_payload, sortino_top_n

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
    ticker, period, active_view, selected_expiry, spx_payload, sortino_top_n = render_sidebar(default_ticker="NVDA")

    if active_view == "Fed Watch":
        with st.spinner("Loading Fed Watch data..."):
            fed_watch_data = fetch_fed_watch_data(period)
        if not fed_watch_data:
            st.error("Fed Watch could not load usable FRED data for the selected history window.")
            st.stop()
        render_fed_watch_header(fed_watch_data)
        st.markdown(
            '<div class="section-note">Active view: <strong>Fed Watch</strong>. This quick view runs independently from the ticker chart stack and uses the shared history window as its macro lookback.</div>',
            unsafe_allow_html=True,
        )
        render_fed_watch_dashboard(fed_watch_data)
        st.caption("Data sources: FRED series for SOFR, reserve balances, Treasury cash, reverse repo usage, Treasury yields, and bank/Fed Treasury holdings.")
        return

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
    need_canary = active_view == "Canary Momentum"
    need_market = active_view == "Market Pulse"
    need_options = active_view == "Options Flow"
    need_etf_sortino = active_view == "ETF Sortino Leadership"

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
        canary_data = compute_canary_momentum_dashboard(period=period) if need_canary else None
        market_data = compute_market_fear_greed() if need_market else None
        options_data = compute_options_analytics(expiry=selected_expiry, payload=spx_payload) if need_options else None
        etf_sortino_data = compute_etf_sortino_leadership(top_n=sortino_top_n) if need_etf_sortino else None

    summary = build_summary(ticker, price_df, elder_df, td_df, stl_df, market_data, options_data)
    render_header(summary, active_view, market_data, options_data, supertrend_data, vix_fix_data, squeeze_data, etf_sortino_data, canary_data)
    render_data_status(price_df, price_source, price_symbol, stl_df, stl_source, stl_symbol)
    active_view_label = display_view_name(active_view)
    st.markdown(
        f'<div class="section-note">Active view: <strong>{active_view_label}</strong>. Core indicators are selected from the chart picker, while Fear &amp; Greed, Canary, Option Gamma, ETF Sortino, and Fed Watch run from the quick-access buttons.</div>',
        unsafe_allow_html=True,
    )

    if active_view == "Elder Impulse":
        render_chart("Elder Impulse and Trend Filter", build_elder_figure(elder_df), mobile_builder=lambda: build_mobile_elder_figure(elder_df))
    elif active_view == "TD Sequential":
        render_chart("TD Sequential with Trend Context", build_td_figure(td_df), mobile_builder=lambda: build_mobile_td_figure(td_df))
    elif active_view == "Robust STL":
        if stl_df is None or stl_df.dropna(subset=["Trend", "Cycle_Score"]).empty:
            st.info("Robust STL could not build a valid cycle series for this ticker after trying both FinanceDataReader and Yahoo Finance.")
        else:
            render_chart("Robust STL Cycle Dashboard", build_stl_figure(stl_df), mobile_builder=lambda: build_mobile_stl_figure(stl_df))
            st.caption(f"STL source: {stl_source} via `{stl_symbol}`")
    elif active_view == "SMC":
        render_chart("Smart Money Concepts", build_smc_figure(smc_data), mobile_builder=lambda: build_mobile_smc_figure(smc_data))
    elif active_view == "SuperTrend":
        render_chart("SuperTrend Regime", build_supertrend_figure(supertrend_data), mobile_builder=lambda: build_mobile_supertrend_figure(supertrend_data))
    elif active_view == "Williams Vix Fix":
        render_chart("Williams Vix Fix / Inverse", build_vix_fix_figure(vix_fix_data), mobile_builder=lambda: build_mobile_vix_fix_figure(vix_fix_data))
    elif active_view == "Squeeze Momentum":
        render_chart("Squeeze Momentum", build_squeeze_figure(squeeze_data), mobile_builder=lambda: build_mobile_squeeze_figure(squeeze_data))
    elif active_view == "Canary Momentum":
        if not canary_data:
            st.info("Canary momentum data could not be loaded from Yahoo Finance for the required universe.")
        else:
            render_canary_dashboard(canary_data)
    elif active_view == "Market Pulse":
        if not market_data:
            st.info("Fear & Greed data could not be loaded from Yahoo Finance for the required macro basket.")
        else:
            if mobile_charts_enabled():
                st.markdown("#### Macro Fear and Greed Dashboard")
                market_fig = build_mobile_market_figure(market_data)
                st.pyplot(market_fig, use_container_width=True)
                plt.close(market_fig)
            else:
                render_plotly_chart(build_market_figure(market_data))
    elif active_view == "Options Flow":
        if not options_data:
            st.info("SPX option data could not be loaded from the CBOE delayed quotes feed.")
        else:
            zero_gamma_level = float(options_data.get("zero_gamma_level", np.nan))
            zero_gamma_value = "n/a" if np.isnan(zero_gamma_level) else f"{zero_gamma_level:,.1f}"
            zero_gamma_subtitle = (
                "No gamma sign change found"
                if np.isnan(zero_gamma_level)
                else "Spot above gamma flip"
                if options_data["spot"] >= zero_gamma_level
                else "Spot below gamma flip"
            )
            zero_gamma_tone = (
                "neutral"
                if np.isnan(zero_gamma_level)
                else "bull"
                if options_data["spot"] >= zero_gamma_level
                else "bear"
            )

            option_cards = st.columns(5)
            with option_cards[0]:
                render_metric_card("Underlying", f"{options_data['underlying']} {options_data['spot']:,.1f}", options_data["expiry"], "neutral")
            with option_cards[1]:
                render_metric_card("Put/Call Volume", f"{options_data['put_call_ratio']:.2f}", "Above 1 implies defensive demand", "accent")
            with option_cards[2]:
                render_metric_card("Max Pain", f"{options_data['max_pain']:.2f}", "Strike with minimum aggregate pain", "neutral")
            with option_cards[3]:
                render_metric_card("Net GEX", f"{options_data['net_gex'] / 1e9:,.2f}B", "Positive often dampens volatility", "bull" if options_data["net_gex"] >= 0 else "bear")
            with option_cards[4]:
                render_metric_card("Zero Gamma", zero_gamma_value, zero_gamma_subtitle, zero_gamma_tone)
            render_chart(
                f"SPX Options Positioning ({options_data['expiry']})",
                build_options_figure(options_data, options_data["spot"]),
                mobile_builder=lambda: build_mobile_options_figure(options_data),
            )
    elif active_view == "ETF Sortino Leadership":
        if not etf_sortino_data:
            st.info("ETF Sortino leadership data could not be constructed from the current Yahoo Finance ETF universe and holdings coverage.")
        else:
            render_etf_sortino_dashboard(etf_sortino_data)

    st.caption("Data sources: Yahoo Finance, FinanceDataReader, CBOE delayed quotes, and FRED. ETF Sortino leadership depends on Yahoo ETF universe metadata, price history, and holdings coverage.")


if __name__ == "__main__":
    main()
