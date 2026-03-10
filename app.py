import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
import matplotlib.patches as patches
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy.signal import argrelextrema
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
plt.style.use('dark_background')
DARK_COLOR = '#131722'

st.set_page_config(page_title="Ultimate Quant Dashboard", layout="wide")

# ==========================================
# 💾 [데이터 캐싱]
# ==========================================
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, days=600):
    end_date = datetime.now() + timedelta(days=1)
    df = yf.download(ticker, start=end_date-timedelta(days=days), end=end_date, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()

@st.cache_data(ttl=3600)
def fetch_macro_data():
    tickers = ['SPY', '^VIX', 'HYG', 'IEF', 'RSP', 'XLY', 'XLP', 'DX=F'] # DX=F 안정성 확보
    data = yf.download(tickers, start='2020-01-01', progress=False)
    df = data['Close'].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
    return df.ffill().bfill()

# ==========================================
# 🧮 1. GEX 옵션 분석 엔진 (복구됨)
# ==========================================
def bs_greeks(S, K, T, r, q, sigma, cp_type):
    if T <= 0.0001 or sigma <= 0.001: return {'delta': 0, 'gamma': 0, 'vanna': 0, 'charm': 0}
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    delta = np.exp(-q * T) * norm.cdf(d1) if cp_type == 'C' else np.exp(-q * T) * (norm.cdf(d1) - 1)
    gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    vanna = -np.exp(-q * T) * norm.pdf(d1) * (d2 / sigma)
    term1 = q * np.exp(-q * T) * norm.cdf(d1 if cp_type == 'C' else -d1)
    term2 = np.exp(-q * T) * norm.pdf(d1) * ( (r - q) / (sigma * np.sqrt(T)) - d2 / (2 * T) )
    charm = (term1 - term2) if cp_type == 'C' else (-term1 - term2)
    return {'delta': delta, 'gamma': gamma, 'vanna': vanna, 'charm': charm}

def plot_options_dashboard(target_date):
    live_r = 0.04; live_sigma_fallback = 0.18; div_yield = 0.014
    url = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
    try:
        data = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).json()
    except:
        return st.error("CBOE 옵션 데이터를 불러오는데 실패했습니다.")
    
    spot = data['data']['current_price'] or data['data']['close']
    options = data['data']['options']
    target_date_str = target_date.strftime("%y%m%d")
    T = max((pd.Timestamp(target_date) - pd.Timestamp.now().normalize()).total_seconds() / (365.25 * 24 * 3600), 0.001)

    strikes_data = {}; total_c_vol = 0; total_p_vol = 0
    for opt in options:
        symbol = opt['option']
        if target_date_str not in symbol: continue
        cp, strike = symbol[-9], int(symbol[-8:]) / 1000.0
        oi, vol = opt['open_interest'], opt['volume']
        sigma = opt.get('implied_volatility') or opt.get('volatility') or live_sigma_fallback
        
        if cp == 'C': total_c_vol += vol
        else: total_p_vol += vol
        
        if oi == 0: continue
        if strike not in strikes_data: strikes_data[strike] = {'C_OI': 0, 'P_OI': 0, 'GEX': 0, 'Vanna': 0, 'Charm': 0}
        
        greeks = bs_greeks(spot, strike, T, live_r, div_yield, sigma, cp)
        gex = greeks['gamma'] * oi * spot * spot * 0.01 * 100
        vanna = greeks['vanna'] * oi * spot * 0.01 * 100
        charm = greeks['charm'] * oi * 100
        
        if cp == 'C':
            strikes_data[strike]['C_OI'] += oi; strikes_data[strike]['GEX'] += gex
            strikes_data[strike]['Vanna'] += vanna; strikes_data[strike]['Charm'] += charm
        else:
            strikes_data[strike]['P_OI'] += oi; strikes_data[strike]['GEX'] -= gex
            strikes_data[strike]['Vanna'] -= vanna; strikes_data[strike]['Charm'] -= charm

    if not strikes_data: return st.error("해당 만기일의 SPX 데이터가 없습니다.")

    df = pd.DataFrame.from_dict(strikes_data, orient='index').sort_index()
    df = df[(df.index >= spot * 0.85) & (df.index <= spot * 1.15)]
    pain = [sum(df['C_OI'] * np.maximum(0, s - df.index)) + sum(df['P_OI'] * np.maximum(0, df.index - s)) for s in df.index]
    max_pain = df.index[np.argmin(pain)]
    call_wall, put_wall = df['GEX'].idxmax(), df['GEX'].idxmin()

    st.markdown(f"**🎯 현재가:** {spot:,.2f} | **🧱 Call Wall:** {call_wall:,.0f} | **🛏️ Put Wall:** {put_wall:,.0f} | **🧲 Max Pain:** {max_pain:,.0f}")
    
    # [차트 크기 최적화] height 700으로 축소
    fig = make_subplots(rows=2, cols=2, subplot_titles=("Dealer Net GEX", "Max Pain", "Vanna", "Charm"), vertical_spacing=0.15)
    fig.add_trace(go.Bar(x=df.index, y=df['GEX']/1e9, marker_color='#1f77b4', name='GEX'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=pain, fill='tozeroy', line_color='#d62728', name='Pain'), row=1, col=2)
    fig.add_trace(go.Bar(x=df.index, y=df['Vanna']/1e9, marker_color='#2ca02c', name='Vanna'), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Charm'], marker_color='#ff7f0e', name='Charm'), row=2, col=2)

    for r, c in [(1,1), (1,2), (2,1), (2,2)]:
        fig.add_vline(x=spot, line_dash="dot", line_color="black", row=r, col=c)
    fig.add_vline(x=call_wall, line_width=2, line_color="red", row=1, col=1)
    fig.add_vline(x=put_wall, line_width=2, line_color="blue", row=1, col=1)
    fig.update_layout(height=650, showlegend=False, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 🧮 2. SMC 분석 엔진 (풀버전 복구)
# ==========================================
class LabelManager:
    def __init__(self): self.labels = []
    def add(self, y, text, color): self.labels.append({'real_y': y, 'text': text, 'color': color})
    def optimize_and_draw(self, ax, x_pos, y_range_max, y_range_min):
        if not self.labels: return
        total_range = y_range_max - y_range_min
        min_dist = total_range * 0.04
        self.labels.sort(key=lambda x: x['real_y'], reverse=True)
        placed = []
        for l in self.labels:
            curr_y = l['real_y']
            if placed and (placed[-1]['visual_y'] - curr_y < min_dist): curr_y = placed[-1]['visual_y'] - min_dist
            l['visual_y'] = curr_y; placed.append(l)
        for i in placed:
            if abs(i['real_y'] - i['visual_y']) > (total_range * 0.01):
                ax.plot([x_pos, x_pos], [i['real_y'], i['visual_y']], color=i['color'], linestyle=':', alpha=0.7)
            ax.text(x_pos, i['visual_y'], f" {i['text']}", color=i['color'], fontsize=8, fontweight='bold', va='center', bbox=dict(facecolor=DARK_COLOR, edgecolor=i['color'], boxstyle='square,pad=0.2', alpha=0.8))

def get_smart_obs(data):
    bull, bear = [], []
    body = abs(data['Close'] - data['Open']); rng = data['High'] - data['Low']
    for i in range(20, len(data)):
        curr_atr = data['atr'].iloc[i]; vol_ma = data['vol_ma'].iloc[i]
        if pd.isna(curr_atr): continue
        if (rng.iloc[i] > curr_atr * 1.2) or (body.iloc[i] > body.rolling(10).mean().iloc[i] * 1.5):
            prev_o, prev_c = data['Open'].iloc[i-1], data['Close'].iloc[i-1]
            prev_h, prev_l = data['High'].iloc[i-1], data['Low'].iloc[i-1]
            if data['Close'].iloc[i] < data['Open'].iloc[i] and prev_c >= prev_o:
                bear.append({'date': data.index[i-1], 'top': prev_h, 'bottom': prev_l, 'type': 'bear', 'strength': rng.iloc[i]})
            elif data['Close'].iloc[i] > data['Open'].iloc[i] and prev_c <= prev_o:
                bull.append({'date': data.index[i-1], 'top': prev_h, 'bottom': prev_l, 'type': 'bull', 'strength': rng.iloc[i]})
    return bull, bear

def get_smart_fvgs(data):
    bull, bear = [], []
    h, l = data['High'], data['Low']
    for i in range(20, len(data)):
        curr_atr = data['atr'].iloc[i]
        if pd.isna(curr_atr): continue
        if l.iloc[i] > h.iloc[i-2] and (l.iloc[i] - h.iloc[i-2]) > curr_atr * 0.3:
            bull.append({'date': data.index[i-1], 'top': l.iloc[i], 'bottom': h.iloc[i-2], 'type': 'bull', 'size': l.iloc[i] - h.iloc[i-2]})
        if h.iloc[i] < l.iloc[i-2] and (l.iloc[i-2] - h.iloc[i]) > curr_atr * 0.3:
            bear.append({'date': data.index[i-1], 'top': l.iloc[i-2], 'bottom': h.iloc[i], 'type': 'bear', 'size': l.iloc[i-2] - h.iloc[i]})
    return bull, bear

def plot_smc_chart(ticker):
    df = fetch_stock_data(ticker, days=300)
    if df.empty: return None
    df['tr'] = np.maximum((df['High']-df['Low']), np.maximum(abs(df['High']-df['Close'].shift(1)), abs(df['Low']-df['Close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean(); df['vol_ma'] = df['Volume'].rolling(20).mean()
    df = df.iloc[-150:].copy() # 시각적 복잡도 완화
    
    bull_ob, bear_ob = get_smart_obs(df)
    bull_fvg, bear_fvg = get_smart_fvgs(df)

    # [차트 크기 최적화] 16x10 -> 12x6으로 축소
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_COLOR); ax1.set_facecolor(DARK_COLOR)
    
    up, down = df[df.Close >= df.Open], df[df.Close < df.Open]
    ax1.vlines(up.index, up.Low, up.High, color='#089981', linewidth=1)
    ax1.vlines(down.index, down.Low, down.High, color='#F23645', linewidth=1)
    ax1.bar(up.index, up.Close-up.Open, bottom=up.Open, width=0.6, color='#089981')
    ax1.bar(down.index, down.Close-down.Open, bottom=down.Open, width=0.6, color='#F23645')

    track_x = mdates.date2num(df.index[-1] + timedelta(days=5))
    label_mgr = LabelManager()

    def add_zone(zones, color, name):
        for z in zones[-3:]: # 최근 3개만 표시
            start_num = mdates.date2num(z['date'])
            ax1.add_patch(patches.Rectangle((start_num, z['bottom']), track_x-start_num, z['top']-z['bottom'], facecolor=color, alpha=0.25, edgecolor='none'))
            label_mgr.add((z['top']+z['bottom'])/2, name, color)

    add_zone(bull_ob, "#00E676", "Bull OB"); add_zone(bear_ob, "#FF1744", "Bear OB")
    add_zone(bull_fvg, "#2979FF", "Bull FVG"); add_zone(bear_fvg, "#FFC107", "Bear FVG")

    label_mgr.optimize_and_draw(ax1, mdates.date2num(df.index[-1] + timedelta(days=6)), df['High'].max(), df['Low'].min())
    ax1.set_xlim(df.index[0], df.index[-1] + timedelta(days=15))
    ax1.set_title(f"[{ticker}] Smart Money Concepts", color='white', pad=15)
    ax1.tick_params(colors='white'); ax1.grid(True, alpha=0.1)
    plt.tight_layout()
    return fig

# ==========================================
# 🧮 3. 그 외 엔진 (크기 축소 및 통합)
# ==========================================
def plot_smart_money_chart():
    df = fetch_macro_data()
    df['Smart_Score'] = (get_probability(df['SPY'].rolling(20).mean(), 252)*0.1 + get_probability((df['RSP']/df['SPY']), 252)*0.4 + get_probability(df['^VIX'], 252, inverse=True)*0.5)
    df = df.dropna().tail(250)
    # [차트 크기 축소]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(DARK_COLOR); ax1.set_facecolor(DARK_COLOR)
    ax1.plot(df.index, df['Smart_Score'], color='#00E676', linewidth=2)
    ax1.fill_between(df.index, 0.8, 1.0, color='red', alpha=0.15); ax1.fill_between(df.index, 0.0, 0.2, color='green', alpha=0.15)
    ax1.set_title('Macro Fear & Greed Index', color='white'); ax1.tick_params(colors='white'); ax1.grid(alpha=0.1)
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['SPY'], color='white', alpha=0.4); ax2.tick_params(colors='white')
    return fig, df['Smart_Score'].iloc[-1]

def get_probability(series, w, inverse=False):
    z = (series - series.rolling(w).mean()) / (series.rolling(w).std() + 1e-9)
    return 1 - z.apply(norm.cdf) if inverse else z.apply(norm.cdf)

def plot_td_chart(ticker):
    df = fetch_stock_data(ticker, days=250)
    df['MA21'] = df['Close'].rolling(21).mean(); df['MA50'] = df['Close'].rolling(50).mean()
    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350', edge='inherit', wick='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, facecolor=DARK_COLOR, edgecolor='#363c4e')
    # [차트 크기 축소] 16x8 -> 12x5
    fig, _ = mpf.plot(df.tail(150), type='candle', style=s, addplot=[mpf.make_addplot(df['MA21'].tail(150), color='#FFEB3B'), mpf.make_addplot(df['MA50'].tail(150), color='#FB8C00')], figsize=(12, 5), returnfig=True)
    fig.patch.set_facecolor(DARK_COLOR)
    return fig

# ==========================================
# 🖥️ 메인 UI 렌더링
# ==========================================
st.title("🚀 Ultimate Quant Trading Dashboard")

st.sidebar.header("⚙️ 검색 설정")
ticker_input = st.sidebar.text_input("종목 티커 (SMC, TD용)", value="NVDA").upper()
target_date = st.sidebar.date_input("옵션 만기일 (SPX GEX용)", value=pd.to_datetime('2026-03-20'))

tab1, tab2, tab3, tab4 = st.tabs(["📊 옵션 GEX (SPX)", "🏦 SMC (매물대/FVG)", "🌐 거시경제 (Smart Money)", "📈 TD 추세 (단기)"])

with tab1:
    st.markdown("### 📊 SPX Dealer Gamma & Options Analysis")
    plot_options_dashboard(target_date)

with tab2:
    st.markdown(f"### 🏦 {ticker_input} - Smart Money Concepts (OB & FVG)")
    fig_smc = plot_smc_chart(ticker_input)
    if fig_smc: st.pyplot(fig_smc)

with tab3:
    st.markdown("### 🌐 스마트 머니 공포/탐욕 인덱스 (S&P 500 기반)")
    fig_macro, score = plot_smart_money_chart()
    st.metric("현재 매크로 스코어 (0=공포, 1=탐욕)", f"{score:.2f}")
    st.pyplot(fig_macro)

with tab4:
    st.markdown(f"### 📈 {ticker_input} - 단기 추세 차트")
    fig_td = plot_td_chart(ticker_input)
    if fig_td: st.pyplot(fig_td)
