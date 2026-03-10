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
st.title("🚀 Ultimate Quant Trading Dashboard (5-in-1)")

# ==========================================
# 💾 0. 데이터 캐싱 및 전처리
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
    tickers = ['SPY', '^VIX', 'HYG', 'IEF', 'RSP', 'XLY', 'XLP', 'DX=F']
    data = yf.download(tickers, start='2020-01-01', progress=False)
    df = data['Close'].copy() if isinstance(data.columns, pd.MultiIndex) else data.copy()
    return df.ffill().bfill()

# ==========================================
# 🧮 1. GEX 옵션 분석 엔진
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
    return {'delta': delta, 'gamma': gamma, 'vanna': vanna, 'charm': (term1 - term2) if cp_type == 'C' else (-term1 - term2)}

def render_options_dashboard(target_date):
    live_r = 0.04; live_sigma_fallback = 0.18; div_yield = 0.014
    url = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
    try: data = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10).json()
    except: return st.error("CBOE 데이터 통신 실패.")
    
    spot = data['data']['current_price'] or data['data']['close']
    T = max((pd.Timestamp(target_date) - pd.Timestamp.now().normalize()).total_seconds() / (365.25 * 24 * 3600), 0.001)
    target_date_str = target_date.strftime("%y%m%d")
    
    strikes_data = {}; total_c_vol = 0; total_p_vol = 0
    for opt in data['data']['options']:
        sym = opt['option']
        if target_date_str not in sym: continue
        cp, strike = sym[-9], int(sym[-8:]) / 1000.0
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

    df = pd.DataFrame.from_dict(strikes_data, orient='index').sort_index()
    lower_bound, upper_bound = spot * 0.85, spot * 1.15
    df = df[(df.index >= lower_bound) & (df.index <= upper_bound)]
    
    pain = [sum(df['C_OI'] * np.maximum(0, s - df.index)) + sum(df['P_OI'] * np.maximum(0, df.index - s)) for s in df.index]
    max_pain = df.index[np.argmin(pain)]
    call_wall, put_wall = df['GEX'].idxmax(), df['GEX'].idxmin()
    pcr = total_p_vol / total_c_vol if total_c_vol > 0 else 1.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재가 (Spot)", f"{spot:,.2f} pt")
    col2.metric("Call Wall (저항)", f"{call_wall:,.0f} pt")
    col3.metric("Put Wall (지지)", f"{put_wall:,.0f} pt")
    col4.metric("Max Pain", f"{max_pain:,.0f} pt")

    if spot >= call_wall:
        st.error(f"🚨 [과매수] Call Wall 돌파! 딜러 매도세 유입 예상. 👉 단기 숏(매도) 대응 또는 전량 익절 권장.")
    elif spot <= put_wall:
        st.success(f"🟢 [과매도] Put Wall 도달! 딜러 숏커버링 예상. 👉 단기 롱(매수) 진입 권장 (Buy the Dip).")
    else:
        st.info(f"📈 [박스권] 현재 지수가 Put Wall({put_wall:,.0f})과 Call Wall({call_wall:,.0f}) 사이에서 안정적으로 움직이고 있습니다.")

    fig = make_subplots(rows=3, cols=2, subplot_titles=("Dealer Net GEX", "Max Pain", "Vanna", "Charm", "VIX", "PCR"), vertical_spacing=0.12, specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, None]])
    fig.add_trace(go.Bar(x=df.index, y=df['GEX']/1e9, marker_color='#1f77b4'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=pain, fill='tozeroy', line_color='#d62728'), row=1, col=2)
    fig.add_trace(go.Bar(x=df.index, y=df['Vanna']/1e9, marker_color='#2ca02c'), row=2, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Charm'], marker_color='#ff7f0e'), row=2, col=2)
    fig.add_trace(go.Scatter(x=['9D', '30D', '3M'], y=[18, 19, 20], mode='lines+markers', line_color='#9467bd'), row=3, col=1) 

    for r, c in [(1,1), (1,2), (2,1), (2,2)]: fig.add_vline(x=spot, line_dash="dot", line_color="black", row=r, col=c)
    fig.add_vline(x=call_wall, line_width=2, line_color="red", row=1, col=1)
    fig.add_vline(x=put_wall, line_width=2, line_color="blue", row=1, col=1)
    fig.add_vline(x=max_pain, line_dash="dash", line_color="purple", row=1, col=2)

    fig.update_layout(height=700, showlegend=False, template='plotly_white')
    indicator = go.Indicator(mode="gauge+number", value=pcr, domain={'x': [0.58, 0.98], 'y': [0.0, 0.23]}, gauge={'axis': {'range': [0, 2]}, 'steps': [{'range': [0, 0.7], 'color': 'lightgreen'}, {'range': [1.3, 2], 'color': 'lightcoral'}]})
    st.plotly_chart(go.Figure(data=list(fig.data) + [indicator], layout=fig.layout), use_container_width=True)

# ==========================================
# 🧮 2. SMC 엔진
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

def calculate_poc(data):
    if data.empty: return np.array([]), np.array([])
    price_bins = np.linspace(data['Low'].min(), data['High'].max(), 100)
    vol_profile, bin_edges = np.histogram((data['High'] + data['Low']) / 2, bins=price_bins, weights=data['Volume'])
    return (bin_edges[:-1] + bin_edges[1:]) / 2, vol_profile

def get_smart_obs(data):
    bull, bear = []; body = abs(data['Close'] - data['Open']); rng = data['High'] - data['Low']
    for i in range(20, len(data)):
        if pd.isna(data['atr'].iloc[i]): continue
        if (rng.iloc[i] > data['atr'].iloc[i] * 1.2) or (body.iloc[i] > body.rolling(10).mean().iloc[i] * 1.5):
            prev_o, prev_c, prev_h, prev_l = data['Open'].iloc[i-1], data['Close'].iloc[i-1], data['High'].iloc[i-1], data['Low'].iloc[i-1]
            if data['Close'].iloc[i] < data['Open'].iloc[i] and prev_c >= prev_o:
                bear.append({'date': data.index[i-1], 'top': prev_h, 'bottom': prev_l, 'type': 'bear', 'strength': rng.iloc[i]})
            elif data['Close'].iloc[i] > data['Open'].iloc[i] and prev_c <= prev_o:
                bull.append({'date': data.index[i-1], 'top': prev_h, 'bottom': prev_l, 'type': 'bull', 'strength': rng.iloc[i]})
    return bull, bear

def get_smart_fvgs(data):
    bull, bear = []; h, l = data['High'], data['Low']
    for i in range(20, len(data)):
        if pd.isna(data['atr'].iloc[i]): continue
        gap_up, gap_down = l.iloc[i] - h.iloc[i-2], l.iloc[i-2] - h.iloc[i]
        if l.iloc[i] > h.iloc[i-2] and gap_up > data['atr'].iloc[i] * 0.3: bull.append({'date': data.index[i-1], 'top': l.iloc[i], 'bottom': h.iloc[i-2], 'type': 'bull'})
        if h.iloc[i] < l.iloc[i-2] and gap_down > data['atr'].iloc[i] * 0.3: bear.append({'date': data.index[i-1], 'top': l.iloc[i-2], 'bottom': h.iloc[i], 'type': 'bear'})
    return bull, bear

def render_smc_dashboard(ticker):
    df_full = fetch_stock_data(ticker, days=600)
    df_full['tr'] = np.maximum((df_full['High']-df_full['Low']), np.maximum(abs(df_full['High']-df_full['Close'].shift(1)), abs(df_full['Low']-df_full['Close'].shift(1))))
    df_full['atr'] = df_full['tr'].rolling(14).mean(); df_full['vol_ma'] = df_full['Volume'].rolling(20).mean()
    df = df_full.iloc[-200:].copy() 

    bull_ob, bear_ob = get_smart_obs(df); bull_fvg, bear_fvg = get_smart_fvgs(df)
    bin_centers, vol_profile = calculate_poc(df)
    poc_price = bin_centers[vol_profile.argmax()] if len(bin_centers) > 0 else 0

    fig, ax1 = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(DARK_COLOR); ax1.set_facecolor(DARK_COLOR)
    
    ax_vol = ax1.twiny()
    ax_vol.barh(bin_centers, vol_profile, height=(bin_centers[1]-bin_centers[0]), color='gray', alpha=0.15)
    ax_vol.set_xlim(0, vol_profile.max() * 3); ax_vol.axis('off')

    up, down = df[df.Close >= df.Open], df[df.Close < df.Open]
    ax1.vlines(up.index, up.Low, up.High, color='#089981', linewidth=1); ax1.vlines(down.index, down.Low, down.High, color='#F23645', linewidth=1)
    ax1.bar(up.index, up.Close-up.Open, bottom=up.Open, width=0.6, color='#089981'); ax1.bar(down.index, down.Close-down.Open, bottom=down.Open, width=0.6, color='#F23645')

    high_idx = argrelextrema(df['High'].values, np.greater, order=5)[0]
    low_idx = argrelextrema(df['Low'].values, np.less, order=5)[0]
    swings = sorted([{'idx': i, 'price': df['High'].iloc[i], 'type': 'H', 'date': df.index[i]} for i in high_idx] + [{'idx': i, 'price': df['Low'].iloc[i], 'type': 'L', 'date': df.index[i]} for i in low_idx], key=lambda x: x['idx'])
    
    if len(swings) >= 2:
        range_high, range_low = max([s['price'] for s in swings[-10:]]), min([s['price'] for s in swings[-10:]])
        eq_price = (range_high + range_low) / 2
        ax1.axhline(eq_price, color='#FFD700', linestyle='-', linewidth=1.5, alpha=0.8)
        ax1.text(df.index[-20], eq_price, " EQ (50%)", color='black', fontsize=9, fontweight='bold', bbox=dict(facecolor='#FFD700', edgecolor='none', pad=1))

    track_x = mdates.date2num(df.index[-1] + timedelta(days=5))
    label_mgr = LabelManager()
    def add_zone(zones, color, name):
        for z in zones[-4:]: 
            start_num = mdates.date2num(z['date'])
            ax1.add_patch(patches.Rectangle((start_num, z['bottom']), track_x-start_num, z['top']-z['bottom'], facecolor=color, alpha=0.25, edgecolor='none'))
            label_mgr.add((z['top']+z['bottom'])/2, name, color)

    add_zone(bull_ob, "#00E676", "Bull OB"); add_zone(bear_ob, "#FF1744", "Bear OB")
    add_zone(bull_fvg, "#2979FF", "Bull FVG"); add_zone(bear_fvg, "#FFC107", "Bear FVG")

    swept = []
    for i in range(len(df)):
        if not swings or i < swings[0]['idx'] or pd.isna(df['atr'].iloc[i]): continue
        curr = df.iloc[i]
        for sh in [s for s in swings if s['idx'] < i and s['type'] == 'H'][-3:]:
            if sh['idx'] not in swept and curr['High'] > sh['price'] and curr['Close'] < sh['price']:
                ax1.text(curr.name, curr['High'], "▼", color='#FF5252', ha='center', va='bottom', fontsize=10); swept.append(sh['idx'])
        for sl in [s for s in swings if s['idx'] < i and s['type'] == 'L'][-3:]:
            if sl['idx'] not in swept and curr['Low'] < sl['price'] and curr['Close'] > sl['price']:
                ax1.text(curr.name, curr['Low'], "▲", color='#00E676', ha='center', va='top', fontsize=10); swept.append(sl['idx'])

    ax1.axhline(poc_price, color='white', linestyle='--', linewidth=1, alpha=0.6)
    label_mgr.optimize_and_draw(ax1, mdates.date2num(df.index[-1] + timedelta(days=6)), df['High'].max(), df['Low'].min())
    ax1.set_xlim(df.index[0], df.index[-1] + timedelta(days=20))
    ax1.set_title(f"[{ticker}] Smart Money Concepts (Pro)", color='white', pad=15)
    ax1.tick_params(colors='white'); ax1.grid(True, alpha=0.1)
    st.pyplot(fig)

# ==========================================
# 🧮 3. TD Sequential 
# ==========================================
def render_td_dashboard(ticker):
    df = fetch_stock_data(ticker, days=300)
    df['Buy_Setup'] = 0; df['Sell_Setup'] = 0; df['Buy_Countdown'] = 0; df['Sell_Countdown'] = 0
    active_b_ct, b_idx = False, 0
    active_s_ct, s_idx = False, 0
    
    for i in range(4, len(df)):
        curr = df['Close'].iloc[i]; prev4 = df['Close'].iloc[i-4]
        df.iloc[i, df.columns.get_loc('Buy_Setup')] = df['Buy_Setup'].iloc[i-1] + 1 if curr < prev4 else 0
        df.iloc[i, df.columns.get_loc('Sell_Setup')] = df['Sell_Setup'].iloc[i-1] + 1 if curr > prev4 else 0

        if df['Buy_Setup'].iloc[i] == 9: active_b_ct, b_idx = True, 0
        if df['Sell_Setup'].iloc[i] == 9: active_s_ct, s_idx = True, 0

        if active_b_ct and curr <= df['Close'].iloc[i-2]:
            b_idx += 1; df.iloc[i, df.columns.get_loc('Buy_Countdown')] = b_idx
            if b_idx == 13: active_b_ct = False
        if active_s_ct and curr >= df['Close'].iloc[i-2]:
            s_idx += 1; df.iloc[i, df.columns.get_loc('Sell_Countdown')] = s_idx
            if s_idx == 13: active_s_ct = False

    df['MA21'] = df['Close'].rolling(21).mean(); df['MA50'] = df['Close'].rolling(50).mean()
    df_vis = df.iloc[-150:]
    
    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350', edge='inherit', wick='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, facecolor=DARK_COLOR, edgecolor='#363c4e')
    fig, axlist = mpf.plot(df_vis, type='candle', style=s, addplot=[mpf.make_addplot(df_vis['MA21'], color='#FFEB3B'), mpf.make_addplot(df_vis['MA50'], color='#FB8C00')], figsize=(14, 7), returnfig=True)
    fig.patch.set_facecolor(DARK_COLOR); ax_main = axlist[0]
    
    for i in range(len(df_vis)):
        low_v, high_v = df_vis['Low'].iloc[i], df_vis['High'].iloc[i]
        b_s, s_s = df_vis['Buy_Setup'].iloc[i], df_vis['Sell_Setup'].iloc[i]
        
        if 6 <= b_s <= 8: ax_main.text(i, low_v, str(int(b_s)), color='#66BB6A', fontsize=8, ha='center', va='top')
        elif b_s == 9: ax_main.text(i, low_v, 'B9', color='#00FF00', fontsize=10, fontweight='bold', ha='center', va='top')
        if df_vis['Buy_Countdown'].iloc[i] == 13: ax_main.text(i, low_v * 0.98, 'BUY 13', color='#00FFFF', fontsize=9, fontweight='bold', ha='center', va='top')
        
        if 6 <= s_s <= 8: ax_main.text(i, high_v, str(int(s_s)), color='#FF8A65', fontsize=8, ha='center', va='bottom')
        elif s_s == 9: ax_main.text(i, high_v, 'S9', color='#FF3D00', fontsize=10, fontweight='bold', ha='center', va='bottom')
        if df_vis['Sell_Countdown'].iloc[i] == 13: ax_main.text(i, high_v * 1.02, 'S13', color='#FF00FF', fontsize=9, fontweight='bold', ha='center', va='bottom')
        
    ax_main.set_title(f"[{ticker}] TD Sequential", color="white")
    st.pyplot(fig)

# ==========================================
# 🧮 4. 매크로 
# ==========================================
def render_macro_dashboard():
    df = fetch_macro_data()
    def get_prob(series, inv=False):
        z = (series - series.rolling(252).mean()) / (series.rolling(252).std() + 1e-9)
        return 1 - z.apply(norm.cdf) if inv else z.apply(norm.cdf)
    df['Smart_Score'] = (get_prob(df['SPY'].rolling(20).mean())*0.2 + get_prob((df['RSP']/df['SPY']))*0.4 + get_prob(df['^VIX'], inv=True)*0.4)
    df = df.dropna().tail(250)
    
    fig, ax1 = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(DARK_COLOR); ax1.set_facecolor(DARK_COLOR)
    ax1.plot(df.index, df['Smart_Score'], color='#00E676', linewidth=2.5)
    ax1.fill_between(df.index, 0.8, 1.0, color='red', alpha=0.15); ax1.fill_between(df.index, 0.0, 0.2, color='green', alpha=0.15)
    ax1.set_title('Macro Fear & Greed Index (SPX)', color='white'); ax1.tick_params(colors='white'); ax1.grid(alpha=0.1)
    ax2 = ax1.twinx(); ax2.plot(df.index, df['SPY'], color='white', alpha=0.5); ax2.tick_params(colors='white')
    
    st.metric("현재 매크로 스코어 (0=극단적 공포, 1=극단적 탐욕)", f"{df['Smart_Score'].iloc[-1]:.2f}")
    st.pyplot(fig)

# ==========================================
# 🧮 5. Elder Impulse (대망의 완벽 복구)
# ==========================================
def render_elder_dashboard(ticker):
    df = fetch_stock_data(ticker, days=400)
    if df.empty: return st.error("데이터가 없습니다.")

    df['EMA13'] = df['Close'].ewm(span=13, adjust=False).mean()
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd_line - signal_line
    df['EMA65'] = df['Close'].ewm(span=65, adjust=False).mean()
    df['Long_Term_Up'] = df['Close'] > df['EMA65']

    ema_diff = df['EMA13'].diff()
    hist_diff = df['MACD_Hist'].diff()

    colors, impulse_states = [], []
    for i in range(len(df)):
        if i == 0:
            colors.append('blue'); impulse_states.append(0); continue
        if ema_diff.iloc[i] > 0 and hist_diff.iloc[i] > 0:
            colors.append('green'); impulse_states.append(1)
        elif ema_diff.iloc[i] < 0 and hist_diff.iloc[i] < 0:
            colors.append('red'); impulse_states.append(-1)
        else:
            colors.append('blue'); impulse_states.append(0)

    df['Impulse_Color'] = colors
    df['Impulse_State'] = impulse_states

    buy_signals, sell_signals = [], []
    prev_buy, prev_sell = False, False
    for i in range(len(df)):
        curr_buy = (df['Impulse_State'].iloc[i] == 1) and df['Long_Term_Up'].iloc[i]
        curr_sell = (df['Impulse_State'].iloc[i] == -1) and (not df['Long_Term_Up'].iloc[i])
        
        buy_signals.append(df['Low'].iloc[i] * 0.98 if curr_buy and not prev_buy else np.nan)
        sell_signals.append(df['High'].iloc[i] * 1.02 if curr_sell and not prev_sell else np.nan)
        prev_buy, prev_sell = curr_buy, curr_sell

    plot_df = df.iloc[-200:].copy()
    apds = [
        mpf.make_addplot(plot_df['EMA65'], color='purple', linestyle='--', width=1.5, label='EMA65'),
        mpf.make_addplot(buy_signals[-200:], type='scatter', markersize=100, marker='^', color='lime'),
        mpf.make_addplot(sell_signals[-200:], type='scatter', markersize=100, marker='v', color='orange'),
        mpf.make_addplot(plot_df['EMA13'], color='gray', width=0.8)
    ]
    
    mc = mpf.make_marketcolors(up='black', down='black', edge='inherit', wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', facecolor=DARK_COLOR, edgecolor='#363c4e')
    
    fig, _ = mpf.plot(plot_df, type='candle', style=s, title=f'\n[{ticker}] Elder Impulse System', 
                      volume=True, addplot=apds, marketcolor_overrides=plot_df['Impulse_Color'].tolist(), 
                      figsize=(14, 7), returnfig=True)
    fig.patch.set_facecolor(DARK_COLOR)
    st.pyplot(fig)


# ==========================================
# 🖥️ 탭 메뉴 렌더링
# ==========================================
st.sidebar.header("⚙️ 분석 설정")
ticker_input = st.sidebar.text_input("종목 티커 (SMC, TD, Elder용)", value="NVDA").upper()
target_date = st.sidebar.date_input("옵션 만기일 (SPX GEX용)", value=pd.to_datetime('2026-03-20'))

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 옵션 GEX (SPX)", 
    "🏦 SMC (스마트머니)", 
    "📈 TD 추세 (단기)", 
    "🌐 거시경제 (매크로)",
    "🎯 Elder (엘더 추세)"
])

with tab1: render_options_dashboard(target_date)
with tab2: render_smc_dashboard(ticker_input)
with tab3: render_td_dashboard(ticker_input)
with tab4: render_macro_dashboard()
with tab5: render_elder_dashboard(ticker_input)
