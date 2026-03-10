import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import matplotlib.dates as mdates
import matplotlib.patches as patches
from scipy.signal import argrelextrema
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")
plt.style.use('dark_background')
DARK_COLOR = '#131722'

# 페이지 기본 설정
st.set_page_config(page_title="Ultimate Quant Dashboard", layout="wide", page_icon="📈")

# ==========================================
# 💾 [데이터 캐싱] 속도 최적화
# ==========================================
@st.cache_data(ttl=3600)
def fetch_stock_data(ticker, days=600):
    end_date = datetime.now() + timedelta(days=1)
    start_date = end_date - timedelta(days=days)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    cols_to_numeric = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in cols_to_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df.dropna()

@st.cache_data(ttl=3600)
def fetch_macro_data():
    # DX-Y.NYB 대신 데이터 수신이 안정적인 달러 선물(DX=F) 사용
    tickers = ['SPY', '^VIX', 'HYG', 'IEF', 'RSP', 'XLY', 'XLP', 'DX=F']
    data = yf.download(tickers, start='2020-01-01', progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        df = data['Close'].copy()
    else:
        df = data.copy()
    # ffill(앞 값으로 채우기) 후 bfill(뒷 값으로 채우기)까지 적용하여 데이터 증발 원천 차단
    return df.ffill().bfill()

# ==========================================
# 🧮 1. TD Sequential 로직
# ==========================================
def calculate_td_indicators(df):
    df = df.copy()
    df['Buy_Setup'] = 0; df['Sell_Setup'] = 0
    df['Buy_Countdown'] = 0; df['Sell_Countdown'] = 0
    active_b_ct, b_idx = False, 0
    active_s_ct, s_idx = False, 0

    for i in range(4, len(df)):
        curr_close = float(df['Close'].iloc[i])
        if curr_close < df['Close'].iloc[i-4]:
            df.iloc[i, df.columns.get_loc('Buy_Setup')] = df['Buy_Setup'].iloc[i-1] + 1
        else:
            df.iloc[i, df.columns.get_loc('Buy_Setup')] = 0
            
        if curr_close > df['Close'].iloc[i-4]:
            df.iloc[i, df.columns.get_loc('Sell_Setup')] = df['Sell_Setup'].iloc[i-1] + 1
        else:
            df.iloc[i, df.columns.get_loc('Sell_Setup')] = 0

        if df['Buy_Setup'].iloc[i] == 9: active_b_ct, b_idx = True, 0
        if df['Sell_Setup'].iloc[i] == 9: active_s_ct, s_idx = True, 0

        if active_b_ct and curr_close <= df['Close'].iloc[i-2]:
            b_idx += 1
            df.iloc[i, df.columns.get_loc('Buy_Countdown')] = b_idx
            if b_idx == 13: active_b_ct = False
        if active_s_ct and curr_close >= df['Close'].iloc[i-2]:
            s_idx += 1
            df.iloc[i, df.columns.get_loc('Sell_Countdown')] = s_idx
            if s_idx == 13: active_s_ct = False

    df['MA21'] = df['Close'].rolling(window=21).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()

    nine_high = df['High'].rolling(window=9).max()
    nine_low = df['Low'].rolling(window=9).min()
    df['Tenkan_Sen'] = (nine_high + nine_low) / 2
    twenty_six_high = df['High'].rolling(window=26).max()
    twenty_six_low = df['Low'].rolling(window=26).min()
    df['Kijun_Sen'] = (twenty_six_high + twenty_six_low) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
    fifty_two_high = df['High'].rolling(window=52).max()
    fifty_two_low = df['Low'].rolling(window=52).min()
    df['Senkou_Span_B'] = ((fifty_two_high + fifty_two_low) / 2).shift(26)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df['RSI'] = 100 - (100 / (1 + (gain / loss)))
    return df

def plot_td_chart(ticker):
    df = fetch_stock_data(ticker, days=500)
    if df.empty: return None
    df = calculate_td_indicators(df)
    df_vis = df.iloc[-200:].copy()

    mc = mpf.make_marketcolors(up='#26A69A', down='#EF5350', edge='inherit', wick='inherit', volume='inherit')
    s = mpf.make_mpf_style(marketcolors=mc, base_mpf_style='charles', facecolor=DARK_COLOR, edgecolor='#363c4e', gridcolor='#2a2e39')

    apds = [
        mpf.make_addplot(df_vis['MA21'], color='#FFEB3B', width=1.2, label='MA 21'),
        mpf.make_addplot(df_vis['MA50'], color='#FB8C00', width=1.5, label='MA 50'),
        mpf.make_addplot(df_vis['MA200'], color='#E0E0E0', width=2.0, label='MA 200'),
        mpf.make_addplot(df_vis['Senkou_Span_A'], color='#26A69A', width=0.5, alpha=0.1),
        mpf.make_addplot(df_vis['Senkou_Span_B'], color='#EF5350', width=0.5, alpha=0.1),
        mpf.make_addplot(df_vis['RSI'], panel=1, color='#7E57C2', width=1.5, ylabel='RSI'),
        mpf.make_addplot([70]*len(df_vis), panel=1, color='#EF5350', width=0.8, linestyle='dashed'),
        mpf.make_addplot([30]*len(df_vis), panel=1, color='#26A69A', width=0.8, linestyle='dashed'),
    ]

    fig, axlist = mpf.plot(df_vis, type='candle', style=s, addplot=apds, figsize=(16, 8), returnfig=True,
                           fill_between=dict(y1=df_vis['Senkou_Span_A'].values, y2=df_vis['Senkou_Span_B'].values, alpha=0.08, color='#9b9b9b'))
    
    fig.patch.set_facecolor(DARK_COLOR)
    ax_main, ax_rsi = axlist[0], axlist[2]

    # TD 신호 텍스트 추가
    for i in range(len(df_vis)):
        low_v, high_v = df_vis['Low'].iloc[i], df_vis['High'].iloc[i]
        b_s, s_s = df_vis['Buy_Setup'].iloc[i], df_vis['Sell_Setup'].iloc[i]
        if 6 <= b_s <= 8: ax_main.text(i, low_v, str(int(b_s)), color='#66BB6A', fontsize=8, ha='center', va='top')
        elif b_s == 9: ax_main.text(i, low_v, 'B9', color='#00FF00', fontsize=10, fontweight='bold', ha='center', va='top')
        if df_vis['Buy_Countdown'].iloc[i] == 13:
            ax_main.text(i, low_v * 0.98, 'BUY 13', color='#00FFFF', fontsize=9, fontweight='bold', ha='center', va='top')
        
        if 6 <= s_s <= 8: ax_main.text(i, high_v, str(int(s_s)), color='#FF8A65', fontsize=8, ha='center', va='bottom')
        elif s_s == 9: ax_main.text(i, high_v, 'S9', color='#FF3D00', fontsize=10, fontweight='bold', ha='center', va='bottom')
        if df_vis['Sell_Countdown'].iloc[i] == 13:
            ax_main.text(i, high_v * 1.02, 'S13', color='#FF00FF', fontsize=9, fontweight='bold', ha='center', va='bottom')

    ax_main.set_title(f'[{ticker}] TD Sequential PRO', color='white')
    return fig

# ==========================================
# 🧮 2. SMC 분석 로직 
# ==========================================
class LabelManager:
    def __init__(self): self.labels = []
    def add(self, y, text, color): self.labels.append({'real_y': y, 'text': text, 'color': color})
    def optimize_and_draw(self, ax, x_pos, y_max, y_min):
        if not self.labels: return
        total_range = y_max - y_min
        min_dist = total_range * 0.04
        self.labels.sort(key=lambda x: x['real_y'], reverse=True)
        placed = []
        for l in self.labels:
            curr_y = l['real_y']
            if placed and (placed[-1]['visual_y'] - curr_y < min_dist):
                curr_y = placed[-1]['visual_y'] - min_dist
            l['visual_y'] = curr_y
            placed.append(l)
        for i in placed:
            if abs(i['real_y'] - i['visual_y']) > (total_range * 0.01):
                ax.plot([x_pos, x_pos], [i['real_y'], i['visual_y']], color=i['color'], linestyle=':', alpha=0.7)
            ax.text(x_pos, i['visual_y'], f" {i['text']}", color=i['color'], fontsize=8, fontweight='bold',
                    va='center', ha='left', bbox=dict(facecolor=DARK_COLOR, edgecolor=i['color'], boxstyle='square,pad=0.2', alpha=0.8))

def get_vol_label(vol, vol_ma):
    if pd.isna(vol_ma) or vol_ma == 0: return ""
    ratio = vol / vol_ma
    if ratio >= 2.5: return " (★★★)"
    if ratio >= 1.5: return " (★)"
    return ""

def plot_smc_chart(ticker):
    df_full = fetch_stock_data(ticker, days=600)
    if df_full.empty: return None

    df_full['tr'] = np.maximum((df_full['High'] - df_full['Low']),
                    np.maximum(abs(df_full['High'] - df_full['Close'].shift(1)), abs(df_full['Low'] - df_full['Close'].shift(1))))
    df_full['atr'] = df_full['tr'].rolling(14).mean()
    df_full['vol_ma'] = df_full['Volume'].rolling(20).mean()
    df_full['EMA_21'] = df_full['Close'].ewm(span=21, adjust=False).mean()
    df_full['SMA_200'] = df_full['Close'].rolling(window=200).mean()

    df = df_full.iloc[-200:].copy()
    fig, ax1 = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor(DARK_COLOR)
    ax1.set_facecolor(DARK_COLOR)

    up = df[df.Close >= df.Open]
    down = df[df.Close < df.Open]
    ax1.vlines(up.index, up.Low, up.High, color='#089981', linewidth=1)
    ax1.vlines(down.index, down.Low, down.High, color='#F23645', linewidth=1)
    ax1.bar(up.index, up.Close - up.Open, bottom=up.Open, width=0.6, color='#089981')
    ax1.bar(down.index, down.Close - down.Open, bottom=down.Open, width=0.6, color='#F23645')

    ax1.plot(df.index, df['EMA_21'], color='yellow', linewidth=1.5, alpha=0.8)
    ax1.plot(df.index, df['SMA_200'], color='purple', linewidth=2, alpha=0.8)
    ax1.set_title(f"[{ticker}] Smart Money Concepts", color='white', pad=20)
    ax1.grid(True, linestyle='--', alpha=0.1)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    
    return fig

# ==========================================
# 🧮 3. Elder Impulse 로직
# ==========================================
def plot_elder_chart(ticker):
    df = fetch_stock_data(ticker, days=300)
    if df.empty: return None

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
        mpf.make_addplot(plot_df['EMA65'], color='purple', linestyle='--', width=1.5),
        mpf.make_addplot(buy_signals[-200:], type='scatter', markersize=100, marker='^', color='lime'),
        mpf.make_addplot(sell_signals[-200:], type='scatter', markersize=100, marker='v', color='orange'),
        mpf.make_addplot(plot_df['EMA13'], color='gray', width=0.8)
    ]
    
    mc = mpf.make_marketcolors(up='black', down='black', edge='inherit', wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', facecolor=DARK_COLOR, edgecolor='#363c4e')
    
    fig, _ = mpf.plot(plot_df, type='candle', style=s, title=f'\n[{ticker}] Elder Impulse System', 
                      volume=True, addplot=apds, marketcolor_overrides=plot_df['Impulse_Color'].tolist(), 
                      figsize=(16, 8), returnfig=True)
    fig.patch.set_facecolor(DARK_COLOR)
    return fig

# ==========================================
# 🧮 4. Smart Money F&G Index (매크로)
# ==========================================
def get_probability(series, w, inverse=False):
    roll_mean = series.rolling(window=w).mean()
    roll_std = series.rolling(window=w).std()
    z_score = (series - roll_mean) / (roll_std + 1e-9)
    prob = z_score.apply(norm.cdf)
    return 1 - prob if inverse else prob

def calc_smart_money_index():
    df = fetch_macro_data()
    period, std_dev, window = 20, 2, 252

    ma20 = df['SPY'].rolling(window=period).mean()
    std20 = df['SPY'].rolling(window=period).std()
    upper, lower = ma20 + (std20 * std_dev), ma20 - (std20 * std_dev)
    df['F_BB'] = (df['SPY'] - lower) / (upper - lower)
    
    d = df['SPY'].diff(1)
    g = (d.where(d > 0, 0)).rolling(window=14).mean()
    l = (-d.where(d < 0, 0)).rolling(window=14).mean()
    df['F_RSI'] = 100 - (100 / (1 + (g / (l + 1e-9))))
    
    df['F_MA125'] = (df['SPY'] - df['SPY'].rolling(125).mean()) / df['SPY'].rolling(125).mean()
    df['F_Breadth'] = df['RSP'] / df['SPY']
    df['F_Sector'] = df['XLY'] / df['XLP']
    df['F_Credit'] = df['HYG'] / df['IEF']
    
    factors = pd.DataFrame(index=df.index)
    factors['BB_Pos']   = get_probability(df['F_BB'], window)
    factors['RSI_Mom']  = get_probability(df['F_RSI'], window)
    factors['MA125_Div']= get_probability(df['F_MA125'], window)
    factors['Breadth']  = get_probability(df['F_Breadth'], window)
    factors['Sector']   = get_probability(df['F_Sector'], window)
    factors['Credit']   = get_probability(df['F_Credit'], window)
    factors['VIX_Inv']  = get_probability(df['^VIX'], window, inverse=True)
    factors['DXY_Inv']  = get_probability(df['DX=F'], window, inverse=True)
    
    final_score = (
        0.10 * factors['BB_Pos'] + 0.10 * factors['RSI_Mom'] + 0.10 * factors['MA125_Div'] +
        0.15 * factors['Breadth'] + 0.15 * factors['Sector'] + 0.10 * factors['Credit'] +
        0.15 * factors['VIX_Inv'] + 0.15 * factors['DXY_Inv']
    )
    df['Smart_Score'] = final_score
    valid_df = df.dropna(subset=['Smart_Score'])
    valid_factors = factors.loc[valid_df.index]
    
    return valid_df.tail(300), valid_factors.tail(300)
    
def plot_smart_money_chart(plot_df, factors):
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(DARK_COLOR)
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(DARK_COLOR)
    ax1.set_title('Smart Money Fear & Greed Index', color='white', fontsize=16)
    ax1.fill_between(plot_df.index, 0.8, 1.0, color='red', alpha=0.15)
    ax1.fill_between(plot_df.index, 0.0, 0.2, color='green', alpha=0.15)
    ax1.plot(plot_df.index, plot_df['Smart_Score'], color='#00E676', linewidth=2.5)
    ax1.axhline(0.8, color='red', linestyle='--', alpha=0.5)
    ax1.axhline(0.2, color='green', linestyle='--', alpha=0.5)
    ax1.tick_params(colors='white')

    ax1_sub = ax1.twinx()
    ax1_sub.plot(plot_df.index, plot_df['SPY'], color='white', alpha=0.4, linewidth=1.5)
    ax1_sub.tick_params(colors='white')

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.set_facecolor(DARK_COLOR)
    ax2.plot(plot_df.index, factors['MA125_Div'], label='125MA Div', color='magenta')
    ax2.plot(plot_df.index, factors['RSI_Mom'], label='RSI Mom', color='gray')
    ax2.axhline(0.5, linestyle=':', color='gray')
    ax2.legend(loc='upper left', frameon=False, labelcolor='white')
    ax2.tick_params(colors='white')

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.set_facecolor(DARK_COLOR)
    ax3.plot(plot_df.index, factors['Breadth'], label='Breadth (RSP/SPY)', color='orange')
    ax3.plot(plot_df.index, factors['Sector'], label='Sector (XLY/XLP)', color='#2979FF')
    ax3.plot(plot_df.index, factors['Credit'], label='Credit (HYG/IEF)', color='#00E676', linestyle='--')
    ax3.legend(loc='upper left', ncol=3, frameon=False, labelcolor='white')
    ax3.tick_params(colors='white')

    plt.tight_layout()
    return fig

# ==========================================
# 🖥️ 메인 UI 렌더링
# ==========================================
st.title("🚀 Ultimate Quant Trading Dashboard")

# 좌측 사이드바 종목 입력
ticker_input = st.sidebar.text_input("종목 티커 (Ticker)", value="NVDA").upper()
st.sidebar.markdown("---")
st.sidebar.info("💡 **TIPS:**\n- **TD Sequential:** 단기 추세 반전(9, 13) 포착\n- **SMC:** 기관 매물대 및 FVG 탐색\n- **Elder Impulse:** 주간 추세 필터링 타점\n- **Smart Money:** S&P500 매크로 환경 스코어")

# 메인 탭 생성
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Smart Money 인덱스 (매크로)", 
    "📊 TD Sequential (반전 포착)", 
    "🏦 SMC Analysis (스마트머니)", 
    "📈 Elder Impulse (추세 추종)"
])

# 1. Smart Money Index (개별 종목 무관, 전체 매크로 시황)
with tab1:
    st.markdown("### 🌐 스마트 머니 공포/탐욕 인덱스 (S&P 500 기반)")
    with st.spinner('매크로 데이터를 연산 중입니다...'):
        sm_df, sm_factors = calc_smart_money_index()
        
        # [수정] 데이터가 비어있을 경우를 대비한 방어 로직
        if sm_df.empty:
            st.error("⚠️ 야후 파이낸스 통신 오류로 데이터를 불러오지 못했습니다. 잠시 후 새로고침 해주세요.")
        else:
            last_score = sm_df['Smart_Score'].iloc[-1]
            
            # 상태에 따른 색상 및 메세지
            if last_score > 0.8: status, color = "🔴 Extreme Greed (과매수/매도 검토)", "red"
            elif last_score > 0.6: status, color = "🟠 Greed (탐욕)", "orange"
            elif last_score < 0.2: status, color = "🟢 Extreme Fear (극단적 공포/분할 매수)", "green"
            elif last_score < 0.4: status, color = "🔵 Fear (공포)", "blue"
            else: status, color = "⚪ Neutral (중립)", "gray"

            col1, col2, col3 = st.columns(3)
            col1.metric("종합 Smart Score", f"{last_score:.2f} / 1.00")
            col2.markdown(f"**현재 시장 상태:**<br><span style='color:{color}; font-size:1.5rem; font-weight:bold;'>{status}</span>", unsafe_allow_html=True)
            col3.metric("기준일", sm_df.index[-1].strftime('%Y-%m-%d'))
            
            st.pyplot(plot_smart_money_chart(sm_df, sm_factors))

# 2. TD Sequential
with tab2:
    st.markdown(f"### 📊 {ticker_input} - TD Sequential Pro Chart")
    with st.spinner(f'{ticker_input} 데이터를 불러오는 중...'):
        fig_td = plot_td_chart(ticker_input)
        if fig_td: st.pyplot(fig_td)
        else: st.error("데이터를 불러올 수 없습니다.")

# 3. SMC Analysis
with tab3:
    st.markdown(f"### 🏦 {ticker_input} - Smart Money Concepts (SMC)")
    with st.spinner('SMC 존을 스캔 중입니다...'):
        fig_smc = plot_smc_chart(ticker_input)
        if fig_smc: st.pyplot(fig_smc)
        else: st.error("데이터를 불러올 수 없습니다.")

# 4. Elder Impulse
with tab4:
    st.markdown(f"### 📈 {ticker_input} - Elder Impulse & Trend Filter")
    with st.spinner('임펄스 신호를 계산 중입니다...'):
        fig_elder = plot_elder_chart(ticker_input)
        if fig_elder: st.pyplot(fig_elder)
        else: st.error("데이터를 불러올 수 없습니다.")
