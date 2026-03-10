import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from scipy.stats import norm
import datetime

# 페이지 기본 설정
st.set_page_config(page_title="SPX 퀀트 대시보드", layout="wide")
st.title("📈 SPX 실전 퀀트 트레이딩 대시보드")

# ==========================================
# 🧮 1. 수학 엔진 (Greeks)
# ==========================================
def bs_greeks(S, K, T, r, q, sigma, cp_type):
    if T <= 0.0001 or sigma <= 0.001:
        return {'delta': 0, 'gamma': 0, 'vanna': 0, 'charm': 0}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if cp_type == 'C': delta = np.exp(-q * T) * norm.cdf(d1)
    else: delta = np.exp(-q * T) * (norm.cdf(d1) - 1)

    gamma = (np.exp(-q * T) * norm.pdf(d1)) / (S * sigma * np.sqrt(T))
    vanna = -np.exp(-q * T) * norm.pdf(d1) * (d2 / sigma)

    term1 = q * np.exp(-q * T) * norm.cdf(d1 if cp_type == 'C' else -d1)
    term2 = np.exp(-q * T) * norm.pdf(d1) * ( (r - q) / (sigma * np.sqrt(T)) - d2 / (2 * T) )
    charm = (term1 - term2) if cp_type == 'C' else (-term1 - term2)

    return {'delta': delta, 'gamma': gamma, 'vanna': vanna, 'charm': charm}

# ==========================================
# 📊 2. 메인 대시보드 렌더링 함수
# ==========================================
def run_dashboard(target_date, spot_range_pct=0.15):
    with st.spinner("글로벌 매크로 및 옵션 체인 데이터를 동기화 중입니다..."):
        try:
            live_r = yf.Ticker('^IRX').history(period='1d')['Close'].iloc[-1] / 100.0
            vix_all = yf.download(['^VIX9D', '^VIX', '^VIX3M'], period='5d', progress=False, auto_adjust=False)['Close']
            vix9d, vix30d, vix3m = vix_all['^VIX9D'].iloc[-1], vix_all['^VIX'].iloc[-1], vix_all['^VIX3M'].iloc[-1]
            live_sigma_fallback = vix30d / 100.0
        except Exception:
            live_r, live_sigma_fallback = 0.04, 0.18
            vix9d, vix30d, vix3m = 18.0, 19.0, 20.0

        div_yield = 0.014 

        url = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = response.json()

        spot = data['data']['current_price'] or data['data']['close']
        options = data['data']['options']
        
        # 스트림릿 date_input 객체를 문자열로 변환 (YYMMDD 형식)
        target_date_str = target_date.strftime("%y%m%d")
        
        # 만기까지 남은 시간 계산
        T = max((pd.Timestamp(target_date) - pd.Timestamp.now().normalize()).total_seconds() / (365.25 * 24 * 3600), 0.001)

        strikes_data = {}
        total_c_vol, total_p_vol = 0, 0

        for opt in options:
            symbol = opt['option']
            if target_date_str not in symbol: continue

            cp, strike = symbol[-9], int(symbol[-8:]) / 1000.0
            oi, vol = opt['open_interest'], opt['volume']
            
            opt_iv = opt.get('implied_volatility') or opt.get('volatility')
            sigma = opt_iv if opt_iv and opt_iv > 0 else live_sigma_fallback

            if cp == 'C': total_c_vol += vol
            else: total_p_vol += vol

            if oi == 0: continue
            if strike not in strikes_data: 
                strikes_data[strike] = {'C_OI': 0, 'P_OI': 0, 'GEX': 0, 'Vanna': 0, 'Charm': 0}

            greeks = bs_greeks(spot, strike, T, live_r, div_yield, sigma, cp)
            multiplier = spot * spot * 0.01 * 100
            gex = greeks['gamma'] * oi * multiplier
            vanna = greeks['vanna'] * oi * spot * 0.01 * 100
            charm = greeks['charm'] * oi * 100

            if cp == 'C':
                strikes_data[strike]['C_OI'] += oi
                strikes_data[strike]['GEX'] += gex
                strikes_data[strike]['Vanna'] += vanna
                strikes_data[strike]['Charm'] += charm
            else:
                strikes_data[strike]['P_OI'] += oi
                strikes_data[strike]['GEX'] -= gex
                strikes_data[strike]['Vanna'] -= vanna
                strikes_data[strike]['Charm'] -= charm

        if not strikes_data:
            st.error("❌ 선택하신 만기일의 옵션 데이터가 없습니다. CBOE에 상장된 유효한 만기일(예: 3번째 금요일)을 선택해주세요.")
            return

        df = pd.DataFrame.from_dict(strikes_data, orient='index').sort_index()
        lower_bound = spot * (1 - spot_range_pct)
        upper_bound = spot * (1 + spot_range_pct)
        df = df[(df.index >= lower_bound) & (df.index <= upper_bound)]

        pain = [sum(df['C_OI'] * np.maximum(0, s - df.index)) + sum(df['P_OI'] * np.maximum(0, df.index - s)) for s in df.index]
        max_pain_strike = df.index[np.argmin(pain)]
        
        call_wall = df['GEX'].idxmax()
        put_wall = df['GEX'].idxmin()
        pcr = total_p_vol / total_c_vol if total_c_vol > 0 else 1.0

        # UI: 메트릭 카드 렌더링
        st.markdown("### 🎯 실전 트레이딩 타점 분석")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("현재가 (Spot)", f"{spot:,.2f} pt")
        col2.metric("Call Wall (저항)", f"{call_wall:,.0f} pt")
        col3.metric("Put Wall (지지)", f"{put_wall:,.0f} pt")
        col4.metric("Max Pain", f"{max_pain_strike:,.0f} pt")

        # UI: AI 매매 시그널 로직
        st.markdown("---")
        if spot >= call_wall:
            st.error(f"**🚨 [과매수 구간] 지수가 Call Wall을 돌파했습니다.** 콜옵션 딜러들의 헷징 물량(매도세)이 쏟아지는 강력한 저항선입니다.\n\n**👉 전략:** 단기 숏(매도) 포지션 진입 또는 롱(매수) 포지션 전량 익절 권장.")
        elif spot <= put_wall:
            st.success(f"**🟢 [과매도 구간] 지수가 Put Wall 아래로 내려갔습니다.** 풋옵션 딜러들의 환매수(숏 커버링)가 들어오는 강력한 지지선입니다.\n\n**👉 전략:** 과감한 단기 롱(매수) 진입 및 '눌림목 매매(Buy the Dip)' 권장.")
        else:
            st.info(f"**📈 [박스권 장세] 지수가 Put Wall과 Call Wall 사이에서 움직이고 있습니다.**")
            if pcr > 1.4 and vix9d > vix30d:
                st.warning(f"**🔥 [강력 매수 시그널] PCR 극단적 쏠림 + 단기 VIX 백워데이션 상태입니다.**\n\n**👉 전략:** 시장 심리가 극도로 억눌려 있습니다. 곧 Max Pain({max_pain_strike:,.0f}) 방향으로 튀어오르는 '숏 스퀴즈 반등'에 대비해 롱 비중을 늘리세요.")
            elif spot < max_pain_strike:
                st.write(f"**👉 전략:** 지수가 Max Pain({max_pain_strike:,.0f})보다 낮으므로 상방 우위의 스윙 매매가 유리합니다.")
            else:
                st.write(f"**👉 전략:** 지수가 Max Pain({max_pain_strike:,.0f})보다 높으므로 기관들의 차익 실현에 주의하며 짧게 끊어 치세요.")

        # UI: Plotly 대시보드 렌더링
        st.markdown("---")
        st.markdown("### 📊 6-Panel 심층 퀀트 대시보드")
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "1. Dealer Net GEX (Support & Resistance)", "2. Max Pain Profile", 
                "3. Vanna (Billions)", "4. Charm", 
                "5. VIX Term Structure", "6. Market Sentiment (PCR)"
            ),
            vertical_spacing=0.12, horizontal_spacing=0.1,
            specs=[[{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, {"type": "xy"}], [{"type": "xy"}, None]]
        )

        fig.add_trace(go.Bar(x=df.index, y=df['GEX']/1e9, marker_color='#1f77b4', name='GEX'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=pain, fill='tozeroy', line_color='#d62728', name='Pain'), row=1, col=2)
        fig.add_trace(go.Bar(x=df.index, y=df['Vanna']/1e9, marker_color='#2ca02c', name='Vanna'), row=2, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['Charm'], marker_color='#ff7f0e', name='Charm'), row=2, col=2)
        fig.add_trace(go.Scatter(x=['9D', '30D', '3M'], y=[vix9d, vix30d, vix3m], mode='lines+markers', line=dict(width=3, color='#9467bd'), fill='tozeroy'), row=3, col=1)

        for r, c in [(1,1), (1,2), (2,1), (2,2)]:
            fig.add_vline(x=spot, line_dash="dot", line_color="black", opacity=0.8, annotation_text="현재가", annotation_position="top left", row=r, col=c)
            
        fig.add_vline(x=call_wall, line_width=2, line_dash="solid", line_color="red", annotation_text="Call Wall", annotation_position="top right", row=1, col=1)
        fig.add_vline(x=put_wall, line_width=2, line_dash="solid", line_color="blue", annotation_text="Put Wall", annotation_position="top left", row=1, col=1)
        fig.add_vline(x=max_pain_strike, line_width=2, line_dash="dash", line_color="purple", annotation_text="Max Pain", annotation_position="top right", row=1, col=2)

        fig.update_layout(height=900, showlegend=False, template='plotly_white')
        
        for r in [1, 2]:
            for c in [1, 2]:
                fig.update_xaxes(range=[lower_bound, upper_bound], row=r, col=c)

        indicator = go.Indicator(
            mode="gauge+number", value=pcr,
            domain={'x': [0.58, 0.98], 'y': [0.0, 0.23]},
            gauge={'axis': {'range': [0, 2]}, 'bar': {'color': 'black'},
                   'steps': [{'range': [0, 0.7], 'color': 'lightgreen'}, {'range': [1.3, 2], 'color': 'lightcoral'}]}
        )

        final_fig = go.Figure(data=list(fig.data) + [indicator], layout=fig.layout)
        
        # 스트림릿 환경에 맞게 차트 출력
        st.plotly_chart(final_fig, use_container_width=True)

# 사이드바 UI 구성
st.sidebar.header("⚙️ 설정")
default_date = pd.to_datetime('2026-03-20')
selected_date = st.sidebar.date_input("옵션 만기일 선택", value=default_date)
spot_range = st.sidebar.slider("행사가 탐색 범위 (%)", min_value=5, max_value=30, value=15, step=5) / 100.0

if st.sidebar.button("데이터 분석 실행 🚀"):
    run_dashboard(selected_date, spot_range)
else:
    st.info("👈 왼쪽 사이드바에서 만기일을 설정하고 '데이터 분석 실행' 버튼을 눌러주세요.")
