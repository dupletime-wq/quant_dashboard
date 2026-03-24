# Quant Fusion Dashboard

Quant Fusion Dashboard 는 Streamlit 기반 시장/종목 해석 대시보드입니다.  
목적은 지표를 많이 나열하는 것이 아니라, 지금 시장과 종목이 어떤 상태인지 빠르게 분류하고 리스크를 관리할 수 있게 만드는 데 있습니다.

이 프로젝트는 크게 두 축으로 구성됩니다.

- 코어 차트: 개별 종목의 추세, 피로도, 구조를 해석
- 스페셜 대시보드: 시장 배경, 리스크 온/오프, 옵션 포지셔닝, ETF 리더십을 해석

현재 포함된 모듈은 아래와 같습니다.

- 코어 차트: `Elder Impulse`, `TD Sequential`, `Robust STL`, `SMC`, `SuperTrend`, `Williams Vix Fix`, `Squeeze Momentum`
- 스페셜 대시보드: `Fear & Greed`, `Canary`, `Option Gamma`, `ETF Sortino`

## 1. 이 대시보드를 어떻게 써야 하나

이 대시보드는 "정답을 찍는 도구"보다 "현재 상태를 분류하는 도구"에 가깝습니다.  
실전에서는 보통 아래 순서로 읽는 것이 가장 자연스럽습니다.

1. 먼저 `Fear & Greed` 또는 `Canary`로 시장이 위험 선호 국면인지 방어 국면인지 확인합니다.
2. 개별 종목에서는 `SuperTrend`, `Elder Impulse`로 방향성과 추세 지속 여부를 봅니다.
3. `TD Sequential`, `Robust STL`, `Williams Vix Fix`, `Squeeze Momentum`으로 과열/과매도 또는 압축 해제를 보완합니다.
4. `SMC`로 현재 가격이 어떤 구조적 구간에 있는지 확인합니다.
5. 인덱스나 대형주를 볼 때는 `Option Gamma`로 옵션 수급이 가격 움직임을 누르는지, 증폭시키는지 확인합니다.
6. `ETF Sortino`로 최근 6개월 동안 리스크 대비 가장 강한 ETF가 어디에 몰려 있는지 보고, 시장의 주도 섹터와 리스크 집중도를 확인합니다.

즉 이 프로젝트는 아래 네 가지 질문에 답하기 위한 도구입니다.

- 방향: 지금 추세가 위인지 아래인지
- 피로도: 지금 추세가 과열인지 눌림인지
- 위치: 지금 가격이 구조상 어디에 있는지
- 배경: 시장 전체가 지금 그 방향을 지지하는지

## 2. 빠른 실행 방법

### 설치

```bash
pip install streamlit pandas numpy plotly yfinance FinanceDataReader requests scipy statsmodels
```

### 실행

```bash
streamlit run Project1.py
```

## 3. 사이드바와 화면 구성

사이드바는 크게 두 부분으로 나뉩니다.

- `Ticker`: 분석할 종목 입력
- `History window`: 가격 히스토리 길이 선택
- `Chart View`: 개별 종목용 코어 차트 선택
- `Quick Access`: 스페셜 대시보드 바로가기

Quick Access에는 아래 4개가 들어 있습니다.

- `Fear & Greed`: 크로스에셋 위험 선호/회피 환경
- `Canary`: 리스크 온/오프 전환과 공격 자산 순위
- `Option Gamma`: SPX 옵션 기반 dealer positioning
- `ETF Sortino`: 미국 상장 주식형 ETF의 리스크 대비 강도 순위와 섹터 집중도

## 4. 코어 차트 해석

### 4.1 Elder Impulse

핵심 아이디어:

- `EMA13`의 기울기와 `MACD Histogram`의 기울기를 동시에 봅니다.
- 여기에 `EMA65` 위/아래 여부를 붙여 추세 필터를 겹칩니다.

어떻게 읽나:

- `Bullish impulse`: 단기 모멘텀과 추세가 같이 받쳐주는 상태
- `Bearish impulse`: 단기 모멘텀이 꺾이고 추세도 약한 상태
- 중립/엇갈림: 방향성이 약하거나 아직 추세 필터와 충돌하는 상태

### 4.2 TD Sequential

핵심 아이디어:

- 연속된 카운트를 통해 추세의 피로도를 봅니다.

어떻게 읽나:

- `Setup 9`: 단기 피로가 쌓이는 단계
- `Countdown 13`: 더 강한 소진 신호

주의:

- 이것은 방향 지표보다 피로도 지표에 가깝습니다.
- 단독 매수/매도 신호로 보기보다, 추세 지표와 같이 봐야 합니다.

### 4.3 Robust STL

핵심 아이디어:

- 가격을 추세와 순환 성분으로 나눠, 현재 가격이 통계적으로 얼마나 과열/과매도인지 추정합니다.

어떻게 읽나:

- `Cycle Score`가 높으면 과열 구간 경계
- `Cycle Score`가 낮으면 과매도/눌림 가능성

주의:

- 강한 추세에서는 높은 점수가 곧바로 하락을 의미하지는 않습니다.

### 4.4 SMC

핵심 아이디어:

- `Order Block`, `Fair Value Gap`, `POC`, 최근 스윙 레벨을 같이 보여줍니다.

어떻게 읽나:

- 현재 가격이 수요 구역 위인지, 공급 구역 아래인지
- 최근 유효한 구조 구간이 아직 살아 있는지
- `POC`, `EQ(50%)` 근처에서 균형 회복이 일어나는지

### 4.5 SuperTrend

핵심 아이디어:

- ATR 기반 밴드로 현재 추세가 유지되는지 확인합니다.

어떻게 읽나:

- 가격이 `SuperTrend` 위에 있으면 추세 지지
- 아래에 있으면 하락 추세 또는 약세 구간
- `Long flip`, `Short flip`은 추세 체인지 이벤트로 봅니다

### 4.6 Williams Vix Fix

핵심 아이디어:

- 종목 내부의 공포/과열 스파이크를 포착합니다.

어떻게 읽나:

- `Oversold`: 급락 스트레스가 극단으로 간 상태
- `Overbought`: 과열/안도 심리가 강한 상태
- `exit`는 극단 구간 종료 후 반응을 보는 보조 신호입니다

### 4.7 Squeeze Momentum

핵심 아이디어:

- Bollinger Band와 Keltner Channel 관계로 압축과 확장을 봅니다.

어떻게 읽나:

- `Squeeze On`: 에너지 압축 구간
- `Squeeze Off`: 압축 해제 이후 확장 가능성
- 히스토그램 방향은 확장 편향을 보조합니다

## 5. 스페셜 대시보드 해석

### 5.1 Fear & Greed

핵심 아이디어:

- SPY, VIX, 섹터, breadth, credit, dollar 등 크로스에셋 데이터를 결합해 시장 심리를 점수화합니다.

어떻게 읽나:

- 높은 점수: 위험 선호가 강하지만 과열일 수도 있음
- 낮은 점수: 방어 심리가 강하지만 극단 공포일 수도 있음

이 화면은 개별 종목 매수 타점보다 시장 배경 해석에 더 적합합니다.

### 5.2 Canary

핵심 아이디어:

- 카나리아 자산군의 모멘텀을 통해 리스크 온/오프 여부를 봅니다.
- 동시에 공격 자산 모멘텀 순위를 보여줘 다음 리밸런싱 후보를 확인할 수 있습니다.

어떻게 읽나:

- `Completed Month`: 현재 확정 포트폴리오 상태
- `Today`: 오늘 기준 실시간 전환 가능성
- `Expected Rotation`: 강한 공격 자산 후보

### 5.3 Option Gamma

핵심 아이디어:

- SPX 옵션 체인에서 `Net GEX`, `Vanna`, `Charm`, `Max Pain`, `Put/Call Ratio`를 계산합니다.

어떻게 읽나:

- `Net GEX` 양수: 변동성 완충 가능성
- `Net GEX` 음수: 변동성 증폭 가능성
- `PCR` 상승: 방어 수요 증가 가능성
- `Max Pain`: 가격이 자주 끌리는 중심 구간 참고값

### 5.4 ETF Sortino

핵심 아이디어:

- 미국 상장 주식형 ETF를 넓게 모아 최근 6개월 `Sortino Ratio`를 계산합니다.
- 단순 수익률이 아니라 하방 변동성을 감안한 리스크 대비 강도를 순위화합니다.
- 상위 ETF들의 섹터 비중을 집계해 현재 시장의 주도 섹터와 집중 리스크를 같이 보여줍니다.

화면에서 볼 수 있는 것:

- ETF Sortino 순위표
- 6개월 수익률
- downside volatility
- AUM 또는 거래 규모 대용치
- ETF별 theme/sector label
- sector weight coverage
- 상위 ETF 기반 sector share

실전 해석:

- 상위권 ETF가 특정 섹터에 과도하게 몰리면, 시장 주도주는 강하지만 리더십이 좁을 수 있습니다.
- 반대로 상위권이 여러 섹터로 분산되면, 리더십이 넓고 추세 확산이 건강할 가능성이 있습니다.
- coverage ratio가 낮으면 holdings 기반 섹터 통계의 신뢰도가 떨어질 수 있으니 같이 봐야 합니다.

## 6. 이 대시보드를 실전에서 같이 읽는 예시

### 추세 지속형

아래 조합이면 추세 지속 가능성이 높다고 해석할 수 있습니다.

- `SuperTrend` 상승
- `Elder Impulse` bullish
- `SMC`에서 수요 구간 위 유지
- `Squeeze Momentum`이 압축 해제 후 양수 확장

### 피로 누적형

아래 조합이면 단기 과열 또는 조정 가능성을 경계해야 합니다.

- `TD Sequential` 고카운트
- `Robust STL` 과열 점수
- `Williams Vix Fix`의 반대편 과열 신호

### 시장 배경 확인형

아래 조합이면 종목 신호보다 시장 배경을 더 중요하게 봐야 합니다.

- 종목 차트는 강해 보임
- 하지만 `Fear & Greed`가 극단 과열
- `Canary`가 방어 전환 조짐
- `Option Gamma`가 변동성 확대 가능성 시사

이 경우 기대 수익보다 보유 기간과 리스크 관리 기준을 더 보수적으로 잡는 편이 좋습니다.

## 7. 지원 심볼 예시

- 미국 주식/ETF: `NVDA`, `SPY`, `QQQ`, `TSLA`
- 한국 주식: `005930.KS`, `035420.KQ`, 일반 6자리 코드
- 기타: `BTC-USD`

## 8. 데이터 소스

- 가격 데이터: Yahoo Finance
- STL 보조 소스: FinanceDataReader
- 옵션 데이터: CBOE delayed quotes
- ETF 리더십/메타데이터: Yahoo Finance ETF metadata 및 holdings 정보

주의:

- 옵션 데이터는 지연 데이터입니다.
- ETF holdings 기반 sector coverage는 일부 ETF에서 비어 있을 수 있습니다.
- 데이터 제공 상태에 따라 일부 모듈은 `Not loaded` 또는 안내 메시지로 대체될 수 있습니다.

## 9. 모바일과 UI 관련 메모

- 모바일에서는 사이드바가 자동으로 접히도록 구성되어 있습니다.
- 차트는 반응형으로 렌더링되며, 밝은 배경 표면 기준으로 표시됩니다.
- 화면이 좁을수록 카드와 표는 세로 배치가 많아집니다.

## 10. 실행 파일

메인 앱 파일은 [app.py] 입니다.
