# Quant Dashboard App

Streamlit 기반의 멀티-모듈 시장 분석 대시보드입니다.  
단일 티커 차트 분석과 거시/옵션/브레드스 특화 대시보드를 한 앱 안에서 함께 사용할 수 있도록 구성되어 있습니다.

이 프로젝트는 미국 주식/ETF뿐 아니라 한국 주식 코드도 함께 다루며, `Breadth Thrust` 같은 무거운 화면은 진입 시 자동 계산하지 않고 `Query` 버튼을 눌렀을 때만 실행되도록 설계되어 있습니다.

## 무엇을 할 수 있나

### Core chart views

- `Elder Impulse`: 단기 모멘텀과 추세 필터를 함께 확인
- `TD Sequential`: 셋업/카운트다운 기반 exhaustion 맥락 확인
- `Robust STL`: 추세 대비 사이클 stretch 확인
- `SMC`: Smart Money Concept 관점의 value area / zone / structure 확인
- `SuperTrend`: ATR 기반 추세 전환선 확인
- `Williams Vix Fix`: 공포/안도 스파이크 확인
- `Squeeze Momentum`: Bollinger vs Keltner 압축 상태 확인

### Special dashboards

- `Market Pulse`: 크로스에셋 기반 risk appetite 체크
- `Canary Momentum`: risk-on / risk-off 회전 신호 점검
- `Breadth Thrust`:  
  `KOSPI200`, `Nasdaq-100`, `S&P 500`에 대해 아래 항목을 한 번에 조회
  - Zweig Breadth Thrust
  - 200일 이평선 상회 종목 비율
  - 200일 이평선 평균 괴리율
  - 최근 breadth / 200DMA tape 및 diagnostics
- `Fed Watch`: FRED 기반 유동성/자금시장 상태 점검
- `Options Flow`: SPX 옵션 포지셔닝, max pain, zero gamma, put/call 흐름 확인
- `JHEQX Collar`: JHEQX collar 구조 재구성
- `ETF Sortino Leadership`: 대형 미국 ETF 리더십과 섹터 흐름 확인

## 주요 특징

- 사이드바에서 `Core Charts`와 `Special dashboards`를 compact selector로 분리
- `Breadth Thrust`는 on-demand 실행
  - 진입만으로는 무거운 계산이 돌지 않음
  - `Query` 클릭 시에만 데이터 수집/계산 실행
  - 진행 중 `progress`와 상태 문구 표시
  - 이전 성공 결과는 새 결과가 끝날 때까지 유지
- `Mobile-friendly charts` 옵션 제공
- `Force refresh cached data` 옵션 제공
- `Yahoo Finance`, `FinanceDataReader`, `FRED`, `CBOE`, 공식/공개 index constituent 소스를 조합해 사용

## 지원 티커 예시

- 미국 주식/ETF: `NVDA`, `QQQ`, `SPY`, `TSLA`
- 한국 주식: `005930.KS`, `035420.KQ`
- 한국 6자리 코드도 입력 가능: `005930`
- 크립토 페어: `BTC-USD`

## 프로젝트 구조

- [app.py](./app.py): Streamlit 실행용 얇은 런처
- [app.py.txt](./app.py.txt): 메인 애플리케이션 구현
- [tests/test_breadth_thrust.py](./tests/test_breadth_thrust.py): breadth / 200DMA 계산 회귀 테스트
- [README-pythonbox.md](./README-pythonbox.md): PythonBox 환경 셋업 안내
- [bootstrap.ps1](./bootstrap.ps1): PythonBox 부트스트랩 스크립트

## 빠른 시작

### 1. 일반 Python 환경

Python 3.12 기준을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install streamlit numpy pandas matplotlib plotly scipy statsmodels yfinance finance-datareader requests lxml pytest
python -m streamlit run .\app.py
```

브라우저에서 기본적으로 `http://localhost:8501`이 열립니다.

### 2. PythonBox 환경

VDI / Citrix 환경에서 사용자 프로필 바깥의 `T:\PythonBox`에 Python과 캐시를 두고 싶다면:

```powershell
powershell -ExecutionPolicy Bypass -File T:\Project\bootstrap.ps1
T:\PythonBox\venv\Scripts\python.exe -m streamlit run T:\Project\app.py
```

자세한 셋업은 [README-pythonbox.md](./README-pythonbox.md)를 참고하세요.

## 사용 방법

1. 사이드바에서 티커와 `History window`를 선택합니다.
2. `Chart View`에서 코어 차트를 선택하거나, `Special Dashboard`에서 특화 화면을 고릅니다.
3. `Open Special` 버튼으로 special dashboard를 열 수 있습니다.
4. `Breadth Thrust`는 시장 선택 후 `Query`를 눌러야 계산이 시작됩니다.
5. 캐시를 무시하고 새로 받고 싶으면 `Force refresh cached data`를 사용합니다.

## 테스트와 검증

### 문법 확인

```powershell
python -m py_compile .\app.py .\app.py.txt .\tests\test_breadth_thrust.py
```

### breadth / 200DMA 테스트

```powershell
python -m pytest .\tests\test_breadth_thrust.py -q
```

현재 테스트에는 다음이 포함됩니다.

- valid / invalid Zweig breadth thrust detection
- missing history가 있는 구성종목 처리
- 200DMA 상회 비율 계산
- 200DMA 평균 괴리율 계산
- 일부 시장 실패 시 payload usable 여부

## 데이터 소스

화면에 따라 다음 소스를 사용합니다.

- Yahoo Finance
- FinanceDataReader
- FRED
- CBOE delayed quotes
- Nasdaq-100 companies page
- Wikipedia constituent snapshots
- SEC / JPM / NEOS 공개 자료 일부

각 special dashboard 하단 caption에서 실제 사용 소스를 추가로 확인할 수 있습니다.

## 구현 메모

- `app.py.txt`가 본체이고, `app.py`는 `streamlit run` 호환성을 위한 wrapper입니다.
- `Breadth Thrust`의 index constituent는 현재 snapshot 기준입니다.
  따라서 과거 신호에는 survivorship bias가 존재할 수 있습니다.
- 200DMA 참여율과 평균 괴리율은 해당 날짜에 유효한 `SMA200`가 존재하는 종목만 포함해 계산합니다.
- `JHEQX Collar`는 관련 모듈/소스가 준비되지 않으면 일부 기능이 제한될 수 있습니다.

## 권장 개선 포인트

- `requirements.txt` 또는 `pyproject.toml` 추가
- 앱 스모크 테스트 자동화
- `use_container_width` 관련 Streamlit deprecation 정리
- breadth constituent snapshot 소스의 장기 안정성 보강
