# app.py
import streamlit as st
import requests
import pandas as pd
import time
import numpy as np

# ---------- CONFIG ----------
coins = {
    "Bitcoin": {"symbol": "BTCUSDT", "cg_id": "bitcoin", "cp_id": "btc-bitcoin"},
    "Ethereum": {"symbol": "ETHUSDT", "cg_id": "ethereum", "cp_id": "eth-ethereum"},
    "Solana": {"symbol": "SOLUSDT", "cg_id": "solana", "cp_id": "sol-solana"},
    "Cardano": {"symbol": "ADAUSDT", "cg_id": "cardano", "cp_id": "ada-cardano"}
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------- DEBUG ----------
def log(msg):
    st.session_state.logs.append(msg)

def init_logs():
    if "logs" not in st.session_state:
        st.session_state.logs = []

# ---------- HELPERS ----------
def safe_request(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            log(f"❌ HTTP {r.status_code}: {url}")
    except Exception as e:
        log(f"❌ Error: {e}")
    return None

def fill_missing_prices(prices, days):
    if not prices:
        return []
    while len(prices) < days:
        prices.append(prices[-1])
    return prices[:days]

def generate_fake_prices(current_price, days):
    if current_price == 0:
        return [0]*days
    prices = []
    for _ in range(days):
        change = np.random.uniform(-0.02, 0.02)
        current_price *= (1 + change)
        prices.append(current_price)
    return prices

# ---------- DATA SOURCES ----------

@st.cache_data(ttl=60)
def get_price_coinpaprika(cp_id, days):
    log("🟡 Trying CoinPaprika...")
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=days)
    url = f"https://api.coinpaprika.com/v1/coins/{cp_id}/ohlcv/historical?start={start.strftime('%Y-%m-%dT00:00:00Z')}&end={end.strftime('%Y-%m-%dT23:59:59Z')}"
    data = safe_request(url)
    if isinstance(data, list) and len(data) > 0:
        log("✅ CoinPaprika success")
        return [d['close'] for d in data if 'close' in d]
    log("❌ CoinPaprika failed")
    return []

@st.cache_data(ttl=60)
def get_price_coingecko(cg_id, days):
    log("🟡 Trying CoinGecko...")
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days={days}"
    data = safe_request(url)
    if data and "prices" in data:
        df = pd.DataFrame(data["prices"], columns=["t","p"])
        df['d'] = pd.to_datetime(df['t'], unit='ms').dt.date
        result = df.groupby('d')['p'].last().tolist()
        log("✅ CoinGecko success")
        return result
    log("❌ CoinGecko failed")
    return []

@st.cache_data(ttl=60)
def get_price_binance(symbol, days):
    log("🟡 Trying Binance...")
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={days}"
    data = safe_request(url)
    if isinstance(data, list) and len(data) > 0:
        log("✅ Binance success")
        return [float(d[4]) for d in data]
    log("❌ Binance failed")
    return []

@st.cache_data(ttl=30)
def get_current_price(coin):
    cg = coins[coin]["cg_id"]
    cp = coins[coin]["cp_id"]
    sym = coins[coin]["symbol"]

    # CoinGecko
    data = safe_request(f"https://api.coingecko.com/api/v3/simple/price?ids={cg}&vs_currencies=usd")
    if data and cg in data:
        return data[cg]['usd']

    # Binance
    data = safe_request(f"https://api.binance.com/api/v3/ticker/price?symbol={sym}")
    if data and "price" in data:
        return float(data["price"])

    # CoinPaprika
    data = safe_request(f"https://api.coinpaprika.com/v1/tickers/{cp}")
    if data and "quotes" in data:
        return data['quotes']['USD']['price']

    return 0

# ---------- UNIFIED FETCH ----------
def get_price(coin, days):
    cg = coins[coin]["cg_id"]
    cp = coins[coin]["cp_id"]
    sym = coins[coin]["symbol"]

    prices = []

    # 1. CoinPaprika
    prices = get_price_coinpaprika(cp, days)

    # 2. CoinGecko
    if len(prices) < 2:
        prices = get_price_coingecko(cg, days)

    # 3. Binance
    if len(prices) < 2:
        prices = get_price_binance(sym, days)

    # 4. Fix / fallback
    current = get_current_price(coin)

    if len(prices) < 2:
        log("⚠️ Using synthetic fallback data")
        prices = generate_fake_prices(current, days)
    else:
        prices = fill_missing_prices(prices, days)

    return prices

# ---------- ANALYSIS ----------
def analyze(prices):
    if len(prices) < 2:
        return "No Data", 0
    change = ((prices[-1]-prices[0])/prices[0])*100
    if change > 2: return "Bullish 📈", change
    elif change < -2: return "Bearish 📉", change
    else: return "Neutral ➡️", change

def estimate_next_price(prices):
    if len(prices) < 2:
        return 0
    x = np.arange(len(prices))
    y = np.array(prices)
    slope, intercept = np.polyfit(x, y, 1)
    return slope * len(prices) + intercept

# ---------- UI ----------
st.title("📊 Crypto Dashboard (Stable Free Version)")
init_logs()

if st.button("Analyze Market"):
    st.session_state.logs = []

    for name in coins:
        st.subheader(name)

        current = get_current_price(name)
        st.write(f"Current Price: ${current:,.2f}")

        prices = get_price(name, 7)

        trend, change = analyze(prices)
        st.write(f"7d Trend: {trend} ({change:.2f}%)")

        est = estimate_next_price(prices)
        if est > 0:
            st.write(f"Estimated Next Price: ${est:,.2f}")

        df = pd.DataFrame(prices, columns=["Price"])
        df.loc[len(df)] = est
        st.line_chart(df)

        st.markdown("---")

    # DEBUG PANEL
    with st.expander("🔍 Debug Logs"):
        for l in st.session_state.logs:
            st.text(l)
