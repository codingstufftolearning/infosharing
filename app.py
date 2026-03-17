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
    "BNB": {"symbol": "BNBUSDT", "cg_id": "binancecoin", "cp_id": "bnb-binance-coin"},
    "XRP": {"symbol": "XRPUSDT", "cg_id": "ripple", "cp_id": "xrp-xrp"}
}

# ---------- FUNCTIONS ----------

# ---- CoinGecko ----
def get_price_coingecko(cg_id, days, retries=3, wait=2):
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days={days}"
    for _ in range(retries):
        try:
            data = requests.get(url, timeout=10).json()
            if "prices" in data and len(data["prices"]) > 0:
                return [p[1] for p in data["prices"]]
        except:
            time.sleep(wait)
    return []

def get_current_price_coingecko(cg_id):
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
        data = requests.get(url, timeout=10).json()
        return data[cg_id]['usd']
    except:
        return None

# ---- Binance ----
def get_price_binance(symbol, interval='1h', limit=100, retries=3, wait=2):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    for _ in range(retries):
        try:
            data = requests.get(url, timeout=10).json()
            if len(data) > 0:
                return [float(item[4]) for item in data]
        except:
            time.sleep(wait)
    return []

def get_current_price_binance(symbol):
    prices = get_price_binance(symbol, interval='1m', limit=1)
    return prices[-1] if prices else None

# ---- CoinPaprika ----
def get_price_coinpaprika(cp_id, days, retries=3, wait=2):
    start = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
    end = pd.Timestamp.now().strftime("%Y-%m-%d")
    url = f"https://api.coinpaprika.com/v1/coins/{cp_id}/ohlcv/historical?start={start}&end={end}"
    for _ in range(retries):
        try:
            data = requests.get(url, timeout=10).json()
            if len(data) > 0:
                return [item['close'] for item in data]
        except:
            time.sleep(wait)
    return []

def get_current_price_coinpaprika(cp_id):
    url = f"https://api.coinpaprika.com/v1/tickers/{cp_id}"
    try:
        data = requests.get(url, timeout=10).json()
        return data['quotes']['USD']['price']
    except:
        return None

# ---- Unified price fetch ----
def get_price(coin_name, days):
    cg_id = coins[coin_name]["cg_id"]
    binance_symbol = coins[coin_name]["symbol"]
    cp_id = coins[coin_name]["cp_id"]

    # Try CoinGecko
    prices = get_price_coingecko(cg_id, days)
    if prices:
        return prices

    st.warning(f"CoinGecko failed for {coin_name}, trying Binance fallback...")
    interval = '1h'
    limit = days*24
    prices = get_price_binance(binance_symbol, interval=interval, limit=limit)
    if prices:
        return prices

    st.warning(f"Binance failed for {coin_name}, trying CoinPaprika fallback...")
    prices = get_price_coinpaprika(cp_id, days)
    if prices:
        return prices

    st.warning(f"No historical data available for {coin_name}, using placeholder")
    return [0]

def get_current_price(coin_name):
    cg_id = coins[coin_name]["cg_id"]
    binance_symbol = coins[coin_name]["symbol"]
    cp_id = coins[coin_name]["cp_id"]

    # Try CoinGecko
    price = get_current_price_coingecko(cg_id)
    if price:
        return price

    # Fallback to Binance
    price = get_current_price_binance(binance_symbol)
    if price:
        return price

    # Fallback to CoinPaprika
    price = get_current_price_coinpaprika(cp_id)
    if price:
        return price

    return 0

# ---- Analysis & Estimation ----
def analyze(prices):
    if len(prices) < 2 or all(p == 0 for p in prices):
        return "No Data", 0
    change = ((prices[-1] - prices[0]) / prices[0]) * 100
    if change > 2:
        return "Bullish 📈", change
    elif change < -2:
        return "Bearish 📉", change
    else:
        return "Neutral ➡️", change

def final_signal(trends):
    scores = {"Bullish 📈": 1, "Neutral ➡️": 0, "Bearish 📉": -1, "No Data": 0}
    total = sum(scores[t] for t in trends)
    if total > 1:
        return "Overall Bullish 📈"
    elif total < -1:
        return "Overall Bearish 📉"
    else:
        return "Overall Neutral ➡️"

def estimate_next_price(prices):
    if len(prices) < 2 or all(p == 0 for p in prices):
        return 0
    x = np.arange(len(prices))
    y = np.array(prices)
    slope, intercept = np.polyfit(x, y, 1)
    return slope * (len(prices)) + intercept

# ---------- STREAMLIT UI ----------
st.title("📊 Crypto Dashboard with Real-Time & Historical Data")
st.write("Fetches current price, historical trends, and estimates next price using fallback sources.")

if st.button("Analyze Market"):
    for name in coins:
        st.subheader(name)

        # Current price
        current_price = get_current_price(name)
        st.markdown(f"**Current Price:** ${current_price:,.2f}")

        # Historical prices
        prices_1d  = get_price(name, 1)
        prices_7d  = get_price(name, 7)
        prices_30d = get_price(name, 30)

        # Analyze trends
        day   = analyze(prices_1d)
        week  = analyze(prices_7d)
        month = analyze(prices_30d)
        trends = [day[0], week[0], month[0]]

        st.write(f"24h Trend: {day[0]} ({day[1]:.2f}%)")
        st.write(f"7d Trend: {week[0]} ({week[1]:.2f}%)")
        st.write(f"30d Trend: {month[0]} ({month[1]:.2f}%)")
        st.markdown(f"**Final Estimation Signal:** {final_signal(trends)}")

        # Estimate next price
        estimated_price = estimate_next_price(prices_7d)
        if estimated_price > 0:
            st.markdown(f"**Estimated Next Price (7d trend):** ${estimated_price:,.2f}")

        # 7-day chart with projected next price
        if prices_7d and all(p > 0 for p in prices_7d):
            df = pd.DataFrame(prices_7d, columns=["Price"])
            df_proj = df.copy()
            df_proj.loc[len(df)] = estimated_price
            st.line_chart(df_proj)
        else:
            st.write("No chart available for 7-day price.")

        st.markdown("---")
