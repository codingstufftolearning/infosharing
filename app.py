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
    end = pd.Timestamp.now()
    start = end - pd.Timedelta(days=days)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    url = f"https://api.coinpaprika.com/v1/coins/{cp_id}/ohlcv/historical?start={start_str}&end={end_str}"

    for _ in range(retries):
        try:
            data = requests.get(url, timeout=10).json()
            if isinstance(data, list) and len(data) > 0:
                closes = [item['close'] for item in data if 'close' in item]
                if closes:
                    return closes
        except:
            time.sleep(wait)
    # Fallback: use current price repeated
    current = get_current_price_coinpaprika(cp_id)
    if current:
        return [current] * max(1, days)
    return [0] * max(1, days)

def get_current_price_coinpaprika(cp_id):
    try:
        url = f"https://api.coinpaprika.com/v1/tickers/{cp_id}"
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

    # Binance fallback
    interval = '1h'
    limit = days*24
    prices = get_price_binance(binance_symbol, interval=interval, limit=limit)
    if prices:
        return prices

    # CoinPaprika fallback
    prices = get_price_coinpaprika(cp_id, days)
    return prices

def get_current_price(coin_name):
    cg_id = coins[coin_name]["cg_id"]
    binance_symbol = coins[coin_name]["symbol"]
    cp_id = coins[coin_name]["cp_id"]

    price = get_current_price_coingecko(cg_id)
    if price: return price

    price = get_current_price_binance(binance_symbol)
    if price: return price

    price = get_current_price_coinpaprika(cp_id)
    if price: return price

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
st.title("📊 Crypto Dashboard with Robust Data Fetch")
st.write("Real-time price + historical data + trend estimation using multiple fallbacks.")

if st.button("Analyze Market"):
    for name in coins:
        st.subheader(name)

        current_price = get_current_price(name)
        st.markdown(f"**Current Price:** ${current_price:,.2f}")

        prices_1d  = get_price(name, 1)
        prices_7d  = get_price(name, 7)
        prices_30d = get_price(name, 30)

        day   = analyze(prices_1d)
        week  = analyze(prices_7d)
        month = analyze(prices_30d)
        trends = [day[0], week[0], month[0]]

        st.write(f"24h Trend: {day[0]} ({day[1]:.2f}%)")
        st.write(f"7d Trend: {week[0]} ({week[1]:.2f}%)")
        st.write(f"30d Trend: {month[0]} ({month[1]:.2f}%)")
        st.markdown(f"**Final Estimation Signal:** {final_signal(trends)}")

        estimated_price = estimate_next_price(prices_7d)
        if estimated_price > 0:
            st.markdown(f"**Estimated Next Price (7d trend):** ${estimated_price:,.2f}")

        if prices_7d and all(p > 0 for p in prices_7d):
            df = pd.DataFrame(prices_7d, columns=["Price"])
            df_proj = df.copy()
            df_proj.loc[len(df)] = estimated_price
            st.line_chart(df_proj)
        else:
            st.write("No chart available for 7-day price.")

        st.markdown("---")
