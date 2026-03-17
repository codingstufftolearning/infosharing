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

# ---------- FUNCTIONS ----------

# Fill missing prices with last available
def fill_missing_prices(prices, days):
    if not prices:
        return [0]*days
    filled = prices.copy()
    while len(filled) < days:
        filled.append(filled[-1])
    return filled[:days]

# ---- CoinGecko ----
def get_price_coingecko(cg_id, days, retries=3, wait=2):
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days={days}"
    for _ in range(retries):
        try:
            data = requests.get(url, timeout=10).json()
            if "prices" in data and len(data["prices"]) > 0:
                df = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
                daily_prices = df.groupby('date')['price'].last().tolist()
                return daily_prices[-days:]
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
def get_price_binance(symbol, interval='1d', limit=100, retries=3, wait=2):
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
    start_str = start.strftime("%Y-%m-%dT00:00:00Z")
    end_str = end.strftime("%Y-%m-%dT23:59:59Z")
    url = f"https://api.coinpaprika.com/v1/coins/{cp_id}/ohlcv/historical?start={start_str}&end={end_str}"
    for _ in range(retries):
        try:
            data = requests.get(url, timeout=10).json()
            if isinstance(data, list) and len(data) > 0:
                closes = [item['close'] for item in data if 'close' in item]
                return closes[-days:]
        except:
            time.sleep(wait)
    return []

def get_current_price_coinpaprika(cp_id):
    try:
        url = f"https://api.coinpaprika.com/v1/tickers/{cp_id}"
        data = requests.get(url, timeout=10).json()
        return data['quotes']['USD']['price']
    except:
        return None

# ---- Unified historical fetch ----
def get_price(coin_name, days):
    cg_id = coins[coin_name]["cg_id"]
    binance_symbol = coins[coin_name]["symbol"]
    cp_id = coins[coin_name]["cp_id"]

    prices = []

    # CoinPaprika
    cp_prices = get_price_coinpaprika(cp_id, days)
    if cp_prices:
        prices = cp_prices

    # CoinGecko fill missing
    if len(prices) < days:
        cg_prices = get_price_coingecko(cg_id, days)
        for i in range(days):
            if i < len(prices): continue
            if i < len(cg_prices): prices.append(cg_prices[i])

    # Binance fill remaining
    if len(prices) < days:
        bin_prices = get_price_binance(binance_symbol, interval='1d', limit=days)
        for i in range(days):
            if i < len(prices): continue
            if i < len(bin_prices): prices.append(bin_prices[i])

    # Last resort: fill with current price
    current = get_current_price(coin_name)
    while len(prices) < days:
        prices.append(current)

    # Ensure full length
    prices = fill_missing_prices(prices, days)
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

# ---- Analysis ----
def analyze(prices):
    if len(prices) < 2 or all(p == 0 for p in prices):
        return "No Data", 0
    change = ((prices[-1]-prices[0])/prices[0])*100
    if change > 2: return "Bullish 📈", change
    elif change < -2: return "Bearish 📉", change
    else: return "Neutral ➡️", change

def final_signal(trends):
    scores = {"Bullish 📈": 1, "Neutral ➡️":0, "Bearish 📉": -1, "No Data":0}
    total = sum(scores[t] for t in trends)
    if total>1: return "Overall Bullish 📈"
    elif total<-1: return "Overall Bearish 📉"
    else: return "Overall Neutral ➡️"

def estimate_next_price(prices):
    if len(prices)<2 or all(p==0 for p in prices): return 0
    x = np.arange(len(prices))
    y = np.array(prices)
    slope, intercept = np.polyfit(x, y, 1)
    return slope*(len(prices))+intercept

# ---------- STREAMLIT UI ----------
st.title("📊 Crypto Dashboard with ADA + Free Robust Historical Data")
st.write("Historical + current prices + trend analysis using free APIs.")

if st.button("Analyze Market"):
    for name in coins:
        st.subheader(name)
        current_price = get_current_price(name)
        st.markdown(f"**Current Price:** ${current_price:,.2f}")

        prices_1d = get_price(name, 1)
        prices_7d = get_price(name, 7)
        prices_30d = get_price(name, 30)

        day = analyze(prices_1d)
        week = analyze(prices_7d)
        month = analyze(prices_30d)
        trends = [day[0], week[0], month[0]]

        st.write(f"24h Trend: {day[0]} ({day[1]:.2f}%)")
        st.write(f"7d Trend: {week[0]} ({week[1]:.2f}%)")
        st.write(f"30d Trend: {month[0]} ({month[1]:.2f}%)")
        st.markdown(f"**Final Estimation Signal:** {final_signal(trends)}")

        estimated_price = estimate_next_price(prices_7d)
        if estimated_price>0:
            st.markdown(f"**Estimated Next Price (7d trend):** ${estimated_price:,.2f}")

        if prices_7d and any(p!=0 for p in prices_7d):
            df = pd.DataFrame(prices_7d, columns=["Price"])
            df_proj = df.copy()
            df_proj.loc[len(df)] = estimated_price
            st.line_chart(df_proj)
        else:
            st.write("No chart available for 7-day price.")

        st.markdown("---")
