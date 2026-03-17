# app.py
import streamlit as st
import requests
import pandas as pd
import time
import numpy as np

# ---------- CONFIG ----------
coins = {
    "Bitcoin": {"symbol": "BTCUSDT", "cg_id": "bitcoin"},
    "Ethereum": {"symbol": "ETHUSDT", "cg_id": "ethereum"},
    "Solana": {"symbol": "SOLUSDT", "cg_id": "solana"},
    "BNB": {"symbol": "BNBUSDT", "cg_id": "binancecoin"},
    "XRP": {"symbol": "XRPUSDT", "cg_id": "ripple"}
}

# ---------- FUNCTIONS ----------
def get_current_price(coin_name):
    cg_id = coins[coin_name]["cg_id"]
    binance_symbol = coins[coin_name]["symbol"]
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={cg_id}&vs_currencies=usd"
        data = requests.get(url, timeout=10).json()
        price = data[cg_id]['usd']
        return price
    except:
        prices = get_price_binance(binance_symbol, interval='1m', limit=1)
        if prices:
            return prices[-1]
    return 0

def get_price_coingecko(cg_id, days, retries=3, wait=2):
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days={days}"
    for attempt in range(retries):
        try:
            data = requests.get(url, timeout=10).json()
            if "prices" in data and len(data["prices"]) > 0:
                return [p[1] for p in data["prices"]]
        except:
            time.sleep(wait)
    return []

def get_price_binance(symbol, interval='1h', limit=100, retries=3, wait=2):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    for attempt in range(retries):
        try:
            data = requests.get(url, timeout=10).json()
            if len(data) > 0:
                return [float(item[4]) for item in data]
        except:
            time.sleep(wait)
    return []

def get_price(coin_name, days):
    cg_id = coins[coin_name]["cg_id"]
    binance_symbol = coins[coin_name]["symbol"]
    prices = get_price_coingecko(cg_id, days)
    if not prices:
        st.warning(f"CoinGecko failed for {coin_name}, using Binance fallback")
        interval = '1h'
        limit = days*24
        prices = get_price_binance(binance_symbol, interval=interval, limit=limit)
    if not prices:
        st.warning(f"No price data available for {coin_name}, using placeholder")
        prices = [0]
    return prices

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
st.title("📊 Crypto Dashboard with Estimated Next Price")
st.write("Shows current price, trend estimations, and simple next-price estimate with projected line.")

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

        # Trends
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
            # Add projected next price as last point
            df_proj = df.copy()
            df_proj.loc[len(df)] = estimated_price
            st.line_chart(df_proj)
        else:
            st.write("No chart available for 7-day price.")

        st.markdown("---")
