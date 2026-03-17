# app.py
import streamlit as st
import requests
import pandas as pd
import time

# ---------- CONFIG ----------
coins = {
    "Bitcoin": "BTCUSDT",
    "Ethereum": "ETHUSDT",
    "Solana": "SOLUSDT",
    "BNB": "BNBUSDT",
    "XRP": "XRPUSDT"
}

# ---------- FUNCTIONS ----------
def get_price_coingecko(coin_id, days, retries=3, wait=2):
    """Fetch price from CoinGecko, return list of prices"""
    cg_ids = {
        "BTCUSDT": "bitcoin",
        "ETHUSDT": "ethereum",
        "SOLUSDT": "solana",
        "BNBUSDT": "binancecoin",
        "XRPUSDT": "ripple"
    }
    url = f"https://api.coingecko.com/api/v3/coins/{cg_ids[coin_id]}/market_chart?vs_currency=usd&days={days}"
    for attempt in range(retries):
        try:
            data = requests.get(url, timeout=10).json()
            if "prices" in data and len(data["prices"]) > 0:
                return [p[1] for p in data["prices"]]
        except:
            time.sleep(wait)
    return []

def get_price_binance(symbol, interval='1h', limit=100, retries=3, wait=2):
    """Fetch fallback price from Binance public API"""
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    for attempt in range(retries):
        try:
            data = requests.get(url, timeout=10).json()
            if len(data) > 0:
                return [float(item[4]) for item in data]  # closing prices
        except:
            time.sleep(wait)
    return []

def get_price(coin, days):
    """Try CoinGecko first, fallback to Binance"""
    prices = get_price_coingecko(coin, days)
    if not prices:
        st.warning(f"CoinGecko failed for {coin}, using Binance fallback")
        prices = get_price_binance(coin, limit=days*24)  # approx hourly candles
    if not prices:
        st.warning(f"No price data available for {coin}, using placeholder")
        prices = [0]
    return prices

def analyze(prices):
    """Simple trend analysis"""
    if len(prices) < 2:
        return "No Data", 0
    change = ((prices[-1] - prices[0]) / prices[0]) * 100
    if change > 2:
        return "Bullish 📈", change
    elif change < -2:
        return "Bearish 📉", change
    else:
        return "Neutral ➡️", change

# ---------- STREAMLIT UI ----------
st.title("📊 Crypto Dashboard (Public Sources Only)")
st.write("Click the button to fetch multi-timeframe crypto analysis using CoinGecko + Binance fallback.")

if st.button("Analyze Market"):
    for coin in coins:
        st.subheader(coin)

        # Fetch price data
        prices_30min = get_price(coin, 1)[-10:]   # last ~30 min
        prices_1d = get_price(coin, 1)            # 24h
        prices_7d = get_price(coin, 7)            # 7d
        prices_30d = get_price(coin, 30)          # 30d

        # Analyze trends
        short = analyze(prices_30min)
        day = analyze(prices_1d)
        week = analyze(prices_7d)
        month = analyze(prices_30d)

        # Display results
        st.write(f"30 min: {short[0]} ({short[1]:.2f}%)")
        st.write(f"24h: {day[0]} ({day[1]:.2f}%)")
        st.write(f"7d: {week[0]} ({week[1]:.2f}%)")
        st.write(f"30d: {month[0]} ({month[1]:.2f}%)")

        # Chart for 7-day price
        if prices_7d and all(p > 0 for p in prices_7d):
            df = pd.DataFrame(prices_7d, columns=["Price"])
            st.line_chart(df)
        else:
            st.write("No chart available due to missing price data.")

        st.markdown("---")
