# app.py
import streamlit as st
import requests
import pandas as pd
import time  # <- FIX: needed for retries

# ---------- CONFIG ----------
coins = {
    "Bitcoin": "bitcoin",
    "Ethereum": "ethereum",
    "Solana": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple"
}

# ---------- FUNCTIONS ----------
def get_price(coin, days, retries=3, wait=2):
    """Fetch price history from CoinGecko with retry"""
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
    for attempt in range(retries):
        try:
            data = requests.get(url, timeout=10).json()
            if "prices" in data and len(data["prices"]) > 0:
                return [p[1] for p in data["prices"]]
        except Exception as e:
            st.warning(f"Attempt {attempt+1} failed for {coin}: {e}")
        time.sleep(wait)
    st.warning(f"No price data returned for {coin}. Using placeholder.")
    return [0]  # fallback so app doesn’t crash

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
st.title("📊 Crypto Dashboard (CoinGecko Only)")
st.write("Click the button to fetch multi-timeframe crypto analysis.")

if st.button("Analyze Market"):
    for name, coin in coins.items():
        st.subheader(name)

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
