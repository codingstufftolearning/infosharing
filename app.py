# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.arima.model import ARIMA
import firebase_admin
from firebase_admin import credentials, db

# ---------------------------
# 🔐 FIREBASE INIT
# ---------------------------
firebase_admin_initialized = False
try:
    firebase_dict = dict(st.secrets["firebase"])
    firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(firebase_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            "databaseURL": firebase_dict["databaseURL"]
        })
    firebase_admin_initialized = True
except Exception as e:
    st.error(f"Firebase init failed: {type(e).__name__}: {e}")
    import traceback
    st.text(traceback.format_exc())

# ---------------------------
# 🔹 FETCH PRICE DATA (COINGECKO)
# ---------------------------
def fetch_historical_daily(symbol, days=30):
    """Fetch last `days` daily prices for the coin."""
    coin = symbol.replace("USDT","").lower()
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
    try:
        data = requests.get(url, timeout=10).json()
        prices = [x[1] for x in data["prices"]]
        dates = [datetime.fromtimestamp(x[0]/1000) for x in data["prices"]]
        return np.array(prices), dates
    except Exception as e:
        st.warning(f"Failed fetching daily historical for {symbol}: {e}")
        return np.array([]), []

def fetch_recent_hourly(symbol, hours=24):
    """Fetch last `hours` of hourly price for the coin."""
    coin = symbol.replace("USDT","").lower()
    url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=1"
    try:
        data = requests.get(url, timeout=10).json()
        prices = [x[1] for x in data["prices"]][-hours:]
        dates = [datetime.fromtimestamp(x[0]/1000) for x in data["prices"]][-hours:]
        return np.array(prices), dates
    except Exception as e:
        st.warning(f"Failed fetching hourly recent for {symbol}: {e}")
        return np.array([]), []

# ---------------------------
# 📈 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    if len(prices) < period: return np.zeros(len(prices))
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100/(1+rs))
    return np.concatenate([np.zeros(1), rsi])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.values, signal.values

# ---------------------------
# 📰 SENTIMENT
# ---------------------------
def get_sentiment(texts=None):
    if texts is None:
        texts = [
            "Bitcoin is rising fast",
            "Crypto market crash fears",
            "Strong bullish momentum ahead"
        ]
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for text in texts:
        vader = analyzer.polarity_scores(text)["compound"]
        blob = TextBlob(text).sentiment.polarity
        scores.append((vader + blob)/2)
    return np.mean(scores)

# ---------------------------
# 🔮 SAFE FORECAST
# ---------------------------
def forecast_price(prices):
    if len(prices)<10: return float(prices[-1])
    try:
        model = ARIMA(prices, order=(1,1,1))
        model_fit = model.fit()
        return float(model_fit.forecast(steps=1)[0])
    except:
        return float(prices[-1])

# ---------------------------
# 🧠 WEIGHTS
# ---------------------------
def load_weights():
    if not firebase_admin_initialized: return {"rsi":1,"macd":1,"trend":1,"sentiment":1}
    ref = db.reference("weights")
    data = ref.get() or {}
    if not data: return {"rsi":1,"macd":1,"trend":1,"sentiment":1}
    return data

def save_weights(weights):
    if firebase_admin_initialized:
        db.reference("weights").set(weights)

# ---------------------------
# 📊 SMART PREDICTION
# ---------------------------
def smart_prediction(prices, rsi, macd, signal, sentiment, weights):
    score = 0
    if rsi[-1] < 30: score += 2*weights.get("rsi",1)
    elif rsi[-1] > 70: score -= 2*weights.get("rsi",1)
    if macd[-1] > signal[-1]: score += 1*weights.get("macd",1)
    else: score -= 1*weights.get("macd",1)
    if prices[-1] > prices.mean(): score += 1*weights.get("trend",1)
    else: score -= 1*weights.get("trend",1)
    if sentiment > 0.2: score += 2*weights.get("sentiment",1)
    elif sentiment < -0.2: score -= 2*weights.get("sentiment",1)
    if score>=3: return "BUY", score
    elif score<=-3: return "SELL", score
    else: return "HOLD", score

# ---------------------------
# ☁️ HISTORY
# ---------------------------
def save_data(symbol, price, prediction, timeframe):
    if not firebase_admin_initialized: return
    ref = db.reference(f"history/{symbol}/{timeframe}")
    ref.push({"time":datetime.utcnow().isoformat(), "price":float(price), "prediction":prediction})

def load_history(symbol, timeframe):
    if not firebase_admin_initialized: return []
    ref = db.reference(f"history/{symbol}/{timeframe}")
    data = ref.get() or {}
    return [v["price"] for v in data.values()]

def calculate_win_rate(symbol):
    if not firebase_admin_initialized: return 0
    ref = db.reference(f"history/{symbol}/hourly")
    data = ref.get() or {}
    if len(data)<2: return 0
    values = list(data.values())
    wins, total = 0, 0
    for i in range(len(values)-1):
        current = values[i]
        next_price = values[i+1]["price"]
        if current["prediction"]=="BUY" and next_price>current["price"]: wins+=1
        elif current["prediction"]=="SELL" and next_price<current["price"]: wins+=1
        total+=1
    return round((wins/total)*100,2) if total>0 else 0

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 AI Crypto Trading Bot")

# Coins selection
symbols = st.multiselect("Select Coins", ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT"], default=["BTCUSDT"])
chart_type = st.selectbox("Chart Type", ["Line", "Candlestick"])

if st.button("Analyze"):
    weights = load_weights()
    for symbol in symbols:
        st.subheader(f"Analysis for {symbol}")

        # 1️⃣ Load historical daily
        daily_prices, daily_dates = fetch_historical_daily(symbol, days=30)
        if len(daily_prices)==0:
            st.warning(f"No historical daily data for {symbol}, skipping.")
            continue
        # Save daily if first time
        for i, price in enumerate(daily_prices):
            save_data(symbol, price, "HISTORICAL", "daily")

        # 2️⃣ Load recent hourly
        hourly_prices, hourly_dates = fetch_recent_hourly(symbol)
        if len(hourly_prices)==0:
            st.warning(f"No hourly data for {symbol}, skipping recent update.")
            continue
        for i, price in enumerate(hourly_prices):
            save_data(symbol, price, "HISTORICAL", "hourly")

        # Combine for calculation
        prices = np.concatenate([daily_prices, hourly_prices])
        rsi = calculate_rsi(prices)
        macd, signal = calculate_macd(prices)
        sentiment = get_sentiment()
        prediction, score = smart_prediction(prices, rsi, macd, signal, sentiment, weights)
        next_price = forecast_price(prices)
        save_weights(weights)

        # Chart
        fig = go.Figure()
        if chart_type=="Line":
            fig.add_trace(go.Scatter(x=list(daily_dates)+list(hourly_dates), y=prices, mode="lines", name="Price"))
            fig.add_trace(go.Scatter(x=[hourly_dates[-1], hourly_dates[-1]+timedelta(hours=1)],
                                     y=[prices[-1], next_price],
                                     mode="lines", name="Forecast"))
        else:
            # Candlestick simplified (open/high/low/close as current ± small range)
            ohlc = []
            for p in prices:
                ohlc.append([p*0.995, p*1.005, p*0.995, p*1.005])
            fig.add_trace(go.Candlestick(
                x=list(daily_dates)+list(hourly_dates),
                open=[x[0] for x in ohlc],
                high=[x[1] for x in ohlc],
                low=[x[2] for x in ohlc],
                close=[x[3] for x in ohlc],
                name="Price"
            ))
        st.plotly_chart(fig)

        # Metrics
        st.metric("Prediction", prediction)
        st.metric("Score", score)
        st.metric("Next Price", round(next_price,2))
        st.metric("Win Rate (%)", calculate_win_rate(symbol))
