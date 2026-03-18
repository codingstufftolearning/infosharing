import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime

# Sentiment
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Firebase
import firebase_admin
from firebase_admin import credentials, db

# Statsmodels
from statsmodels.tsa.arima.model import ARIMA

# ---------------------------
# 🔐 FIREBASE INIT (FROM SECRETS)
# ---------------------------
if not firebase_admin._apps:
    cred = credentials.Certificate(dict(st.secrets["firebase"]))
    firebase_admin.initialize_app(cred, {
        "databaseURL": st.secrets["firebase"]["databaseURL"]
    })

# ---------------------------
# 📊 FETCH PRICE DATA (BINANCE)
# ---------------------------
def get_price_data(symbol="BTCUSDT"):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit=7"
    data = requests.get(url).json()

    prices = [float(x[4]) for x in data]
    dates = [x[0] for x in data]

    return np.array(prices), dates

# ---------------------------
# 📈 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))

    avg_gain = np.mean(gain)
    avg_loss = np.mean(loss)

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return np.append(np.zeros(len(prices)-1), 100 - (100 / (1 + rs)))

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12).mean()
    exp2 = pd.Series(prices).ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    return macd.values, signal.values

# ---------------------------
# 📰 SENTIMENT
# ---------------------------
def get_sentiment():
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
        scores.append((vader + blob) / 2)

    return np.mean(scores)

# ---------------------------
# 🔮 SAFE FORECAST
# ---------------------------
def forecast_price(prices):
    try:
        model = ARIMA(prices, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return float(forecast[0])
    except:
        return float(prices[-1])

# ---------------------------
# 🧠 AI-LIKE LEARNING (WEIGHTS)
# ---------------------------
def load_weights():
    ref = db.reference("weights")
    data = ref.get()
    if data:
        return data
    return {"rsi":1, "macd":1, "trend":1, "sentiment":1}

def save_weights(weights):
    db.reference("weights").set(weights)

# ---------------------------
# 📊 SMART PREDICTION
# ---------------------------
def smart_prediction(prices, rsi, macd, signal, sentiment, weights):
    score = 0

    if rsi[-1] < 30:
        score += 2 * weights["rsi"]
    elif rsi[-1] > 70:
        score -= 2 * weights["rsi"]

    if macd[-1] > signal[-1]:
        score += 1 * weights["macd"]
    else:
        score -= 1 * weights["macd"]

    if prices[-1] > prices.mean():
        score += 1 * weights["trend"]
    else:
        score -= 1 * weights["trend"]

    if sentiment > 0.2:
        score += 2 * weights["sentiment"]
    elif sentiment < -0.2:
        score -= 2 * weights["sentiment"]

    if score >= 3:
        return "BUY", score
    elif score <= -3:
        return "SELL", score
    else:
        return "HOLD", score

# ---------------------------
# ☁️ SAVE HISTORY
# ---------------------------
def save_data(symbol, price, prediction):
    ref = db.reference(f"history/{symbol}")
    ref.push({
        "time": datetime.utcnow().isoformat(),
        "price": float(price),
        "prediction": prediction
    })

# ---------------------------
# 📊 LOAD HISTORY
# ---------------------------
def load_history(symbol):
    ref = db.reference(f"history/{symbol}")
    data = ref.get()

    if not data:
        return []

    return [v["price"] for v in data.values()]

# ---------------------------
# 📈 WIN RATE TRACKER
# ---------------------------
def calculate_win_rate(symbol):
    ref = db.reference(f"history/{symbol}")
    data = ref.get()

    if not data or len(data) < 2:
        return 0

    values = list(data.values())

    wins = 0
    total = 0

    for i in range(len(values)-1):
        current = values[i]
        next_price = values[i+1]["price"]

        if current["prediction"] == "BUY" and next_price > current["price"]:
            wins += 1
        elif current["prediction"] == "SELL" and next_price < current["price"]:
            wins += 1

        total += 1

    return round((wins/total)*100, 2) if total > 0 else 0

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 AI Trading System")

symbol = st.selectbox("Select Coin", ["BTCUSDT", "ETHUSDT"])

if st.button("Analyze"):
    prices, dates = get_price_data(symbol)

    # Load history
    history = load_history(symbol)
    if history:
        prices = np.concatenate([history, prices])

    rsi = calculate_rsi(prices)
    macd, signal = calculate_macd(prices)
    sentiment = get_sentiment()
    weights = load_weights()

    prediction, score = smart_prediction(prices, rsi, macd, signal, sentiment, weights)
    next_price = forecast_price(prices)

    # Save data
    save_data(symbol, prices[-1], prediction)

    # Chart
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=prices, mode='lines', name='Price'))
    fig.add_trace(go.Scatter(
        x=[len(prices)-1, len(prices)],
        y=[prices[-1], next_price],
        mode='lines',
        name='Prediction'
    ))

    st.plotly_chart(fig)

    # Metrics
    st.metric("Prediction", prediction)
    st.metric("Score", score)
    st.metric("Next Price", round(next_price, 2))

    # Win rate
    win_rate = calculate_win_rate(symbol)
    st.metric("Win Rate (%)", win_rate)
