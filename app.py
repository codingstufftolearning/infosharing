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
import streamlit as st
import firebase_admin
from firebase_admin import credentials, db

firebase_admin_initialized = False
try:
    firebase_dict = dict(st.secrets["firebase"])
    firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(firebase_dict)
    firebase_admin.initialize_app(cred, {
        "databaseURL": firebase_dict["databaseURL"]
    })
    firebase_admin_initialized = True
except Exception as e:
    st.error(f"Firebase init failed: {type(e).__name__}: {e}")
    import traceback
    st.text(traceback.format_exc())
# ---------------------------
# 📊 FETCH PRICE DATA (BINANCE + FALLBACK)
# ---------------------------
@st.cache_data(ttl=300)
def get_price_data(symbol="BTCUSDT", limit=30):
    """
    Fetch historical price data from Binance.
    If Binance fails, fallback to CoinGecko.
    Returns:
        prices: np.array
        dates: list of datetime
    """
    prices, dates = [], []

    # First attempt: Binance API
    try:
        url_binance = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
        data_binance = requests.get(url_binance, timeout=5).json()
        prices = [float(x[4]) for x in data_binance]
        dates = [datetime.fromtimestamp(x[0]/1000) for x in data_binance]
        if prices:
            return np.array(prices), dates
    except Exception as e:
        st.warning(f"Binance fetch failed for {symbol}: {e}")

    # Fallback: CoinGecko API
    try:
        coin = symbol.replace("USDT","").lower()
        url_coingecko = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={limit}"
        data_cg = requests.get(url_coingecko, timeout=5).json()
        prices = [x[1] for x in data_cg["prices"]]
        dates = [datetime.fromtimestamp(x[0]/1000) for x in data_cg["prices"]]
        if prices:
            return np.array(prices), dates
    except Exception as e:
        st.warning(f"CoinGecko fetch failed for {symbol}: {e}")

    # Last fallback: empty arrays
    st.error(f"Failed to fetch price data for {symbol}.")
    return np.array([]), []

# ---------------------------
# 📈 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    if len(prices) < period:
        return np.zeros(len(prices))
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    rsi_full = np.concatenate([np.zeros(1), rsi])  # diff reduces length by 1
    return rsi_full

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.values, signal.values

# ---------------------------
# 📰 SENTIMENT ANALYSIS
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
        scores.append((vader + blob) / 2)
    return np.mean(scores)

# ---------------------------
# 🔮 SAFE FORECAST (ARIMA)
# ---------------------------
def forecast_price(prices):
    if len(prices) < 10:
        return float(prices[-1])
    try:
        model = ARIMA(prices, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return float(forecast[0])
    except:
        return float(prices[-1])

# ---------------------------
# 🧠 AUTO-LEARNING WEIGHTS
# ---------------------------
def load_weights():
    ref = db.reference("weights")
    data = ref.get() or {}
    if not data:
        return {"rsi":1, "macd":1, "trend":1, "sentiment":1}
    return data

def save_weights(weights):
    db.reference("weights").set(weights)

# ---------------------------
# 📊 SMART PREDICTION
# ---------------------------
def smart_prediction(prices, rsi, macd, signal, sentiment, weights):
    score = 0
    # RSI rules
    if rsi[-1] < 30: score += 2 * weights.get("rsi",1)
    elif rsi[-1] > 70: score -= 2 * weights.get("rsi",1)
    # MACD rules
    if macd[-1] > signal[-1]: score += 1 * weights.get("macd",1)
    else: score -= 1 * weights.get("macd",1)
    # Trend
    if prices[-1] > prices.mean(): score += 1 * weights.get("trend",1)
    else: score -= 1 * weights.get("trend",1)
    # Sentiment
    if sentiment > 0.2: score += 2 * weights.get("sentiment",1)
    elif sentiment < -0.2: score -= 2 * weights.get("sentiment",1)

    if score >= 3: return "BUY", score
    elif score <= -3: return "SELL", score
    else: return "HOLD", score

# ---------------------------
# ☁️ SAVE & LOAD HISTORY
# ---------------------------
def save_data(symbol, price, prediction):
    ref = db.reference(f"history/{symbol}")
    ref.push({
        "time": datetime.utcnow().isoformat(),
        "price": float(price),
        "prediction": prediction
    })

def load_history(symbol):
    ref = db.reference(f"history/{symbol}")
    data = ref.get() or {}
    if not data: return []
    return [v["price"] for v in data.values()]

def calculate_win_rate(symbol):
    ref = db.reference(f"history/{symbol}")
    data = ref.get() or {}
    if len(data) < 2: return 0
    values = list(data.values())
    wins, total = 0, 0
    for i in range(len(values)-1):
        current = values[i]
        next_price = values[i+1]["price"]
        if current["prediction"] == "BUY" and next_price > current["price"]: wins += 1
        elif current["prediction"] == "SELL" and next_price < current["price"]: wins += 1
        total += 1
    return round((wins/total)*100,2) if total>0 else 0

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 AI Crypto Trading Bot")

# Select multiple coins
symbols = st.multiselect("Select Coins to Analyze", ["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT"], default=["BTCUSDT"])

if st.button("Analyze"):
    weights = load_weights()  # global weights for all coins
    for symbol in symbols:
        st.subheader(f"Analysis for {symbol}")

        # Fetch prices with fallback
        prices, dates = get_price_data(symbol)
        if len(prices) == 0:
            st.warning(f"No price data for {symbol}, skipping.")
            continue

        # Append history
        history = load_history(symbol)
        if history:
            prices = np.concatenate([np.array(history), prices])

        # Calculate indicators
        rsi = calculate_rsi(prices)
        macd, signal = calculate_macd(prices)
        sentiment = get_sentiment()
        prediction, score = smart_prediction(prices, rsi, macd, signal, sentiment, weights)
        next_price = forecast_price(prices)

        # Save data
        save_data(symbol, prices[-1], prediction)
        save_weights(weights)  # update weights globally

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Price'))
        fig.add_trace(go.Scatter(
            x=[dates[-1], dates[-1] + timedelta(days=1)],
            y=[prices[-1], next_price],
            mode='lines',
            name='Prediction'
        ))
        st.plotly_chart(fig)

        # Metrics
        st.metric("Prediction", prediction)
        st.metric("Score", score)
        st.metric("Next Price", round(next_price,2))
        st.metric("Win Rate (%)", calculate_win_rate(symbol))
