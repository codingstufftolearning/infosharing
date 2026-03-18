import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import firebase_admin
from firebase_admin import credentials, db

# ---------------------------
# 🔐 FIREBASE INIT
# ---------------------------
if not firebase_admin._apps:
    firebase_dict = dict(st.secrets["firebase"])
    firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(firebase_dict)
    firebase_admin.initialize_app(cred, {
        "databaseURL": firebase_dict["databaseURL"]
    })

# ---------------------------
# ⏱️ FETCH HOURLY DATA
# ---------------------------
def get_hourly_data(symbol="BTCUSDT", hours=72):
    fsym = symbol.replace("USDT", "")
    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={hours-1}"
    data = requests.get(url).json()

    prices, dates = [], []
    if "Data" in data and "Data" in data["Data"]:
        raw = data["Data"]["Data"]
        prices = [x["close"] for x in raw]
        dates = [datetime.fromtimestamp(x["time"]) for x in raw]

    return np.array(prices), dates

# ---------------------------
# 📊 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[50], rsi.fillna(50)])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12).mean()
    exp2 = pd.Series(prices).ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    return macd.values, signal.values

# ---------------------------
# 📰 REAL SENTIMENT
# ---------------------------
def get_sentiment():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data = requests.get(url, timeout=5).json()

        if "Data" not in data:
            return 0

        analyzer = SentimentIntensityAnalyzer()
        scores = []

        for article in data["Data"][:8]:
            title = article.get("title", "")
            body = article.get("body", "")[:150]
            text = f"{title}. {body}"

            vader = analyzer.polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity

            scores.append((vader + blob) / 2)

        if not scores:
            return 0

        return max(-1, min(np.mean(scores), 1))

    except:
        return 0

# ---------------------------
# 🔮 ARIMA
# ---------------------------
def arima_forecast(prices, steps):
    try:
        model = ARIMA(prices, order=(2,1,2))
        model_fit = model.fit()
        return list(model_fit.forecast(steps=steps))
    except:
        return [prices[-1]] * steps

# ---------------------------
# 🔮 PROPHET
# ---------------------------
def prophet_forecast(dates, prices, steps):
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": prices})
    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=steps, freq='H')
    forecast = model.predict(future)

    return (
        forecast["yhat"].tail(steps).values,
        forecast["yhat_upper"].tail(steps).values,
        forecast["yhat_lower"].tail(steps).values
    )

# ---------------------------
# 🧠 HYBRID FORECAST
# ---------------------------
def hybrid_forecast(prices, dates, steps):
    arima_preds = arima_forecast(prices, steps)
    prophet_preds, upper, lower = prophet_forecast(dates, prices, steps)

    final = [(a + p) / 2 for a, p in zip(arima_preds, prophet_preds)]
    return final, upper, lower

# ---------------------------
# 🧠 LOAD/SAVE WEIGHTS
# ---------------------------
def load_weights():
    ref = db.reference("weights")
    data = ref.get()
    if not data:
        return {"rsi":1.0,"macd":1.0,"trend":1.0,"sentiment":1.0}
    return data

def save_weights(weights):
    db.reference("weights").set(weights)

# ---------------------------
# 🧠 SMART SIGNAL + CONTRIBUTIONS
# ---------------------------
def smart_signal(prices, rsi, macd, signal, sentiment, weights):
    score = 0
    contributions = {}

    # RSI
    if rsi[-1] < 30:
        val = 2 * weights["rsi"]
    elif rsi[-1] > 70:
        val = -2 * weights["rsi"]
    else:
        val = 0
    score += val
    contributions["rsi"] = val

    # MACD
    val = (1 if macd[-1] > signal[-1] else -1) * weights["macd"]
    score += val
    contributions["macd"] = val

    # Trend
    val = (1 if prices[-1] > np.mean(prices) else -1) * weights["trend"]
    score += val
    contributions["trend"] = val

    # Sentiment
    if sentiment > 0.2:
        val = 2 * weights["sentiment"]
    elif sentiment < -0.2:
        val = -2 * weights["sentiment"]
    else:
        val = 0
    score += val
    contributions["sentiment"] = val

    if score >= 3:
        return "BUY", score, contributions
    elif score <= -3:
        return "SELL", score, contributions
    return "HOLD", score, contributions

# ---------------------------
# 🔁 AUTO-LEARNING
# ---------------------------
def update_weights(symbol, weights):
    ref = db.reference(f"history/{symbol}")
    data = list((ref.get() or {}).values())

    if len(data) < 2:
        return weights

    prev = data[-2]
    curr = data[-1]

    if prev["signal"] == "HOLD":
        return weights

    actual_up = curr["price"] > prev["price"]
    lr = 0.03

    contributions = prev.get("contributions", {})

    for key in weights:
        contrib = contributions.get(key, 0)
        if contrib == 0:
            continue

        if (contrib > 0 and actual_up) or (contrib < 0 and not actual_up):
            weights[key] *= (1 + lr)
        else:
            weights[key] *= (1 - lr)

        weights[key] = max(0.2, min(weights[key], 3))

    return weights

# ---------------------------
# ☁️ SAVE + WIN RATE
# ---------------------------
def save_prediction(symbol, price, pred_price, signal, contributions):
    ref = db.reference(f"history/{symbol}")
    ref.push({
        "time": datetime.utcnow().isoformat(),
        "price": float(price),
        "predicted": float(pred_price),
        "signal": signal,
        "contributions": contributions
    })

def calculate_win_rate(symbol):
    ref = db.reference(f"history/{symbol}")
    data = list((ref.get() or {}).values())

    if len(data) < 2:
        return 0

    wins, total = 0, 0
    for i in range(len(data)-1):
        curr = data[i]
        nxt = data[i+1]

        if curr["signal"] == "BUY" and nxt["price"] > curr["price"]:
            wins += 1
        elif curr["signal"] == "SELL" and nxt["price"] < curr["price"]:
            wins += 1

        total += 1

    return round((wins/total)*100,2)

# ---------------------------
# 🎨 UI
# ---------------------------
st.title("🚀 Adaptive AI Crypto Bot")

symbol = st.selectbox("Coin", ["BTCUSDT","ETHUSDT","BNBUSDT"])

if st.button("Analyze"):

    now = datetime.now()
    start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    prices, dates = get_hourly_data(symbol)

    filtered = [(d,p) for d,p in zip(dates,prices) if d >= start_today]
    dates = [x[0] for x in filtered]
    prices = np.array([x[1] for x in filtered])

    if len(prices) < 10:
        st.warning("Not enough data")
        st.stop()

    weights = load_weights()

    rsi = calculate_rsi(prices)
    macd, signal_line = calculate_macd(prices)
    sentiment = get_sentiment()

    trade_signal, score, contributions = smart_signal(
        prices, rsi, macd, signal_line, sentiment, weights
    )

    remaining_hours = 24 - now.hour
    future_prices, upper, lower = hybrid_forecast(prices, dates, remaining_hours)

    future_dates = [dates[-1] + timedelta(hours=i+1) for i in range(remaining_hours)]

    save_prediction(symbol, prices[-1], future_prices[0], trade_signal, contributions)

    weights = update_weights(symbol, weights)
    save_weights(weights)

    # ---------------------------
    # 📊 PLOT
    # ---------------------------
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers', name='Actual'))

    fig.add_trace(go.Scatter(
        x=[dates[-1]] + future_dates,
        y=[prices[-1]] + future_prices,
        mode='lines+markers',
        name='Forecast',
        line=dict(dash='dash', color='red')
    ))

    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(upper) + list(lower[::-1]),
        fill='toself',
        name='Confidence',
        opacity=0.2
    ))

    st.plotly_chart(fig)

    # ---------------------------
    # 📊 METRICS
    # ---------------------------
    st.metric("Signal", trade_signal)
    st.metric("Score", round(score,2))
    st.metric("Win Rate (%)", calculate_win_rate(symbol))
    st.metric("Next Hour Price", round(future_prices[0],2))

    # ---------------------------
    # 🧠 DEBUG PANEL
    # ---------------------------
    with st.expander("🧠 Debug Info"):
        st.write("Weights:", weights)
        st.write("Sentiment:", sentiment)
        st.write("Contributions:", contributions)
