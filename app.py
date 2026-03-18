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
    firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n","\n")
    cred = credentials.Certificate(firebase_dict)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {"databaseURL":firebase_dict["databaseURL"]})
    firebase_admin_initialized = True
except Exception as e:
    st.error(f"Firebase init failed: {type(e).__name__}: {e}")

# ---------------------------
# 📊 FETCH PRICE DATA (MULTI-SOURCE + FALLBACK)
# ---------------------------
def get_price_data(symbol="BTCUSDT", limit=30):
    """
    Fetch price data from multiple sources.
    Returns np.array(prices), list(dates)
    """
    prices, dates = [], []

    # 1️⃣ CoinGecko
    try:
        coin = symbol.replace("USDT","").lower()
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={limit}"
        res = requests.get(url, timeout=10).json()
        if "prices" in res:
            prices = [x[1] for x in res["prices"]]
            dates = [datetime.fromtimestamp(x[0]/1000) for x in res["prices"]]
            if prices: return np.array(prices), dates
        else:
            st.warning(f"CoinGecko returned unexpected data for {symbol}: {res}")
    except Exception as e:
        st.warning(f"CoinGecko fetch failed for {symbol}: {e}")

    # 2️⃣ CryptoCompare (Fallback)
    try:
        coin_short = symbol.replace("USDT","")
        url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={coin_short}&tsym=USD&limit={limit-1}"
        res = requests.get(url, timeout=10).json()
        if "Data" in res and "Data" in res["Data"]:
            data = res["Data"]["Data"]
            prices = [x["close"] for x in data]
            dates = [datetime.fromtimestamp(x["time"]) for x in data]
            if prices: return np.array(prices), dates
        else:
            st.warning(f"CryptoCompare returned unexpected data for {symbol}: {res}")
    except Exception as e:
        st.warning(f"CryptoCompare fetch failed for {symbol}: {e}")

    # 3️⃣ Last fallback: empty arrays
    st.error(f"Failed to fetch price data for {symbol}.")
    return np.array([]), []

# ---------------------------
# 📈 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    if len(prices)<period: return np.zeros(len(prices))
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain/(avg_loss + 1e-9)
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
    scores=[]
    for t in texts:
        vader = analyzer.polarity_scores(t)["compound"]
        blob = TextBlob(t).sentiment.polarity
        scores.append((vader+blob)/2)
    return np.mean(scores)

# ---------------------------
# 🔮 FORECAST
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
    if firebase_admin_initialized: db.reference("weights").set(weights)

# ---------------------------
# 📊 SMART PREDICTION
# ---------------------------
def smart_prediction(prices,rsi,macd,signal,sentiment,weights):
    score=0
    if rsi[-1]<30: score+=2*weights.get("rsi",1)
    elif rsi[-1]>70: score-=2*weights.get("rsi",1)
    if macd[-1]>signal[-1]: score+=1*weights.get("macd",1)
    else: score-=1*weights.get("macd",1)
    if prices[-1]>prices.mean(): score+=1*weights.get("trend",1)
    else: score-=1*weights.get("trend",1)
    if sentiment>0.2: score+=2*weights.get("sentiment",1)
    elif sentiment<-0.2: score-=2*weights.get("sentiment",1)
    if score>=3: return "BUY",score
    elif score<=-3: return "SELL",score
    else: return "HOLD",score

# ---------------------------
# ☁️ SAVE & LOAD HISTORY
# ---------------------------
def save_data(symbol, price, prediction):
    if not firebase_admin_initialized: return
    ref = db.reference(f"history/{symbol}")
    ref.push({"time":datetime.utcnow().isoformat(),"price":float(price),"prediction":prediction})

def load_history(symbol):
    if not firebase_admin_initialized: return []
    ref = db.reference(f"history/{symbol}")
    data = ref.get() or {}
    return [v["price"] for v in data.values()]

def calculate_win_rate(symbol):
    if not firebase_admin_initialized: return 0
    ref = db.reference(f"history/{symbol}")
    data = ref.get() or {}
    if len(data)<2: return 0
    values=list(data.values())
    wins,total=0,0
    for i in range(len(values)-1):
        current=values[i]
        next_price=values[i+1]["price"]
        if current["prediction"]=="BUY" and next_price>current["price"]: wins+=1
        elif current["prediction"]=="SELL" and next_price<current["price"]: wins+=1
        total+=1
    return round((wins/total)*100,2) if total>0 else 0

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 AI Crypto Trading Bot")

symbols = st.multiselect("Select Coins", ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT"], default=["BTCUSDT"])
chart_type = st.selectbox("Chart Type", ["Line","Candlestick"])

if st.button("Analyze"):
    weights = load_weights()
    for symbol in symbols:
        st.subheader(f"Analysis for {symbol}")
        prices, dates = get_price_data(symbol, limit=30)
        if len(prices)==0:
            st.warning(f"No data for {symbol}, skipping.")
            continue

        # Append history if any
        history = load_history(symbol)
        if history: prices = np.concatenate([np.array(history), prices])

        # Indicators
        rsi = calculate_rsi(prices)
        macd, signal = calculate_macd(prices)
        sentiment = get_sentiment()
        prediction,score = smart_prediction(prices,rsi,macd,signal,sentiment,weights)
        next_price = forecast_price(prices)

        # Save
        save_data(symbol, prices[-1], prediction)
        save_weights(weights)

        # Chart
        fig=go.Figure()
        if chart_type=="Line":
            fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name="Price"))
            fig.add_trace(go.Scatter(x=[dates[-1], dates[-1]+timedelta(days=1)],
                                     y=[prices[-1], next_price],
                                     mode="lines", name="Forecast"))
        else:
            ohlc=[]
            for p in prices:
                ohlc.append([p*0.995,p*1.005,p*0.995,p*1.005])
            fig.add_trace(go.Candlestick(x=dates,
                                         open=[x[0] for x in ohlc],
                                         high=[x[1] for x in ohlc],
                                         low=[x[2] for x in ohlc],
                                         close=[x[3] for x in ohlc],
                                         name="Price"))
        st.plotly_chart(fig)

        st.metric("Prediction",prediction)
        st.metric("Score",score)
        st.metric("Next Price",round(next_price,2))
        st.metric("Win Rate (%)",calculate_win_rate(symbol))
