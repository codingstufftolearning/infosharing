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
if not firebase_admin._apps:
    try:
        firebase_dict = dict(st.secrets["firebase"])
        firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n","\n")
        cred = credentials.Certificate(firebase_dict)
        firebase_admin.initialize_app(cred, {
            "databaseURL": firebase_dict["databaseURL"]
        })
    except Exception as e:
        with st.expander("Firebase Initialization Error"):
            st.error(f"{type(e).__name__}: {e}")
            import traceback
            st.text(traceback.format_exc())

# ---------------------------
# 📊 FETCH PRICE DATA (MULTI-SOURCE)
# ---------------------------
@st.cache_data(ttl=300)
def get_daily_historical(symbol="BTCUSDT", days=30):
    """
    Fetch daily historical price for last `days` days.
    Returns prices, dates, errors.
    """
    prices, dates, errors = [], [], []
    coin = symbol.replace("USDT","").lower()
    
    # CoinGecko daily
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}&interval=daily"
        data = requests.get(url, timeout=10).json()
        if "prices" in data:
            prices = [x[1] for x in data["prices"]]
            dates = [datetime.fromtimestamp(x[0]/1000) for x in data["prices"]]
            return np.array(prices), dates, errors
        else:
            errors.append(f"CoinGecko daily unexpected for {symbol}: {data}")
    except Exception as e:
        errors.append(f"CoinGecko daily fetch failed for {symbol}: {e}")
    
    # CryptoCompare daily fallback
    try:
        url_cc = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol[:3]}&tsym=USD&limit={days-1}"
        data_cc = requests.get(url_cc, timeout=10).json()
        if "Data" in data_cc and "Data" in data_cc["Data"]:
            prices = [x["close"] for x in data_cc["Data"]["Data"]]
            dates = [datetime.fromtimestamp(x["time"]) for x in data_cc["Data"]["Data"]]
            return np.array(prices), dates, errors
        else:
            errors.append(f"CryptoCompare daily unexpected for {symbol}: {data_cc}")
    except Exception as e:
        errors.append(f"CryptoCompare daily fetch failed for {symbol}: {e}")
    
    errors.append(f"Failed to fetch daily historical for {symbol}.")
    return np.array([]), [], errors

@st.cache_data(ttl=300)
def get_hourly_data(symbol="BTCUSDT", hours=24):
    """
    Fetch hourly price for last `hours` hours.
    Returns prices, dates, errors.
    """
    prices, dates, errors = [], [], []
    coin = symbol.replace("USDT","").lower()
    
    # CoinGecko hourly
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=1&interval=hourly"
        data = requests.get(url, timeout=10).json()
        if "prices" in data:
            prices = [x[1] for x in data["prices"][-hours:]]
            dates = [datetime.fromtimestamp(x[0]/1000) for x in data["prices"][-hours:]]
            return np.array(prices), dates, errors
        else:
            errors.append(f"CoinGecko hourly unexpected for {symbol}: {data}")
    except Exception as e:
        errors.append(f"CoinGecko hourly fetch failed for {symbol}: {e}")
    
    # CryptoCompare hourly fallback
    try:
        url_cc = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={symbol[:3]}&tsym=USD&limit={hours-1}"
        data_cc = requests.get(url_cc, timeout=10).json()
        if "Data" in data_cc and "Data" in data_cc["Data"]:
            prices = [x["close"] for x in data_cc["Data"]["Data"]]
            dates = [datetime.fromtimestamp(x["time"]) for x in data_cc["Data"]["Data"]]
            return np.array(prices), dates, errors
        else:
            errors.append(f"CryptoCompare hourly unexpected for {symbol}: {data_cc}")
    except Exception as e:
        errors.append(f"CryptoCompare hourly fetch failed for {symbol}: {e}")
    
    errors.append(f"Failed to fetch hourly data for {symbol}.")
    return np.array([]), [], errors

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
        scores.append((vader+blob)/2)
    return np.mean(scores)

# ---------------------------
# 🔮 FORECAST
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
# 🧠 WEIGHTS
# ---------------------------
def load_weights():
    try:
        ref = db.reference("weights")
        data = ref.get() or {}
        if not data: return {"rsi":1,"macd":1,"trend":1,"sentiment":1}
        return data
    except:
        return {"rsi":1,"macd":1,"trend":1,"sentiment":1}

def save_weights(weights):
    try: db.reference("weights").set(weights)
    except: pass

# ---------------------------
# 📊 SMART PREDICTION
# ---------------------------
def smart_prediction(prices, rsi, macd, signal, sentiment, weights):
    score=0
    if rsi[-1]<30: score+=2*weights.get("rsi",1)
    elif rsi[-1]>70: score-=2*weights.get("rsi",1)
    if macd[-1]>signal[-1]: score+=1*weights.get("macd",1)
    else: score-=1*weights.get("macd",1)
    if prices[-1]>prices.mean(): score+=1*weights.get("trend",1)
    else: score-=1*weights.get("trend",1)
    if sentiment>0.2: score+=2*weights.get("sentiment",1)
    elif sentiment<-0.2: score-=2*weights.get("sentiment",1)
    if score>=3: return "BUY", score
    elif score<=-3: return "SELL", score
    else: return "HOLD", score

# ---------------------------
# ☁️ SAVE/LOAD HISTORY
# ---------------------------
def save_data(symbol, price, prediction):
    try:
        ref = db.reference(f"history/{symbol}")
        ref.push({
            "time": datetime.utcnow().isoformat(),
            "price": float(price),
            "prediction": prediction
        })
    except: pass

def load_history(symbol):
    try:
        ref = db.reference(f"history/{symbol}")
        data = ref.get() or {}
        if not data: return []
        return [v["price"] for v in data.values()]
    except: return []

def calculate_win_rate(symbol):
    try:
        ref = db.reference(f"history/{symbol}")
        data = ref.get() or {}
        if len(data)<2: return 0
        values=list(data.values())
        wins,total=0,0
        for i in range(len(values)-1):
            current = values[i]
            next_price = values[i+1]["price"]
            if current["prediction"]=="BUY" and next_price>current["price"]: wins+=1
            elif current["prediction"]=="SELL" and next_price<current["price"]: wins+=1
            total+=1
        return round((wins/total)*100,2) if total>0 else 0
    except:
        return 0

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 AI Crypto Trading Bot")

# Select coins
symbols = st.multiselect("Select Coins to Analyze", ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XAI"], default=["BTCUSDT"])

# Persist results
if "results" not in st.session_state: st.session_state.results = {}
if "errors" not in st.session_state: st.session_state.errors = []

# Select timeframe
timeframe = st.selectbox("Chart Timeframe", ["Hourly (24h)", "Daily (30d)"])

if st.button("Analyze"):
    st.session_state.results={}
    st.session_state.errors=[]
    weights=load_weights()
    for symbol in symbols:
        # Fetch daily historical
        daily_prices,daily_dates,errors_daily = get_daily_historical(symbol,30)
        st.session_state.errors+=errors_daily
        if len(daily_prices)==0:
            st.warning(f"No daily historical for {symbol}, skipping.")
            continue

        # Save daily history in DB if not exists
        history=load_history(symbol)
        if not history:
            for i,p in enumerate(daily_prices):
                save_data(symbol,p,"HOLD")  # default HOLD

        # Fetch hourly last 24h
        hourly_prices,hourly_dates,errors_hourly = get_hourly_data(symbol,24)
        st.session_state.errors+=errors_hourly

        # Combine hourly with daily for chart (hourly last day)
        combined_prices = np.concatenate([daily_prices[:-1], hourly_prices])
        combined_dates = daily_dates[:-1]+hourly_dates

        # Indicators & prediction
        rsi = calculate_rsi(combined_prices)
        macd,signal = calculate_macd(combined_prices)
        sentiment = get_sentiment()
        prediction, score = smart_prediction(combined_prices, rsi, macd, signal, sentiment, weights)
        next_price = forecast_price(combined_prices)

        # Save latest price
        save_data(symbol, combined_prices[-1], prediction)
        save_weights(weights)

        st.session_state.results[symbol]={
            "prices": combined_prices,
            "dates": combined_dates,
            "prediction": prediction,
            "score": score,
            "next_price": next_price
        }

# Display results
for symbol,result in st.session_state.results.items():
    st.subheader(f"Analysis for {symbol}")
    prices=result["prices"]
    dates=result["dates"]
    next_price=result["next_price"]

    extended_dates=dates+[dates[-1]+timedelta(hours=1)]
    extended_prices=np.append(prices,next_price)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=dates,y=prices,mode='lines+markers',name='Price'))
    fig.add_trace(go.Scatter(
        x=[dates[-1],dates[-1]+timedelta(hours=1)],
        y=[prices[-1],next_price],
        mode='lines+markers',
        name='Estimated Price',
        line=dict(dash='dash',color='red')
    ))
    fig.update_yaxes(range=[min(extended_prices)*0.98,max(extended_prices)*1.02])
    st.plotly_chart(fig)

    st.metric("Prediction",result["prediction"])
    st.metric("Score",result["score"])
    st.metric("Next Price",round(next_price,2))
    st.metric("Win Rate (%)",calculate_win_rate(symbol))

# Show errors in expander
if st.session_state.errors:
    with st.expander("Data Fetch Errors"):
        for err in st.session_state.errors:
            st.warning(err)
