# app.py - Multi-Coin Adaptive AI Crypto Bot with Statistics
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
        firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(firebase_dict)
        firebase_admin.initialize_app(cred, {
            "databaseURL": firebase_dict["databaseURL"]
        })
    except Exception as e:
        with st.expander("Firebase Initialization Error"):
            st.error(f"{type(e).__name__}: {e}")

# ---------------------------
# 📊 FETCH PRICE DATA
# ---------------------------
def get_price_data(symbol="BTCUSDT", days=30):
    prices, dates, errors = [], [], []
    coin = symbol.replace("USDT","").lower()

    try:
        url_cg = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
        data_cg = requests.get(url_cg, timeout=5).json()
        if "prices" in data_cg:
            prices = [x[1] for x in data_cg["prices"]]
            dates = [datetime.fromtimestamp(x[0]/1000) for x in data_cg["prices"]]
            return np.array(prices), dates, errors
    except: pass
    try:
        url_cc = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol[:3]}&tsym=USD&limit={days-1}"
        data_cc = requests.get(url_cc, timeout=5).json()
        if "Data" in data_cc and "Data" in data_cc["Data"]:
            prices = [x["close"] for x in data_cc["Data"]["Data"]]
            dates = [datetime.fromtimestamp(x["time"]) for x in data_cc["Data"]["Data"]]
            return np.array(prices), dates, errors
    except: pass
    errors.append(f"Failed to fetch {symbol}")
    return np.array([]), [], errors

# ---------------------------
# ⏱️ HOURLY DATA FOR TODAY
# ---------------------------
def get_hourly_data(symbol="BTCUSDT", hours=72):
    fsym = symbol.replace("USDT", "")
    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={hours-1}"
    try:
        data = requests.get(url, timeout=5).json()
        prices, dates = [], []
        if "Data" in data and "Data" in data["Data"]:
            raw = data["Data"]["Data"]
            prices = [x["close"] for x in raw]
            dates = [datetime.fromtimestamp(x["time"]) for x in raw]
        return np.array(prices), dates
    except:
        return np.array([]), []

# ---------------------------
# 📈 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    avg_gain = pd.Series(gain).rolling(period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[50], rsi.fillna(50)])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.values, signal.values

# ---------------------------
# 📰 SENTIMENT
# ---------------------------
def get_sentiment():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data = requests.get(url, timeout=5).json()
        if "Data" not in data: return 0
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for article in data["Data"][:8]:
            text = f"{article.get('title','')}. {article.get('body','')[:150]}"
            vader = analyzer.polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity
            scores.append((vader + blob)/2)
        return max(-1, min(np.mean(scores), 1)) if scores else 0
    except:
        return 0

# ---------------------------
# 🔮 FORECAST
# ---------------------------
def arima_forecast(prices, steps):
    try:
        model = ARIMA(prices, order=(2,1,2))
        model_fit = model.fit()
        return list(model_fit.forecast(steps=steps))
    except:
        return [prices[-1]]*steps

def hybrid_forecast(prices, dates, steps):
    return arima_forecast(prices, steps)

# ---------------------------
# 🧠 SMART SIGNAL & WEIGHTS
# ---------------------------
def load_weights():
    try:
        ref = db.reference("weights")
        data = ref.get()
        return data if data else {"rsi":1.0,"macd":1.0,"trend":1.0,"sentiment":1.0}
    except: return {"rsi":1.0,"macd":1.0,"trend":1.0,"sentiment":1.0}

def save_weights(weights):
    try: db.reference("weights").set(weights)
    except: pass

def smart_signal(prices, rsi, macd, signal, sentiment, weights):
    score=0
    contributions={}
    val=0
    if rsi[-1]<30: val=2*weights["rsi"]
    elif rsi[-1]>70: val=-2*weights["rsi"]
    score+=val
    contributions["rsi"]=val
    val=(1 if macd[-1]>signal[-1] else -1)*weights["macd"]
    score+=val
    contributions["macd"]=val
    val=(1 if prices[-1]>np.mean(prices) else -1)*weights["trend"]
    score+=val
    contributions["trend"]=val
    val=0
    if sentiment>0.2: val=2*weights["sentiment"]
    elif sentiment<-0.2: val=-2*weights["sentiment"]
    score+=val
    contributions["sentiment"]=val
    if score>=3: return "BUY",score,contributions
    elif score<=-3: return "SELL",score,contributions
    return "HOLD",score,contributions

def update_weights(symbol, weights):
    try:
        ref=db.reference(f"history/{symbol}")
        data=list((ref.get() or {}).values())
        if len(data)<2: return weights
        prev=data[-2]; curr=data[-1]
        signal=prev.get("signal","HOLD")
        if signal=="HOLD": return weights
        actual_up=curr["price"]>prev["price"]
        lr=0.03
        contributions=prev.get("contributions",{})
        for key in weights:
            contrib=contributions.get(key,0)
            if contrib==0: continue
            if (contrib>0 and actual_up) or (contrib<0 and not actual_up): weights[key]*=(1+lr)
            else: weights[key]*=(1-lr)
            weights[key]=max(0.2,min(weights[key],3))
        return weights
    except: return weights

def save_prediction(symbol, price, pred_price, signal, contributions):
    try:
        ref=db.reference(f"history/{symbol}")
        ref.push({
            "time": datetime.utcnow().isoformat(),
            "price": float(price),
            "predicted": float(pred_price),
            "signal": signal,
            "contributions": contributions
        })
    except: pass

def calculate_win_rate(symbol):
    try:
        ref=db.reference(f"history/{symbol}")
        data=list((ref.get() or {}).values())
        if len(data)<2: return 0
        wins,total=0,0
        for i in range(len(data)-1):
            curr,nxt=data[i],data[i+1]
            if curr.get("signal","HOLD")=="BUY" and nxt["price"]>curr["price"]: wins+=1
            elif curr.get("signal","HOLD")=="SELL" and nxt["price"]<curr["price"]: wins+=1
            total+=1
        return round((wins/total)*100,2) if total>0 else 0
    except: return 0

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 Adaptive Multi-Coin AI Crypto Bot")

coins_list = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XAIUSD","SOLUSDT"]
symbols = st.multiselect("Select Coins", coins_list, default=["BTCUSDT"])

timeframe = st.selectbox("Select Timeframe", ["1 Day","3 Days","5 Days","1 Month","Today Real-Time"])

if st.button("Analyze"):
    for symbol in symbols:
        # ----- Historical Prices -----
        prices, dates, errors = get_price_data(symbol, days=30)
        if len(prices)==0:
            st.warning(f"No data for {symbol}")
            continue

        # ----- Live Today Prices -----
        if timeframe=="Today Real-Time":
            hourly_prices, hourly_dates = get_hourly_data(symbol)
            start_today=datetime.utcnow().replace(hour=0,minute=0,second=0,microsecond=0)
            filtered=[(d,p) for d,p in zip(hourly_dates,hourly_prices) if d>=start_today]
            if filtered:
                hourly_dates=[x[0] for x in filtered]
                hourly_prices=np.array([x[1] for x in filtered])
                dates=hourly_dates; prices=hourly_prices

        # ----- Indicators & Sentiment -----
        rsi=calculate_rsi(prices)
        macd,signal_line=calculate_macd(prices)
        sentiment=get_sentiment()
        weights=load_weights()
        signal, score, contributions=smart_signal(prices,rsi,macd,signal_line,sentiment,weights)

        # ----- Forecast -----
        now=datetime.utcnow()
        remaining_hours=24-now.hour
        future_prices=[]
        future_dates=[]
        if remaining_hours>0:
            future_prices=hybrid_forecast(prices, dates, remaining_hours)
            future_dates=[dates[-1]+timedelta(hours=i+1) for i in range(remaining_hours)]

        # ----- Save Prediction & Update Weights -----
        if future_prices:
            save_prediction(symbol, prices[-1], future_prices[0], signal, contributions)
        weights=update_weights(symbol, weights)
        save_weights(weights)

        # ----- Statistics -----
        pct_change_today = round((prices[-1]-prices[0])/prices[0]*100,2) if len(prices)>1 else 0
        trend_today = "Rising" if pct_change_today>=0 else "Dropping"
        forecast_pct = round((future_prices[-1]-prices[-1])/prices[-1]*100,2) if future_prices else 0

        # ----- Plot -----
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=dates,y=prices,mode='lines+markers',name='Actual'))
        if future_prices:
            fig.add_trace(go.Scatter(
                x=[dates[-1]]+future_dates,
                y=[prices[-1]]+future_prices,
                mode='lines+markers',
                name='Forecast',
                line=dict(dash='dash', color='red')
            ))
        st.subheader(f"{symbol} Analysis")
        st.plotly_chart(fig)

        # ----- Metrics -----
        st.metric("Signal", signal)
        st.metric("Score", round(score,2))
        st.metric("Win Rate (%)", calculate_win_rate(symbol))
        st.metric("Today's Change (%)", pct_change_today)
        st.metric("Trend Today", trend_today)
        if future_prices:
            st.metric("Forecast Change (%)", forecast_pct)

        # ----- Debug -----
        with st.expander(f"Debug Info - {symbol}"):
            st.write("Weights:", weights)
            st.write("Sentiment:", sentiment)
            st.write("Contributions:", contributions)
