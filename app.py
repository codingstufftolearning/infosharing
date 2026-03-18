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
    firebase_admin.initialize_app(cred, {"databaseURL": firebase_dict["databaseURL"]})

# ---------------------------
# ⏱️ DATA FETCH
# ---------------------------
def get_historical_data(symbol="BTCUSDT"):
    prices, dates, errors = [], [], []
    try:
        ref = db.reference(f"historical/{symbol}")
        data = ref.get() or {}
        if data:
            sorted_data = sorted(data.items(), key=lambda x: x[1]['time'])
            prices = [v["price"] for _,v in sorted_data]
            dates = [datetime.fromisoformat(v["time"]) for _,v in sorted_data]
            return np.array(prices), dates, errors
    except Exception as e:
        errors.append(f"Firebase read failed: {e}")
    
    # fallback API
    fsym = symbol.replace("USDT","")
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit=72"
        data = requests.get(url, timeout=5).json()
        raw = data["Data"]["Data"]
        prices = [x["close"] for x in raw]
        dates = [datetime.fromtimestamp(x["time"]) for x in raw]
    except Exception as e:
        errors.append(f"API fetch failed: {e}")

    # Save to Firebase
    try:
        ref = db.reference(f"historical/{symbol}")
        for i in range(len(prices)):
            ref.push({"time": dates[i].isoformat(),"price":float(prices[i])})
    except: pass

    return np.array(prices), dates, errors

def get_live_price(symbol="BTCUSDT"):
    fsym = symbol.replace("USDT","")
    try:
        url = f"https://min-api.cryptocompare.com/data/price?fsym={fsym}&tsyms=USD"
        data = requests.get(url, timeout=5).json()
        return data.get("USD", None)
    except:
        return None

# ---------------------------
# 📊 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100/(1+rs))
    return np.concatenate([[50], rsi.fillna(50)])

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
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data = requests.get(url, timeout=5).json()
        if "Data" not in data: return 0
        analyzer = SentimentIntensityAnalyzer()
        scores=[]
        for article in data["Data"][:8]:
            title = article.get("title","")
            body = article.get("body","")[:150]
            text = f"{title}. {body}"
            vader = analyzer.polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity
            scores.append((vader+blob)/2)
        return max(-1,min(np.mean(scores) if scores else 0,1))
    except: return 0

# ---------------------------
# 🔮 FORECAST
# ---------------------------
def arima_forecast(prices, steps):
    try:
        model = ARIMA(prices, order=(2,1,2))
        model_fit = model.fit()
        return list(model_fit.forecast(steps=steps))
    except: return [prices[-1]]*steps

def prophet_forecast(dates, prices, steps):
    df = pd.DataFrame({"ds": pd.to_datetime(dates),"y":prices})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq='H')
    forecast = model.predict(future)
    return forecast["yhat"].tail(steps).values, forecast["yhat_upper"].tail(steps).values, forecast["yhat_lower"].tail(steps).values

def hybrid_forecast(prices, dates, steps):
    arima_preds = arima_forecast(prices, steps)
    prophet_preds, upper, lower = prophet_forecast(dates, prices, steps)
    final = [(a+p)/2 for a,p in zip(arima_preds,prophet_preds)]
    return final, upper, lower

# ---------------------------
# 🧠 WEIGHTS + SIGNAL
# ---------------------------
def load_weights():
    ref = db.reference("weights")
    data = ref.get()
    if not data: return {"rsi":1.0,"macd":1.0,"trend":1.0,"sentiment":1.0}
    return data

def save_weights(weights):
    db.reference("weights").set(weights)

def smart_signal(prices,rsi,macd,signal,sentiment,weights):
    score=0; contributions={}
    val=0
    if rsi[-1]<30: val=2*weights["rsi"]
    elif rsi[-1]>70: val=-2*weights["rsi"]
    score+=val; contributions["rsi"]=val
    val=(1 if macd[-1]>signal[-1] else -1)*weights["macd"]
    score+=val; contributions["macd"]=val
    val=(1 if prices[-1]>np.mean(prices) else -1)*weights["trend"]
    score+=val; contributions["trend"]=val
    val=0
    if sentiment>0.2: val=2*weights["sentiment"]
    elif sentiment<-0.2: val=-2*weights["sentiment"]
    score+=val; contributions["sentiment"]=val
    if score>=3: return "BUY",score,contributions
    elif score<=-3: return "SELL",score,contributions
    return "HOLD",score,contributions

# ---------------------------
# 🔁 AUTO LEARNING
# ---------------------------
def update_weights(symbol, weights):
    ref=db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    if len(data)<2: return weights
    prev=data[-2]; curr=data[-1]
    if prev["signal"]=="HOLD": return weights
    actual_up = curr["price"]>prev["price"]
    lr=0.03
    contributions = prev.get("contributions",{})
    for key in weights:
        contrib=contributions.get(key,0)
        if contrib==0: continue
        if (contrib>0 and actual_up) or (contrib<0 and not actual_up):
            weights[key]*=(1+lr)
        else: weights[key]*=(1-lr)
        weights[key]=max(0.2,min(weights[key],3))
    return weights

# ---------------------------
# ☁️ SAVE + WIN RATE
# ---------------------------
def save_prediction(symbol, price, pred_price, signal, contributions):
    ref=db.reference(f"history/{symbol}")
    ref.push({"time":datetime.utcnow().isoformat(),
              "price":float(price),
              "predicted":float(pred_price),
              "signal":signal,
              "contributions":contributions})

def calculate_win_rate(symbol):
    ref=db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    if len(data)<2: return 0
    wins,total=0,0
    for i in range(len(data)-1):
        curr=data[i]; nxt=data[i+1]
        if curr["signal"]=="BUY" and nxt["price"]>curr["price"]: wins+=1
        elif curr["signal"]=="SELL" and nxt["price"]<curr["price"]: wins+=1
        total+=1
    return round((wins/total)*100,2)

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 Adaptive AI Crypto Bot")
symbol = st.selectbox("Select Coin", ["BTCUSDT","ETHUSDT","BNBUSDT"])
timeframe = st.selectbox("Select Timeframe", ["1 Day","3 Days","5 Days","1 Month","Today Real-Time"])

if st.button("Analyze"):

    now=datetime.utcnow()
    prices, dates, errors = get_historical_data(symbol)
    filtered_dates = dates.copy()
    filtered_prices = prices.copy()

    live_price = None
    future_prices,future_dates=[],[]
    if timeframe=="Today Real-Time":
        live_price = get_live_price(symbol)
        if live_price:
            filtered_dates.append(datetime.utcnow())
            filtered_prices=np.append(filtered_prices, live_price)

        remaining_hours = 24 - filtered_dates[-1].hour
        if remaining_hours>0:
            future_prices, _, _ = hybrid_forecast(filtered_prices, filtered_dates, remaining_hours)
            future_dates = [filtered_dates[-1]+timedelta(hours=i+1) for i in range(remaining_hours)]

    if len(filtered_prices)<10:
        st.warning("Not enough data")
        st.stop()

    weights = load_weights()
    rsi = calculate_rsi(filtered_prices)
    macd, signal_line = calculate_macd(filtered_prices)
    sentiment = get_sentiment()
    trade_signal, score, contributions = smart_signal(filtered_prices,rsi,macd,signal_line,sentiment,weights)

    save_prediction(symbol, filtered_prices[-1], future_prices[0] if future_prices else filtered_prices[-1], trade_signal, contributions)
    weights = update_weights(symbol, weights)
    save_weights(weights)

    # ---------------------------
    # 📊 PLOT (highlight today + forecast)
    # ---------------------------
    fig = go.Figure()
    # Historical full
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines', name='Historical', line=dict(color='gray')))
    # Today + live
    if timeframe=="Today Real-Time":
        today_mask = [d.date()==datetime.utcnow().date() for d in dates]
        today_dates = [d for d,m in zip(dates,today_mask) if m]
        today_prices = [p for p,m in zip(prices,today_mask) if m]
        if live_price:
            today_dates.append(datetime.utcnow())
            today_prices.append(live_price)
        fig.add_trace(go.Scatter(x=today_dates,y=today_prices,mode='lines+markers',name='Today + Live',line=dict(color='blue',width=2)))
        # Forecast
        if future_prices:
            fig.add_trace(go.Scatter(x=[today_dates[-1]]+future_dates,y=[today_prices[-1]]+list(future_prices),mode='lines+markers',name='Forecast',line=dict(dash='dash',color='red',width=2)))
    st.plotly_chart(fig)

    # ---------------------------
    # 📊 METRICS
    # ---------------------------
    st.metric("Signal", trade_signal)
    st.metric("Score", round(score,2))
    st.metric("Win Rate (%)", calculate_win_rate(symbol))
    if future_prices: st.metric("Next Hour Price", round(future_prices[0],2))

    # ---------------------------
    # 🧠 DEBUG
    # ---------------------------
    with st.expander("🧠 Debug Info"):
        st.write("Weights:", weights)
        st.write("Sentiment:", sentiment)
        st.write("Contributions:", contributions)
