# app.py - Polished Adaptive AI Crypto Bot with clean forecast line
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
    try:
        firebase_dict = dict(st.secrets["firebase"])
        firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(firebase_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": firebase_dict["databaseURL"]})
    except Exception as e:
        st.error(f"Firebase Init Error: {type(e).__name__}: {e}")

# ---------------------------
# ⏱️ DATA FETCHING
# ---------------------------
def get_price_data(symbol="BTCUSDT", days=30):
    prices, dates, errors = [], [], []
    coin = symbol.replace("USDT", "").lower()
    try:
        url_cg = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
        data = requests.get(url_cg, timeout=5).json()
        if "prices" in data:
            prices = [x[1] for x in data["prices"]]
            dates = [datetime.fromtimestamp(x[0]/1000) for x in data["prices"]]
            return np.array(prices), dates, errors
        else:
            errors.append(f"CoinGecko returned unexpected data for {symbol}")
    except Exception as e:
        errors.append(f"CoinGecko fetch failed: {e}")
    return np.array(prices), dates, errors

def get_hourly_data(symbol="BTCUSDT", hours=72):
    fsym = symbol.replace("USDT", "")
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={hours-1}"
        data = requests.get(url, timeout=5).json()
        if "Data" in data and "Data" in data["Data"]:
            raw = data["Data"]["Data"]
            prices = [x["close"] for x in raw]
            dates = [datetime.fromtimestamp(x["time"]) for x in raw]
            return np.array(prices), dates
    except:
        pass
    return np.array([]), []

# ---------------------------
# 📊 HISTORICAL DATA HANDLER
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
    # Fetch fallback
    prices, dates, fetch_errors = get_price_data(symbol, days=30)
    errors += fetch_errors
    try:
        ref = db.reference(f"historical/{symbol}")
        for i in range(len(prices)):
            ref.push({"time": dates[i].isoformat(), "price": float(prices[i])})
    except:
        pass
    return np.array(prices), dates, errors

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
    return np.concatenate([[50], rsi])

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
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for article in data.get("Data", [])[:8]:
            text = f"{article.get('title','')}. {article.get('body','')[:150]}"
            vader = analyzer.polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity
            scores.append((vader + blob)/2)
        if scores:
            return max(-1, min(np.mean(scores), 1))
    except:
        return 0
    return 0

# ---------------------------
# 🔮 FORECASTING
# ---------------------------
def arima_forecast(prices, steps):
    try:
        model = ARIMA(prices, order=(2,1,2))
        model_fit = model.fit()
        return list(model_fit.forecast(steps=steps))
    except:
        return [prices[-1]] * steps

def prophet_forecast(dates, prices, steps):
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": prices})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq='H')
    forecast = model.predict(future)
    return (forecast["yhat"].tail(steps).values,
            forecast["yhat_upper"].tail(steps).values,
            forecast["yhat_lower"].tail(steps).values)

def hybrid_forecast(prices, dates, steps):
    arima_preds = arima_forecast(prices, steps)
    prophet_preds, upper, lower = prophet_forecast(dates, prices, steps)
    final = [(a + p)/2 for a,p in zip(arima_preds, prophet_preds)]
    return final, upper, lower

# ---------------------------
# 🧠 WEIGHTS & SMART SIGNAL
# ---------------------------
def load_weights():
    ref = db.reference("weights")
    data = ref.get()
    if not data: return {"rsi":1.0,"macd":1.0,"trend":1.0,"sentiment":1.0}
    return data

def save_weights(weights):
    db.reference("weights").set(weights)

def smart_signal(prices, rsi, macd, signal_line, sentiment, weights):
    score = 0
    contributions = {}
    val = 0
    if rsi[-1]<30: val=2*weights["rsi"]
    elif rsi[-1]>70: val=-2*weights["rsi"]
    score+=val; contributions["rsi"]=val
    val=(1 if macd[-1]>signal_line[-1] else -1)*weights["macd"]
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
# 🔁 AUTO-WEIGHT UPDATE
# ---------------------------
def update_weights(symbol, weights):
    ref = db.reference(f"history/{symbol}")
    data = list((ref.get() or {}).values())
    if len(data)<2: return weights
    prev = data[-2]; curr = data[-1]
    if prev.get("signal")=="HOLD": return weights
    actual_up = curr["price"]>prev["price"]
    lr = 0.03
    contributions = prev.get("contributions",{})
    for key in weights:
        contrib = contributions.get(key,0)
        if contrib==0: continue
        if (contrib>0 and actual_up) or (contrib<0 and not actual_up): weights[key]*=(1+lr)
        else: weights[key]*=(1-lr)
        weights[key]=max(0.2,min(weights[key],3))
    return weights

# ---------------------------
# ☁️ SAVE HISTORY + WIN RATE
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
    if len(data)<2: return 0
    wins,total = 0,0
    for i in range(len(data)-1):
        curr,nxt = data[i],data[i+1]
        if curr.get("signal")=="BUY" and nxt["price"]>curr["price"]: wins+=1
        elif curr.get("signal")=="SELL" and nxt["price"]<curr["price"]: wins+=1
        total+=1
    return round((wins/total)*100,2)

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 Adaptive AI Crypto Bot")

# 💱 USDT → IDR
try:
    url_idr = "https://api.exchangerate.host/latest?base=USD&symbols=IDR"
    idr_data = requests.get(url_idr, timeout=5).json()
    usdt_idr = round(idr_data["rates"]["IDR"],2)
except:
    usdt_idr = None
st.markdown(f"**USDT → IDR Rate:** {'N/A' if usdt_idr is None else usdt_idr}")

# Select coins and timeframe
coins_list = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XAIUSD","SOLUSDT"]
symbols = st.multiselect("Select Coins", coins_list, default=["BTCUSDT"])
timeframe = st.selectbox("Select timeframe", ["1 Day","3 Days","5 Days","1 Month"])

if st.button("Analyze"):
    results_summary=[]
    for symbol in symbols:
        # Historical + hourly
        hist_prices, hist_dates, errors = get_historical_data(symbol)
        hourly_prices, hourly_dates = get_hourly_data(symbol)
        prices, dates = np.concatenate([hist_prices,hourly_prices]), hist_dates+hourly_dates
        if len(prices)==0: continue

        # Indicators + sentiment
        weights = load_weights()
        rsi = calculate_rsi(prices)
        macd, signal_line = calculate_macd(prices)
        sentiment = get_sentiment()
        signal, score, contributions = smart_signal(prices,rsi,macd,signal_line,sentiment,weights)

        # Filter by timeframe
        now = datetime.now()
        cutoff_map={"1 Day":1,"3 Days":3,"5 Days":5,"1 Month":30}
        cutoff = now - timedelta(days=cutoff_map.get(timeframe,1))
        filtered = [(d,p) for d,p in zip(dates,prices) if d>=cutoff]
        if not filtered: continue
        filtered_dates=[x[0] for x in filtered]
        filtered_prices=np.array([x[1] for x in filtered])

        # Forecast
        remaining_hours=24-now.hour
        future_prices, upper, lower = hybrid_forecast(filtered_prices, filtered_dates, remaining_hours)
        future_dates=[filtered_dates[-1]+timedelta(hours=i+1) for i in range(remaining_hours)]

        # Save
        save_prediction(symbol, filtered_prices[-1], future_prices[0], signal, contributions)
        weights = update_weights(symbol, weights)
        save_weights(weights)

        # ---------------------------
        # Charts - clean separation
        # ---------------------------
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=filtered_dates, y=filtered_prices, mode='lines+markers', name='Actual', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=[filtered_dates[-1]]+future_dates, y=[filtered_prices[-1]]+future_prices, mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')))
        fig.add_trace(go.Scatter(x=future_dates+future_dates[::-1], y=list(upper)+list(lower[::-1]), fill='toself', name='Confidence', opacity=0.2))

        st.subheader(f"Analysis for {symbol}")
        st.plotly_chart(fig)
        st.metric("Signal", signal)
        st.metric("Score", round(score,2))
        st.metric("Win Rate (%)", calculate_win_rate(symbol))
        st.metric("Next Hour Price", round(future_prices[0],2))
        pct_change=((filtered_prices[-1]-filtered_prices[0])/filtered_prices[0])*100
        st.metric("Day Change (%)", round(pct_change,2))

        results_summary.append({"Coin":symbol,"Signal":signal,"Score":round(score,2),"Next Price":round(future_prices[0],2),"Day Change (%)":round(pct_change,2)})

    if results_summary:
        st.subheader("📋 Summary Table")
        st.dataframe(pd.DataFrame(results_summary))
