# app.py - Full Polished Version
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
        firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n","\n")
        cred = credentials.Certificate(firebase_dict)
        firebase_admin.initialize_app(cred, {
            "databaseURL": firebase_dict["databaseURL"]
        })
    except Exception as e:
        st.error(f"Firebase Initialization Error: {e}")

# ---------------------------
# ⏱️ FETCH DATA
# ---------------------------
def get_price_data(symbol="BTCUSDT", days=30):
    prices, dates, errors = [], [], []
    coin = symbol.replace("USDT","").lower()

    # CoinGecko
    try:
        url_cg = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
        data_cg = requests.get(url_cg, timeout=5).json()
        if "prices" in data_cg:
            prices = [x[1] for x in data_cg["prices"]]
            dates = [datetime.fromtimestamp(x[0]/1000) for x in data_cg["prices"]]
            return np.array(prices), dates, errors
        else:
            errors.append(f"CoinGecko returned unexpected data: {data_cg}")
    except Exception as e:
        errors.append(f"CoinGecko fetch failed: {e}")

    # CryptoCompare fallback
    try:
        url_cc = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol[:3]}&tsym=USD&limit={days-1}"
        data_cc = requests.get(url_cc, timeout=5).json()
        if "Data" in data_cc and "Data" in data_cc["Data"]:
            prices = [x["close"] for x in data_cc["Data"]["Data"]]
            dates = [datetime.fromtimestamp(x["time"]) for x in data_cc["Data"]["Data"]]
            return np.array(prices), dates, errors
        else:
            errors.append(f"CryptoCompare returned unexpected data: {data_cc}")
    except Exception as e:
        errors.append(f"CryptoCompare fetch failed: {e}")

    errors.append(f"Failed to fetch price data for {symbol}.")
    return np.array([]), [], errors

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
        errors.append(f"Firebase read failed for {symbol}: {e}")

    prices, dates, fetch_errors = get_price_data(symbol, 30)
    errors += fetch_errors
    try:
        ref = db.reference(f"historical/{symbol}")
        for i in range(len(prices)):
            ref.push({"time": dates[i].isoformat(), "price": float(prices[i])})
    except Exception as e:
        errors.append(f"Firebase save failed: {e}")

    return np.array(prices), dates, errors

# ---------------------------
# 📈 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period, min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    rsi = 100-(100/(1+rs))
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
        if "Data" not in data: return 0
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for article in data["Data"][:8]:
            title = article.get("title","")
            body = article.get("body","")[:150]
            text = f"{title}. {body}"
            vader = analyzer.polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity
            scores.append((vader+blob)/2)
        return np.mean(scores) if scores else 0
    except:
        return 0

# ---------------------------
# 🔮 FORECASTS
# ---------------------------
def arima_forecast(prices, steps):
    try:
        model = ARIMA(prices, order=(2,1,2))
        model_fit = model.fit()
        return list(model_fit.forecast(steps=steps))
    except:
        return [prices[-1]]*steps

def prophet_forecast(dates, prices, steps):
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": prices})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq='H')
    forecast = model.predict(future)
    return forecast["yhat"].tail(steps).values, forecast["yhat_upper"].tail(steps).values, forecast["yhat_lower"].tail(steps).values

def hybrid_forecast(prices, dates, steps):
    arima_preds = arima_forecast(prices, steps)
    prophet_preds, upper, lower = prophet_forecast(dates, prices, steps)
    final = [(a+p)/2 for a,p in zip(arima_preds, prophet_preds)]
    return final, upper, lower

# ---------------------------
# 🧠 WEIGHTS & SIGNAL
# ---------------------------
def load_weights():
    ref = db.reference("weights")
    data = ref.get()
    return data if data else {"rsi":1.0,"macd":1.0,"trend":1.0,"sentiment":1.0}

def save_weights(weights):
    db.reference("weights").set(weights)

def smart_signal(prices,rsi,macd,signal,sentiment,weights):
    score=0
    contribs={}
    # RSI
    if rsi[-1]<30: val=2*weights["rsi"]
    elif rsi[-1]>70: val=-2*weights["rsi"]
    else: val=0
    score+=val; contribs["rsi"]=val
    # MACD
    val=(1 if macd[-1]>signal[-1] else -1)*weights["macd"]
    score+=val; contribs["macd"]=val
    # Trend
    val=(1 if prices[-1]>np.mean(prices) else -1)*weights["trend"]
    score+=val; contribs["trend"]=val
    # Sentiment
    if sentiment>0.2: val=2*weights["sentiment"]
    elif sentiment<-0.2: val=-2*weights["sentiment"]
    else: val=0
    score+=val; contribs["sentiment"]=val

    if score>=3: return "BUY",score,contribs
    elif score<=-3: return "SELL",score,contribs
    return "HOLD",score,contribs

def update_weights(symbol, weights):
    ref = db.reference(f"history/{symbol}")
    data = list((ref.get() or {}).values())
    if len(data)<2: return weights
    prev, curr = data[-2], data[-1]
    if prev.get("signal","HOLD")=="HOLD": return weights
    actual_up = curr["price"]>prev["price"]
    lr=0.03
    contributions=prev.get("contributions",{})
    for key in weights:
        contrib = contributions.get(key,0)
        if contrib==0: continue
        if (contrib>0 and actual_up) or (contrib<0 and not actual_up):
            weights[key]*=(1+lr)
        else:
            weights[key]*=(1-lr)
        weights[key]=max(0.2,min(weights[key],3))
    return weights

def save_prediction(symbol, price, pred_price, signal, contribs):
    ref = db.reference(f"history/{symbol}")
    ref.push({"time":datetime.utcnow().isoformat(),"price":float(price),"predicted":float(pred_price),"signal":signal,"contributions":contribs})

def calculate_win_rate(symbol):
    ref = db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    if len(data)<2: return 0
    wins,total=0,0
    for i in range(len(data)-1):
        curr,nxt=data[i],data[i+1]
        if curr["signal"]=="BUY" and nxt["price"]>curr["price"]: wins+=1
        elif curr["signal"]=="SELL" and nxt["price"]<curr["price"]: wins+=1
        total+=1
    return round((wins/total)*100,2)

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 Adaptive AI Crypto Bot")

# --- USDT → IDR ---
try:
    data = requests.get("https://api.exchangerate.host/latest?base=USD&symbols=IDR").json()
    usdt_idr = round(data["rates"]["IDR"],2)
except:
    usdt_idr = None

if usdt_idr:
    st.metric("USDT → IDR", usdt_idr)
else:
    st.warning("Failed to fetch USDT → IDR data")

# --- Coins & Timeframe ---
coins_list = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XAIUSD","SOLUSDT"]
symbols = st.multiselect("Select Coins", coins_list, default=["BTCUSDT"])
timeframe = st.selectbox("Select Timeframe", ["1 Day","3 Days","5 Days","1 Month"])

now = datetime.now()

if st.button("Analyze"):
    for idx, symbol in enumerate(symbols):
        if symbol=="USDTIDR":
            prices, dates = get_usd_idr_history(hours=72)
            if len(prices)==0: continue
        else:
            prices, dates, errors = get_historical_data(symbol)
            new_prices,new_dates = get_hourly_data(symbol)
            if len(new_prices)>0:
                prices=np.concatenate([prices,new_prices])
                dates=dates+new_dates
                try:
                    ref = db.reference(f"historical/{symbol}")
                    for i in range(len(new_prices)):
                        ref.push({"time":new_dates[i].isoformat(),"price":float(new_prices[i])})
                except: pass
        if len(prices)<2: continue

        # Filter by timeframe
        if timeframe=="1 Day": cutoff=now-timedelta(days=1)
        elif timeframe=="3 Days": cutoff=now-timedelta(days=3)
        elif timeframe=="5 Days": cutoff=now-timedelta(days=5)
        else: cutoff=now-timedelta(days=30)
        filtered_prices = [p for d,p in zip(dates,prices) if d>=cutoff]
        filtered_dates = [d for d in dates if d>=cutoff]
        if len(filtered_prices)<2: continue

        # Forecast & indicators
        next_price, upper, lower = hybrid_forecast(filtered_prices, filtered_dates, 1)
        rsi = calculate_rsi(filtered_prices)
        macd_vals, signal_line = calculate_macd(filtered_prices)
        sentiment = get_sentiment() if symbol!="USDTIDR" else 0
        weights = load_weights()
        trade_signal, score, contributions = smart_signal(filtered_prices,rsi,macd_vals,signal_line,sentiment,weights)

        save_prediction(symbol, filtered_prices[-1], next_price[0], trade_signal, contributions)
        weights = update_weights(symbol, weights)
        save_weights(weights)

        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=filtered_dates, y=filtered_prices, mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(
            x=[filtered_dates[-1], filtered_dates[-1]+timedelta(days=1)],
            y=[filtered_prices[-1], next_price[0]],
            mode='lines+markers', name='Forecast', line=dict(dash='dash', color='red')
        ))
        fig.add_trace(go.Scatter(
            x=[filtered_dates[-1], filtered_dates[-1]+timedelta(days=1), filtered_dates[-1]+timedelta(days=1), filtered_dates[-1]],
            y=[upper[0], upper[0], lower[0], lower[0]],
            fill='toself', name='Confidence', opacity=0.2
        ))
        st.subheader(f"Analysis for {symbol}")
        st.plotly_chart(fig)
        st.metric("Signal", trade_signal)
        st.metric("Score", round(score,2))
        st.metric("Next Price", round(next_price[0],2))
        st.metric("Win Rate (%)", calculate_win_rate(symbol))
        st.metric("Daily % Change", round((filtered_prices[-1]-filtered_prices[0])/filtered_prices[0]*100,2))
