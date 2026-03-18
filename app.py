import streamlit as st
from streamlit_autorefresh import st_autorefresh
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

# =========================
# 🔄 AUTO REFRESH (30 MIN)
# =========================
st_autorefresh(interval=1800000, key="refresh")

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
        st.error(f"Firebase Initialization Error: {e}")

# ---------------------------
# 📊 FETCH PRICE DATA
# ---------------------------
def get_price_data(symbol="BTCUSDT", days=30):
    prices, dates, errors = [], [], []
    coin = symbol.replace("USDT","").lower()

    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
        data = requests.get(url, timeout=5).json()
        if "prices" in data:
            prices = [x[1] for x in data["prices"]]
            dates = [datetime.fromtimestamp(x[0]/1000) for x in data["prices"]]
            return np.array(prices), dates, errors
    except Exception as e:
        errors.append(str(e))

    try:
        url_cc = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol[:3]}&tsym=USD&limit={days-1}"
        data_cc = requests.get(url_cc, timeout=5).json()
        raw = data_cc.get("Data", {}).get("Data", [])
        prices = [x["close"] for x in raw]
        dates = [datetime.fromtimestamp(x["time"]) for x in raw]
        return np.array(prices), dates, errors
    except Exception as e:
        errors.append(str(e))

    return np.array([]), [], errors

# ---------------------------
# 📊 HISTORICAL + FIREBASE
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
        errors.append(str(e))

    prices, dates, _ = get_price_data(symbol, 30)

    try:
        ref = db.reference(f"historical/{symbol}")
        for i in range(len(prices)):
            ref.push({"time": dates[i].isoformat(), "price": float(prices[i])})
    except:
        pass

    return np.array(prices), dates, errors

# ---------------------------
# ⏱️ HOURLY DATA
# ---------------------------
def get_hourly_data(symbol="BTCUSDT", hours=72):
    fsym = symbol.replace("USDT","")
    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={hours-1}"
    data = requests.get(url, timeout=5).json()
    prices, dates = [], []
    for x in data.get("Data", {}).get("Data", []):
        prices.append(x["close"])
        dates.append(datetime.fromtimestamp(x["time"]))
    return np.array(prices), dates

# ---------------------------
# 📈 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    avg_gain = pd.Series(gain).rolling(period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period, min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    rsi = 100 - (100/(1+rs))
    return np.concatenate([[50], rsi])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12,adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26,adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9,adjust=False).mean()
    return macd.values, signal.values

# ---------------------------
# 📰 SENTIMENT
# ---------------------------
def get_sentiment():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data = requests.get(url, timeout=5).json()
        scores=[]
        for a in data.get("Data", [])[:8]:
            text = f"{a.get('title','')}"
            vader = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity
            scores.append((vader + blob)/2)
        return np.mean(scores) if scores else 0
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

def prophet_forecast(dates, prices, steps):
    df = pd.DataFrame({"ds": pd.to_datetime(dates),"y":prices})
    model=Prophet(daily_seasonality=True)
    model.fit(df)
    future=model.make_future_dataframe(periods=steps,freq="H")
    f= model.predict(future)
    return (f["yhat"].tail(steps).values,
            f["yhat_upper"].tail(steps).values,
            f["yhat_lower"].tail(steps).values)

def hybrid_forecast(prices, dates, steps):
    ar = arima_forecast(prices, steps)
    pr, upper, lower = prophet_forecast(dates, prices, steps)
    combined = [(a+p)/2 for a,p in zip(ar, pr)]
    return combined, upper, lower

# ---------------------------
# 🧠 AI LOGIC
# ---------------------------
def load_weights():
    ref=db.reference("weights")
    data=ref.get() or {}
    return data if data else {"rsi":1,"macd":1,"trend":1,"sentiment":1}

def save_weights(weights):
    db.reference("weights").set(weights)

def smart_signal(prices, rsi, macd, signal, sentiment, weights):
    score=0; contrib={}
    val=2*weights["rsi"] if rsi[-1]<30 else (-2*weights["rsi"] if rsi[-1]>70 else 0)
    score+=val; contrib["rsi"]=val
    val=(1 if macd[-1]>signal[-1] else -1)*weights["macd"]
    score+=val; contrib["macd"]=val
    val=(1 if prices[-1]>np.mean(prices) else -1)*weights["trend"]
    score+=val; contrib["trend"]=val
    val=2*weights["sentiment"] if sentiment>0.2 else (-2*weights["sentiment"] if sentiment<-0.2 else 0)
    score+=val; contrib["sentiment"]=val
    if score>=3: return "BUY",score,contrib
    elif score<=-3: return "SELL",score,contrib
    return "HOLD",score,contrib

def update_weights(symbol, weights):
    ref=db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    if len(data)<2: return weights
    prev,curr=data[-2],data[-1]
    actual_up = curr["price"]>prev["price"]
    lr=0.03
    for key in weights:
        contrib=prev.get("contributions",{}).get(key,0)
        if contrib==0: continue
        if (contrib>0 and actual_up) or (contrib<0 and not actual_up):
            weights[key]*=(1+lr)
        else:
            weights[key]*=(1-lr)
        weights[key]=max(0.2,min(weights[key],3))
    return weights

def save_prediction(symbol, price, pred, sig, contrib):
    ref=db.reference(f"history/{symbol}")
    ref.push({
        "time":datetime.utcnow().isoformat(),
        "price":float(price),
        "predicted":float(pred),
        "signal":sig,
        "contributions":contrib
    })

def calculate_win_rate(symbol):
    ref=db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    wins,total=0,0
    for i in range(len(data)-1):
        curr,nxt=data[i],data[i+1]
        sig=curr.get("signal","HOLD")
        cp=curr.get("price",0); np_=nxt.get("price",0)
        if sig=="BUY" and np_>cp: wins+=1
        elif sig=="SELL" and np_<cp: wins+=1
        total+=1
    return round((wins/total)*100,2) if total>0 else 0

def calculate_confidence(upper, lower, price):
    return round(max(0,min(1,1-(upper-lower)/price))*100,2)

# =========================
# 🚀 BACKGROUND COLLECTOR
# =========================
COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","MATICUSDT","XAIUSD"]

for sym in COINS:
    prices, dates, _ = get_historical_data(sym)
    if len(prices) < 2:
        continue

    hr_prices, hr_dates = get_hourly_data(sym)
    if len(hr_prices) > 0:
        prices = np.concatenate([prices, hr_prices])
        dates = dates + hr_dates

    next_p, upper, lower = hybrid_forecast(prices, dates, 1)

    rsi = calculate_rsi(prices)
    macd_v, sig_line = calculate_macd(prices)
    sentiment = get_sentiment()
    weights = load_weights()

    sig, score, con = smart_signal(prices, rsi, macd_v, sig_line, sentiment, weights)

    save_prediction(sym, prices[-1], next_p[0], sig, con)

    weights = update_weights(sym, weights)
    save_weights(weights)

# =========================
# 🎨 UI
# =========================
st.title("🚀 AI Crypto Bot")

portfolio={}
buy={}
with st.expander("💰 Portfolio"):
    for c in COINS:
        portfolio[c]=st.number_input(f"{c} amount",0.0,key=f"h{c}")
        buy[c]=st.number_input(f"{c} buy price",0.0,key=f"b{c}")

symbols=st.multiselect("Select Coins", COINS, default=["BTCUSDT","ETHUSDT"])

summary_data=[]

for sym in symbols:
    prices, dates, _ = get_historical_data(sym)
    hr_prices, hr_dates = get_hourly_data(sym)

    if len(hr_prices)>0:
        prices=np.concatenate([prices,hr_prices])
        dates=dates+hr_dates

    # FIXED CHART
    combined = sorted(zip(dates, prices), key=lambda x: x[0])
    seen=set(); fd=[]; fp=[]
    for d,p in combined:
        if d not in seen:
            fd.append(d); fp.append(p); seen.add(d)

    if len(fp)<2: continue

    next_p, upper, lower = hybrid_forecast(fp, fd, 1)

    rsi=calculate_rsi(fp)
    macd_v, sig_line=calculate_macd(fp)
    sentiment=get_sentiment()
    weights=load_weights()

    sig,score,con=smart_signal(fp,rsi,macd_v,sig_line,sentiment,weights)

    save_prediction(sym,fp[-1],next_p[0],sig,con)

    weights=update_weights(sym,weights)
    save_weights(weights)

    wr=calculate_win_rate(sym)
    conf=calculate_confidence(upper[0],lower[0],fp[-1])

    summary_data.append({
        "Coin":sym,
        "Price":fp[-1],
        "Signal":sig,
        "Score":score,
        "WinRate":wr,
        "Confidence":conf
    })

# =========================
# 📊 SUMMARY TOP
# =========================
st.subheader("📊 Summary")
st.dataframe(pd.DataFrame(summary_data))

# =========================
# 📈 DETAILS
# =========================
for sym in symbols:
    prices, dates, _ = get_historical_data(sym)
    hr_prices, hr_dates = get_hourly_data(sym)

    if len(hr_prices)>0:
        prices=np.concatenate([prices,hr_prices])
        dates=dates+hr_dates

    combined = sorted(zip(dates, prices), key=lambda x: x[0])
    fd=[x[0] for x in combined]
    fp=[x[1] for x in combined]

    next_p, _, _ = hybrid_forecast(fp, fd, 1)

    st.markdown(f"### {sym} — Current Price: ${round(fp[-1],2)}")

    col1,col2=st.columns([3,1])

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=fd,y=fp,name=sym))
    fig.add_trace(go.Scatter(
        x=[fd[-1],fd[-1]+timedelta(days=1)],
        y=[fp[-1],next_p[0]],
        name="Forecast",line=dict(dash="dash")
    ))

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        color={"BUY":"green","SELL":"red","HOLD":"#777777"}
        st.markdown(
            f"<div style='padding:8px;border-radius:8px;background:{color[sig]};color:white;text-align:center;font-weight:bold'>{sig}</div>",
            unsafe_allow_html=True
        )
