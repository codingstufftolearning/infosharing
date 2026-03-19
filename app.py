import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import firebase_admin
from firebase_admin import credentials, db
import warnings

# 🔥 NEW
import time as t
import random
import concurrent.futures
import threading
import websocket
from sklearn.linear_model import LogisticRegression

# =========================
# AUTO REFRESH
# =========================
st_autorefresh(interval=1800000, key="refresh")

# =========================
# FIREBASE INIT
# =========================
if not firebase_admin._apps:
    try:
        firebase_dict = dict(st.secrets["firebase"])
        firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(firebase_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": firebase_dict["databaseURL"]})
    except Exception as e:
        st.error(f"Firebase Error: {e}")

# =========================
# GLOBALS
# =========================
source_health = {"binance":1,"cryptocompare":1,"coingecko":1}
ws_prices = {}

# =========================
# SAFE REQUEST
# =========================
def safe_request(url, params=None):
    headers = {"User-Agent": "Mozilla/5.0"}
    for i in range(3):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=10)
            if r.status_code == 200:
                return r.json()
            elif r.status_code in [429,418]:
                t.sleep(2*(i+1))
        except:
            t.sleep(1)
    return None

# =========================
# WEBSOCKET
# =========================
def start_ws(symbol):
    def on_message(ws, message):
        try:
            data = eval(message)
            if "c" in data:
                ws_prices[symbol] = float(data["c"])
        except:
            pass

    url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
    ws = websocket.WebSocketApp(url, on_message=on_message)
    threading.Thread(target=ws.run_forever, daemon=True).start()

# =========================
# DATA FETCH
# =========================
def get_ohlc_binance(symbol, interval, limit):
    bases = ["https://api.binance.com","https://api1.binance.com"]
    for b in bases:
        data = safe_request(b+"/api/v3/klines", {"symbol":symbol,"interval":interval,"limit":limit})
        if data:
            try:
                d,o,h,l,c,v=[],[],[],[],[],[]
                for k in data:
                    d.append(datetime.fromtimestamp(k[0]/1000))
                    o.append(float(k[1])); h.append(float(k[2]))
                    l.append(float(k[3])); c.append(float(k[4]))
                    v.append(float(k[5]))
                return d,o,h,l,c,v
            except:
                pass
    return None

def get_ohlc_cryptocompare(symbol, interval, limit):
    try:
        fsym=symbol.replace("USDT","")
        url=f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={limit}"
        data=safe_request(url)
        raw=data.get("Data",{}).get("Data",[]) if data else []
        d,o,h,l,c,v=[],[],[],[],[],[]
        for x in raw:
            d.append(datetime.fromtimestamp(x["time"]))
            o.append(x["open"]); h.append(x["high"])
            l.append(x["low"]); c.append(x["close"])
            v.append(x.get("volumeto",0))
        return d,o,h,l,c,v
    except:
        return None

def get_ohlc(symbol, timeframe):
    mapping={"15 Min":("15m",96),"Hourly":("1h",48),"Daily":("1d",180)}
    interval,limit=mapping[timeframe]

    funcs=[
        lambda:get_ohlc_binance(symbol,interval,limit),
        lambda:get_ohlc_cryptocompare(symbol,interval,limit)
    ]

    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures=[ex.submit(f) for f in funcs]
        for f in concurrent.futures.as_completed(futures):
            data=f.result()
            if data:
                return data,"multi"
    return None,"None"

# =========================
# INDICATORS (FIXED)
# =========================
def calculate_rsi(prices):
    prices=np.array(prices)
    if len(prices)<15:
        return np.array([50]*len(prices))
    delta=np.diff(prices)
    gain=np.maximum(delta,0)
    loss=np.abs(np.minimum(delta,0))
    rs=pd.Series(gain).rolling(14).mean()/(pd.Series(loss).rolling(14).mean()+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def calculate_macd(prices):
    prices=np.array(prices)
    if len(prices)<26:
        return np.zeros(len(prices)),np.zeros(len(prices))
    exp1=pd.Series(prices).ewm(span=12).mean()
    exp2=pd.Series(prices).ewm(span=26).mean()
    macd=exp1-exp2
    signal=macd.ewm(span=9).mean()
    return macd.values,signal.values

# =========================
# AI MODEL
# =========================
def train_ai(prices):
    if len(prices)<50: return None
    X=[];y=[]
    for i in range(20,len(prices)-1):
        w=prices[i-20:i]
        X.append([np.mean(w),np.std(w),w[-1]-w[0]])
        y.append(1 if prices[i+1]>prices[i] else 0)
    m=LogisticRegression()
    m.fit(X,y)
    return m

def ai_predict(m,prices):
    if not m or len(prices)<20: return 0.5
    w=prices[-20:]
    return m.predict_proba([[np.mean(w),np.std(w),w[-1]-w[0]]])[0][1]

# =========================
# SENTIMENT
# =========================
@st.cache_data(ttl=600)
def get_sentiment():
    try:
        data=safe_request("https://min-api.cryptocompare.com/data/v2/news/?lang=EN")
        scores=[]
        for a in data.get("Data",[])[:5]:
            text=a["title"]
            v=SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
            b=TextBlob(text).sentiment.polarity
            scores.append((v+b)/2)
        return np.mean(scores) if scores else 0
    except:
        return 0

# =========================
# UI
# =========================
st.title("🚀 AI Crypto Bot FULL")

st.sidebar.header("💰 Portfolio")
portfolio={}
coins=["BTCUSDT","ETHUSDT"]
for sym in coins:
    amt=st.sidebar.number_input(f"{sym} Amount",0.0)
    buy=st.sidebar.number_input(f"{sym} Buy Price",0.0)
    if amt>0 and buy>0:
        portfolio[sym]={"amt":amt,"buy":buy}

symbols=st.multiselect("Coins",coins,default=["BTCUSDT"])
timeframe=st.selectbox("Timeframe",["15 Min","Hourly","Daily"])

debug=[]

for sym in symbols:

    if sym not in ws_prices:
        start_ws(sym)

    data,source=get_ohlc(sym,timeframe)
    if not data:
        debug.append(f"{sym}: data fail")
        continue

    fd,o,h,l,c,v=data
    c=np.array(c)

    if len(c)<30:
        debug.append(f"{sym}: not enough data")
        continue

    rsi=calculate_rsi(c)
    macd,signal=calculate_macd(c)

    model=train_ai(c)
    ai_prob=ai_predict(model,c)

    sentiment=get_sentiment()

    score=(50-rsi[-1])/50+(macd[-1]-signal[-1])+sentiment+(ai_prob-0.5)*2
    sig="BUY" if score>1 else "SELL" if score<-1 else "HOLD"

    live=ws_prices.get(sym,c[-1])

    # CHART
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=fd,open=o,high=h,low=l,close=c))
    st.plotly_chart(fig,use_container_width=True)

    st.write(f"### {sym}")
    st.write(f"Live: {live}")
    st.write(f"Signal: {sig}")
    st.write(f"AI: {round(ai_prob*100,2)}%")

# DEBUG
with st.expander("Debug"):
    for d in debug:
        st.write(d)
