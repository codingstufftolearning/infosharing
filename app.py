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

import time as t
import concurrent.futures
import threading
import websocket
from sklearn.linear_model import LogisticRegression

# =========================
# AUTO REFRESH
# =========================
st_autorefresh(interval=1800000, key="refresh")

# =========================
# FIREBASE
# =========================
if not firebase_admin._apps:
    firebase_dict = dict(st.secrets["firebase"])
    firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n","\n")
    cred = credentials.Certificate(firebase_dict)
    firebase_admin.initialize_app(cred, {"databaseURL": firebase_dict["databaseURL"]})

# =========================
# GLOBAL
# =========================
ws_prices = {}

# =========================
# SAFE REQUEST
# =========================
def safe_request(url, params=None):
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            t.sleep(1)
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
def get_ohlc(symbol, interval="1h", limit=120):
    url = "https://api.binance.com/api/v3/klines"
    data = safe_request(url, {"symbol":symbol,"interval":interval,"limit":limit})
    if not data:
        return None
    d,o,h,l,c,v=[],[],[],[],[],[]
    for k in data:
        d.append(datetime.fromtimestamp(k[0]/1000))
        o.append(float(k[1])); h.append(float(k[2]))
        l.append(float(k[3])); c.append(float(k[4]))
        v.append(float(k[5]))
    return d,o,h,l,c,v

# =========================
# INDICATORS
# =========================
def calculate_rsi(prices):
    if len(prices)<15:
        return np.array([50]*len(prices))
    delta=np.diff(prices)
    gain=np.maximum(delta,0)
    loss=np.abs(np.minimum(delta,0))
    rs=pd.Series(gain).rolling(14).mean()/(pd.Series(loss).rolling(14).mean()+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def calculate_macd(prices):
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
    X=[]; y=[]
    for i in range(20,len(prices)-1):
        w=prices[i-20:i]
        X.append([np.mean(w), np.std(w), w[-1]-w[0]])
        y.append(1 if prices[i+1]>prices[i] else 0)
    m=LogisticRegression()
    m.fit(X,y)
    return m

def ai_predict(m, prices):
    if not m or len(prices)<20: return 0.5
    w=prices[-20:]
    return m.predict_proba([[np.mean(w),np.std(w),w[-1]-w[0]]])[0][1]

# =========================
# SENTIMENT
# =========================
@st.cache_data(ttl=600)
def get_sentiment():
    data = safe_request("https://min-api.cryptocompare.com/data/v2/news/?lang=EN")
    scores=[]
    for a in data.get("Data",[])[:5]:
        text=a["title"]
        v=SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
        b=TextBlob(text).sentiment.polarity
        scores.append((v+b)/2)
    return np.mean(scores) if scores else 0

# =========================
# UI
# =========================
st.title("🚀 AI Crypto Bot Pro")

COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","MATICUSDT","XAIUSD"]

# Portfolio
st.sidebar.header("💰 Portfolio")
portfolio={}
total_pl=0
for sym in COINS:
    amt=st.sidebar.number_input(f"{sym} Amount",0.0)
    buy=st.sidebar.number_input(f"{sym} Buy Price",0.0)
    if amt>0 and buy>0:
        portfolio[sym]={"amt":amt,"buy":buy}

symbols = st.multiselect("Select Coins", COINS, default=["BTCUSDT","ETHUSDT"])

debug=[]

for sym in symbols:

    if sym not in ws_prices:
        start_ws(sym)

    data = get_ohlc(sym)
    if not data:
        debug.append(f"{sym} fetch failed")
        continue

    fd,o,h,l,c,v = data
    c=np.array(c)

    rsi=calculate_rsi(c)
    macd,signal=calculate_macd(c)

    ai_model=train_ai(c)
    ai_prob=ai_predict(ai_model,c)

    sentiment=get_sentiment()

    # IMPROVED SCORE
    trend = (c[-1]-np.mean(c))/np.std(c)
    macd_score = np.tanh(macd[-1]-signal[-1])
    rsi_score = (50-rsi[-1])/50

    score = rsi_score + macd_score + trend + sentiment + (ai_prob-0.5)*2

    sig = "BUY" if score>1 else "SELL" if score<-1 else "HOLD"

    live = ws_prices.get(sym,c[-1])

    # CONFIDENCE (improved)
    conf = max(0,min(100,100*(1-np.std(c)/c[-1])))

    # =========================
    # CARD CONTAINER
    # =========================
    with st.container():
        st.markdown(
            "<div style='border:1px solid white; padding:10px; border-radius:8px;'>",
            unsafe_allow_html=True
        )

        st.markdown(f"### {sym}")

        col1,col2 = st.columns([3,1])

        # CHART WITH ZOOM ENABLED
        with col1:
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=fd,open=o,high=h,low=l,close=c))
            fig.update_layout(dragmode="zoom")
            st.plotly_chart(fig, use_container_width=True)  # toolbar enabled

        with col2:
            st.write(f"**Signal:** {sig}")
            st.write(f"**Score:** {round(score,2)}")
            st.write(f"**Confidence:** {round(conf,2)}%")
            st.write(f"**RSI:** {round(rsi[-1],2)}")
            st.write(f"**MACD:** {round(macd[-1],2)}")
            st.write(f"**AI:** {round(ai_prob*100,2)}%")
            st.write(f"**Sentiment:** {round(sentiment,2)}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Portfolio calc
    if sym in portfolio:
        p = portfolio[sym]
        pl = (live - p["buy"]) * p["amt"]
        total_pl += pl

# Portfolio summary
st.sidebar.write(f"### Total P/L: ${round(total_pl,2)}")

# Debug panel
with st.expander("🧰 Debug Panel"):
    for d in debug:
        st.write(d)
