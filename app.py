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

# 🔥 NEW IMPORTS
import time
import random
import concurrent.futures
import threading
import websocket
from sklearn.linear_model import LogisticRegression

# =========================
# 🔄 AUTO REFRESH (30 MIN)
# =========================
st_autorefresh(interval=1800000, key="refresh")

# =========================
# 🔐 FIREBASE INIT
# =========================
if not firebase_admin._apps:
    try:
        firebase_dict = dict(st.secrets["firebase"])
        firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(firebase_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": firebase_dict["databaseURL"]})
    except Exception as e:
        st.error(f"Firebase Initialization Error: {e}")

# =========================
# 🔥 GLOBALS
# =========================
# 🔥 NEW
source_health = {"binance":1.0,"cryptocompare":1.0,"coingecko":1.0}
ws_prices = {}

# =========================
# 🔥 SAFE REQUEST (NEW)
# =========================
def safe_request(url, params=None):
    headers = {"User-Agent": "Mozilla/5.0"}
    for i in range(3):
        try:
            res = requests.get(url, params=params, headers=headers, timeout=10)
            if res.status_code == 200:
                return res.json(), None
            elif res.status_code in [429,418]:
                time.sleep(2*(i+1))
            else:
                return None, f"HTTP {res.status_code}"
        except Exception as e:
            time.sleep(1)
            err = str(e)
    return None, err

# =========================
# 🔥 WEBSOCKET (LIVE PRICE)
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
    thread = threading.Thread(target=ws.run_forever, daemon=True)
    thread.start()

# =========================
# 📊 DATA FETCH FUNCTIONS
# =========================
# 🔧 UPDATED
def get_ohlc_binance(symbol, interval, limit):
    BASE_URLS = [
        "https://api.binance.com",
        "https://api1.binance.com",
        "https://api2.binance.com"
    ]

    for base in BASE_URLS:
        url = base + "/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data, err = safe_request(url, params)

        if data:
            try:
                d,o,h,l,c,v = [],[],[],[],[],[]
                for k in data:
                    d.append(datetime.fromtimestamp(k[0]/1000))
                    o.append(float(k[1]))
                    h.append(float(k[2]))
                    l.append(float(k[3]))
                    c.append(float(k[4]))
                    v.append(float(k[5]))
                return d,o,h,l,c,v
            except:
                return None
    return None

# 🔧 UPDATED
def get_ohlc_cryptocompare(symbol, interval, limit):
    try:
        fsym = symbol.replace("USDT","")
        if interval == "15m":
            url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={fsym}&tsym=USD&limit={limit}&aggregate=15"
        elif interval == "1h":
            url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={limit}"
        else:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={fsym}&tsym=USD&limit={limit}"

        data, _ = safe_request(url)
        raw = data.get("Data", {}).get("Data", []) if data else []

        d,o,h,l,c,v = [],[],[],[],[],[]
        for x in raw:
            d.append(datetime.fromtimestamp(x["time"]))
            o.append(x["open"])
            h.append(x["high"])
            l.append(x["low"])
            c.append(x["close"])
            v.append(x.get("volumeto",0))
        return d,o,h,l,c,v
    except:
        return None

def get_ohlc_coingecko(symbol):
    try:
        coin = symbol.replace("USDT","").lower()
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=365"
        data,_ = safe_request(url)
        prices = data.get("prices", []) if data else []

        d,o,h,l,c,v = [],[],[],[],[],[]
        for i in range(1,len(prices)):
            t = datetime.fromtimestamp(prices[i][0]/1000)
            prev = prices[i-1][1]
            curr = prices[i][1]
            d.append(t)
            o.append(prev)
            h.append(max(prev,curr))
            l.append(min(prev,curr))
            c.append(curr)
            v.append(0)
        return d,o,h,l,c,v
    except:
        return None

# 🔥 NEW PARALLEL FETCH
def get_ohlc(symbol, timeframe):
    mapping = {
        "15 Min": ("15m", 96),
        "Hourly": ("1h", 48),
        "Daily": ("1d", 180),
        "3-Day": ("3d", 90)
    }
    interval, limit = mapping[timeframe]

    funcs = [
        lambda: get_ohlc_binance(symbol, interval, limit),
        lambda: get_ohlc_cryptocompare(symbol, interval, limit),
        lambda: get_ohlc_coingecko(symbol)
    ]

    with concurrent.futures.ThreadPoolExecutor() as ex:
        futures = [ex.submit(f) for f in funcs]
        for f in concurrent.futures.as_completed(futures):
            data = f.result()
            if data:
                return data, "multi-source"
    return None, "None"

# =========================
# 📈 INDICATORS
# =========================
def calculate_rsi(prices, period=14):
    prices = np.array(prices)
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    rs = pd.Series(gain).rolling(period).mean()/(pd.Series(loss).rolling(period).mean()+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12).mean()
    exp2 = pd.Series(prices).ewm(span=26).mean()
    macd = exp1-exp2
    signal = macd.ewm(span=9).mean()
    return macd.values, signal.values

# =========================
# 🔥 AI MODEL
# =========================
def train_ai(prices):
    if len(prices)<50: return None
    X=[]; y=[]
    for i in range(20,len(prices)-1):
        w = prices[i-20:i]
        X.append([np.mean(w), np.std(w), w[-1]-w[0]])
        y.append(1 if prices[i+1]>prices[i] else 0)
    model = LogisticRegression()
    model.fit(X,y)
    return model

def ai_predict(model, prices):
    if not model or len(prices)<20: return 0.5
    w = prices[-20:]
    return model.predict_proba([[np.mean(w),np.std(w),w[-1]-w[0]]])[0][1]

# =========================
# 📰 SENTIMENT
# =========================
@st.cache_data(ttl=600)
def get_sentiment():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data,_ = safe_request(url)
        scores=[]
        for a in data.get("Data", [])[:8]:
            text = a.get("title","")
            vader = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity
            scores.append((vader+blob)/2)
        return np.mean(scores) if scores else 0
    except:
        return 0

# =========================
# 🎨 UI
# =========================
st.title("🚀 AI Crypto Bot FULL VERSION")

symbols = st.multiselect("Select Coins", ["BTCUSDT","ETHUSDT"], default=["BTCUSDT"])
timeframe = st.selectbox("Timeframe", ["15 Min","Hourly","Daily"])

debug_info=[]

for sym in symbols:

    # 🔥 start websocket
    if sym not in ws_prices:
        start_ws(sym)

    data, source = get_ohlc(sym, timeframe)
    if data is None:
        debug_info.append(f"{sym}: data fetch failed")
        continue

    fd,o,h,l,c,v = data
    c = np.array(c)

    rsi = calculate_rsi(c)
    macd_v, sig_line = calculate_macd(c)

    # 🔥 AI
    model = train_ai(c)
    ai_prob = ai_predict(model,c)

    sentiment = get_sentiment()

    score = (
        (50-rsi[-1])/50 +
        (macd_v[-1]-sig_line[-1]) +
        sentiment +
        (ai_prob-0.5)*2
    )

    sig = "BUY" if score>1 else "SELL" if score<-1 else "HOLD"

    live_price = ws_prices.get(sym, c[-1])

    # =========================
    # CHART
    # =========================
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=fd, open=o, high=h, low=l, close=c))
    st.plotly_chart(fig, use_container_width=True)

    st.write(f"### {sym}")
    st.write(f"Live Price: {live_price}")
    st.write(f"Signal: {sig}")
    st.write(f"AI Confidence: {round(ai_prob*100,2)}%")
    st.write(f"Sentiment: {round(sentiment,2)}")

# =========================
# DEBUG
# =========================
with st.expander("🧰 Debug Panel"):
    for d in debug_info:
        st.write(d)
