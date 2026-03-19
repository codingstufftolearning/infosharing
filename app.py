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
# 🔄 AUTO REFRESH
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
        firebase_admin.initialize_app(cred, {
            "databaseURL": firebase_dict["databaseURL"]
        })
    except Exception as e:
        st.error(f"Firebase Initialization Error: {e}")

# =========================
# 📡 MULTI SOURCE OHLC
# =========================
def get_ohlc_binance(symbol, interval, limit):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data = requests.get(url, params=params, timeout=5).json()

        if not isinstance(data, list):
            return None

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

def get_ohlc_cryptocompare(symbol, interval, limit):
    try:
        fsym = symbol.replace("USDT","")
        if interval == "15m":
            url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={fsym}&tsym=USD&limit={limit}&aggregate=15"
        elif interval == "1h":
            url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={limit}"
        else:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={fsym}&tsym=USD&limit={limit}"

        data = requests.get(url, timeout=5).json()
        raw = data.get("Data", {}).get("Data", [])
        if not raw:
            return None

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
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=30"
        data = requests.get(url, timeout=5).json()
        prices = data.get("prices", [])

        if not prices:
            return None

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

def get_ohlc(symbol, timeframe):
    mapping = {
        "15 Min": ("15m", 96),
        "Hourly": ("1h", 48),
        "Daily": ("1d", 30),
        "3-Day": ("3d", 20),
        "Weekly": ("1w", 12)
    }

    interval, limit = mapping[timeframe]

    for fn in [get_ohlc_binance, get_ohlc_cryptocompare, get_ohlc_coingecko]:
        data = fn(symbol, interval, limit) if fn != get_ohlc_coingecko else fn(symbol)
        if data:
            return data, fn.__name__

    return None, "None"

# =========================
# 📊 INDICATORS (UNCHANGED)
# =========================
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    avg_gain = pd.Series(gain).rolling(period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period, min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    return np.concatenate([[50], 100-(100/(1+rs))])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12,adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26,adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9,adjust=False).mean()
    return macd.values, signal.values

# =========================
# 📰 SENTIMENT (UNCHANGED)
# =========================
def get_sentiment():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data = requests.get(url, timeout=5).json()
        scores=[]
        for a in data.get("Data", [])[:8]:
            text=a.get("title","")
            vader=SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
            blob=TextBlob(text).sentiment.polarity
            scores.append((vader+blob)/2)
        return np.mean(scores) if scores else 0
    except:
        return 0

# =========================
# 🔮 FORECAST (UNCHANGED)
# =========================
def arima_forecast(prices, steps):
    try:
        model = ARIMA(prices, order=(2,1,2))
        return list(model.fit().forecast(steps=steps))
    except:
        return [prices[-1]]*steps

# =========================
# 🚀 BACKGROUND COLLECTOR (UNCHANGED)
# =========================
COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","MATICUSDT","XAIUSD"]

# =========================
# 🎨 UI
# =========================
st.title("🚀 AI Crypto Bot")

timeframe = st.selectbox("Timeframe", ["15 Min","Hourly","Daily","3-Day","Weekly"])

symbols = st.multiselect("Select Coins", COINS, default=["BTCUSDT","ETHUSDT"])

# =========================
# 📈 DETAILS (FULLY UPGRADED)
# =========================
for sym in symbols:

    data, source = get_ohlc(sym, timeframe)
    if data is None:
        st.error(f"{sym} failed all sources")
        continue

    fd,o,h,l,c,v = data

    # ZOOM SYSTEM
    if f"zoom_{sym}" not in st.session_state:
        st.session_state[f"zoom_{sym}"] = len(fd)

    colz1,colz2 = st.columns(2)

    with colz1:
        if st.button(f"Zoom In {sym}"):
            st.session_state[f"zoom_{sym}"] -= 5

    with colz2:
        if st.button(f"Zoom Out {sym}"):
            st.session_state[f"zoom_{sym}"] += 5

    zoom = st.slider(f"Zoom {sym}", 10, len(fd), st.session_state[f"zoom_{sym}"])

    fd = fd[-zoom:]
    o = o[-zoom:]
    h = h[-zoom:]
    l = l[-zoom:]
    c = c[-zoom:]
    v = v[-zoom:]

    pred = arima_forecast(c, 1)[0]

    fig = go.Figure()

    fig.add_trace(go.Candlestick(x=fd, open=o, high=h, low=l, close=c))
    fig.add_trace(go.Bar(x=fd, y=v, yaxis="y2"))

    fig.add_trace(go.Scatter(
        x=fd, y=c,
        line=dict(dash="dash", color="gray")
    ))

    color = "green" if pred > c[-1] else "red"

    fig.add_trace(go.Scatter(
        x=[fd[-1], fd[-1]+timedelta(hours=1)],
        y=[c[-1], pred],
        line=dict(color=color, width=3)
    ))

    fig.update_layout(
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        yaxis2=dict(overlaying="y", side="right"),
        showlegend=False
    )

    st.subheader(sym)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"Source: {source}")

# =========================
# 🧰 DEBUG PANEL
# =========================
with st.expander("🧰 Debug Panel"):
    st.write("Timeframe:", timeframe)
    st.write("Session State:", st.session_state)
