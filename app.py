import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import threading, websocket, time as t
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.linear_model import LogisticRegression

# =========================
# AUTO REFRESH
# =========================
if "loading" not in st.session_state:
    st.session_state.loading = False

if not st.session_state.loading:
    st_autorefresh(interval=300000, key="refresh")

# =========================
# GLOBALS
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
            t.sleep(2*(i+1))
        except:
            t.sleep(1)
    return None

# =========================
# WEBSOCKET PRICE
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
# FETCH DATA
# =========================
def get_ohlc(symbol, timeframe):
    mapping = {"15 Min":("15m",96),"Hourly":("1h",72),"Daily":("1d",180)}
    interval, limit = mapping[timeframe]

    url = "https://api.binance.com/api/v3/klines"
    data = safe_request(url, {"symbol":symbol,"interval":interval,"limit":limit})
    if not data:
        return None

    d,o,h,l,c,v=[],[],[],[],[],[]
    for k in data:
        d.append(datetime.fromtimestamp(k[0]/1000))
        o.append(float(k[1]))
        h.append(float(k[2]))
        l.append(float(k[3]))
        c.append(float(k[4]))
        v.append(float(k[5]))

    return d,o,h,l,c,v

# =========================
# INDICATORS
# =========================
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(period,min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period,min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.values, signal.values

# =========================
# AI MODEL (LIGHTWEIGHT)
# =========================
def train_ai(prices):
    if len(prices)<50:
        return None

    X=[]; y=[]
    for i in range(20,len(prices)-1):
        w = prices[i-20:i]
        X.append([np.mean(w),np.std(w),w[-1]-w[0]])
        y.append(1 if prices[i+1]>prices[i] else 0)

    model = LogisticRegression()
    model.fit(X,y)
    return model

def ai_predict(model, prices):
    if model is None or len(prices)<20:
        return 0.5
    w = prices[-20:]
    return model.predict_proba([[np.mean(w),np.std(w),w[-1]-w[0]]])[0][1]

# =========================
# SENTIMENT
# =========================
@st.cache_data(ttl=900)
def get_sentiment():
    try:
        data = safe_request("https://min-api.cryptocompare.com/data/v2/news/?lang=EN")
        scores=[]
        for a in data.get("Data",[])[:5]:
            text = a.get("title","")
            v = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
            b = TextBlob(text).sentiment.polarity
            scores.append((v+b)/2)
        return np.mean(scores) if scores else 0
    except:
        return 0

# =========================
# SMART SIGNAL SYSTEM
# =========================
def get_trend(prices):
    s = pd.Series(prices)
    ema50 = s.ewm(span=50).mean()
    ema200 = s.ewm(span=200).mean()
    return "UP" if ema50.iloc[-1] > ema200.iloc[-1] else "DOWN"

def volume_boost(volumes):
    if len(volumes)<20:
        return False
    return volumes[-1] > np.mean(volumes[-20:]) * 1.5

def get_volatility(prices):
    return np.std(prices)/prices[-1]

def generate_signal(c, v, rsi, macd, signal_line, ai_prob):
    trend = get_trend(c)
    vol_ok = volume_boost(v)
    volatility = get_volatility(c)

    macd_up = macd[-1]>signal_line[-1] and macd[-2]<=signal_line[-2]
    macd_down = macd[-1]<signal_line[-1] and macd[-2]>=signal_line[-2]

    signal = "HOLD"

    if volatility < 0.005:
        return "HOLD", trend, volatility

    if rsi[-1]<35 and macd_up and ai_prob>0.55:
        signal = "STRONG BUY" if trend=="UP" and vol_ok else "BUY"

    elif rsi[-1]>65 and macd_down and ai_prob<0.45:
        signal = "STRONG SELL" if trend=="DOWN" and vol_ok else "SELL"

    return signal, trend, volatility

def calculate_confidence(rsi, ai_prob, volatility):
    rsi_score = 1 - abs(rsi[-1]-50)/50
    vol_score = max(0, 1 - volatility*10)
    return max(0,min(100,(ai_prob*0.5 + rsi_score*0.3 + vol_score*0.2)*100))

# =========================
# UI
# =========================
st.title("🚀 AI Crypto Bot PRO")

COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]

timeframe = st.selectbox("Timeframe", ["15 Min","Hourly","Daily"])
symbols = st.multiselect("Coins", COINS, default=COINS[:2])

st.session_state.loading = True

for sym in symbols:

    if sym not in ws_prices:
        start_ws(sym)

    data = get_ohlc(sym, timeframe)
    if not data:
        continue

    fd,o,h,l,c,v = data
    c = np.array(c)

    # Indicators
    rsi = calculate_rsi(c)
    macd, signal_line = calculate_macd(c)
    sentiment = get_sentiment()
    ai_model = train_ai(c)
    ai_prob = ai_predict(ai_model,c)

    # Smart Signal
    signal, trend, volatility = generate_signal(c, v, rsi, macd, signal_line, ai_prob)
    confidence = calculate_confidence(rsi, ai_prob, volatility)

    # UI Card
    with st.container():
        st.markdown("<div style='border:1px solid white;padding:10px;border-radius:8px;margin-bottom:10px;'>",unsafe_allow_html=True)
        st.markdown(f"### {sym}")

        col1,col2 = st.columns([3,1])

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=fd,open=o,high=h,low=l,close=c))
            fig.update_layout(dragmode="zoom")
            st.plotly_chart(fig,use_container_width=True)

        with col2:
            color = "white"
            if "BUY" in signal: color="green"
            if "SELL" in signal: color="red"

            st.markdown(f"**Signal:** <span style='color:{color}'>{signal}</span>", unsafe_allow_html=True)
            st.write(f"Trend: {trend}")
            st.write(f"Confidence: {round(confidence,2)}%")
            st.write(f"RSI: {round(rsi[-1],2)}")
            st.write(f"AI: {round(ai_prob*100,2)}%")
            st.write(f"Volatility: {round(volatility,4)}")
            st.write(f"Sentiment: {round(sentiment,2)}")

        st.markdown("</div>",unsafe_allow_html=True)

st.session_state.loading = False
