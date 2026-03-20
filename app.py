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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.linear_model import LogisticRegression

# =========================
# AUTO REFRESH
# =========================
if "loading" not in st.session_state:
    st.session_state.loading = False

if not st.session_state.loading:
    st_autorefresh(interval=300000, key="refresh")  # 5 min

# =========================
# GLOBALS
# =========================
ws_prices = {}

# =========================
# SAFE REQUEST WITH RETRY
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
# WEBSOCKET LIVE PRICE
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
# FETCH OHLC DATA
# =========================
def get_ohlc_binance(symbol, interval, limit):
    try:
        url = "https://api.binance.com/api/v3/klines"
        data = safe_request(url, {"symbol": symbol,"interval":interval,"limit":limit})
        if not data: return None
        d,o,h,l,c,v=[],[],[],[],[],[]
        for k in data:
            d.append(datetime.fromtimestamp(k[0]/1000))
            o.append(float(k[1])); h.append(float(k[2]))
            l.append(float(k[3])); c.append(float(k[4]))
            v.append(float(k[5]))
        return d,o,h,l,c,v
    except:
        return None

def get_ohlc(symbol, timeframe):
    mapping = {"15 Min":("15m",96),"Hourly":("1h",72),"Daily":("1d",180)}
    interval, limit = mapping[timeframe]
    return get_ohlc_binance(symbol, interval, limit)  # simplified for example

# =========================
# INDICATORS
# =========================
def calculate_rsi(prices, period=14):
    prices = np.array(prices)
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(period,min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period,min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def calculate_macd(prices):
    prices = np.array(prices)
    exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.values, signal.values

# =========================
# LSTM MODEL
# =========================
@st.cache_resource(ttl=3600)
def train_lstm(prices):
    if len(prices)<30: return None, None
    scaler = MinMaxScaler()
    data = scaler.fit_transform(np.array(prices).reshape(-1,1))
    X, y = [], []
    for i in range(20, len(data)):
        X.append(data[i-20:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    model = Sequential([Input(shape=(20,1)), LSTM(50), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=3, verbose=0)
    return model, scaler

def lstm_predict(model, scaler, prices):
    if model is None or scaler is None: return prices[-1]
    data = scaler.transform(np.array(prices).reshape(-1,1))
    seq = data[-20:]
    pred = model.predict(seq.reshape(1,20,1), verbose=0)
    return scaler.inverse_transform(pred)[0][0]

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
# SUPPORT / RESISTANCE
# =========================
def support_resistance(prices):
    return min(prices[-50:]), max(prices[-50:])

# =========================
# SPIKE
# =========================
def detect_spike(prices):
    if len(prices)<5: return "Normal"
    change = (prices[-1]-prices[-5])/prices[-5]
    if change<-0.05: return "🔻 Sharp Drop"
    if change>0.05: return "🚀 Sharp Rise"
    return "Normal"

# =========================
# UI
# =========================
st.title("🚀 AI Crypto Bot Styled Charts")

COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]

st.sidebar.header("💰 Portfolio Tracker")
portfolio={}
total_pl=0
for sym in COINS:
    col1,col2 = st.sidebar.columns(2)
    amt = col1.number_input(f"{sym} Amount",0.0,key=f"{sym}_amt")
    buy_price = col2.number_input(f"{sym} Buy Price",0.0,key=f"{sym}_buy")
    if amt>0 and buy_price>0:
        portfolio[sym] = {"amount": amt, "buy_price": buy_price, "pl":0}

timeframe = st.selectbox("Select Timeframe", ["15 Min","Hourly","Daily"])
symbols = st.multiselect("Select Coins", COINS, default=COINS)

debug_info=[]

st.session_state.loading = True
for sym in symbols:
    if sym not in ws_prices:
        start_ws(sym)
    data = get_ohlc(sym, timeframe)
    if not data: continue
    fd,o,h,l,c,v = data
    c=np.array(c)

    # Indicators
    rsi_val = calculate_rsi(c)
    macd_v, sig_line = calculate_macd(c)
    sentiment = get_sentiment()
    lstm_model, lstm_scaler = train_lstm(c)
    lstm_pred = lstm_predict(lstm_model,lstm_scaler,c)
    sup,res = support_resistance(c)
    spike = detect_spike(c)
    conf = max(0,min(100,100*(1-np.std(c)/c[-1])))

    # =========================
    # CARD UI STYLE (from second script)
    # =========================
    with st.container():
        st.markdown("<div style='border:1px solid white;padding:10px;border-radius:8px;margin-bottom:10px;'>",unsafe_allow_html=True)
        st.markdown(f"### {sym}")

        col1,col2 = st.columns([3,1])

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=fd, open=o, high=h, low=l, close=c))
            fig.add_trace(go.Scatter(x=fd, y=[lstm_pred]*len(fd), line=dict(color="yellow"), name="LSTM Prediction"))
            fig.add_hline(y=sup, line_dash="dash", line_color="green")
            fig.add_hline(y=res, line_dash="dash", line_color="red")
            fig.update_layout(dragmode="zoom")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write(f"LSTM Prediction: {round(lstm_pred,2)}")
            st.write(f"Confidence: {round(conf,2)}%")
            st.write(f"Sentiment: {round(sentiment,2)}")
            st.write(f"Spike: {spike}")

        st.markdown("</div>", unsafe_allow_html=True)

    # Portfolio Update
    if sym in portfolio:
        amt = portfolio[sym]["amount"]
        buy_price = portfolio[sym]["buy_price"]
        portfolio[sym]["pl"] = (c[-1]-buy_price)*amt
        total_pl += portfolio[sym]["pl"]

st.session_state.loading = False

# =========================
# PORTFOLIO SUMMARY
# =========================
st.sidebar.markdown(f"### Total Portfolio P/L: ${round(total_pl,2)}")

# =========================
# DEBUG PANEL
# =========================
with st.expander("🧰 Debug Panel"):
    for line in debug_info:
        st.write(line)
