import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import threading, websocket
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# GLOBALS
# =========================
ws_prices = {}
last_heavy_update = datetime.min

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
# DATA FETCHING (LIMITED HISTORY)
# =========================
def get_ohlc_binance(symbol, interval, limit=30):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data = requests.get(url, params=params, timeout=5).json()
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

def get_ohlc(symbol, timeframe, debug):
    mapping = {"15 Min":"15m","Hourly":"1h","Daily":"1d"}
    interval = mapping[timeframe]
    data = get_ohlc_binance(symbol, interval)
    if data:
        debug.append(f"{symbol}: Binance OK")
        return data
    debug.append(f"{symbol}: Fetch failed")
    return None

# =========================
# SIMPLE LSTM MODEL (cached)
# =========================
@st.cache_resource(ttl=1800)
def train_lstm(prices):
    scaler=MinMaxScaler()
    data=scaler.fit_transform(np.array(prices).reshape(-1,1))
    X,y=[],[]
    for i in range(10,len(data)):
        X.append(data[i-10:i])
        y.append(data[i])
    X,y=np.array(X),np.array(y)
    model=Sequential([LSTM(20,input_shape=(10,1)),Dense(1)])
    model.compile("adam","mse")
    model.fit(X,y,epochs=2,verbose=0)
    return model,scaler

def lstm_predict(model,scaler,prices):
    data=scaler.transform(np.array(prices).reshape(-1,1))
    seq=data[-10:]
    pred=model.predict(seq.reshape(1,10,1),verbose=0)
    return scaler.inverse_transform(pred)[0][0]

# =========================
# UI SETUP
# =========================
st.title("🚀 Lightweight AI Crypto Bot")

COINS=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]
symbol = st.selectbox("Select Coin", COINS)
timeframe = st.selectbox("Select Timeframe", ["15 Min","Hourly","Daily"])

st.sidebar.header("💰 Portfolio Tracker")
amt = st.sidebar.number_input(f"{symbol} Amount",0.0, step=0.01)
buy_price = st.sidebar.number_input(f"{symbol} Buy Price",0.0, step=0.01)

portfolio = {"amount": amt, "buy_price": buy_price, "pl":0}
debug=[]

# =========================
# START WEBSOCKET
# =========================
if symbol not in ws_prices:
    start_ws(symbol)

# =========================
# FETCH DATA
# =========================
data = get_ohlc(symbol, timeframe, debug)
if not data:
    st.warning("Failed to fetch data")
else:
    fd,o,h,l,c,v=data
    c=np.array(c)

    # =========================
    # HEAVY CALCULATION every 5 min
    # =========================
    now=datetime.utcnow()
    global last_heavy_update
    if (now-last_heavy_update).total_seconds()>300 or last_heavy_update==datetime.min:
        try:
            model,scaler=train_lstm(c)
            pred=lstm_predict(model,scaler,c)
        except:
            pred=c[-1]
        last_heavy_update=now
    else:
        pred=c[-1]

    # =========================
    # CARD UI
    # =========================
    with st.container():
        st.markdown("<div style='border:1px solid white;padding:10px;border-radius:8px;margin-bottom:10px;'>",unsafe_allow_html=True)
        st.markdown(f"### {symbol}")

        col1,col2=st.columns([3,1])

        with col1:
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=fd, open=o, high=h, low=l, close=c))
            fig.add_trace(go.Scatter(x=fd, y=[pred]*len(fd), line=dict(color="yellow"), name="Prediction"))
            fig.update_layout(dragmode="zoom")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write(f"Prediction: {round(pred,2)}")
            if amt>0 and buy_price>0:
                portfolio["pl"]=(c[-1]-buy_price)*amt
            st.write(f"P/L: {round(portfolio['pl'],2)}")

        st.markdown("</div>", unsafe_allow_html=True)

# =========================
# DEBUG PANEL
# =========================
with st.expander("🧰 Debug Panel"):
    if not debug:
        st.write("No debug messages")
    else:
        for d in debug:
            st.write(d)
