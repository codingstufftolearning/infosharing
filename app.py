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

import firebase_admin
from firebase_admin import credentials, db

# =========================
# AUTO REFRESH
# =========================
st_autorefresh(interval=300000, key="refresh")

# =========================
# FIREBASE INIT
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
        st.error(f"Firebase error: {e}")

# =========================
# GLOBALS
# =========================
ws_prices = {}

# =========================
# SESSION INIT
# =========================
if "balance" not in st.session_state:
    st.session_state.balance = 10000

if "positions" not in st.session_state:
    st.session_state.positions = {}

if "trade_history" not in st.session_state:
    st.session_state.trade_history = []

# =========================
# SAFE REQUEST
# =========================
def safe_request(url, params=None):
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            t.sleep(2)
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
# MULTI TF DATA
# =========================
def get_multi_tf(symbol):
    tfs = {
        "15m": ("15m", 96),
        "1h": ("1h", 72),
        "4h": ("4h", 60)
    }

    results = {}

    for tf, (interval, limit) in tfs.items():
        url = "https://api.binance.com/api/v3/klines"
        data = safe_request(url, {"symbol":symbol,"interval":interval,"limit":limit})
        if not data:
            return None

        closes = [float(k[4]) for k in data]
        volumes = [float(k[5]) for k in data]

        results[tf] = (np.array(closes), volumes, data)

    return results

# =========================
# INDICATORS
# =========================
def rsi(prices):
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    rs = pd.Series(gain).rolling(14).mean() / (pd.Series(loss).rolling(14).mean()+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def macd(prices):
    exp1 = pd.Series(prices).ewm(span=12).mean()
    exp2 = pd.Series(prices).ewm(span=26).mean()
    m = exp1-exp2
    s = m.ewm(span=9).mean()
    return m.values, s.values

# =========================
# AI
# =========================
def train_ai(prices):
    if len(prices)<50:
        return None
    X,y=[],[]
    for i in range(20,len(prices)-1):
        w = prices[i-20:i]
        X.append([np.mean(w),np.std(w),w[-1]-w[0]])
        y.append(1 if prices[i+1]>prices[i] else 0)
    m = LogisticRegression()
    m.fit(X,y)
    return m

def ai_predict(m, prices):
    if not m or len(prices)<20:
        return 0.5
    w = prices[-20:]
    return m.predict_proba([[np.mean(w),np.std(w),w[-1]-w[0]]])[0][1]

# =========================
# SIGNAL
# =========================
def generate_signal(c, v):
    r = rsi(c)
    m, s = macd(c)
    ai = ai_predict(train_ai(c), c)

    if r[-1] < 35 and m[-1] > s[-1] and ai > 0.55:
        return "BUY"
    if r[-1] > 65 and m[-1] < s[-1] and ai < 0.45:
        return "SELL"
    return "HOLD"

def multi_tf_signal(symbol):
    data = get_multi_tf(symbol)
    if not data:
        return "HOLD"

    signals = []
    for tf in data:
        c, v, _ = data[tf]
        signals.append(generate_signal(c, v))

    if signals.count("BUY") >= 2:
        return "BUY"
    if signals.count("SELL") >= 2:
        return "SELL"
    return "HOLD"

# =========================
# TRADING ENGINE
# =========================
RISK = 0.1

def open_position(sym, signal, price):
    capital = st.session_state.balance * RISK

    st.session_state.positions[sym] = {
        "type": "LONG" if signal=="BUY" else "SHORT",
        "entry": price,
        "size": capital,
        "sl": price*0.97 if signal=="BUY" else price*1.03,
        "tp": price*1.05 if signal=="BUY" else price*0.95
    }

def close_position(sym, price):
    pos = st.session_state.positions.get(sym)
    if not pos:
        return

    entry = pos["entry"]
    size = pos["size"]

    if pos["type"]=="LONG":
        pct = (price-entry)/entry
    else:
        pct = (entry-price)/entry

    profit = size * pct
    st.session_state.balance += profit

    trade = {
        "symbol": sym,
        "type": pos["type"],
        "entry": entry,
        "exit": price,
        "profit": profit,
        "time": str(datetime.now())
    }

    st.session_state.trade_history.append(trade)

    # SAVE TO FIREBASE
    try:
        db.reference("trades").push(trade)
    except:
        pass

    del st.session_state.positions[sym]

# =========================
# UI
# =========================
st.title("🚀 AI Crypto Bot PRO DEMO")

COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","SOLUSDT"]

st.sidebar.header("🧪 Demo Trading")

if st.sidebar.button("Reset"):
    st.session_state.balance = 10000
    st.session_state.positions = {}
    st.session_state.trade_history = []

st.sidebar.write(f"Balance: ${round(st.session_state.balance,2)}")
st.sidebar.write(f"Open Positions: {len(st.session_state.positions)}")

# =========================
# MAIN LOOP
# =========================
for sym in COINS:

    if sym not in ws_prices:
        start_ws(sym)

    data = get_multi_tf(sym)
    if not data:
        continue

    c = data["15m"][0]
    raw = data["15m"][2]

    fd = [datetime.fromtimestamp(k[0]/1000) for k in raw]
    o = [float(k[1]) for k in raw]
    h = [float(k[2]) for k in raw]
    l = [float(k[3]) for k in raw]

    signal = multi_tf_signal(sym)
    price = c[-1]

    # Trading logic
    pos = st.session_state.positions.get(sym)

    if not pos and signal in ["BUY","SELL"]:
        open_position(sym, signal, price)

    if pos:
        if pos["type"]=="LONG" and (price<=pos["sl"] or price>=pos["tp"]):
            close_position(sym, price)
        elif pos["type"]=="SHORT" and (price>=pos["sl"] or price<=pos["tp"]):
            close_position(sym, price)

    # UI
    with st.container():
        st.markdown(f"### {sym}")
        col1,col2 = st.columns([3,1])

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=fd,open=o,high=h,low=l,close=c))
            st.plotly_chart(fig,use_container_width=True)

        with col2:
            st.write(f"Signal: {signal}")
            st.write(f"Price: {round(price,2)}")

# =========================
# PERFORMANCE
# =========================
trades = st.session_state.trade_history

if trades:
    wins = sum(1 for t in trades if t["profit"]>0)
    total = len(trades)
    winrate = wins/total*100
    profit = sum(t["profit"] for t in trades)

    st.markdown("### 📊 Performance")
    st.write(f"Trades: {total}")
    st.write(f"Winrate: {round(winrate,2)}%")
    st.write(f"Profit: ${round(profit,2)}")

    st.dataframe(pd.DataFrame(trades).tail(10))
