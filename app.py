import os
import nltk

# =========================
# NLTK DATA SETUP FOR TEXTBLOB
# =========================
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

for pkg in ['punkt', 'averaged_perceptron_tagger', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_path)

# =========================
# IMPORTS
# =========================
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import threading, websocket, json
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sqlite3
import uuid
import random

# =========================
# DATABASE SETUP (SAFE)
# =========================
conn = sqlite3.connect("trade_history.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS trades (
timestamp TEXT,
coin TEXT,
mode TEXT,
action TEXT,
amount REAL,
entry_price REAL,
exit_price REAL,
pl REAL,
confidence REAL,
tp REAL,
sl REAL,
notes TEXT,
status TEXT DEFAULT 'OPEN',
tp_price REAL DEFAULT 0,
sl_price REAL DEFAULT 0,
trade_id TEXT
)
""")
conn.commit()

# =========================
# AUTO REFRESH
# =========================
if "loading" not in st.session_state:
    st.session_state.loading = False

if not st.session_state.loading:
    st_autorefresh(interval=30000,key="refresh")  # refresh every 30s

# =========================
# GLOBALS
# =========================
ws_prices = {}
TRADE_INTERVAL_MINUTES = 20

# =========================
# LEARNING MEMORY
# =========================
def get_learning_stats(symbol):
    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE coin=? AND status='CLOSED'",
        conn,
        params=(symbol,)
    )
    if df.empty:
        return 0.5
    wins = len(df[df["pl"] > 0])
    losses = len(df[df["pl"] <= 0])
    total = wins + losses
    return wins / total if total > 0 else 0.5

# =========================
# PER COIN TIMER
# =========================
def can_auto_trade_coin(symbol):
    df = pd.read_sql_query("SELECT * FROM trades WHERE coin=?", conn, params=(symbol,))
    if df.empty:
        return True
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    last_trade_time = df["timestamp"].max()
    return datetime.now() - last_trade_time > timedelta(minutes=TRADE_INTERVAL_MINUTES)

# =========================
# WEBSOCKET LIVE PRICE (SAFE)
# =========================
def start_ws(symbol):
    def on_message(ws, message):
        try:
            data = json.loads(message)
            if "c" in data:
                ws_prices[symbol] = float(data["c"])
        except:
            pass
    url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"  # optional, keep for live updates
    ws = websocket.WebSocketApp(url, on_message=on_message)
    threading.Thread(target=ws.run_forever, daemon=True).start()

# =========================
# MULTI-SOURCE DATA FETCH (COINBASE ONLY)
# =========================
def fetch_coinbase(symbol, timeframe):
    try:
        coin = symbol.replace("USDT","-USD")
        tf_map = {"15 Min":900,"Hourly":3600,"Daily":86400}
        gran = tf_map[timeframe]
        url = f"https://api.exchange.coinbase.com/products/{coin}/candles"
        params = {"granularity": gran}
        data = requests.get(url, params=params, timeout=5).json()
        if not isinstance(data,list):
            return None
        data.reverse()
        d,o,h,l,c,v=[],[],[],[],[],[]
        for k in data:
            d.append(datetime.fromtimestamp(k[0]))
            o.append(float(k[3]))
            h.append(float(k[2]))
            l.append(float(k[1]))
            c.append(float(k[4]))
            v.append(float(k[5]))
        return d,o,h,l,c,v
    except:
        return None

def get_ohlc(symbol, timeframe, debug):
    data = fetch_coinbase(symbol, timeframe)
    if data:
        return data
    debug.append(f"{symbol}: Coinbase data failed")
    return None

# =========================
# INDICATORS
# =========================
def rsi(prices):
    delta=np.diff(prices)
    gain=np.maximum(delta,0)
    loss=np.abs(np.minimum(delta,0))
    rs=pd.Series(gain).rolling(14).mean() / (pd.Series(loss).rolling(14).mean()+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def macd(prices):
    exp1=pd.Series(prices).ewm(span=12).mean()
    exp2=pd.Series(prices).ewm(span=26).mean()
    m=exp1-exp2
    s=m.ewm(span=9).mean()
    return m.values,s.values

# =========================
# LSTM TRAIN & PREDICT
# =========================
@st.cache_resource(ttl=1800)
def train_lstm(prices):
    scaler=MinMaxScaler()
    data=scaler.fit_transform(np.array(prices).reshape(-1,1))
    X,y=[],[]
    for i in range(20,len(data)):
        X.append(data[i-20:i])
        y.append(data[i])
    X,y=np.array(X),np.array(y)
    model=Sequential([LSTM(50,input_shape=(20,1)),Dense(1)])
    model.compile("adam","mse")
    model.fit(X,y,epochs=3,verbose=0)
    return model,scaler

def lstm_predict(model,scaler,prices):
    data=scaler.transform(np.array(prices).reshape(-1,1))
    seq=data[-20:]
    pred=model.predict(seq.reshape(1,20,1),verbose=0)
    return scaler.inverse_transform(pred)[0][0]

# =========================
# AI DECISION
# =========================
def ai_trade_decision(symbol, prices, rsi_vals, macd_vals, signal_vals, prediction):
    price=prices[-1]
    r=rsi_vals[-1]
    m=macd_vals[-1]
    s=signal_vals[-1]
    learning_bias=get_learning_stats(symbol)
    if r<35 and m>s and prediction>price and learning_bias>0.4:
        return "BUY"
    if r>70 and m<s and prediction>price and learning_bias>0.4:
        return "SELL"
    return "HOLD"

# =========================
# SIMULATED INFINITE AUTO TRADE
# =========================
def execute_auto_trade(sym, decision, price, confidence):
    trade_id=str(uuid.uuid4())
    amount=1.0
    tp_percent=1.5
    sl_percent=1.0
    if decision=="BUY":
        tp_price=price*(1+tp_percent/100)
        sl_price=price*(1-sl_percent/100)
    elif decision=="SELL":
        tp_price=price*(1-tp_percent/100)
        sl_price=price*(1+sl_percent/100)
    else:
        return
    cursor.execute("""
    INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        datetime.now().isoformat(), sym, "AUTO_SIM", decision, amount, price, 0, 0,
        confidence, tp_percent, sl_percent, "Simulated AI Trade","OPEN",tp_price,sl_price,trade_id
    ))
    conn.commit()

# =========================
# MAIN APP UI
# =========================
st.title("🚀 AI Crypto Bot Demo")

COINS=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]
timeframe=st.selectbox("Select Timeframe", ["15 Min","Hourly","Daily"])
symbols=st.multiselect("Select Coins", COINS, default=["BTCUSDT"])

debug=[]
st.session_state.loading=True

# =========================
# PER COIN DISPLAY (WHITE CARDS)
# =========================
for sym in symbols:
    # Card container
    st.markdown(f"""
        <div style="border:2px solid white; padding:10px; border-radius:10px; margin-bottom:20px;">
    """, unsafe_allow_html=True)

    # Coin title + checkbox
    col_title, col_checkbox = st.columns([3,1])
    with col_title:
        st.markdown(f"### {sym}")
    with col_checkbox:
        auto_mode=st.checkbox("🤖 Enable Auto Demo Trading", key=f"auto_{sym}", value=True)

    if sym not in ws_prices:
        start_ws(sym)

    data=get_ohlc(sym,timeframe,debug)
    if not data:
        st.write("Data not available")
        st.markdown("</div>", unsafe_allow_html=True)
        continue

    fd,o,h,l,c,v=data
    c=np.array(c)
    r=rsi(c)
    m,s=macd(c)
    model,scaler=train_lstm(c)
    pred=lstm_predict(model,scaler,c)
    conf=100*(1-np.std(c)/c[-1])
    decision=ai_trade_decision(sym,c,r,m,s,pred)

    # Simulated trades
    if auto_mode:
        for _ in range(random.randint(1,3)):
            execute_auto_trade(sym, random.choice(["BUY","SELL"]), c[-1], conf)

    # Chart + stats side by side
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=fd, open=o, high=h, low=l, close=c))
    fig.update_layout(height=300,margin=dict(l=0,r=0,t=20,b=20))
    stats_df = pd.DataFrame({
        "Metric":["Current Price","Prediction","RSI","MACD Signal","Decision"],
        "Value":[c[-1], pred, r[-1], m[-1]-s[-1], decision]
    })
    col1,col2=st.columns([2,1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.table(stats_df)

    st.markdown("#### AI Demo Trades")
    df=pd.read_sql_query(f"SELECT * FROM trades WHERE coin='{sym}' ORDER BY timestamp DESC", conn)
    st.dataframe(df)

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

st.session_state.loading=False
