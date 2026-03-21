import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import threading, websocket
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sqlite3
import uuid

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
notes TEXT
)
""")

def add_column_safe(name, coltype):
    try:
        cursor.execute(f"ALTER TABLE trades ADD COLUMN {name} {coltype}")
    except:
        pass

add_column_safe("status","TEXT DEFAULT 'OPEN'")
add_column_safe("tp_price","REAL DEFAULT 0")
add_column_safe("sl_price","REAL DEFAULT 0")
add_column_safe("trade_id","TEXT")

conn.commit()

# =========================
# AUTO REFRESH
# =========================
if "loading" not in st.session_state:
    st.session_state.loading = False

if not st.session_state.loading:
    st_autorefresh(interval=300000,key="refresh")

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
        return 0.5  # neutral confidence

    wins = len(df[df["pl"] > 0])
    losses = len(df[df["pl"] <= 0])

    total = wins + losses

    if total == 0:
        return 0.5

    win_rate = wins / total

    return win_rate

# =========================
# PER COIN TIMER
# =========================
def can_auto_trade_coin(symbol):

    df = pd.read_sql_query(
        "SELECT * FROM trades WHERE coin=?",
        conn,
        params=(symbol,)
    )

    if df.empty:
        return True

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    last_trade_time = df["timestamp"].max()

    if datetime.now() - last_trade_time > timedelta(minutes=TRADE_INTERVAL_MINUTES):
        return True

    return False

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
# MULTI-SOURCE DATA FETCH
# =========================
def fetch_binance(symbol, interval, limit):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data = requests.get(url, params=params, timeout=5).json()
        if not isinstance(data, list):
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
    except:
        return None

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

def fetch_kraken(symbol,timeframe):
    try:
        coin = symbol.replace("USDT","USD")
        pair_map = {"BTCUSD":"XXBTZUSD","ETHUSD":"XETHZUSD"}
        pair = pair_map.get(coin, coin)
        tf_map={"15 Min":15,"Hourly":60,"Daily":1440}
        interval=tf_map[timeframe]
        url="https://api.kraken.com/0/public/OHLC"
        params={"pair":pair,"interval":interval}
        data=requests.get(url, params=params, timeout=5).json()
        result=data.get("result",{})
        key=list(result.keys())[0]
        candles=result[key]
        d,o,h,l,c,v=[],[],[],[],[],[]
        for k in candles:
            d.append(datetime.fromtimestamp(int(k[0])))
            o.append(float(k[1]))
            h.append(float(k[2]))
            l.append(float(k[3]))
            c.append(float(k[4]))
            v.append(float(k[6]))
        return d,o,h,l,c,v
    except:
        return None

def get_ohlc(symbol,timeframe,debug):
    mapping={"15 Min":("15m",96),"Hourly":("1h",72),"Daily":("1d",180)}
    interval,limit=mapping[timeframe]
    data=fetch_binance(symbol, interval, limit)
    if data:
        debug.append(f"{symbol}: Binance OK")
        return data
    data=fetch_coinbase(symbol, timeframe)
    if data:
        debug.append(f"{symbol}: Coinbase OK")
        return data
    data=fetch_kraken(symbol, timeframe)
    if data:
        debug.append(f"{symbol}: Kraken OK")
        return data
    debug.append(f"{symbol}: All Sources Failed")
    return None
# =========================
# INDICATORS
# =========================
def rsi(prices):

    delta=np.diff(prices)

    gain=np.maximum(delta,0)
    loss=np.abs(np.minimum(delta,0))

    rs=pd.Series(gain).rolling(14).mean() / (
        pd.Series(loss).rolling(14).mean()+1e-9
    )

    return np.concatenate(
        [[50],100-(100/(1+rs))]
    )

def macd(prices):

    exp1=pd.Series(prices).ewm(span=12).mean()
    exp2=pd.Series(prices).ewm(span=26).mean()

    m=exp1-exp2
    s=m.ewm(span=9).mean()

    return m.values,s.values

# =========================
# LSTM
# =========================
@st.cache_resource(ttl=1800)
def train_lstm(prices):

    scaler=MinMaxScaler()

    data=scaler.fit_transform(
        np.array(prices).reshape(-1,1)
    )

    X,y=[],[]

    for i in range(20,len(data)):
        X.append(data[i-20:i])
        y.append(data[i])

    X,y=np.array(X),np.array(y)

    model=Sequential([
        LSTM(50,input_shape=(20,1)),
        Dense(1)
    ])

    model.compile("adam","mse")

    model.fit(X,y,epochs=3,verbose=0)

    return model,scaler

def lstm_predict(model,scaler,prices):

    data=scaler.transform(
        np.array(prices).reshape(-1,1)
    )

    seq=data[-20:]

    pred=model.predict(
        seq.reshape(1,20,1),
        verbose=0
    )

    return scaler.inverse_transform(pred)[0][0]

# =========================
# AI DECISION WITH LEARNING
# =========================
def ai_trade_decision(
    symbol,
    prices,
    rsi_vals,
    macd_vals,
    signal_vals,
    prediction
):

    price=prices[-1]

    r=rsi_vals[-1]
    m=macd_vals[-1]
    s=signal_vals[-1]

    learning_bias=get_learning_stats(symbol)

    if (
        r<35 and
        m>s and
        prediction>price and
        learning_bias>0.4
    ):
        return "BUY"

    if (
        r>70 and
        m<s and
        prediction<price and
        learning_bias>0.4
    ):
        return "SELL"

    return "HOLD"

# =========================
# AUTO TRADE
# =========================
def execute_auto_trade(sym,decision,price,confidence):

    if decision=="HOLD":
        return

    trade_id=str(uuid.uuid4())

    amount=0.01

    tp_percent=1.5
    sl_percent=1.0

    if decision=="BUY":

        tp_price=price*(1+tp_percent/100)
        sl_price=price*(1-sl_percent/100)

    else:

        tp_price=price*(1-tp_percent/100)
        sl_price=price*(1+sl_percent/100)

    cursor.execute("""
    INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,
    (
        datetime.now().isoformat(),
        sym,
        "AUTO_DEMO",
        decision,
        amount,
        price,
        0,
        0,
        confidence,
        tp_percent,
        sl_percent,
        "AI AUTO TRADE",
        "OPEN",
        tp_price,
        sl_price,
        trade_id
    ))

    conn.commit()

# =========================
# TP/SL CLOSE
# =========================
def check_open_trades():

    df=pd.read_sql_query(
        "SELECT * FROM trades WHERE status='OPEN'",
        conn
    )

    if df.empty:
        return

    for i,row in df.iterrows():

        coin=row["coin"]

        if coin not in ws_prices:
            continue

        price=ws_prices[coin]

        entry=row["entry_price"]
        amount=row["amount"]

        tp_price=row["tp_price"]
        sl_price=row["sl_price"]

        action=row["action"]
        trade_id=row["trade_id"]

        close=False

        if action=="BUY":

            if price>=tp_price:
                close=True

            if price<=sl_price:
                close=True

            pl=(price-entry)*amount

        elif action=="SELL":

            if price<=tp_price:
                close=True

            if price>=sl_price:
                close=True

            pl=(entry-price)*amount

        if close:

            cursor.execute("""
            UPDATE trades
            SET exit_price=?,pl=?,status='CLOSED'
            WHERE trade_id=?
            """,
            (price,pl,trade_id))

    conn.commit()

# =========================
# UI
# =========================
st.title("🚀 AI Crypto Bot Demo")

auto_mode=st.sidebar.toggle(
"🤖 Enable Auto Demo Trading",
value=False
)

COINS=[
"BTCUSDT",
"ETHUSDT",
"BNBUSDT",
"ADAUSDT",
"SOLUSDT"
]

timeframe=st.selectbox(
"Select Timeframe",
["15 Min","Hourly","Daily"]
)

symbols=st.multiselect(
"Select Coins",
COINS,
default=["BTCUSDT"]
)

debug=[]

st.session_state.loading=True

# =========================
# MAIN LOOP
# =========================
for sym in symbols:

    if sym not in ws_prices:
        start_ws(sym)

    data=get_ohlc(sym,timeframe,debug)

    if not data:
        continue

    fd,o,h,l,c,v=data

    c=np.array(c)

    r=rsi(c)
    m,s=macd(c)

    model,scaler=train_lstm(c)

    pred=lstm_predict(model,scaler,c)

    conf=100*(1-np.std(c)/c[-1])

    decision=ai_trade_decision(
        sym,
        c,
        r,
        m,
        s,
        pred
    )

    if auto_mode:

        if can_auto_trade_coin(sym):

            execute_auto_trade(
                sym,
                decision,
                c[-1],
                conf
            )

check_open_trades()

st.markdown("### Trade History")

df=pd.read_sql_query(
"SELECT * FROM trades ORDER BY timestamp DESC",
conn
)

st.dataframe(df)

with st.expander("🧰 Debug Panel"):

    if not debug:
        st.write("No debug messages")

    else:

        for d in debug:
            st.write(d)

st.session_state.loading=False
