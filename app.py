import os
import nltk
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import sqlite3
import uuid
import random

# =========================
# NLTK SETUP
# =========================

nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

for pkg in ['punkt','averaged_perceptron_tagger','wordnet','omw-1.4']:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_path)

# =========================
# DATABASE
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
status TEXT,
tp_price REAL,
sl_price REAL,
trade_id TEXT
)
""")

conn.commit()

# =========================
# AUTO REFRESH
# =========================

st_autorefresh(interval=30000, key="refresh")

# =========================
# DATA FETCH
# =========================

debug=[]

def fetch_coinbase(symbol,timeframe):

    try:

        coin=symbol.replace("USDT","-USD")

        tf_map={
            "15 Min":900,
            "Hourly":3600,
            "Daily":86400
        }

        gran=tf_map[timeframe]

        url=f"https://api.exchange.coinbase.com/products/{coin}/candles"

        params={"granularity":gran}

        data=requests.get(url,params=params,timeout=5).json()

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

        coin=symbol.replace("USDT","USD")

        pair_map={
            "BTCUSD":"XXBTZUSD",
            "ETHUSD":"XETHZUSD"
        }

        pair=pair_map.get(coin,coin)

        tf_map={
            "15 Min":15,
            "Hourly":60,
            "Daily":1440
        }

        interval=tf_map[timeframe]

        url="https://api.kraken.com/0/public/OHLC"

        params={
            "pair":pair,
            "interval":interval
        }

        data=requests.get(url,params=params,timeout=5).json()

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


def get_ohlc(symbol,timeframe):

    data=fetch_coinbase(symbol,timeframe)

    if data:
        return data

    data=fetch_kraken(symbol,timeframe)

    if data:
        return data

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

    return np.concatenate([[50],100-(100/(1+rs))])


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
# LEARNING STATS
# =========================

def get_learning_stats(symbol):

    df=pd.read_sql_query(
        "SELECT * FROM trades WHERE coin=?",
        conn,
        params=(symbol,)
    )

    if df.empty:
        return 0.5

    wins=len(df[df["pl"]>0])

    losses=len(df[df["pl"]<=0])

    total=wins+losses

    if total==0:
        return 0.5

    return wins/total

# =========================
# AI DECISION (RELAXED)
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

    if r<45 and m>s and prediction>price:
        return "BUY"

    if r>60 and m<s and prediction<price:
        return "SELL"

    return "HOLD"

# =========================
# DEMO TRADE EXECUTION
# =========================

def execute_demo_trade(sym,decision,price,conf):

    trade_id=str(uuid.uuid4())

    amount=random.uniform(0.01,0.05)

    move=random.uniform(-0.02,0.02)

    if decision=="BUY":

        exit_price=price*(1+move)

        pl=(exit_price-price)*amount

    elif decision=="SELL":

        exit_price=price*(1-move)

        pl=(price-exit_price)*amount

    else:

        return

    cursor.execute("""
    INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """,(

        datetime.now().isoformat(),

        sym,

        "AUTO_DEMO",

        decision,

        amount,

        price,

        exit_price,

        pl,

        conf,

        1.5,

        1.0,

        "AI DEMO AUTO",

        "CLOSED",

        exit_price*1.015,

        exit_price*0.985,

        trade_id

    ))

    conn.commit()

# =========================
# STREAMLIT UI
# =========================

st.title("🚀 AI Crypto Bot Demo")

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

# =========================
# MAIN LOOP
# =========================

for sym in symbols:

    col1,col2=st.columns([3,2])

    with col1:
        st.subheader(sym)

    with col2:
        auto_mode=st.toggle(
            "🤖 Auto Demo Trading",
            value=True,
            key=f"auto_{sym}"
        )

    data=get_ohlc(sym,timeframe)

    if not data:
        st.warning("No data available")
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

    # =========================
    # EXPLORATION MODE
    # =========================

    if auto_mode:

        learning_bias=get_learning_stats(sym)

        exploration_rate=max(
            0.3,
            1-learning_bias
        )

        trade_count=random.randint(2,5)

        for _ in range(trade_count):

            trade_decision=decision

            if decision=="HOLD":

                if random.random()<exploration_rate:

                    trade_decision=random.choice(
                        ["BUY","SELL"]
                    )

                else:

                    continue

            execute_demo_trade(
                sym,
                trade_decision,
                c[-1],
                conf
            )

    # =========================
    # CHART
    # =========================

    fig=go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=fd,
            open=o,
            high=h,
            low=l,
            close=c
        )
    )

    chart_col,stat_col=st.columns([3,1])

    with chart_col:
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with stat_col:

        st.metric(
            "Last Price",
            f"{c[-1]:.2f}"
        )

        st.metric(
            "Prediction",
            f"{pred:.2f}"
        )

        st.metric(
            "RSI",
            f"{r[-1]:.2f}"
        )

        st.metric(
            "Win Rate",
            f"{get_learning_stats(sym)*100:.2f}%"
        )

    # =========================
    # TRADE TABLE
    # =========================

    df=pd.read_sql_query(
        f"""
        SELECT
        timestamp,
        action,
        entry_price,
        exit_price,
        pl
        FROM trades
        WHERE coin='{sym}'
        ORDER BY timestamp DESC
        LIMIT 50
        """,
        conn
    )

    with st.expander("Trade History"):

        st.dataframe(
            df,
            use_container_width=True
        )

    total=pd.read_sql_query(
        f"SELECT COUNT(*) as c FROM trades WHERE coin='{sym}'",
        conn
    )

    st.caption(
        f"Total Trades: {total['c'][0]}"
    )
