import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import threading, websocket, sqlite3
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# Auto Refresh
# =========================
if "loading" not in st.session_state:
    st.session_state.loading = False

if not st.session_state.loading:
    st_autorefresh(interval=300000, key="refresh")  # 5 min

# =========================
# Globals
# =========================
ws_prices = {}
demo_trades = []
virtual_capital = 10000

# SQLite DB for trade logging
conn = sqlite3.connect('trade_history.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS trades (
                    coin TEXT, amount REAL, entry REAL, exit REAL, 
                    mode TEXT, sl REAL, tp REAL, profit REAL, timestamp TEXT)''')
conn.commit()

# =========================
# WebSocket price updates
# =========================
def start_ws(symbol):
    def on_message(ws, message):
        try:
            data = eval(message)
            if 'c' in data:
                ws_prices[symbol] = float(data['c'])
        except:
            pass
    url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker"
    ws = websocket.WebSocketApp(url, on_message=on_message)
    threading.Thread(target=ws.run_forever, daemon=True).start()

# =========================
# Data Fetching (Old way)
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

# =========================
# Indicators
# =========================
def rsi(prices):
    delta=np.diff(prices)
    gain=np.maximum(delta,0)
    loss=np.abs(np.minimum(delta,0))
    rs=pd.Series(gain).rolling(14).mean()/(pd.Series(loss).rolling(14).mean()+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def macd(prices):
    exp1=pd.Series(prices).ewm(span=12).mean()
    exp2=pd.Series(prices).ewm(span=26).mean()
    m=exp1-exp2
    s=m.ewm(span=9).mean()
    return m.values,s.values

# =========================
# LSTM Model
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
# Sentiment Analysis
# =========================
@st.cache_data(ttl=900)
def get_sentiment():
    try:
        data=requests.get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN").json()
        scores=[]
        for a in data.get("Data",[])[:5]:
            text=a["title"]
            v=SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
            b=TextBlob(text).sentiment.polarity
            scores.append((v+b)/2)
        return np.mean(scores) if scores else 0
    except:
        return 0

# =========================
# Demo Trade Logic
# =========================
def add_demo_trade(coin, amount, entry, mode, sl, tp):
    demo_trades.append({
        'coin': coin, 'amount': amount, 'entry': entry, 'mode': mode, 'sl': sl, 'tp': tp
    })

# =========================
# UI Setup
# =========================
st.title("🚀 AI Crypto Bot Demo")
COINS=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]

st.sidebar.header("Demo Settings")
virtual_capital = st.sidebar.number_input("Virtual Capital (USD)", 1000, 100000, value=10000)

timeframe=st.selectbox("Select Timeframe",["15 Min","Hourly","Daily"])
symbols=st.multiselect("Select Coins",COINS,default=["BTCUSDT","ETHUSDT"])

# Debug panel
debug=[]

# =========================
# Main Loop
# =========================
st.session_state.loading = True
for sym in symbols:
    if sym not in ws_prices:
        start_ws(sym)

    data=get_ohlc_binance(sym,"1h",72)
    if not data:
        debug.append(f"{sym}: Failed to fetch")
        continue
    fd,o,h,l,c,v=data
    c=np.array(c)

    r=rsi(c)
    m,s=macd(c)
    sentiment=get_sentiment()
    model,scaler=train_lstm(c)
    pred=lstm_predict(model,scaler,c)

    # =========================
    # Chart
    # =========================
    fig=go.Figure()
    fig.add_trace(go.Candlestick(x=fd,open=o,high=h,low=l,close=c,name=sym))
    fig.add_trace(go.Scatter(x=fd,y=[pred]*len(fd),line=dict(color="yellow"),name="Prediction"))
    fig.update_layout(dragmode="zoom")
    st.plotly_chart(fig,use_container_width=True)

    # =========================
    # Demo Panel
    # =========================
    st.markdown(f"### Demo Trading for {sym}")
    mode = st.selectbox(f"Mode ({sym})",["Trade","Futures"])
    entry_price = st.number_input(f"Entry Price ({sym})",value=float(c[-1]))
    sl_perc = st.number_input(f"Stop Loss % ({sym})",value=2.0)
    tp_perc = st.number_input(f"Take Profit % ({sym})",value=5.0)

    if st.button(f"Add Demo Trade ({sym})"):
        amount = virtual_capital*0.1/entry_price  # 10% capital default
        sl = entry_price*(1-sl_perc/100)
        tp = entry_price*(1+tp_perc/100)
        add_demo_trade(sym,amount,entry_price,mode,sl,tp)
        st.success(f"Added demo trade for {sym}: {mode}")

st.session_state.loading=False

# =========================
# Trade History Panel
# =========================
st.markdown("### Demo Trade History")
if demo_trades:
    df=pd.DataFrame(demo_trades)
    st.dataframe(df)
else:
    st.write("No demo trades yet")

# =========================
# Debug Panel
# =========================
with st.expander("🧰 Debug Panel"):
    if not debug:
        st.write("No debug messages")
    else:
        for d in debug:
            st.write(d)
