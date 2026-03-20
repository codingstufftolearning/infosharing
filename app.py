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
# SMART AUTO REFRESH
# =========================
if "loading" not in st.session_state:
    st.session_state.loading = False

if not st.session_state.loading:
    st_autorefresh(interval=300000, key="refresh")  # 5 min

# =========================
# DATABASE SETUP
# =========================
conn = sqlite3.connect("demo_trades.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS demo_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    mode TEXT,
    buy_sell TEXT,
    amount REAL,
    price REAL,
    tp REAL,
    sl REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# =========================
# GLOBALS
# =========================
ws_prices = {}

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
# DATA FETCHING (OLD WAY)
# =========================
def get_ohlc_binance(symbol, interval, limit):
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

def get_ohlc_cryptocompare(symbol, interval, limit):
    try:
        fsym = symbol.replace("USDT","")
        if interval=="15m":
            url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={fsym}&tsym=USD&limit={limit}&aggregate=15"
        elif interval=="1h":
            url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={limit}"
        else:
            url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={fsym}&tsym=USD&limit={limit}"
        data = requests.get(url, timeout=5).json()
        raw = data.get("Data", {}).get("Data", [])
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
        data = requests.get(url, timeout=5).json()
        prices = data.get("prices", [])
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

def get_ohlc(symbol, timeframe, debug):
    mapping = {"15 Min":("15m",96),"Hourly":("1h",72),"Daily":("1d",180)}
    interval, limit = mapping[timeframe]

    data = get_ohlc_binance(symbol, interval, limit)
    if data:
        debug.append(f"{symbol}: Binance OK")
        return data
    data = get_ohlc_cryptocompare(symbol, interval, limit)
    if data:
        debug.append(f"{symbol}: CryptoCompare OK")
        return data
    data = get_ohlc_coingecko(symbol)
    if data:
        debug.append(f"{symbol}: CoinGecko OK")
        return data
    debug.append(f"{symbol}: ALL SOURCES FAILED")
    return None

# =========================
# INDICATORS
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
# LSTM MODEL
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
# SENTIMENT
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
# SUPPORT / RESISTANCE
# =========================
def support_resistance(prices):
    return min(prices[-50:]), max(prices[-50:])

# =========================
# SPIKE + EVENT
# =========================
def detect_spike(prices):
    change=(prices[-1]-prices[-5])/prices[-5]
    if change<-0.05: return "🔻 Sharp Drop"
    if change>0.05: return "🚀 Sharp Rise"
    return "Normal"

def detect_event():
    try:
        data=requests.get("https://min-api.cryptocompare.com/data/v2/news/?lang=EN").json()
        events=[]
        for a in data.get("Data",[])[:10]:
            t=a["title"].lower()
            if "fed" in t: events.append("Fed Impact")
            if "regulation" in t: events.append("Regulation")
        return list(set(events))
    except:
        return []

# =========================
# UI SETUP
# =========================
st.title("🚀 AI Crypto Demo Bot")

COINS=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]

timeframe=st.selectbox("Select Timeframe",["15 Min","Hourly","Daily"])
symbols=st.multiselect("Select Coins",COINS,default=["BTCUSDT","ETHUSDT"])

debug=[]
st.session_state.loading = True

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

    # Indicators + LSTM + sentiment
    r=rsi(c)
    m,s=macd(c)
    sentiment=get_sentiment()
    model,scaler=train_lstm(c)
    pred=lstm_predict(model,scaler,c)
    sup,res=support_resistance(c)
    spike=detect_spike(c)
    events=detect_event()
    conf=100*(1-np.std(c)/c[-1])

    # =========================
    # CARD UI
    # =========================
    with st.container():
        st.markdown("<div style='border:1px solid white;padding:10px;border-radius:8px;margin-bottom:10px;'>",unsafe_allow_html=True)
        st.markdown(f"### {sym}")

        col1,col2=st.columns([3,1])

        with col1:
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=fd,open=o,high=h,low=l,close=c))
            fig.add_trace(go.Scatter(x=fd,y=[pred]*len(fd),line=dict(color="yellow"),name="Prediction"))
            fig.add_hline(y=sup,line_dash="dash",line_color="green")
            fig.add_hline(y=res,line_dash="dash",line_color="red")
            fig.update_layout(dragmode="zoom")
            st.plotly_chart(fig,use_container_width=True)

        with col2:
            st.write(f"Prediction: {round(pred,2)}")
            st.write(f"Confidence: {round(conf,2)}%")
            st.write(f"Sentiment: {round(sentiment,2)}")
            st.write(f"Spike: {spike}")
            st.write(f"Events: {events}")

        st.markdown("</div>",unsafe_allow_html=True)

    # =========================
    # DEMO PANEL
    # =========================
    with st.expander(f"🎮 Demo Trading: {sym}"):
        mode = st.radio("Mode", ["Spot","Futures"], key=f"{sym}_mode")

        buy_sell = st.radio("Action", ["Buy","Sell"] if mode=="Spot" else ["Buy Long","Sell Short"], key=f"{sym}_action")
        amount = st.number_input("Amount (coins or contracts)",0.0,key=f"{sym}_amt")
        price = st.number_input("Price (USDT)",0.0,key=f"{sym}_price")
        tp = st.number_input("Take Profit %",0.0,key=f"{sym}_tp")
        sl = st.number_input("Stop Loss %",0.0,key=f"{sym}_sl")
        if st.button("Execute Demo Trade", key=f"{sym}_exec"):
            c.execute("INSERT INTO demo_trades (symbol, mode, buy_sell, amount, price, tp, sl) VALUES (?,?,?,?,?,?,?)",
                      (sym, mode, buy_sell, amount, price, tp, sl))
            conn.commit()
            st.success("Demo trade executed!")

    # =========================
    # DEMO STATISTICS
    # =========================
    trades = c.execute("SELECT buy_sell, amount, price FROM demo_trades WHERE symbol=?", (sym,)).fetchall()
    total_trades = len(trades)
    profit = 0
    wins = 0
    for t in trades:
        action, amt, p = t
        current_price = ws_prices.get(sym, c[-1])
        if action in ["Buy","Buy Long"]:
            pnl = (current_price - p) * amt / p
        else:
            pnl = (p - current_price) * amt / p
        profit += pnl
        if pnl > 0:
            wins += 1
    winrate = round(wins/total_trades*100,2) if total_trades>0 else 0
    st.write(f"Total Trades: {total_trades} | Winrate: {winrate}% | Profit/Loss: {round(profit,2)} USDT")

st.session_state.loading = False

# =========================
# DEBUG PANEL
# =========================
with st.expander("🧰 Debug Panel"):
    if not debug:
        st.write("No debug messages")
    else:
        for d in debug:
            st.write(d)
