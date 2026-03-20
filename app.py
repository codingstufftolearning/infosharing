import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime
import threading, websocket
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import firebase_admin
from firebase_admin import credentials, db

# =========================
# AUTO REFRESH
# =========================
if "loading" not in st.session_state:
    st.session_state.loading = False
if not st.session_state.loading:
    st_autorefresh(interval=300000, key="refresh")  # 5 minutes

# =========================
# FIREBASE INIT
# =========================
if not firebase_admin._apps:
    try:
        firebase_dict = dict(st.secrets["firebase"])
        firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(firebase_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": firebase_dict["databaseURL"]})
    except Exception as e:
        st.error(f"Firebase Initialization Error: {e}")

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
# SAFE REQUEST
# =========================
def safe_request(url, params=None):
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
        except:
            continue
    return None

# =========================
# OHLC FETCHERS
# =========================
def get_ohlc_binance(symbol, interval, limit):
    try:
        url = "https://api.binance.com/api/v3/klines"
        data = safe_request(url, {"symbol":symbol,"interval":interval,"limit":limit})
        if not data:
            return None
        d,o,h,l,c,v = [],[],[],[],[],[]
        for k in data:
            d.append(datetime.fromtimestamp(k[0]/1000))
            o.append(float(k[1])); h.append(float(k[2]))
            l.append(float(k[3])); c.append(float(k[4]))
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
        data = safe_request(url)
        raw = data.get("Data",{}).get("Data",[]) if data else []
        d,o,h,l,c,v=[],[],[],[],[],[]
        for x in raw:
            d.append(datetime.fromtimestamp(x["time"]))
            o.append(x["open"]); h.append(x["high"])
            l.append(x["low"]); c.append(x["close"])
            v.append(x.get("volumeto",0))
        return d,o,h,l,c,v
    except:
        return None

def get_ohlc_coingecko(symbol):
    try:
        coin = symbol.replace("USDT","").lower()
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=365"
        data = safe_request(url)
        prices = data.get("prices",[]) if data else []
        d,o,h,l,c,v=[],[],[],[],[],[]
        for i in range(1,len(prices)):
            tstamp = datetime.fromtimestamp(prices[i][0]/1000)
            prev = prices[i-1][1]; curr = prices[i][1]
            d.append(tstamp)
            o.append(prev)
            h.append(max(prev,curr))
            l.append(min(prev,curr))
            c.append(curr)
            v.append(0)
        return d,o,h,l,c,v
    except:
        return None

def get_ohlc(symbol, timeframe):
    mapping = {"15 Min":("15m",96),"Hourly":("1h",48),"Daily":("1d",180)}
    interval, limit = mapping[timeframe]
    for fn in [get_ohlc_binance, get_ohlc_cryptocompare, get_ohlc_coingecko]:
        try:
            data = fn(symbol, interval, limit) if fn!=get_ohlc_coingecko else fn(symbol)
            if data: return data, fn.__name__
        except:
            continue
    return None,"None"

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
        data=safe_request("https://min-api.cryptocompare.com/data/v2/news/?lang=EN")
        scores=[]
        for a in data.get("Data",[])[:5]:
            text = a.get("title","")
            v=SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
            b=TextBlob(text).sentiment.polarity
            scores.append((v+b)/2)
        return np.mean(scores) if scores else 0
    except:
        return 0

# =========================
# SUPPORT/RESISTANCE
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
        data = safe_request("https://min-api.cryptocompare.com/data/v2/news/?lang=EN")
        events=[]
        for a in data.get("Data",[])[:10]:
            t = a.get("title","").lower()
            if "fed" in t: events.append("Fed Impact")
            if "regulation" in t: events.append("Regulation")
        return list(set(events))
    except:
        return []

# =========================
# UI START
# =========================
st.title("🚀 AI Crypto Bot Demo")

COINS=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]

timeframe=st.selectbox("Select Timeframe",["15 Min","Hourly","Daily"], index=1)
symbols=st.multiselect("Select Coins",COINS,default=["BTCUSDT","ETHUSDT"])

st.session_state.loading = True
debug=[]

for sym in symbols:
    if sym not in ws_prices:
        start_ws(sym)

    data, source = get_ohlc(sym, timeframe)
    if not data:
        st.warning(f"{sym}: Failed to fetch OHLC data.")
        continue

    fd,o,h,l,c,v = data
    c=np.array(c)

    r=rsi(c)
    m,sig = macd(c)
    sentiment = get_sentiment()
    model,scaler = train_lstm(c)
    pred = lstm_predict(model,scaler,c)
    sup,res = support_resistance(c)
    spike = detect_spike(c)
    events = detect_event()
    conf = 100*(1-np.std(c)/c[-1])

    # =========================
    # Chart Card
    # =========================
    st.markdown(f"### {sym} - Chart")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=fd,open=o,high=h,low=l,close=c))
    fig.add_trace(go.Scatter(x=fd,y=[pred]*len(fd),line=dict(color="yellow"),name="Prediction"))
    fig.add_hline(y=sup,line_dash="dash",line_color="green")
    fig.add_hline(y=res,line_dash="dash",line_color="red")
    fig.update_layout(dragmode="zoom")
    st.plotly_chart(fig,use_container_width=True)

    # =========================
    # Demo Trading Panel
    # =========================
    st.markdown(f"#### Demo Trading Panel: {sym}")
    if sym not in st.session_state:
        st.session_state[f"{sym}_trades"] = []

    mode_key = f"{sym}_mode"
    dir_key = f"{sym}_dir"
    entry_key = f"{sym}_entry"
    amt_key = f"{sym}_amt"
    tp_key = f"{sym}_tp"
    sl_key = f"{sym}_sl"
    btn_key = f"{sym}_open"

    mode = st.selectbox("Mode", ["Spot","Future"], key=mode_key)
    direction = st.selectbox("Action", ["Buy","Sell"] if mode=="Spot" else ["Long","Short"], key=dir_key)
    entry_price = st.number_input("Entry Price", value=float(c[-1]), step=0.01, key=entry_key)
    amount = st.number_input("Amount (USDT)", value=100.0, step=1.0, key=amt_key)
    tp = st.number_input("Take Profit (%)", value=2.0, step=0.1, key=tp_key)
    sl = st.number_input("Stop Loss (%)", value=1.0, step=0.1, key=sl_key)

    if st.button("Open Trade", key=btn_key):
        trade = {
            "mode": mode,
            "direction": direction,
            "entry": entry_price,
            "amount": amount,
            "tp": tp,
            "sl": sl,
            "timestamp": str(datetime.now())
        }
        st.session_state[f"{sym}_trades"].append(trade)
        # Push to Firebase
        try:
            db.reference(f"/trades/{sym}").push(trade)
        except:
            pass

# =========================
# Debug Panel
# =========================
with st.expander("🧰 Debug Panel"):
    for d in debug:
        st.write(d)

st.session_state.loading = False
