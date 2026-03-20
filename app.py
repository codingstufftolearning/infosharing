import streamlit as st
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

# =========================
# GLOBALS
# =========================
ws_prices = {}
last_heavy_update = datetime.min  # fixed: declare global at the top

# =========================
# SMART REFRESH LOCK
# =========================
if "loading" not in st.session_state:
    st.session_state.loading = False

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
# DATA FETCHING
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

def get_ohlc(symbol, timeframe, debug):
    mapping = {"15 Min":("15m",96),"Hourly":("1h",72),"Daily":("1d",180)}
    interval, limit = mapping[timeframe]
    data = get_ohlc_binance(symbol, interval, limit)
    if data:
        debug.append(f"{symbol}: Binance OK")
        return data
    debug.append(f"{symbol}: Fetch failed")
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
# SPIKE / EVENT
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
st.title("🚀 AI Crypto Bot Live")
COINS=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]

st.sidebar.header("💰 Portfolio Tracker")
portfolio={}
total_pl=0
for sym in COINS:
    col1,col2=st.sidebar.columns(2)
    amt=col1.number_input(f"{sym} Amount",0.0,key=f"{sym}_amt")
    buy=col2.number_input(f"{sym} Buy Price",0.0,key=f"{sym}_buy")
    if amt>0 and buy>0:
        portfolio[sym]={"amount":amt,"buy_price":buy,"pl":0}

timeframe=st.selectbox("Select Timeframe",["15 Min","Hourly","Daily"])
symbols=st.multiselect("Select Coins",COINS,default=["BTCUSDT","ETHUSDT"])

debug=[]

# =========================
# MAIN LOOP
# =========================
st.session_state.loading=True

for sym in symbols:
    if sym not in ws_prices:
        start_ws(sym)

    data=get_ohlc(sym,timeframe,debug)
    if not data: continue

    fd,o,h,l,c,v=data
    c=np.array(c)

    # Recalculate heavy models every 5 min
    now=datetime.utcnow()
    global last_heavy_update  # <-- already declared at top; optional here
    if (now-last_heavy_update).total_seconds()>300 or last_heavy_update==datetime.min:
        model, scaler = train_lstm(c)
        pred = lstm_predict(model, scaler, c)
        senti = get_sentiment()
        r = rsi(c)
        m, sig = macd(c)
        last_heavy_update = now
    else:
        pred = c[-1]  # fallback prediction
        senti = 0
        r = np.array([50]*len(c))
        m, sig = np.zeros(len(c)), np.zeros(len(c))

    sup,res=support_resistance(c)
    spike=detect_spike(c)
    events=detect_event()
    conf=100*(1-np.std(c)/c[-1])

    # CARD UI
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
            st.write(f"Sentiment: {round(senti,2)}")
            st.write(f"Spike: {spike}")
            st.write(f"Events: {events}")

        st.markdown("</div>",unsafe_allow_html=True)

    if sym in portfolio:
        amt,buy=portfolio[sym]["amount"],portfolio[sym]["buy_price"]
        portfolio[sym]["pl"]=(c[-1]-buy)*amt
        total_pl+=portfolio[sym]["pl"]

st.session_state.loading=False

# =========================
# PORTFOLIO SUMMARY
# =========================
st.sidebar.markdown(f"### Total Portfolio P/L: ${round(total_pl,2)}")

# =========================
# DEBUG PANEL
# =========================
with st.expander("🧰 Debug Panel"):
    if not debug:
        st.write("No debug messages")
    else:
        for d in debug:
            st.write(d)
