import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, db
import threading, websocket
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# AUTO REFRESH
# =========================
st_autorefresh(interval=5000, key="refresh")

# =========================
# FIREBASE INIT
# =========================
if not firebase_admin._apps:
    try:
        firebase_dict = dict(st.secrets["firebase"])
        firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n","\n")
        cred = credentials.Certificate(firebase_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": firebase_dict["databaseURL"]})
    except:
        pass

# =========================
# GLOBAL
# =========================
ws_prices = {}

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
# SAFE REQUEST
# =========================
def safe_request(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json()
    except:
        return None

# =========================
# FIREBASE CACHE
# =========================
def load_ohlc(symbol, timeframe):
    ref = db.reference(f"ohlc/{symbol}/{timeframe}")
    data = ref.get()
    if not data: return None
    fd,o,h,l,c,v=[],[],[],[],[],[]
    for x in data:
        fd.append(datetime.fromisoformat(x["t"]))
        o.append(x["o"]); h.append(x["h"])
        l.append(x["l"]); c.append(x["c"])
        v.append(x["v"])
    return fd,o,h,l,c,v

def save_ohlc(symbol, timeframe, fd,o,h,l,c,v):
    ref = db.reference(f"ohlc/{symbol}/{timeframe}")
    data=[]
    for i in range(len(fd)):
        data.append({"t":fd[i].isoformat(),"o":o[i],"h":h[i],"l":l[i],"c":c[i],"v":v[i]})
    ref.set(data[-300:])

# =========================
# FETCH
# =========================
def get_ohlc(symbol, interval, limit):
    url="https://api.binance.com/api/v3/klines"
    data=safe_request(url,{"symbol":symbol,"interval":interval,"limit":limit})
    if not data: return None
    d,o,h,l,c,v=[],[],[],[],[],[]
    for k in data:
        d.append(datetime.fromtimestamp(k[0]/1000))
        o.append(float(k[1])); h.append(float(k[2]))
        l.append(float(k[3])); c.append(float(k[4]))
        v.append(float(k[5]))
    return d,o,h,l,c,v

def get_ohlc_smart(symbol, timeframe):
    mapping={"15 Min":("15m",120),"Hourly":("1h",120),"Daily":("1d",200)}
    interval,limit=mapping[timeframe]

    cached=load_ohlc(symbol,timeframe)
    if cached:
        fd,o,h,l,c,v=cached
        new=get_ohlc(symbol,interval,5)
        if new:
            nfd,no,nh,nl,nc,nv=new
            for i in range(len(nfd)):
                if nfd[i]>fd[-1]:
                    fd.append(nfd[i]); o.append(no[i]); h.append(nh[i])
                    l.append(nl[i]); c.append(nc[i]); v.append(nv[i])
        fd,o,h,l,c,v=fd[-300:],o[-300:],h[-300:],l[-300:],c[-300:],v[-300:]
        save_ohlc(symbol,timeframe,fd,o,h,l,c,v)
        return fd,o,h,l,c,v

    data=get_ohlc(symbol,interval,limit)
    if data:
        save_ohlc(symbol,timeframe,*data)
    return data

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
# LSTM (CACHED)
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
# SENTIMENT (CACHED)
# =========================
@st.cache_data(ttl=900)
def get_sentiment():
    data=safe_request("https://min-api.cryptocompare.com/data/v2/news/?lang=EN")
    scores=[]
    for a in data.get("Data",[])[:5]:
        text=a["title"]
        v=SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
        b=TextBlob(text).sentiment.polarity
        scores.append((v+b)/2)
    return np.mean(scores) if scores else 0

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
    data=safe_request("https://min-api.cryptocompare.com/data/v2/news/?lang=EN")
    events=[]
    for a in data.get("Data",[])[:10]:
        t=a["title"].lower()
        if "fed" in t: events.append("Fed News")
        if "regulation" in t: events.append("Regulation")
    return list(set(events))

# =========================
# UI
# =========================
st.title("🚀 AI Crypto Bot (Optimized)")

COINS=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]

# Portfolio
st.sidebar.header("💰 Portfolio")
portfolio={}
total=0
for sym in COINS:
    col1,col2=st.sidebar.columns(2)
    amt=col1.number_input(f"{sym} Amt",0.0)
    buy=col2.number_input(f"{sym} Buy",0.0)
    if amt>0 and buy>0:
        portfolio[sym]=(amt,buy)

timeframe=st.selectbox("Timeframe",["15 Min","Hourly","Daily"])
symbols=st.multiselect("Coins",COINS,default=["BTCUSDT","ETHUSDT"])

# =========================
# MAIN LOOP
# =========================
for sym in symbols:

    if sym not in ws_prices:
        start_ws(sym)

    data=get_ohlc_smart(sym,timeframe)
    if not data: continue

    fd,o,h,l,c,v=data
    c=np.array(c)

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
    # UI CARD
    # =========================
    with st.container():
        st.markdown("<div style='border:1px solid white;padding:10px;border-radius:8px;'>",unsafe_allow_html=True)
        st.markdown(f"### {sym}")

        col1,col2=st.columns([3,1])

        with col1:
            fig=go.Figure()
            fig.add_trace(go.Candlestick(x=fd,open=o,high=h,low=l,close=c))
            fig.add_trace(go.Scatter(x=fd,y=[pred]*len(fd),line=dict(color="yellow")))
            fig.add_hline(y=sup,line_dash="dash",line_color="green")
            fig.add_hline(y=res,line_dash="dash",line_color="red")
            st.plotly_chart(fig,use_container_width=True)

        with col2:
            st.write(f"Prediction: {round(pred,2)}")
            st.write(f"Confidence: {round(conf,2)}%")
            st.write(f"Spike: {spike}")
            st.write(f"Events: {events}")

        st.markdown("</div>",unsafe_allow_html=True)
