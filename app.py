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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import firebase_admin
from firebase_admin import credentials, db

# =========================
# SMART AUTO REFRESH
# =========================
if "loading" not in st.session_state:
    st.session_state.loading = False

if not st.session_state.loading:
    st_autorefresh(interval=300000, key="refresh")  # 5 min

# =========================
# FIREBASE INIT
# =========================
if not firebase_admin._apps:
    try:
        firebase_dict = dict(st.secrets["firebase"])
        firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n","\n")
        cred = credentials.Certificate(firebase_dict)
        firebase_admin.initialize_app(cred, {"databaseURL": firebase_dict["databaseURL"]})
    except Exception as e:
        st.error(f"Firebase Initialization Error: {e}")

# =========================
# GLOBALS
# =========================
ws_prices = {}
demo_trades = {}  # key=symbol, value=list of open trades

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
# FETCH DATA
# =========================
def safe_request(url, params=None):
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=5)
            if r.status_code==200: return r.json()
            t.sleep(2*(i+1))
        except: t.sleep(1)
    return None

def get_ohlc_binance(symbol, interval, limit):
    data = safe_request("https://api.binance.com/api/v3/klines", {"symbol":symbol,"interval":interval,"limit":limit})
    if not data: return None
    d,o,h,l,c,v = [],[],[],[],[],[]
    try:
        for k in data:
            d.append(datetime.fromtimestamp(k[0]/1000))
            o.append(float(k[1])); h.append(float(k[2]))
            l.append(float(k[3])); c.append(float(k[4]))
            v.append(float(k[5]))
        return d,o,h,l,c,v
    except: return None

def get_ohlc_cryptocompare(symbol, interval, limit):
    fsym = symbol.replace("USDT","")
    if interval=="15m": url=f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={fsym}&tsym=USD&limit={limit}&aggregate=15"
    elif interval=="1h": url=f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={limit}"
    else: url=f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={fsym}&tsym=USD&limit={limit}"
    data = safe_request(url)
    raw = data.get("Data",{}).get("Data",[]) if data else []
    d,o,h,l,c,v=[],[],[],[],[],[]
    try:
        for x in raw:
            d.append(datetime.fromtimestamp(x["time"]))
            o.append(x["open"]); h.append(x["high"])
            l.append(x["low"]); c.append(x["close"])
            v.append(x.get("volumeto",0))
        return d,o,h,l,c,v
    except: return None

def get_ohlc_coingecko(symbol):
    coin = symbol.replace("USDT","").lower()
    url=f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=365"
    data = safe_request(url)
    prices = data.get("prices",[]) if data else []
    d,o,h,l,c,v=[],[],[],[],[],[]
    try:
        for i in range(1,len(prices)):
            tstamp=datetime.fromtimestamp(prices[i][0]/1000)
            prev=prices[i-1][1]; curr=prices[i][1]
            d.append(tstamp); o.append(prev); h.append(max(prev,curr))
            l.append(min(prev,curr)); c.append(curr); v.append(0)
        return d,o,h,l,c,v
    except: return None

def get_ohlc(symbol, timeframe):
    mapping={"15 Min":("15m",96),"Hourly":("1h",72),"Daily":("1d",180)}
    interval, limit = mapping[timeframe]
    funcs=[get_ohlc_binance,get_ohlc_cryptocompare,get_ohlc_coingecko]
    for fn in funcs:
        try:
            data = fn(symbol, interval, limit) if fn != get_ohlc_coingecko else fn(symbol)
            if data: return data, fn.__name__
        except: continue
    return None,"None"

# =========================
# INDICATORS
# =========================
def rsi(prices):
    delta=np.diff(prices); gain=np.maximum(delta,0); loss=np.abs(np.minimum(delta,0))
    rs=pd.Series(gain).rolling(14).mean()/(pd.Series(loss).rolling(14).mean()+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def macd(prices):
    exp1=pd.Series(prices).ewm(span=12).mean()
    exp2=pd.Series(prices).ewm(span=26).mean()
    m=exp1-exp2; s=m.ewm(span=9).mean()
    return m.values,s.values

# =========================
# LSTM PREDICTION
# =========================
@st.cache_resource(ttl=1800)
def train_lstm(prices):
    scaler=MinMaxScaler()
    data=scaler.fit_transform(np.array(prices).reshape(-1,1))
    X,y=[],[]
    for i in range(20,len(data)):
        X.append(data[i-20:i]); y.append(data[i])
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
            text=a.get("title","")
            v=SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
            b=TextBlob(text).sentiment.polarity
            scores.append((v+b)/2)
        return np.mean(scores) if scores else 0
    except: return 0

# =========================
# SUPPORT/RESISTANCE + SPIKES
# =========================
def support_resistance(prices): return min(prices[-50:]), max(prices[-50:])

def detect_spike(prices):
    change=(prices[-1]-prices[-5])/prices[-5]
    if change<-0.05: return "🔻 Sharp Drop"
    if change>0.05: return "🚀 Sharp Rise"
    return "Normal"

# =========================
# UI SETUP
# =========================
st.title("🚀 AI Crypto Bot Demo & Trading")

COINS=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]
timeframe = st.selectbox("Select Timeframe", ["15 Min","Hourly","Daily"])
symbols = st.multiselect("Select Coins", COINS, default=["BTCUSDT","ETHUSDT"])

st.session_state.loading=True
debug=[]

# =========================
# MAIN LOOP
# =========================
for sym in symbols:
    if sym not in ws_prices: start_ws(sym)

    data, source = get_ohlc(sym, timeframe)
    if not data: continue
    fd,o,h,l,c,v=data
    c=np.array(c)

    r=rsi(c); m,s=macd(c); sentiment=get_sentiment()
    model,scaler=train_lstm(c); pred=lstm_predict(model,scaler,c)
    sup,res=support_resistance(c); spike=detect_spike(c)
    conf=100*(1-np.std(c)/c[-1])

    # =========================
    # CHART
    # =========================
    with st.container():
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
            st.write(f"Data Source: {source}")

    # =========================
    # DEMO TRADING PANEL
    # =========================
    if sym not in demo_trades: demo_trades[sym]=[]

    st.markdown(f"#### Demo Trading Panel: {sym}")
    mode = st.selectbox(f"{sym} Mode", ["Spot","Future"], key=f"{sym}_mode")
    direction = st.selectbox(f"{sym} Action", ["Buy","Sell"] if mode=="Spot" else ["Long","Short"], key=f"{sym}_dir")
    entry_price = st.number_input("Entry Price", min_value=0.0, value=float(c[-1]), step=0.01, key=f"{sym}_entry")
    amount = st.number_input("Amount (USDT)", min_value=0.0, value=100.0, step=1.0, key=f"{sym}_amt")
    tp = st.number_input("Take Profit (%)", min_value=0.0, max_value=100.0, value=2.0, step=0.1, key=f"{sym}_tp")
    sl = st.number_input("Stop Loss (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1, key=f"{sym}_sl")

    if st.button("Open Trade", key=f"{sym}_open"):
        trade = {"mode":mode,"direction":direction,"entry":entry_price,"amount":amount,"tp":tp,"sl":sl,"timestamp":str(datetime.now())}
        demo_trades[sym].append(trade)
        # save to firebase
        try:
            db.reference(f"trades/{sym}").push(trade)
        except: pass

    # Show all demo trades for this coin
    for idx,t in enumerate(demo_trades[sym]):
        live_price = ws_prices.get(sym,c[-1])
        if t["direction"] in ["Buy","Long"]: pnl=(live_price-t["entry"])*t["amount"]/t["entry"]
        else: pnl=(t["entry"]-live_price)*t["amount"]/t["entry"]
        stop_loss_price = t["entry"]*(1-t["sl"]/100 if t["direction"] in ["Buy","Long"] else 1+t["sl"]/100)
        take_profit_price = t["entry"]*(1+t["tp"]/100 if t["direction"] in ["Buy","Long"] else 1-t["tp"]/100)
        st.markdown(f"- Trade {idx+1}: {t['direction']} {t['amount']} USDT at {t['entry']} | P/L: {round(pnl,2)} | TP: {round(take_profit_price,2)}, SL: {round(stop_loss_price,2)}")
        # Optional: auto close if TP/SL hit
        if (t["direction"] in ["Buy","Long"] and (live_price>=take_profit_price or live_price<=stop_loss_price)) or \
           (t["direction"] in ["Sell","Short"] and (live_price<=take_profit_price or live_price>=stop_loss_price)):
            # log exit
            t["exit_price"]=live_price; t["pnl"]=pnl; t["closed_timestamp"]=str(datetime.now())
            try:
                db.reference(f"trades/{sym}/closed").push(t)
            except: pass
            demo_trades[sym].remove(t)

st.session_state.loading=False

# =========================
# DEBUG PANEL
# =========================
with st.expander("🧰 Debug Panel"):
    if not debug: st.write("No debug messages")
    else:
        for d in debug: st.write(d)
