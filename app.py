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
import firebase_admin
from firebase_admin import credentials, db
import time as t

# =========================
# SMART AUTO REFRESH
# =========================
if "loading" not in st.session_state:
    st.session_state.loading = False

# Refresh only if not loading
if not st.session_state.loading:
    st_autorefresh(interval=300000, key="refresh")  # 5 minutes

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
demo_trades = {}  # track demo trades per coin

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
# FETCH OHLC DATA
# =========================
def safe_request(url, params=None):
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code==200:
                return r.json()
            t.sleep(2*(i+1))
        except:
            t.sleep(1)
    return None

def get_ohlc_binance(symbol, interval, limit):
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol":symbol,"interval":interval,"limit":limit}
        data = safe_request(url, params)
        if not data: return None
        d,o,h,l,c,v=[],[],[],[],[],[]
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
            o.append(prev); h.append(max(prev,curr))
            l.append(min(prev,curr)); c.append(curr)
            v.append(0)
        return d,o,h,l,c,v
    except:
        return None

def get_ohlc(symbol, timeframe):
    mapping = {"15 Min":("15m",96),"Hourly":("1h",48),"Daily":("1d",180)}
    interval, limit = mapping[timeframe]
    funcs = [get_ohlc_binance,get_ohlc_cryptocompare,get_ohlc_coingecko]
    for fn in funcs:
        try:
            data = fn(symbol,interval,limit) if fn!=get_ohlc_coingecko else fn(symbol)
            if data: return data, fn.__name__
        except: continue
    return None,"None"

# =========================
# INDICATORS
# =========================
def rsi(prices, period=14):
    prices=np.array(prices)
    if len(prices)<period+1: return np.array([50]*len(prices))
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(period,min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period,min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def macd(prices):
    prices=np.array(prices)
    if len(prices)<26: return np.zeros(len(prices)), np.zeros(len(prices))
    exp1=pd.Series(prices).ewm(span=12,adjust=False).mean()
    exp2=pd.Series(prices).ewm(span=26,adjust=False).mean()
    m=exp1-exp2
    s=m.ewm(span=9,adjust=False).mean()
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
        data = safe_request("https://min-api.cryptocompare.com/data/v2/news/?lang=EN")
        scores=[]
        for a in data.get("Data",[])[:5]:
            text = a.get("title","")
            v = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
            b = TextBlob(text).sentiment.polarity
            scores.append((v+b)/2)
        return np.mean(scores) if scores else 0
    except:
        return 0

# =========================
# SUPPORT/RESISTANCE & SPIKE
# =========================
def support_resistance(prices): return min(prices[-50:]), max(prices[-50:])
def detect_spike(prices):
    change=(prices[-1]-prices[-5])/prices[-5]
    if change<-0.05: return "🔻 Sharp Drop"
    if change>0.05: return "🚀 Sharp Rise"
    return "Normal"

# =========================
# DEMO TRADE UTILS
# =========================
def calculate_tp_sl(entry_price, tp_percent, sl_percent, long=True):
    if long:
        tp = entry_price*(1+tp_percent/100)
        sl = entry_price*(1-sl_percent/100)
    else:
        tp = entry_price*(1-sl_percent/100)
        sl = entry_price*(1+tp_percent/100)
    return tp,sl

def update_demo_trade(trade, current_price):
    long = trade["long"]
    closed = False
    recommendation = "Hold"
    if long:
        if current_price>=trade["tp"]:
            recommendation="Take Profit"
            closed=True
        elif current_price<=trade["sl"]:
            recommendation="Stop Loss"
            closed=True
    else:
        if current_price<=trade["tp"]:
            recommendation="Take Profit"
            closed=True
        elif current_price>=trade["sl"]:
            recommendation="Stop Loss"
            closed=True
    if closed:
        trade["exit_price"] = current_price
        trade["closed"] = True
        trade["pnl"] = (current_price-trade["entry"])*trade["amount"] if long else (trade["entry"]-current_price)*trade["amount"]
    else:
        trade["pnl"] = (current_price-trade["entry"])*trade["amount"] if long else (trade["entry"]-current_price)*trade["amount"]
    return trade, recommendation

# =========================
# UI SETUP
# =========================
st.title("🚀 AI Crypto Bot Demo")

COINS=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]
timeframe = st.selectbox("Select Timeframe", ["15 Min","Hourly","Daily"])
symbols = st.multiselect("Select Coins", COINS, default=["BTCUSDT","ETHUSDT"])

# =========================
# AUTO SPOT PORTFOLIO
# =========================
portfolio = {sym:{"amount":1,"entry":0,"pnl":0} for sym in COINS}  # 1 coin each demo

# =========================
# MAIN LOOP PER COIN
# =========================
st.session_state.loading=True
for sym in symbols:
    try:
        if sym not in ws_prices: start_ws(sym)
        data, source = get_ohlc(sym,timeframe)
        if data is None: continue
        fd,o,h,l,c,v=data
        c=np.array(c)
        if len(c)<20: continue

        # Indicators
        r = rsi(c)
        m,sig = macd(c)
        sentiment = get_sentiment()
        model,scaler = train_lstm(c)
        pred = lstm_predict(model,scaler,c)
        sup,res = support_resistance(c)
        spike = detect_spike(c)
        conf = max(0,min(100,100*(1-np.std(c)/c[-1])))

        # =========================
        # CHARTS
        # =========================
        with st.container():
            st.markdown(f"### {sym}",unsafe_allow_html=True)
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
                st.write(f"Confidence: {round(conf,2)}%")
                st.write(f"Sentiment: {round(sentiment,2)}")
                st.write(f"Spike: {spike}")
                st.write(f"LSTM Prediction: {round(pred,2)}")

        # =========================
        # DEMO TRADE PANEL
        # =========================
        if sym not in demo_trades: demo_trades[sym]=[]
        st.markdown("**Demo Trading Panel**")
        mode = st.selectbox(f"{sym} Mode", ["Spot","Future"], key=f"mode_{sym}")

        if mode=="Future":
            long_short = st.radio(f"{sym} Direction", ["Long","Short"], key=f"dir_{sym}")
            amount = st.number_input(f"USDT Amount", min_value=1.0, step=1.0, key=f"amt_{sym}")
            tp_percent = st.slider("Take Profit %", 1,20,5,key=f"tp_{sym}")
            sl_percent = st.slider("Stop Loss %",1,20,3,key=f"sl_{sym}")
            entry_price = ws_prices.get(sym,c[-1])
            if st.button(f"Open Future {long_short} Trade", key=f"open_{sym}"):
                demo_trades[sym].append({"mode":"future","long":long_short=="Long",
                                         "entry":entry_price,"tp":0,"sl":0,
                                         "amount":amount,"closed":False,"pnl":0})
                demo_trades[sym][-1]["tp"],demo_trades[sym][-1]["sl"] = calculate_tp_sl(entry_price,tp_percent,sl_percent,long=long_short=="Long")

        # =========================
        # UPDATE EXISTING TRADES
        # =========================
        current_price = ws_prices.get(sym,c[-1])
        for trade in demo_trades[sym]:
            if trade.get("closed"): continue
            trade, rec = update_demo_trade(trade,current_price)
            st.write(f"Trade: {trade['mode']} | Entry: {trade['entry']:.2f} | P/L: {trade['pnl']:.2f} | Recommendation: {rec}")

        # =========================
        # AUTO PORTFOLIO P/L
        # =========================
        portfolio[sym]["entry"] = c[-1]
        portfolio[sym]["pnl"] = c[-1]-c[-2]

    except Exception as e:
        st.write(f"{sym} Exception: {e}")

st.session_state.loading=False
