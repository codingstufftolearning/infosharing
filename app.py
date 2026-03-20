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
from tensorflow.keras.layers import LSTM, Dense, Input
from sklearn.linear_model import LogisticRegression

# =========================
# AUTO REFRESH
# =========================
if "loading" not in st.session_state:
    st.session_state.loading = False

if not st.session_state.loading:
    st_autorefresh(interval=300000, key="refresh")  # 5 min

# =========================
# GLOBALS
# =========================
ws_prices = {}

# =========================
# SAFE REQUEST WITH RETRY
# =========================
def safe_request(url, params=None):
    for i in range(3):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            t.sleep(2*(i+1))
        except:
            t.sleep(1)
    return None

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
def get_ohlc_binance(symbol, interval, limit):
    bases = ["https://api.binance.com","https://api1.binance.com"]
    for b in bases:
        data = safe_request(b+"/api/v3/klines", {"symbol":symbol,"interval":interval,"limit":limit})
        if data:
            try:
                d,o,h,l,c,v=[],[],[],[],[],[]
                for k in data:
                    d.append(datetime.fromtimestamp(k[0]/1000))
                    o.append(float(k[1])); h.append(float(k[2]))
                    l.append(float(k[3])); c.append(float(k[4]))
                    v.append(float(k[5]))
                return d,o,h,l,c,v
            except:
                pass
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
    mapping = {"15 Min":("15m",96),"Hourly":("1h",72),"Daily":("1d",180),"3-Day":("3d",90)}
    interval, limit = mapping[timeframe]
    funcs = [get_ohlc_binance, get_ohlc_cryptocompare, get_ohlc_coingecko]
    for fn in funcs:
        try:
            data = fn(symbol, interval, limit) if fn != get_ohlc_coingecko else fn(symbol)
            if data:
                return data, fn.__name__
        except:
            continue
    return None,"None"

# =========================
# INDICATORS
# =========================
def calculate_rsi(prices, period=14):
    prices = np.array(prices)
    if len(prices) < period+1:
        return np.array([50]*len(prices))
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(period,min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period,min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def calculate_macd(prices):
    prices = np.array(prices)
    if len(prices)<26:
        return np.zeros(len(prices)), np.zeros(len(prices))
    exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.values, signal.values

# =========================
# LSTM MODEL
# =========================
@st.cache_resource(ttl=3600)
def train_lstm(prices):
    if len(prices)<30: return None, None
    scaler = MinMaxScaler()
    data = scaler.fit_transform(np.array(prices).reshape(-1,1))
    X, y = [], []
    for i in range(20, len(data)):
        X.append(data[i-20:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    model = Sequential([Input(shape=(20,1)), LSTM(50), Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=3, verbose=0)
    return model, scaler

def lstm_predict(model, scaler, prices):
    if model is None or scaler is None: return prices[-1]
    data = scaler.transform(np.array(prices).reshape(-1,1))
    seq = data[-20:]
    pred = model.predict(seq.reshape(1,20,1), verbose=0)
    return scaler.inverse_transform(pred)[0][0]

# =========================
# LIGHTWEIGHT AI SIGNAL
# =========================
def train_ai(prices):
    if len(prices)<30: return None
    X=[]; y=[]
    for i in range(20,len(prices)-1):
        w = prices[i-20:i]
        X.append([np.mean(w),np.std(w),w[-1]-w[0]])
        y.append(1 if prices[i+1]>prices[i] else 0)
    m = LogisticRegression()
    m.fit(X,y)
    return m

def ai_predict(m, prices):
    if not m or len(prices)<20: return 0.5
    w = prices[-20:]
    return m.predict_proba([[np.mean(w),np.std(w),w[-1]-w[0]]])[0][1]

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
# SPIKE / SUPPORT
# =========================
def detect_spike(prices):
    if len(prices)<5: return "Normal"
    change = (prices[-1]-prices[-5])/prices[-5]
    if change<-0.05: return "🔻 Sharp Drop"
    if change>0.05: return "🚀 Sharp Rise"
    return "Normal"

def support_resistance(prices):
    if len(prices)<50: return min(prices), max(prices)
    return min(prices[-50:]), max(prices[-50:])

# =========================
# UI
# =========================
st.title("🚀 AI Crypto Bot Fixed")

# Limit default coins to 5 to reduce memory
COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT"]

# Portfolio Sidebar
st.sidebar.header("💰 Portfolio Tracker")
portfolio = {}
total_pl = 0
for sym in COINS:
    col1, col2 = st.sidebar.columns(2)
    amt = col1.number_input(f"{sym} Amount", min_value=0.0, step=0.01, key=f"{sym}_amt")
    buy_price = col2.number_input(f"{sym} Buy Price", min_value=0.0, step=0.01, key=f"{sym}_buy")
    if amt>0 and buy_price>0:
        portfolio[sym] = {"amount": amt, "buy_price": buy_price, "pl":0}

timeframe = st.selectbox("Select Timeframe", ["15 Min","Hourly","Daily","3-Day"])
symbols = st.multiselect("Select Coins", COINS, default=COINS)

debug_info = []

st.session_state.loading = True
for sym in symbols:
    try:
        if sym not in ws_prices:
            start_ws(sym)
        data, source = get_ohlc(sym, timeframe)
        if data is None:
            debug_info.append(f"{sym}: Failed to fetch data from all sources.")
            continue
        fd,o,h,l,c,v = data
        c = np.array(c)
        if len(c)<5:
            debug_info.append(f"{sym}: Not enough data")
            continue

        # Indicators
        rsi = calculate_rsi(c)
        macd_v, sig_line = calculate_macd(c)
        sentiment = get_sentiment()

        # AI & LSTM
        ai_model = train_ai(c)
        ai_prob = ai_predict(ai_model,c)
        try:
            lstm_model, lstm_scaler = train_lstm(c)
            lstm_pred = lstm_predict(lstm_model,lstm_scaler,c)
        except:
            lstm_pred = c[-1]

        # Signal
        trend = (c[-1]-np.mean(c))/np.std(c) if np.std(c)>0 else 0
        score = (50-rsi[-1])/50 + (macd_v[-1]-sig_line[-1]) + trend + sentiment + (ai_prob-0.5)*2
        sig = "BUY" if score>1 else "SELL" if score<-1 else "HOLD"
        live = ws_prices.get(sym,c[-1])
        conf = max(0,min(100,100*(1-np.std(c)/c[-1])))
        sup,res = support_resistance(c)
        spike = detect_spike(c)

        # Coin Container
        with st.container():
            st.markdown("<div style='border:1px solid white; padding:10px; border-radius:8px; margin-bottom:10px;'>", unsafe_allow_html=True)
            st.markdown(f"### {sym} ({source})")
            col_chart, col_stats = st.columns([3,1])

            with col_chart:
                spark_fig = go.Figure()
                spark_fig.add_trace(go.Scatter(x=fd, y=c, mode="lines", line=dict(color="cyan", width=2)))
                spark_fig.update_layout(height=80, margin=dict(l=0,r=0,t=0,b=0), xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
                st.plotly_chart(spark_fig, use_container_width=True, config={"displayModeBar":False})

                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=fd, open=o, high=h, low=l, close=c,
                                             increasing_line_color='green', decreasing_line_color='red', name="Price"))
                fig.add_trace(go.Scatter(x=fd, y=[lstm_pred]*len(fd), line=dict(color="yellow"), name="LSTM Prediction"))
                fig.add_hline(y=sup, line_dash="dash", line_color="green")
                fig.add_hline(y=res, line_dash="dash", line_color="red")
                fig.update_layout(dragmode="zoom", margin=dict(l=20,r=20,t=30,b=20))
                st.plotly_chart(fig, use_container_width=True)

            with col_stats:
                st.markdown(f"**Signal:** {sig}")
                st.markdown(f"**Score:** {round(score,2)}")
                st.markdown(f"**Confidence:** {round(conf,2)}%")
                st.markdown(f"**RSI:** {round(rsi[-1],2)}")
                st.markdown(f"**MACD:** {round(macd_v[-1],2)} / {round(sig_line[-1],2)}")
                st.markdown(f"**Sentiment:** {round(sentiment,2)}")
                st.markdown(f"**AI:** {round(ai_prob*100,2)}%")
                st.markdown(f"**Spike:** {spike}")
                st.markdown(f"**Live Price:** {live}")

            st.markdown("</div>", unsafe_allow_html=True)

        # Portfolio Update
        if sym in portfolio:
            amt = portfolio[sym]["amount"]
            buy_price = portfolio[sym]["buy_price"]
            portfolio[sym]["pl"] = (live-buy_price)*amt
            total_pl += portfolio[sym]["pl"]

    except Exception as e:
        debug_info.append(f"{sym}: Exception - {str(e)}")

st.session_state.loading = False

# Portfolio Summary
st.sidebar.markdown(f"### Total Portfolio P/L: ${round(total_pl,2)}")

# Debug Panel
with st.expander("🧰 Debug Panel"):
    for line in debug_info:
        st.write(line)
