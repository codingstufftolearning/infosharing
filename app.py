import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import firebase_admin
from firebase_admin import credentials, db

# =========================
# 🔄 AUTO REFRESH (30 MIN)
# =========================
st_autorefresh(interval=1800000, key="refresh")

# =========================
# 🔐 FIREBASE INIT
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
# 📊 DATA FETCH FUNCTIONS
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
        if interval == "15m":
            url = f"https://min-api.cryptocompare.com/data/v2/histominute?fsym={fsym}&tsym=USD&limit={limit}&aggregate=15"
        elif interval == "1h":
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
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days=30"
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

def get_ohlc(symbol, timeframe):
    mapping = {"15 Min": ("15m", 96),
               "Hourly": ("1h", 48),
               "Daily": ("1d", 30),
               "3-Day": ("3d", 20),
               "Weekly": ("1w", 12)}
    interval, limit = mapping[timeframe]
    for fn in [get_ohlc_binance, get_ohlc_cryptocompare, get_ohlc_coingecko]:
        data = fn(symbol, interval, limit) if fn != get_ohlc_coingecko else fn(symbol)
        if data: return data, fn.__name__
    return None, "None"

# =========================
# 📈 INDICATORS
# =========================
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    avg_gain = pd.Series(gain).rolling(period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period, min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    return np.concatenate([[50], 100-(100/(1+rs))])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12,adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26,adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9,adjust=False).mean()
    return macd.values, signal.values

# =========================
# 📰 SENTIMENT
# =========================
def get_sentiment():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data = requests.get(url, timeout=5).json()
        scores=[]
        for a in data.get("Data", [])[:8]:
            text = a.get("title","")
            vader = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity
            scores.append((vader + blob)/2)
        return np.mean(scores) if scores else 0
    except:
        return 0

# =========================
# 🔮 FORECAST
# =========================
def arima_forecast(prices, steps):
    try:
        return list(ARIMA(prices, order=(2,1,2)).fit().forecast(steps=steps))
    except:
        return [prices[-1]]*steps

# =========================
# 🧠 AI LOGIC
# =========================
def load_weights():
    ref = db.reference("weights")
    data = ref.get() or {}
    return data if data else {"rsi":1,"macd":1,"trend":1,"sentiment":1}

def save_weights(weights):
    db.reference("weights").set(weights)

def smart_signal(prices,rsi,macd,signal,sentiment,weights):
    score=0; contrib={}
    val=2*weights["rsi"] if rsi[-1]<30 else (-2*weights["rsi"] if rsi[-1]>70 else 0)
    score+=val; contrib["rsi"]=val
    val=(1 if macd[-1]>signal[-1] else -1)*weights["macd"]
    score+=val; contrib["macd"]=val
    val=(1 if prices[-1]>np.mean(prices) else -1)*weights["trend"]
    score+=val; contrib["trend"]=val
    val=2*weights["sentiment"] if sentiment>0.2 else (-2*weights["sentiment"] if sentiment<-0.2 else 0)
    score+=val; contrib["sentiment"]=val
    if score>=3: return "BUY",score,contrib
    elif score<=-3: return "SELL",score,contrib
    return "HOLD",score,contrib

def update_weights(symbol, weights):
    ref=db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    if len(data)<2: return weights
    prev,curr=data[-2],data[-1]
    actual_up = curr["price"]>prev["price"]
    lr=0.03
    for key in weights:
        contrib=prev.get("contributions",{}).get(key,0)
        if contrib==0: continue
        if (contrib>0 and actual_up) or (contrib<0 and not actual_up):
            weights[key]*=(1+lr)
        else:
            weights[key]*=(1-lr)
        weights[key]=max(0.2,min(weights[key],3))
    return weights

def save_prediction(symbol, price, pred, sig, contrib):
    ref=db.reference(f"history/{symbol}")
    ref.push({
        "time":datetime.utcnow().isoformat(),
        "price":float(price),
        "predicted":float(pred),
        "signal":sig,
        "contributions":contrib
    })

def calculate_win_rate(symbol):
    ref=db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    wins,total=0,0
    for i in range(len(data)-1):
        curr,nxt=data[i],data[i+1]
        sig=curr.get("signal","HOLD")
        cp=curr.get("price",0); np_=nxt.get("price",0)
        if sig=="BUY" and np_>cp: wins+=1
        elif sig=="SELL" and np_<cp: wins+=1
        total+=1
    return round((wins/total)*100,2) if total>0 else 0

def calculate_confidence(upper, lower, price):
    return round(max(0,min(1,1-(upper-lower)/price))*100,2)

# =========================
# 🎨 UI
# =========================
st.title("🚀 AI Crypto Bot")

timeframe = st.selectbox("Select Timeframe", ["15 Min","Hourly","Daily","3-Day","Weekly"])
COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","MATICUSDT","XAIUSD"]
symbols = st.multiselect("Select Coins", COINS, default=["BTCUSDT","ETHUSDT"])

for sym in symbols:
    data, source = get_ohlc(sym, timeframe)
    if data is None:
        st.error(f"{sym} failed to fetch data from all sources")
        continue
    fd,o,h,l,c,v = data
    next_p = arima_forecast(c,1)[0]
    rsi = calculate_rsi(c)
    macd_v, sig_line = calculate_macd(c)
    sentiment = get_sentiment()
    weights = load_weights()
    sig,score,con = smart_signal(c,rsi,macd_v,sig_line,sentiment,weights)
    wr = calculate_win_rate(sym)
    conf = calculate_confidence(max(c), min(c), c[-1])

    # Two-column layout
    col_chart, col_stats = st.columns([3,1])

    # --- Chart ---
    with col_chart:
        color = "green" if next_p>c[-1] else "red"
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=fd, open=o, high=h, low=l, close=c,
                                     increasing_line_color='green',
                                     decreasing_line_color='red',
                                     name="Price"))
        fig.add_trace(go.Scatter(x=[fd[-1],fd[-1]+timedelta(hours=1)],
                                 y=[c[-1],next_p],
                                 mode="lines", line=dict(color=color, width=3),
                                 name="Forecast"))
        fig.update_layout(
            dragmode=False,
            margin=dict(l=20,r=20,t=30,b=20),
            updatemenus=[dict(type="buttons", y=1, x=1.05, showactive=False, buttons=[
                dict(label="+", method="relayout", args=["xaxis.range[1]", fd[-1]]),
                dict(label="-", method="relayout", args=["xaxis.range[0]", fd[0]])
            ])]
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption(f"Source: {source}")

    # --- Stats Panel ---
    tooltip = {
        "Signal":"AI trading signal: BUY, SELL, or HOLD based on indicators",
        "Score":"Aggregated score from RSI, MACD, Trend, Sentiment",
        "WinRate":"Historical accuracy percentage of previous signals",
        "Confidence":"Forecast reliability percentage based on upper/lower bounds",
        "RSI":"Relative Strength Index (<30 oversold, >70 overbought)",
        "MACD":"MACD vs Signal line; trend momentum indicator",
        "Sentiment":"Average sentiment score from recent news"
    }

    with col_stats:
        st.markdown(f"### {sym} Stats")
        st.markdown(f"<span title='{tooltip['Signal']}'>**Signal:**</span> {sig}", unsafe_allow_html=True)
        st.markdown(f"<span title='{tooltip['Score']}'>**Score:**</span> {score}", unsafe_allow_html=True)
        st.markdown(f"<span title='{tooltip['WinRate']}'>**WinRate:**</span> {wr}%", unsafe_allow_html=True)
        st.markdown(f"<span title='{tooltip['Confidence']}'>**Confidence:**</span> {conf}%", unsafe_allow_html=True)
        st.markdown(f"<span title='{tooltip['RSI']}'>**RSI:**</span> {round(rsi[-1],2)}", unsafe_allow_html=True)
        st.markdown(f"<span title='{tooltip['MACD']}'>**MACD:**</span> {round(macd_v[-1],2)} / {round(sig_line[-1],2)}", unsafe_allow_html=True)
        st.markdown(f"<span title='{tooltip['Sentiment']}'>**Sentiment:**</span> {round(sentiment,2)}", unsafe_allow_html=True)

# =========================
# 🧰 DEBUG PANEL
# =========================
with st.expander("🧰 Debug Panel"):
    st.write("Selected Timeframe:", timeframe)
    st.write("Session State:", st.session_state)
