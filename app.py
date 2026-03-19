import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta, time
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
# 📈 INDICATORS (safe for short data)
# =========================
def calculate_rsi(prices, period=14):
    prices = np.array(prices)
    if len(prices) < period + 1:
        return np.array([50]*len(prices))  # fallback
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    avg_gain = pd.Series(gain).rolling(period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return np.concatenate([[50], 100 - (100 / (1 + rs))])

def calculate_macd(prices):
    prices = np.array(prices)
    if len(prices) < 26:
        return np.zeros(len(prices)), np.zeros(len(prices))
    exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
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
# 🔮 FORECAST FUNCTIONS
# =========================
def arima_forecast(prices, steps):
    try:
        return list(ARIMA(prices, order=(2,1,2)).fit().forecast(steps=steps))
    except:
        return [prices[-1]]*steps

def prophet_forecast(fd, c, step_delta):
    df = pd.DataFrame({"ds": fd, "y": c})
    m = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False)
    m.fit(df)
    last_date = fd[-1]
    end_of_day = datetime.combine(last_date.date(), time(23,59))
    future_dates = []
    curr = last_date + step_delta
    while curr <= end_of_day:
        future_dates.append(curr)
        curr += step_delta
    if not future_dates:
        return [], []
    future_df = pd.DataFrame({"ds": future_dates})
    forecast = m.predict(future_df)
    return future_dates, forecast["yhat"].values

# =========================
# 🧠 AI LOGIC
# =========================
def load_weights():
    ref = db.reference("weights")
    data = ref.get() or {}
    return data if data else {"rsi":1,"macd":1,"trend":1,"sentiment":1}

def smart_signal(prices,rsi,macd,signal,sentiment,weights):
    if len(macd)==0 or len(signal)==0:
        return "HOLD",0,{}
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

# Portfolio Sidebar
st.sidebar.header("💰 Portfolio Tracker")
COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XRPUSDT","MATICUSDT","XAIUSD"]
portfolio = {}
for sym in COINS:
    col1, col2 = st.sidebar.columns(2)
    amount = col1.number_input(f"{sym} Amount", min_value=0.0, step=0.01, key=f"{sym}_amt")
    buy_price = col2.number_input(f"{sym} Buy Price", min_value=0.0, step=0.01, key=f"{sym}_buy")
    if amount>0 and buy_price>0:
        portfolio[sym] = {"amount": amount, "buy_price": buy_price}

timeframe = st.selectbox("Select Timeframe", ["15 Min","Hourly","Daily","3-Day","Weekly"])
symbols = st.multiselect("Select Coins", COINS, default=["BTCUSDT","ETHUSDT"])

tooltip = {
    "Signal":"AI trading signal: BUY, SELL, or HOLD based on indicators",
    "Score":"Aggregated score from RSI, MACD, Trend, Sentiment",
    "WinRate":"Historical accuracy percentage of previous signals",
    "Confidence":"Forecast reliability percentage based on upper/lower bounds",
    "RSI":"Relative Strength Index (<30 oversold, >70 overbought)",
    "MACD":"MACD vs Signal line; trend momentum indicator",
    "Sentiment":"Average sentiment score from recent news"
}

# Forecast step mapping
step_mapping = {
    "15 Min": timedelta(minutes=15),
    "Hourly": timedelta(hours=1),
    "Daily": timedelta(days=1),
    "3-Day": timedelta(days=3),
    "Weekly": timedelta(weeks=1)
}
forecast_step = step_mapping[timeframe]

debug_info = []

for sym in symbols:
    try:
        data, source = get_ohlc(sym, timeframe)
        if data is None:
            debug_info.append(f"{sym}: Failed to fetch data from all sources.")
            continue
        fd,o,h,l,c,v = data
        c = np.array(c)
        if len(c) < 14:  # minimum for RSI
            debug_info.append(f"{sym}: Insufficient data for analysis.")
            continue
        rsi = calculate_rsi(c)
        macd_v, sig_line = calculate_macd(c)
        sentiment = get_sentiment()
        weights = load_weights()
        sig,score,con = smart_signal(c,rsi,macd_v,sig_line,sentiment,weights)
        wr = calculate_win_rate(sym)
        conf = calculate_confidence(max(c), min(c), c[-1])

        # ARIMA forecast
        future_dates_arima = []
        curr = fd[-1] + forecast_step
        while curr <= datetime.combine(fd[-1].date(), time(23,59)):
            future_dates_arima.append(curr)
            curr += forecast_step
        future_prices_arima = arima_forecast(c, len(future_dates_arima))

        # Prophet forecast
        future_dates_prophet, future_prices_prophet = prophet_forecast(fd, c, forecast_step)
        if len(future_dates_prophet)==0:
            future_dates_prophet = future_dates_arima
            future_prices_prophet = future_prices_arima

        # Combine forecasts
        future_prices = (future_prices_arima + future_prices_prophet)/2
        future_dates = future_dates_arima

        # Coin Box
        with st.container():
            st.markdown(
                f"<div style='border:1px solid white; padding:10px; border-radius:8px; margin-bottom:10px;'>",
                unsafe_allow_html=True
            )
            st.markdown(f"### {sym}")

            # Chart + Stats Layout
            col_chart, col_stats = st.columns([3,1])

            # --- Chart ---
            with col_chart:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=fd, open=o, high=h, low=l, close=c,
                                             increasing_line_color='green',
                                             decreasing_line_color='red',
                                             name="Price"))
                fig.add_trace(go.Scatter(
                    x=future_dates, y=future_prices,
                    mode="lines+markers",
                    line=dict(color="yellow", width=3),
                    marker=dict(size=6, symbol="circle"),
                    name="Forecast"
                ))
                fig.update_layout(
                    dragmode=False,
                    margin=dict(l=20,r=20,t=30,b=20)
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                st.caption(f"Source: {source}")

            # --- Stats Panel ---
            with col_stats:
                st.markdown(f"<span title='{tooltip['Signal']}'>**Signal:**</span> {sig}", unsafe_allow_html=True)
                st.markdown(f"<span title='{tooltip['Score']}'>**Score:**</span> {score}", unsafe_allow_html=True)
                st.markdown(f"<span title='{tooltip['WinRate']}'>**WinRate:**</span> {wr}%", unsafe_allow_html=True)
                st.markdown(f"<span title='{tooltip['Confidence']}'>**Confidence:**</span> {conf}%", unsafe_allow_html=True)
                st.markdown(f"<span title='{tooltip['RSI']}'>**RSI:**</span> {round(rsi[-1],2)}", unsafe_allow_html=True)
                st.markdown(f"<span title='{tooltip['MACD']}'>**MACD:**</span> {round(macd_v[-1],2)} / {round(sig_line[-1],2)}", unsafe_allow_html=True)
                st.markdown(f"<span title='{tooltip['Sentiment']}'>**Sentiment:**</span> {round(sentiment,2)}", unsafe_allow_html=True)

                # Portfolio P/L
                if sym in portfolio:
                    info = portfolio[sym]
                    profit_loss = (c[-1]-info["buy_price"])*info["amount"]
                    pl_color = "green" if profit_loss>=0 else "red"
                    st.markdown(f"<span>**Portfolio P/L:**</span> <span style='color:{pl_color}'>${profit_loss:.2f}</span>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)
            debug_info.append(f"{sym}: min={min(c)}, max={max(c)}, last={c[-1]}")

    except Exception as e:
        debug_info.append(f"{sym}: Exception occurred - {e}")

# =========================
# 🧰 Debug Panel at bottom
# =========================
with st.expander("🧰 Debug Panel"):
    for line in debug_info:
        st.write(line)
