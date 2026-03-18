import streamlit as st
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

# ---------------------------
# 🔐 Firebase Initialization
# ---------------------------
if not firebase_admin._apps:
    try:
        firebase_dict = dict(st.secrets["firebase"])
        firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")
        cred = credentials.Certificate(firebase_dict)
        firebase_admin.initialize_app(cred, {
            "databaseURL": firebase_dict["databaseURL"]
        })
    except Exception as e:
        st.error(f"Firebase Initialization Error: {e}")

# ---------------------------
# 🔧 Config
# ---------------------------
COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XAIUSD","SOLUSDT"]
TIMEFRAMES = {"1 Day":1,"3 Days":3,"5 Days":5,"1 Month":30}
RSI_PERIOD = 14
ARIMA_ORDER = (2,1,2)
ROLLING_DAYS_FOR_INDICATORS = 7

# ---------------------------
# 📊 Data Fetching with Caching
# ---------------------------
@st.cache_data(ttl=600)
def fetch_usdt_idr_rate():
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=tether&vs_currencies=idr"
        data = requests.get(url, timeout=5).json()
        rate = data.get("tether", {}).get("idr", None)
        return round(rate,2) if rate else None
    except:
        return None

@st.cache_data(ttl=600)
def fetch_historical(symbol="BTCUSDT", days=30):
    # Firebase read first
    prices, dates = [], []
    try:
        ref = db.reference(f"historical/{symbol}")
        data = ref.get() or {}
        if data:
            sorted_data = sorted(data.items(), key=lambda x: x[1]['time'])
            prices = [v["price"] for _,v in sorted_data]
            dates = [datetime.fromisoformat(v["time"]) for _,v in sorted_data]
            return np.array(prices), dates
    except:
        pass
    # API fallback
    coin = symbol.replace("USDT","").lower()
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
        data = requests.get(url, timeout=5).json()
        prices = [x[1] for x in data.get("prices",[])]
        dates = [datetime.fromtimestamp(x[0]/1000) for x in data.get("prices",[])]
        # Save to Firebase
        try:
            ref = db.reference(f"historical/{symbol}")
            for i in range(len(prices)):
                ref.push({"time": dates[i].isoformat(), "price": float(prices[i])})
        except:
            pass
        return np.array(prices), dates
    except:
        return np.array([]), []

@st.cache_data(ttl=300)
def fetch_hourly(symbol="BTCUSDT", hours=72):
    fsym = symbol.replace("USDT","")
    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={hours-1}"
    try:
        data = requests.get(url, timeout=5).json()
        prices = [x["close"] for x in data.get("Data", {}).get("Data",[])]
        dates = [datetime.fromtimestamp(x["time"]) for x in data.get("Data", {}).get("Data",[])]
        return np.array(prices), dates
    except:
        return np.array([]), []

# ---------------------------
# 📈 Technical Indicators
# ---------------------------
def calculate_rsi(prices, period=RSI_PERIOD):
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(period,min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period,min_periods=1).mean()
    rs = avg_gain/(avg_loss+1e-9)
    rsi = 100 - (100/(1+rs))
    return np.concatenate([[50], rsi])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12,adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26,adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9,adjust=False).mean()
    return macd.values, signal.values

# ---------------------------
# 📰 Sentiment Analysis
# ---------------------------
@st.cache_data(ttl=600)
def get_sentiment():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data = requests.get(url, timeout=5).json()
        scores=[]
        for a in data.get("Data", [])[:8]:
            text = f"{a.get('title','')}. {a.get('body','')[:150]}"
            vader = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity
            scores.append((vader+blob)/2)
        return np.mean(scores) if scores else 0
    except:
        return 0

# ---------------------------
# 🔮 Forecasting
# ---------------------------
def arima_forecast(prices, steps):
    try:
        model = ARIMA(prices, order=ARIMA_ORDER)
        model_fit = model.fit()
        return list(model_fit.forecast(steps=steps))
    except:
        return [prices[-1]]*steps

def prophet_forecast(dates, prices, steps):
    df = pd.DataFrame({"ds": pd.to_datetime(dates),"y":prices})
    model=Prophet(daily_seasonality=True)
    model.fit(df)
    future=model.make_future_dataframe(periods=steps,freq="H")
    f= model.predict(future)
    return (f["yhat"].tail(steps).values,
            f["yhat_upper"].tail(steps).values,
            f["yhat_lower"].tail(steps).values)

def hybrid_forecast(prices, dates, steps=1):
    ar = arima_forecast(prices, steps)
    pr, upper, lower = prophet_forecast(dates, prices, steps)
    combined = [(a+p)/2 for a,p in zip(ar,pr)]
    return combined, upper, lower

# ---------------------------
# 🧠 Smart Signal & Weights
# ---------------------------
def load_weights():
    ref=db.reference("weights")
    data=ref.get() or {}
    return data if data else {"rsi":1,"macd":1,"trend":1,"sentiment":1}

def save_weights(weights):
    db.reference("weights").set(weights)

def smart_signal(prices, rsi, macd, signal, sentiment, weights):
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
    if prev.get("signal","HOLD")=="HOLD": return weights
    actual_up = curr["price"]>prev["price"]
    lr=0.03
    for key in weights:
        contrib=prev.get("contributions",{}).get(key,0)
        if contrib==0: continue
        if (contrib>0 and actual_up) or (contrib<0 and not actual_up): weights[key]*=(1+lr)
        else: weights[key]*=(1-lr)
        weights[key]=max(0.2,min(weights[key],3))
    return weights

def save_prediction(symbol, price, pred, sig, contrib):
    ref=db.reference(f"history/{symbol}")
    ref.push({"time":datetime.utcnow().isoformat(),"price":float(price),
              "predicted":float(pred),"signal":sig,"contributions":contrib})

def calculate_win_rate(symbol):
    ref=db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    valid_data = [d for d in data if "price" in d]
    if len(valid_data) < 2: return 0
    wins,total=0,0
    for i in range(len(valid_data)-1):
        curr,nxt=valid_data[i],valid_data[i+1]
        sig=curr.get("signal","HOLD")
        cp,nxt_p = curr.get("price",0), nxt.get("price",0)
        if sig=="BUY" and nxt_p>cp: wins+=1
        elif sig=="SELL" and nxt_p<cp: wins+=1
        total+=1
    return round((wins/total)*100,2) if total>0 else 0

def calculate_confidence(upper, lower, last_price):
    conf = max(0, min(1, 1 - (upper[0] - lower[0])/last_price))
    return round(conf*100,2)

# ---------------------------
# 🎨 Streamlit UI
# ---------------------------
st.title("🚀 AI Crypto Dashboard + Live USDT→IDR")

usdt_idr = fetch_usdt_idr_rate()
if usdt_idr:
    st.metric("USDT → IDR", usdt_idr)
else:
    st.warning("Failed to fetch USDT → IDR rate")

symbols = st.multiselect("Select Coins", COINS, default=["BTCUSDT"])
timeframe = st.selectbox("Select Timeframe", list(TIMEFRAMES.keys()))

# Helper: color-coded table
def apply_colors(val):
    if val == "BUY":
        return "background-color: #b6fcb6; color: black"
    elif val == "SELL":
        return "background-color: #fcb6b6; color: black"
    elif val == "HOLD":
        return "background-color: #fff79a; color: black"
    try:
        num = float(val)
        if num > 0: return "color: green"
        elif num < 0: return "color: red"
    except: pass
    return ""

if st.button("Analyze"):
    with st.spinner("Analyzing coins..."):
        summary_data=[]
        for sym in symbols:
            prices, dates = fetch_historical(sym)
            hr_prices, hr_dates = fetch_hourly(sym)
            if len(hr_prices)>0:
                prices=np.concatenate([prices, hr_prices])
                dates=dates+hr_dates

            cutoff = datetime.utcnow() - timedelta(days=TIMEFRAMES[timeframe])
            fp = [p for d,p in zip(dates,prices) if d>=cutoff]
            fd = [d for d in dates if d>=cutoff]
            if len(fp)<2: continue

            # Forecast
            next_p, upper, lower = hybrid_forecast(fp, fd, 1)

            # Indicators & sentiment
            recent_prices = fp[-ROLLING_DAYS_FOR_INDICATORS*24:] if len(fp)>24 else fp
            rsi = calculate_rsi(recent_prices)
            macd_v, sig_line = calculate_macd(recent_prices)
            sentiment = get_sentiment()
            weights = load_weights()

            # Signal
            sig, score, contrib = smart_signal(recent_prices, rsi, macd_v, sig_line, sentiment, weights)
            save_prediction(sym, fp[-1], next_p[0], sig, contrib)
            weights = update_weights(sym, weights)
            save_weights(weights)

            # Metrics
            win_rate = calculate_win_rate(sym)
            change_pct = ((fp[-1]-fp[0])/fp[0])*100
            current_price = fp[-1]
            confidence = calculate_confidence(upper, lower, current_price)

            summary_data.append({
                "Coin": sym,
                "Current Price": round(current_price,2),
                "Signal": sig,
                "Score": round(score,2),
                "Win Rate (%)": win_rate,
                "Change (%)": round(change_pct,2),
                "Confidence (%)": confidence
            })

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fd, y=fp, mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(
                x=[fd[-1], fd[-1]+timedelta(days=1)],
                y=[fp[-1], next_p[0]],
                mode='lines+markers', name='Forecast',
                line=dict(dash='dash', color='red')
            ))
            # Upper/lower bounds
            fig.add_trace(go.Scatter(
                x=[fd[-1], fd[-1]+timedelta(days=1)],
                y=[upper[0], upper[0]],
                mode='lines', line=dict(dash='dot', color='orange'), name='Upper')
            )
            fig.add_trace(go.Scatter(
                x=[fd[-1], fd[-1]+timedelta(days=1)],
                y=[lower[0], lower[0]],
                mode='lines', line=dict(dash='dot', color='orange'), name='Lower')
            )

            col1, col2 = st.columns([3,1])
            with col1:
                st.subheader(f"{sym} Chart")
                st.plotly_chart(fig)
            with col2:
                st.metric("Current Price", round(current_price,2))
                st.metric("Signal", sig)
                st.metric("Score", round(score,2))
                st.metric("Win Rate (%)", win_rate)
                st.metric("Change (%)", round(change_pct,2))
                st.metric("Confidence (%)", confidence)

        # Summary Table
        if summary_data:
            st.subheader("📊 Summary Table")
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary.style.applymap(apply_colors))
