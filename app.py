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
COINS = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","SOLUSDT","XAIUSD","XRPUSDT","LUNAUSDT","MATICUSDT"]  # DOGE removed
TIMEFRAMES = {"1 Day":1,"3 Days":3,"5 Days":5,"1 Month":30}
RSI_PERIOD = 14
ARIMA_ORDER = (2,1,2)
ROLLING_HOURS = 7*24

# ---------------------------
# 📊 Data Fetching
# ---------------------------
@st.cache_data(ttl=600)
def fetch_historical(symbol="BTCUSDT", days=30):
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
# 📈 Indicators
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
# 📰 Sentiment
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
# 🧠 Signals
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
st.title("🚀 AI Crypto Dashboard")

# Portfolio Input Collapsible
st.subheader("💰 Portfolio Tracker (Optional)")
portfolio = {}
portfolio_buy = {}
with st.expander("Enter Holdings & Buy Price"):
    for coin in COINS:
        portfolio[coin] = st.number_input(f"Holdings for {coin}", min_value=0.0, step=0.01, key=f"hold_{coin}")
        portfolio_buy[coin] = st.number_input(f"Buy Price for {coin}", min_value=0.0, step=0.01, key=f"buy_{coin}")

symbols = st.multiselect("Select Coins", COINS, default=["BTCUSDT"])
timeframe = st.selectbox("Select Timeframe", list(TIMEFRAMES.keys()))

# Color and UI helpers
def apply_colors(val):
    if val == "BUY": return "background-color: #b6fcb6; color: black"
    elif val == "SELL": return "background-color: #fcb6b6; color: black"
    elif val == "HOLD": return "background-color: #fff79a; color: black"
    try:
        num = float(val)
        if num > 0: return "color: green"
        elif num < 0: return "color: red"
    except: pass
    return ""

def metric_with_tooltip(name, value, tooltip):
    st.markdown(f"<span title='{tooltip}'>{name}: {value}</span>", unsafe_allow_html=True)

# ---------------------------
# Analyze Button
# ---------------------------
if st.button("Analyze"):
    with st.spinner("Analyzing coins..."):
        summary_data=[]
        coin_details=[]
        for sym in symbols:
            prices, dates = fetch_historical(sym)
            hr_prices, hr_dates = fetch_hourly(sym)
            # Combine daily + hourly
            all_prices = np.concatenate([prices, hr_prices]) if len(hr_prices)>0 else prices
            all_dates = dates + hr_dates if len(hr_dates)>0 else dates
            # Sort & remove duplicates
            sorted_pairs = sorted(zip(all_dates, all_prices), key=lambda x: x[0])
            seen = set()
            fp, fd = [], []
            for d,p in sorted_pairs:
                if d not in seen:
                    fd.append(d)
                    fp.append(p)
                    seen.add(d)
            if len(fp)<2: continue

            # Forecast
            next_p, upper, lower = hybrid_forecast(fp, fd, 1)

            # Indicators
            recent_prices = fp[-ROLLING_HOURS:] if len(fp)>ROLLING_HOURS else fp
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
            portfolio_value = portfolio[sym]*current_price
            profit_loss = portfolio_value - (portfolio[sym]*portfolio_buy[sym])

            summary_data.append({
                "Coin": sym,
                "Current Price": round(current_price,2),
                "Signal": sig,
                "Score": round(score,2),
                "Win Rate (%)": win_rate,
                "Change (%)": round(change_pct,2),
                "Confidence (%)": confidence,
                "Portfolio Value": round(portfolio_value,2),
                "P/L": round(profit_loss,2)
            })

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fd, y=fp, mode='lines+markers', name=f"{sym} Price"))
            fig.add_trace(go.Scatter(
                x=[fd[-1], fd[-1]+timedelta(days=1)],
                y=[fp[-1], next_p[0]],
                mode='lines+markers',
                name='Forecast',
                line=dict(dash='dash', color='red')
            ))

            coin_details.append((sym, fig, current_price, sig, score, win_rate, change_pct, confidence, portfolio_value, profit_loss))

        # Summary Table at Top
        if summary_data:
            st.subheader("📊 Summary Table")
            df_summary = pd.DataFrame(summary_data)
            st.dataframe(df_summary.style.applymap(apply_colors))

        # Per-Coin Charts & Stats
        for (sym, fig, current_price, sig, score, win_rate, change_pct, confidence, portfolio_value, profit_loss) in coin_details:
            with st.container():
                st.markdown(f"""
                    <div style="
                        border:1px solid #ccc;
                        border-radius:10px;
                        padding:15px;
                        margin-bottom:15px;
                        background-color:#f7f7f7;">
                        <h3 style='margin-bottom:10px'>{sym} - ${round(current_price,2)}</h3>
                    """, unsafe_allow_html=True)
                col1, col2 = st.columns([3,1])
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    metric_with_tooltip("Signal", sig, "BUY/SELL/HOLD")
                    metric_with_tooltip("Score", round(score,2), "Weighted indicator score")
                    metric_with_tooltip("Win Rate (%)", win_rate, "Historical signal accuracy")
                    metric_with_tooltip("Change (%)", round(change_pct,2), "Price change in timeframe")
                    metric_with_tooltip("Confidence (%)", confidence, "Forecast confidence")
                    if portfolio[sym]>0:
                        with st.expander("Portfolio Info"):
                            metric_with_tooltip("Holdings", portfolio[sym], "Number of coins held")
                            metric_with_tooltip("Buy Price", portfolio_buy[sym], "Price purchased")
                            metric_with_tooltip("Current Value", round(portfolio_value,2), "Current value")
                            metric_with_tooltip("P/L", round(profit_loss,2), "Profit/Loss")
                st.markdown("</div>", unsafe_allow_html=True)
