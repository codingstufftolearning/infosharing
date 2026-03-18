# 🚀 AI Crypto Trading Bot (Full Version, Expanded)
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
# 🔐 FIREBASE INIT
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
        st.error(f"Firebase Initialization Error: {type(e).__name__}: {e}")

# ---------------------------
# 🌐 FETCH COIN PRICE DATA (DAILY & HOURLY)
# ---------------------------
def get_price_data(symbol="BTCUSDT", days=30):
    """Fetch daily historical prices from CoinGecko or fallback to CryptoCompare"""
    prices, dates, errors = [], [], []
    coin = symbol.replace("USDT","").lower()

    # CoinGecko
    try:
        url_cg = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart?vs_currency=usd&days={days}"
        data_cg = requests.get(url_cg, timeout=5).json()
        if "prices" in data_cg:
            prices = [x[1] for x in data_cg["prices"]]
            dates = [datetime.fromtimestamp(x[0]/1000) for x in data_cg["prices"]]
            return np.array(prices), dates, errors
        else:
            errors.append(f"CoinGecko returned unexpected data for {symbol}: {data_cg}")
    except Exception as e:
        errors.append(f"CoinGecko fetch failed: {e}")

    # CryptoCompare fallback
    try:
        url_cc = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol[:3]}&tsym=USD&limit={days-1}"
        data_cc = requests.get(url_cc, timeout=5).json()
        if "Data" in data_cc and "Data" in data_cc["Data"]:
            prices = [x["close"] for x in data_cc["Data"]["Data"]]
            dates = [datetime.fromtimestamp(x["time"]) for x in data_cc["Data"]["Data"]]
            return np.array(prices), dates, errors
        else:
            errors.append(f"CryptoCompare returned unexpected data for {symbol}: {data_cc}")
    except Exception as e:
        errors.append(f"CryptoCompare fetch failed: {e}")

    errors.append(f"Failed to fetch price data for {symbol}.")
    return np.array([]), [], errors

def get_hourly_data(symbol="BTCUSDT", hours=72):
    """Fetch hourly prices (used for today)"""
    fsym = symbol.replace("USDT", "")
    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={hours-1}"
    try:
        data = requests.get(url, timeout=5).json()
        prices, dates = [], []
        if "Data" in data and "Data" in data["Data"]:
            raw = data["Data"]["Data"]
            prices = [x["close"] for x in raw]
            dates = [datetime.fromtimestamp(x["time"]) for x in raw]
        return np.array(prices), dates
    except:
        return np.array([]), []

# ---------------------------
# 📊 HISTORICAL DATA HANDLER (DATABASE)
# ---------------------------
def get_historical_data(symbol="BTCUSDT"):
    """Get historical data from Firebase or fetch and save if not present"""
    prices, dates, errors = [], [], []

    # Firebase
    try:
        ref = db.reference(f"historical/{symbol}")
        data = ref.get() or {}
        if data:
            sorted_data = sorted(data.items(), key=lambda x: x[1]['time'])
            prices = [v["price"] for _,v in sorted_data]
            dates = [datetime.fromisoformat(v["time"]) for _,v in sorted_data]
            return np.array(prices), dates, errors
    except Exception as e:
        errors.append(f"Firebase read failed: {e}")

    # Fetch from API
    prices, dates, fetch_errors = get_price_data(symbol, days=30)
    errors += fetch_errors

    # Save to Firebase
    try:
        ref = db.reference(f"historical/{symbol}")
        for i in range(len(prices)):
            ref.push({"time": dates[i].isoformat(), "price": float(prices[i])})
    except Exception as e:
        errors.append(f"Firebase save failed: {e}")

    return np.array(prices), dates, errors

# ---------------------------
# 📈 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    if len(prices)<period: return np.zeros(len(prices))
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1+rs))
    return np.concatenate([[50], rsi])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.values, signal.values

# ---------------------------
# 📰 SENTIMENT
# ---------------------------
def get_sentiment():
    """Fetch crypto news sentiment from CryptoCompare"""
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data = requests.get(url, timeout=5).json()
        if "Data" not in data: return 0
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for article in data["Data"][:8]:
            title = article.get("title","")
            body = article.get("body","")[:150]
            text = f"{title}. {body}"
            vader = analyzer.polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity
            scores.append((vader+blob)/2)
        return max(-1, min(np.mean(scores),1))
    except:
        return 0

# ---------------------------
# 🔮 FORECASTS (ARIMA + PROPHET + HYBRID)
# ---------------------------
def arima_forecast(prices, steps):
    try:
        model = ARIMA(prices, order=(2,1,2))
        fit = model.fit()
        return list(fit.forecast(steps=steps))
    except:
        return [prices[-1]]*steps

def prophet_forecast(dates, prices, steps):
    if len(prices)<2: return [prices[-1]]*steps, [prices[-1]]*steps, [prices[-1]]*steps
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": prices})
    model = Prophet(daily_seasonality=True)
    model.fit(df)
    future = model.make_future_dataframe(periods=steps, freq='H')
    forecast = model.predict(future)
    return forecast["yhat"].tail(steps).values, forecast["yhat_upper"].tail(steps).values, forecast["yhat_lower"].tail(steps).values

def hybrid_forecast(prices, dates, steps):
    arima_preds = arima_forecast(prices, steps)
    prophet_preds, upper, lower = prophet_forecast(dates, prices, steps)
    final = [(a+p)/2 for a,p in zip(arima_preds, prophet_preds)]
    return final, upper, lower

# ---------------------------
# 🧠 SMART SIGNAL + WEIGHTS
# ---------------------------
def load_weights():
    try:
        ref = db.reference("weights")
        data = ref.get() or {}
        return data if data else {"rsi":1,"macd":1,"trend":1,"sentiment":1}
    except:
        return {"rsi":1,"macd":1,"trend":1,"sentiment":1}

def save_weights(weights):
    try:
        db.reference("weights").set(weights)
    except:
        pass

def smart_signal(prices, rsi, macd, signal, sentiment, weights):
    score = 0
    contribs = {}
    # RSI
    val = 0
    if rsi[-1]<30: val=2*weights["rsi"]
    elif rsi[-1]>70: val=-2*weights["rsi"]
    score+=val; contribs["rsi"]=val
    # MACD
    val=(1 if macd[-1]>signal[-1] else -1)*weights["macd"]
    score+=val; contribs["macd"]=val
    # Trend
    val=(1 if prices[-1]>np.mean(prices) else -1)*weights["trend"]
    score+=val; contribs["trend"]=val
    # Sentiment
    val=0
    if sentiment>0.2: val=2*weights["sentiment"]
    elif sentiment<-0.2: val=-2*weights["sentiment"]
    score+=val; contribs["sentiment"]=val
    # Final
    if score>=3: return "BUY",score,contribs
    elif score<=-3: return "SELL",score,contribs
    return "HOLD",score,contribs

def update_weights(symbol, weights):
    ref = db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    if len(data)<2: return weights
    prev = data[-2]; curr = data[-1]
    if prev.get("signal","HOLD")=="HOLD": return weights
    actual_up = curr["price"]>prev["price"]
    lr = 0.03
    contribs=prev.get("contributions",{})
    for k in weights:
        c=contribs.get(k,0)
        if c==0: continue
        if (c>0 and actual_up) or (c<0 and not actual_up):
            weights[k]*=(1+lr)
        else:
            weights[k]*=(1-lr)
        weights[k]=max(0.2,min(weights[k],3))
    return weights

def save_prediction(symbol, price, pred_price, signal, contribs):
    ref = db.reference(f"history/{symbol}")
    ref.push({"time":datetime.utcnow().isoformat(),
              "price":float(price),
              "predicted":float(pred_price),
              "signal":signal,
              "contributions":contribs})

def calculate_win_rate(symbol):
    ref = db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    if len(data)<2: return 0
    wins,total=0,0
    for i in range(len(data)-1):
        curr=data[i]; nxt=data[i+1]
        if curr.get("signal","HOLD")=="BUY" and nxt["price"]>curr["price"]: wins+=1
        elif curr.get("signal","HOLD")=="SELL" and nxt["price"]<curr["price"]: wins+=1
        total+=1
    return round((wins/total)*100,2)

# ---------------------------
# 🌏 USDT → IDR RATE
# ---------------------------
def get_usdt_idr():
    try:
        data=requests.get("https://api.exchangerate.host/convert?from=USD&to=IDR").json()
        return data.get("result",None)
    except:
        return None

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 Adaptive AI Crypto Bot")

# Show USDT → IDR
idr_rate = get_usdt_idr()
st.markdown(f"**USDT → IDR Rate:** {idr_rate if idr_rate else 'N/A'}")

# Coin selector
coins_list=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XAIUSD","SOLUSDT"]
symbols=st.multiselect("Select Coins",coins_list,default=["BTCUSDT"])

# Timeframe selector
timeframe=st.selectbox("Select timeframe",["1 Day","3 Days","5 Days","1 Month"])

# Results state
if "results" not in st.session_state: st.session_state.results={}

# Analyze button
if st.button("Analyze"):
    st.session_state.results={}
    weights=load_weights()

    for symbol in symbols:
        # Historical daily data
        prices, dates, errors = get_historical_data(symbol)
        if len(prices)==0:
            with st.expander(f"No Data / Errors for {symbol}"):
                for e in errors: st.warning(e)
            continue

        # Hourly data today
        hourly_prices, hourly_dates = get_hourly_data(symbol, hours=72)
        today_start = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
        filtered=[(d,p) for d,p in zip(hourly_dates,hourly_prices) if d>=today_start]
        if filtered:
            h_dates=[x[0] for x in filtered]
            h_prices=np.array([x[1] for x in filtered])
            # Merge with historical daily if needed
            prices = np.concatenate([prices,h_prices])
            dates = dates + h_dates

        # Indicators
        rsi = calculate_rsi(prices)
        macd, signal_line = calculate_macd(prices)
        sentiment = get_sentiment()

        # Smart signal
        trade_signal, score, contribs = smart_signal(prices, rsi, macd, signal_line, sentiment, weights)

        # Forecast remaining hours today
        now = datetime.now()
        remaining_hours = 24 - now.hour
        future_prices, upper, lower = hybrid_forecast(prices, dates, remaining_hours)
        future_dates = [dates[-1]+timedelta(hours=i+1) for i in range(remaining_hours)]

        # Save prediction
        save_prediction(symbol, prices[-1], future_prices[0], trade_signal, contribs)
        weights=update_weights(symbol,weights)
        save_weights(weights)

        # Store result
        st.session_state.results[symbol]={"prices":prices,"dates":dates,
                                          "future_prices":future_prices,"future_dates":future_dates,
                                          "signal":trade_signal,"score":score,"contribs":contribs}

# ---------------------------
# 📊 DISPLAY RESULTS
# ---------------------------
if st.session_state.results:
    # Summary table
    summary=[]
    for s,r in st.session_state.results.items():
        today_prices = [p for d,p in zip(r["dates"],r["prices"]) if d.date()==datetime.now().date()]
        if today_prices:
            start=today_prices[0]; end=today_prices[-1]
            change_pct = ((end-start)/start)*100
            summary.append({"Coin":s,"Signal":r["signal"],"Score":round(r["score"],2),
                            "Change (%)":round(change_pct,2)})
    if summary:
        st.subheader("📋 Summary")
        st.dataframe(pd.DataFrame(summary))

    # Individual charts
    for s,r in st.session_state.results.items():
        st.subheader(f"Analysis for {s}")

        fig = go.Figure()
        # Historical + hourly prices
        fig.add_trace(go.Scatter(x=r["dates"],y=r["prices"],mode='lines+markers',name='Price',line=dict(color='blue')))
        # Forecast
        fig.add_trace(go.Scatter(x=[r["dates"][-1]]+r["future_dates"],
                                 y=[r["prices"][-1]]+r["future_prices"],
                                 mode='lines+markers',name='Forecast',
                                 line=dict(dash='dash',color='red')))
        # Confidence band
        fig.add_trace(go.Scatter(x=r["future_dates"]+r["future_dates"][::-1],
                                 y=list(r["future_prices"]+r["future_prices"][::-1]),
                                 fill='toself',name='Confidence',opacity=0.2))

        st.plotly_chart(fig)

        # Metrics
        st.metric("Signal", r["signal"])
        st.metric("Score", round(r["score"],2))
        st.metric("Next Hour Price", round(r["future_prices"][0],2))
        st.metric("Win Rate (%)", calculate_win_rate(s))
