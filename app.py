# app.py – Full polished version
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
        with st.expander("Firebase Initialization Error"):
            st.error(f"{type(e).__name__}: {e}")

# ---------------------------
# 📊 FETCH PRICE DATA (MULTI-SOURCE)
# ---------------------------
def get_price_data(symbol="BTCUSDT", days=30):
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
        errors.append(f"CoinGecko fetch failed for {symbol}: {e}")
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
        errors.append(f"CryptoCompare fetch failed for {symbol}: {e}")
    errors.append(f"Failed to fetch price data for {symbol}.")
    return np.array([]), [], errors

# ---------------------------
# 📊 HISTORICAL DATA HANDLER
# ---------------------------
def get_historical_data(symbol="BTCUSDT"):
    prices, dates, errors = [], [], []
    try:
        ref = db.reference(f"historical/{symbol}")
        data = ref.get() or {}
        if data:
            sorted_data = sorted(data.items(), key=lambda x: x[1]['time'])
            prices = [v["price"] for _,v in sorted_data]
            dates = [datetime.fromisoformat(v["time"]) for _,v in sorted_data]
            return np.array(prices), dates, errors
    except Exception as e:
        errors.append(f"Firebase read failed for {symbol}: {e}")
    # Fetch from API
    prices, dates, fetch_errors = get_price_data(symbol, days=30)
    errors += fetch_errors
    # Save to Firebase (only if not existing)
    try:
        ref = db.reference(f"historical/{symbol}")
        existing = ref.get() or {}
        existing_times = set(v["time"] for v in existing.values())
        for i in range(len(prices)):
            if dates[i].isoformat() not in existing_times:
                ref.push({"time": dates[i].isoformat(), "price": float(prices[i])})
    except Exception as e:
        errors.append(f"Firebase save failed for {symbol}: {e}")
    return np.array(prices), dates, errors

# ---------------------------
# ⏱️ HOURLY DATA (LAST 72 HOURS)
# ---------------------------
def get_hourly_data(symbol="BTCUSDT", hours=72):
    if symbol == "USDTIDR":
        return get_usd_idr_history(hours)
    fsym = symbol.replace("USDT","")
    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={hours-1}"
    data = requests.get(url, timeout=5).json()
    prices, dates = [], []
    if "Data" in data and "Data" in data["Data"]:
        raw = data["Data"]["Data"]
        prices = [x["close"] for x in raw]
        dates = [datetime.fromtimestamp(x["time"]) for x in raw]
    return np.array(prices), dates

# ---------------------------
# 💵 USD → IDR
# ---------------------------
def get_usd_idr_history(hours=72):
    try:
        start = (datetime.utcnow() - timedelta(hours=hours)).date()
        end = datetime.utcnow().date()
        url = f"https://api.exchangerate.host/timeseries?start_date={start}&end_date={end}&base=USD&symbols=IDR"
        data = requests.get(url, timeout=5).json()
        dates, prices = [], []
        for date_str, rate_data in data.get("rates", {}).items():
            dates.append(datetime.fromisoformat(date_str))
            prices.append(rate_data["IDR"])
        return np.array(prices), dates
    except:
        return np.array([]), []

# ---------------------------
# 📈 TECHNICAL INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[50], rsi.fillna(50)])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12).mean()
    exp2 = pd.Series(prices).ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    return macd.values, signal.values

# ---------------------------
# 📰 SENTIMENT
# ---------------------------
def get_sentiment():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data = requests.get(url, timeout=5).json()
        if "Data" not in data:
            return 0
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for article in data["Data"][:8]:
            title = article.get("title","")
            body = article.get("body","")[:150]
            text = f"{title}. {body}"
            vader = analyzer.polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity
            scores.append((vader + blob)/2)
        return max(-1, min(np.mean(scores),1))
    except:
        return 0

# ---------------------------
# 🔮 FORECAST
# ---------------------------
def arima_forecast(prices, steps):
    try:
        model = ARIMA(prices, order=(2,1,2))
        model_fit = model.fit()
        return list(model_fit.forecast(steps=steps))
    except:
        return [prices[-1]] * steps

def prophet_forecast(dates, prices, steps):
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
# 🧠 WEIGHTS
# ---------------------------
def load_weights():
    ref = db.reference("weights")
    data = ref.get()
    if not data:
        return {"rsi":1.0,"macd":1.0,"trend":1.0,"sentiment":1.0}
    return data

def save_weights(weights):
    db.reference("weights").set(weights)

# ---------------------------
# 🧠 SMART SIGNAL
# ---------------------------
def smart_signal(prices, rsi, macd, signal, sentiment, weights):
    score = 0
    contributions = {}
    val = 0
    if rsi[-1]<30: val = 2*weights["rsi"]
    elif rsi[-1]>70: val = -2*weights["rsi"]
    score += val; contributions["rsi"]=val
    val = (1 if macd[-1]>signal[-1] else -1)*weights["macd"]
    score += val; contributions["macd"]=val
    val = (1 if prices[-1]>np.mean(prices) else -1)*weights["trend"]
    score += val; contributions["trend"]=val
    if sentiment>0.2: val = 2*weights["sentiment"]
    elif sentiment<-0.2: val=-2*weights["sentiment"]
    else: val=0
    score += val; contributions["sentiment"]=val
    if score>=3: return "BUY", score, contributions
    elif score<=-3: return "SELL", score, contributions
    return "HOLD", score, contributions

# ---------------------------
# 🔁 AUTO-LEARNING
# ---------------------------
def update_weights(symbol, weights):
    ref = db.reference(f"history/{symbol}")
    data = list((ref.get() or {}).values())
    if len(data)<2: return weights
    prev = data[-2]; curr = data[-1]
    if prev.get("signal","HOLD")=="HOLD": return weights
    actual_up = curr["price"]>prev["price"]
    lr = 0.03
    contributions = prev.get("contributions",{})
    for key in weights:
        contrib = contributions.get(key,0)
        if contrib==0: continue
        if (contrib>0 and actual_up) or (contrib<0 and not actual_up):
            weights[key]*=(1+lr)
        else:
            weights[key]*=(1-lr)
        weights[key] = max(0.2,min(weights[key],3))
    return weights

# ---------------------------
# ☁️ SAVE HISTORY
# ---------------------------
def save_prediction(symbol, price, pred_price, signal, contributions):
    ref = db.reference(f"history/{symbol}")
    ref.push({
        "time": datetime.utcnow().isoformat(),
        "price": float(price),
        "predicted": float(pred_price),
        "signal": signal,
        "contributions": contributions
    })

def calculate_win_rate(symbol):
    ref = db.reference(f"history/{symbol}")
    data = list((ref.get() or {}).values())
    if len(data)<2: return 0
    wins,total = 0,0
    for i in range(len(data)-1):
        curr = data[i]; nxt = data[i+1]
        if curr.get("signal","HOLD")=="BUY" and nxt["price"]>curr["price"]: wins+=1
        elif curr.get("signal","HOLD")=="SELL" and nxt["price"]<curr["price"]: wins+=1
        total+=1
    return round((wins/total)*100,2)

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 Adaptive AI Crypto Bot + USDT → IDR")

coins_list = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XAIUSD","SOLUSDT","USDTIDR"]
symbols = st.multiselect("Select Coins", coins_list, default=["BTCUSDT"])

if st.button("Analyze"):
    for symbol in symbols:
        now = datetime.utcnow()
        if symbol=="USDTIDR":
            prices, dates = get_usd_idr_history(hours=72)
            if len(prices)==0:
                st.warning("Failed to fetch USDT → IDR data")
                continue
        else:
            prices, dates, errors = get_historical_data(symbol)
            # fetch latest 1 day hourly for crypto coins
            new_prices, new_dates = get_hourly_data(symbol)
            if len(new_prices)>0:
                prices = np.concatenate([prices,new_prices])
                dates = dates+new_dates
                try:
                    ref=db.reference(f"historical/{symbol}")
                    for i in range(len(new_prices)):
                        ref.push({"time":new_dates[i].isoformat(),"price":float(new_prices[i])})
                except: pass
            if errors:
                with st.expander(f"Errors for {symbol}"):
                    for err in errors: st.warning(err)

        # ---------------------------
        # Technicals + Signal
        # ---------------------------
        if len(prices)<10:
            st.warning(f"Not enough data for {symbol}")
            continue
        weights = load_weights()
        rsi = calculate_rsi(prices)
        macd_vals, signal_line = calculate_macd(prices)
        sentiment = get_sentiment() if symbol!="USDTIDR" else 0
        signal, score, contributions = smart_signal(prices, rsi, macd_vals, signal_line, sentiment, weights)

        # ---------------------------
        # Forecast
        # ---------------------------
        remaining_hours = 24 - now.hour
        future_prices, upper, lower = hybrid_forecast(prices, dates, remaining_hours)
        future_dates = [dates[-1] + timedelta(hours=i+1) for i in range(remaining_hours)]

        save_prediction(symbol, prices[-1], future_prices[0], signal, contributions)
        weights = update_weights(symbol, weights)
        save_weights(weights)

        # ---------------------------
        # Daily Change % + State
        # ---------------------------
        day_start_price = prices[0]
        percent_change = ((prices[-1]-day_start_price)/day_start_price)*100
        status = "Rising 📈" if percent_change>0 else "Dropping 📉"

        # ---------------------------
        # Plot
        # ---------------------------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dates, y=prices, mode='lines+markers', name='Actual'))
        fig.add_trace(go.Scatter(x=[dates[-1]]+future_dates, y=[prices[-1]]+future_prices,
                                 mode='lines+markers', name='Forecast', line=dict(dash='dash',color='red')))
        if symbol!="USDTIDR":
            fig.add_trace(go.Scatter(x=future_dates+future_dates[::-1],
                                     y=list(upper)+list(lower[::-1]), fill='toself',
                                     name='Confidence', opacity=0.2))
        st.subheader(f"{symbol} Analysis")
        st.plotly_chart(fig)

        # ---------------------------
        # Metrics
        # ---------------------------
        st.metric("Signal", signal)
        st.metric("Score", round(score,2))
        st.metric("Win Rate (%)", calculate_win_rate(symbol))
        st.metric("Current Price", round(prices[-1],2))
        st.metric("Change Today (%)", round(percent_change,2))
        st.metric("Current State", status)
