# app.py (Updated with selectable timeframes)
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.arima.model import ARIMA
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
# 📊 FETCH PRICE DATA (MULTI-SOURCE FALLBACK)
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

    # CryptoCompare
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
        errors.append(f"Firebase read failed for {symbol}: {e}")

    # Fetch from API
    prices, dates, fetch_errors = get_price_data(symbol, days=30)
    errors += fetch_errors

    # Save to Firebase
    try:
        ref = db.reference(f"historical/{symbol}")
        for i in range(len(prices)):
            ref.push({"time": dates[i].isoformat(), "price": float(prices[i])})
    except Exception as e:
        errors.append(f"Firebase save failed for {symbol}: {e}")

    return np.array(prices), dates, errors

# ---------------------------
# 📈 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    if len(prices) < period: return np.zeros(len(prices))
    delta = np.diff(prices)
    gain = np.maximum(delta, 0)
    loss = np.abs(np.minimum(delta, 0))
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([np.zeros(1), rsi])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd.values, signal.values

# ---------------------------
# 📰 SENTIMENT
# ---------------------------
def get_sentiment(texts=None):
    if texts is None:
        texts = [
            "Bitcoin is rising fast",
            "Crypto market crash fears",
            "Strong bullish momentum ahead"
        ]
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for text in texts:
        vader = analyzer.polarity_scores(text)["compound"]
        blob = TextBlob(text).sentiment.polarity
        scores.append((vader + blob)/2)
    return np.mean(scores)

# ---------------------------
# 🔮 FORECAST
# ---------------------------
def forecast_price(prices):
    if len(prices)<10: return float(prices[-1])
    try:
        model = ARIMA(prices, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return float(forecast[0])
    except:
        return float(prices[-1])

# ---------------------------
# 🧠 WEIGHTS
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

# ---------------------------
# 📊 SMART PREDICTION
# ---------------------------
def smart_prediction(prices, rsi, macd, signal, sentiment, weights):
    score=0
    if rsi[-1]<30: score+=2*weights.get("rsi",1)
    elif rsi[-1]>70: score-=2*weights.get("rsi",1)
    if macd[-1]>signal[-1]: score+=1*weights.get("macd",1)
    else: score-=1*weights.get("macd",1)
    if prices[-1]>prices.mean(): score+=1*weights.get("trend",1)
    else: score-=1*weights.get("trend",1)
    if sentiment>0.2: score+=2*weights.get("sentiment",1)
    elif sentiment<-0.2: score-=2*weights.get("sentiment",1)
    if score>=3: return "BUY",score
    elif score<=-3: return "SELL",score
    else: return "HOLD",score

# ---------------------------
# ☁️ SAVE & LOAD HISTORY
# ---------------------------
def save_data(symbol, price, prediction):
    try:
        ref = db.reference(f"history/{symbol}")
        ref.push({"time": datetime.utcnow().isoformat(),"price":float(price),"prediction":prediction})
    except:
        pass

def load_history(symbol):
    try:
        ref = db.reference(f"history/{symbol}")
        data = ref.get() or {}
        if not data: return []
        return [v["price"] for v in data.values()]
    except:
        return []

def calculate_win_rate(symbol):
    try:
        ref = db.reference(f"history/{symbol}")
        data = ref.get() or {}
        if len(data)<2: return 0
        values=list(data.values())
        wins,total=0,0
        for i in range(len(values)-1):
            current=values[i]
            next_price=values[i+1]["price"]
            if current["prediction"]=="BUY" and next_price>current["price"]: wins+=1
            elif current["prediction"]=="SELL" and next_price<current["price"]: wins+=1
            total+=1
        return round((wins/total)*100,2) if total>0 else 0
    except:
        return 0

# ---------------------------
# 🎨 STREAMLIT UI
# ---------------------------
st.title("🚀 AI Crypto Trading Bot")
coins_list = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XAIUSD"]
symbols = st.multiselect("Select Coins", coins_list, default=["BTCUSDT"])

# Timeframe selector
timeframe = st.selectbox("Select timeframe", ["1 Day","3 Days","5 Days","1 Month"])

# Session state
if "results" not in st.session_state: st.session_state.results = {}

if st.button("Analyze"):
    st.session_state.results = {}
    weights = load_weights()
    for symbol in symbols:
        prices, dates, errors = get_historical_data(symbol)
        if len(prices)==0:
            with st.expander(f"No Data / Errors for {symbol}"):
                for err in errors: st.warning(err)
            continue

        # Fetch latest 1 day data
        new_prices,new_dates,new_errors=get_price_data(symbol,days=1)
        if len(new_prices)>0:
            prices=np.concatenate([prices,new_prices])
            dates=dates+new_dates
            try:
                ref=db.reference(f"historical/{symbol}")
                for i in range(len(new_prices)):
                    ref.push({"time":new_dates[i].isoformat(),"price":float(new_prices[i])})
            except: pass
        if new_errors: errors+=new_errors

        rsi=calculate_rsi(prices)
        macd,signal=calculate_macd(prices)
        sentiment=get_sentiment()
        prediction,score=smart_prediction(prices,rsi,macd,signal,sentiment,weights)
        next_price=forecast_price(prices)
        save_data(symbol,prices[-1],prediction)
        save_weights(weights)

        st.session_state.results[symbol] = {
            "prices": prices,
            "dates": dates,
            "prediction": prediction,
            "score": score,
            "next_price": next_price,
            "errors": errors
        }

# Display results
for symbol,result in st.session_state.results.items():
    st.subheader(f"Analysis for {symbol}")
    prices = result["prices"]
    dates = result["dates"]
    next_price = result["next_price"]

    # Filter by selected timeframe
    now = datetime.utcnow()
    if timeframe=="1 Day": cutoff=now - timedelta(days=1)
    elif timeframe=="3 Days": cutoff=now - timedelta(days=3)
    elif timeframe=="5 Days": cutoff=now - timedelta(days=5)
    else: cutoff=now - timedelta(days=30)
    filtered_prices=[p for d,p in zip(dates,prices) if d>=cutoff]
    filtered_dates=[d for d in dates if d>=cutoff]
    if len(filtered_prices)==0: continue

    extended_dates=filtered_dates + [filtered_dates[-1]+timedelta(days=1)]
    extended_prices=np.append(filtered_prices,next_price)

    fig=go.Figure()
    fig.add_trace(go.Scatter(x=filtered_dates,y=filtered_prices,mode='lines+markers',name='Historical Price'))
    fig.add_trace(go.Scatter(x=[filtered_dates[-1],filtered_dates[-1]+timedelta(days=1)],
                             y=[filtered_prices[-1],next_price],
                             mode='lines+markers',name='Estimated Price',
                             line=dict(dash='dash',color='red')))
    fig.update_yaxes(range=[min(extended_prices)*0.98,max(extended_prices)*1.02])
    st.plotly_chart(fig)

    st.metric("Prediction", result["prediction"])
    st.metric("Score", result["score"])
    st.metric("Next Price", round(next_price,2))
    st.metric("Win Rate (%)", calculate_win_rate(symbol))

    if result["errors"]:
        with st.expander("Errors / Warnings"):
            for err in result["errors"]:
                st.warning(err)
