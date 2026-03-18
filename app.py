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
# 📊 FETCH PRICE DATA
# ---------------------------
def get_price_data(symbol="BTCUSDT", days=30):
    prices, dates, errors = [], [], []
    coin = symbol.replace("USDT","").lower()
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
    prices, dates, fetch_errors = get_price_data(symbol, days=30)
    errors += fetch_errors
    try:
        ref = db.reference(f"historical/{symbol}")
        for i in range(len(prices)):
            ref.push({"time": dates[i].isoformat(), "price": float(prices[i])})
    except Exception as e:
        errors.append(f"Firebase save failed for {symbol}: {e}")
    return np.array(prices), dates, errors

# ---------------------------
# ⏱️ HOURLY DATA
# ---------------------------
def get_hourly_data(symbol="BTCUSDT", hours=72):
    fsym = symbol.replace("USDT", "")
    url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={hours-1}"
    data = requests.get(url).json()
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
        now = datetime.utcnow()
        start = now - timedelta(hours=hours)
        url = f"https://api.exchangerate.host/timeseries?start_date={start.date()}&end_date={now.date()}&base=USD&symbols=IDR"
        data = requests.get(url, timeout=5).json()
        dates, prices = [], []
        for date_str, rate_data in data.get("rates", {}).items():
            dates.append(datetime.fromisoformat(date_str))
            prices.append(rate_data["IDR"])
        if len(prices) > 1:
            prices_interp = np.interp(
                pd.date_range(start=dates[0], end=dates[-1], freq='H').astype(np.int64),
                pd.Series(dates).astype(np.int64),
                prices
            )
            dates_interp = list(pd.date_range(start=dates[0], end=dates[-1], freq='H'))
            return np.array(prices_interp), dates_interp
        return np.array(prices), dates
    except:
        return np.array([]), []

# ---------------------------
# 📈 INDICATORS
# ---------------------------
def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return np.concatenate([[50], rsi.fillna(50)])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12).mean()
    exp2 = pd.Series(prices).ewm(span=26).mean()
    macd = exp1-exp2
    signal = macd.ewm(span=9).mean()
    return macd.values, signal.values

# ---------------------------
# 📰 SENTIMENT
# ---------------------------
def get_sentiment():
    try:
        url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data = requests.get(url, timeout=5).json()
        if "Data" not in data: return 0
        analyzer = SentimentIntensityAnalyzer()
        scores=[]
        for article in data["Data"][:8]:
            text=f"{article.get('title','')}. {article.get('body','')[:150]}"
            vader=analyzer.polarity_scores(text)["compound"]
            blob=TextBlob(text).sentiment.polarity
            scores.append((vader+blob)/2)
        return max(-1, min(np.mean(scores),1))
    except:
        return 0

# ---------------------------
# 🔮 FORECASTS
# ---------------------------
def arima_forecast(prices, steps):
    try:
        model = ARIMA(prices, order=(2,1,2))
        model_fit = model.fit()
        return list(model_fit.forecast(steps=steps))
    except: return [prices[-1]]*steps

def prophet_forecast(dates, prices, steps):
    df=pd.DataFrame({"ds":pd.to_datetime(dates),"y":prices})
    model=Prophet(daily_seasonality=True)
    model.fit(df)
    future=model.make_future_dataframe(periods=steps,freq='H')
    forecast=model.predict(future)
    return forecast["yhat"].tail(steps).values, forecast["yhat_upper"].tail(steps).values, forecast["yhat_lower"].tail(steps).values

def hybrid_forecast(prices, dates, steps):
    arima_preds = arima_forecast(prices, steps)
    prophet_preds, upper, lower = prophet_forecast(dates, prices, steps)
    final = [(a+p)/2 for a,p in zip(arima_preds, prophet_preds)]
    return final, upper, lower

# ---------------------------
# 🧠 SMART SIGNAL
# ---------------------------
def load_weights():
    ref=db.reference("weights")
    data=ref.get() or {}
    return data if data else {"rsi":1,"macd":1,"trend":1,"sentiment":1}

def save_weights(weights):
    db.reference("weights").set(weights)

def smart_signal(prices,rsi,macd,signal_line,sentiment,weights):
    score=0
    contributions={}
    val=2*weights["rsi"] if rsi[-1]<30 else (-2*weights["rsi"] if rsi[-1]>70 else 0)
    score+=val; contributions["rsi"]=val
    val=(1 if macd[-1]>signal_line[-1] else -1)*weights["macd"]
    score+=val; contributions["macd"]=val
    val=(1 if prices[-1]>np.mean(prices) else -1)*weights["trend"]
    score+=val; contributions["trend"]=val
    val=2*weights["sentiment"] if sentiment>0.2 else (-2*weights["sentiment"] if sentiment<-0.2 else 0)
    score+=val; contributions["sentiment"]=val
    if score>=3: return "BUY", score, contributions
    elif score<=-3: return "SELL", score, contributions
    return "HOLD", score, contributions

def update_weights(symbol, weights):
    ref=db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    if len(data)<2: return weights
    prev=data[-2]; curr=data[-1]
    if prev["signal"]=="HOLD": return weights
    actual_up=curr["price"]>prev["price"]
    lr=0.03
    contributions=prev.get("contributions",{})
    for key in weights:
        contrib=contributions.get(key,0)
        if contrib==0: continue
        if (contrib>0 and actual_up) or (contrib<0 and not actual_up): weights[key]*=(1+lr)
        else: weights[key]*=(1-lr)
        weights[key]=max(0.2,min(weights[key],3))
    return weights

def save_prediction(symbol, price, pred_price, signal, contributions):
    ref=db.reference(f"history/{symbol}")
    ref.push({"time":datetime.utcnow().isoformat(),"price":float(price),"predicted":float(pred_price),"signal":signal,"contributions":contributions})

def calculate_win_rate(symbol):
    ref=db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    if len(data)<2: return 0
    wins,total=0,0
    for i in range(len(data)-1):
        curr=data[i]; nxt=data[i+1]
        if curr["signal"]=="BUY" and nxt["price"]>curr["price"]: wins+=1
        elif curr["signal"]=="SELL" and nxt["price"]<curr["price"]: wins+=1
        total+=1
    return round((wins/total)*100,2)

# ---------------------------
# 🎨 UI
# ---------------------------
st.title("🚀 Adaptive AI Crypto Bot + USDT → IDR")
coins_list=["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XAIUSD","SOLUSDT","USDTIDR"]
symbols=st.multiselect("Select Coins", coins_list, default=["BTCUSDT","USDTIDR"])
timeframe=st.selectbox("Select Timeframe", ["1 Day","3 Days","5 Days","1 Month"])

if st.button("Analyze"):
    dashboard_data=[]
    now=datetime.utcnow()
    # First pass: collect summary data for dashboard
    for symbol in symbols:
        if symbol=="USDTIDR":
            prices, dates = get_usd_idr_history(hours=72)
            if len(prices)==0: continue
        else:
            prices, dates, errors = get_historical_data(symbol)
            new_prices, new_dates = get_hourly_data(symbol)
            if len(new_prices)>0:
                prices=np.concatenate([prices,new_prices])
                dates=dates+new_dates
                try:
                    ref=db.reference(f"historical/{symbol}")
                    for i in range(len(new_prices)):
                        ref.push({"time":new_dates[i].isoformat(),"price":float(new_prices[i])})
                except: pass
        if len(prices)<2: continue
        # Filter by timeframe
        if timeframe=="1 Day": cutoff=now-timedelta(days=1)
        elif timeframe=="3 Days": cutoff=now-timedelta(days=3)
        elif timeframe=="5 Days": cutoff=now-timedelta(days=5)
        else: cutoff=now-timedelta(days=30)
        filtered_prices=[p for d,p in zip(dates,prices) if d>=cutoff]
        if len(filtered_prices)<2: continue
        percent_change=((filtered_prices[-1]-filtered_prices[0])/filtered_prices[0])*100
        weights=load_weights()
        rsi=calculate_rsi(filtered_prices)
        macd_vals, signal_line = calculate_macd(filtered_prices)
        sentiment = get_sentiment() if symbol!="USDTIDR" else 0
        signal, score, contributions=smart_signal(filtered_prices,rsi,macd_vals,signal_line,sentiment,weights)
        dashboard_data.append({
            "Coin":symbol,
            "Current Price": round(filtered_prices[-1],2),
            "Change (%)": round(percent_change,2),
            "Signal": signal
        })
    # Show dashboard
    if dashboard_data:
        st.subheader("📊 Dashboard Summary")
        df_dash=pd.DataFrame(dashboard_data)
        st.dataframe(df_dash)

    col1,col2=st.columns(2)
    for idx,symbol in enumerate(symbols):
