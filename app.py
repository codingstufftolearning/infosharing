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

if not firebase_admin._apps:
    firebase_dict = dict(st.secrets["firebase"])
    firebase_dict["private_key"] = firebase_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(firebase_dict)
    firebase_admin.initialize_app(cred, {"databaseURL": firebase_dict["databaseURL"]})

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
        else: errors.append(f"CoinGecko returned unexpected data for {symbol}")
    except Exception as e: errors.append(f"CoinGecko fetch failed for {symbol}: {e}")
    try:
        url_cc = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol[:3]}&tsym=USD&limit={days-1}"
        data_cc = requests.get(url_cc, timeout=5).json()
        if "Data" in data_cc and "Data" in data_cc["Data"]:
            prices = [x["close"] for x in data_cc["Data"]["Data"]]
            dates = [datetime.fromtimestamp(x["time"]) for x in data_cc["Data"]["Data"]]
            return np.array(prices), dates, errors
        else: errors.append(f"CryptoCompare returned unexpected data for {symbol}")
    except Exception as e: errors.append(f"CryptoCompare fetch failed for {symbol}: {e}")
    errors.append(f"Failed to fetch price data for {symbol}.")
    return np.array([]), [], errors

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
    except Exception as e: errors.append(f"Firebase read failed for {symbol}: {e}")
    prices, dates, fetch_errors = get_price_data(symbol, days=30)
    errors += fetch_errors
    try:
        ref = db.reference(f"historical/{symbol}")
        for i in range(len(prices)):
            ref.push({"time": dates[i].isoformat(), "price": float(prices[i])})
    except Exception as e: errors.append(f"Firebase save failed for {symbol}: {e}")
    return np.array(prices), dates, errors

def get_hourly_data(symbol="BTCUSDT", hours=72):
    fsym = symbol.replace("USDT", "")
    try:
        url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={hours-1}"
        data = requests.get(url, timeout=5).json()
        prices, dates = [], []
        if "Data" in data and "Data" in data["Data"]:
            raw = data["Data"]["Data"]
            prices = [x["close"] for x in raw]
            dates = [datetime.fromtimestamp(x["time"]) for x in raw]
            return np.array(prices), dates
    except: pass
    return np.array([]), []

def calculate_rsi(prices, period=14):
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain/(avg_loss+1e-9)
    rsi = 100-(100/(1+rs))
    return np.concatenate([[50], rsi.fillna(50)])

def calculate_macd(prices):
    exp1 = pd.Series(prices).ewm(span=12).mean()
    exp2 = pd.Series(prices).ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    return macd.values, signal.values

def get_sentiment():
    try:
        url="https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
        data=requests.get(url, timeout=5).json()
        if "Data" not in data: return 0
        analyzer = SentimentIntensityAnalyzer()
        scores=[]
        for article in data["Data"][:8]:
            title = article.get("title","")
            body = article.get("body","")[:150]
            text = f"{title}. {body}"
            vader = analyzer.polarity_scores(text)["compound"]
            blob = TextBlob(text).sentiment.polarity
            scores.append((vader+blob)/2)
        return max(-1,min(np.mean(scores),1)) if scores else 0
    except: return 0

def arima_forecast(prices, steps):
    try:
        model=ARIMA(prices, order=(2,1,2))
        model_fit=model.fit()
        return list(model_fit.forecast(steps=steps))
    except: return [prices[-1]]*steps

def prophet_forecast(dates, prices, steps):
    df=pd.DataFrame({"ds":pd.to_datetime(dates),"y":prices})
    model=Prophet(daily_seasonality=True)
    if len(df)<2: return np.array([prices[-1]]*steps), np.array([prices[-1]]*steps), np.array([prices[-1]]*steps)
    model.fit(df)
    future=model.make_future_dataframe(periods=steps,freq='H')
    forecast=model.predict(future)
    return forecast["yhat"].tail(steps).values, forecast["yhat_upper"].tail(steps).values, forecast["yhat_lower"].tail(steps).values

def hybrid_forecast(prices, dates, steps):
    arima_preds = arima_forecast(prices, steps)
    prophet_preds, upper, lower = prophet_forecast(dates, prices, steps)
    final = [(a+p)/2 for a,p in zip(arima_preds, prophet_preds)]
    return final, upper, lower

def load_weights():
    try: data=db.reference("weights").get()
    except: data=None
    if not data: return {"rsi":1,"macd":1,"trend":1,"sentiment":1}
    return data

def save_weights(weights):
    try: db.reference("weights").set(weights)
    except: pass

def smart_signal(prices,rsi,macd,signal,sentiment,weights):
    score=0
    contrib={}
    val=0
    if rsi[-1]<30: val=2*weights["rsi"]
    elif rsi[-1]>70: val=-2*weights["rsi"]
    score+=val; contrib["rsi"]=val
    val=(1 if macd[-1]>signal[-1] else -1)*weights["macd"]
    score+=val; contrib["macd"]=val
    val=(1 if prices[-1]>np.mean(prices) else -1)*weights["trend"]
    score+=val; contrib["trend"]=val
    if sentiment>0.2: val=2*weights["sentiment"]
    elif sentiment<-0.2: val=-2*weights["sentiment"]
    else: val=0
    score+=val; contrib["sentiment"]=val
    if score>=3: return "BUY",score,contrib
    elif score<=-3: return "SELL",score,contrib
    return "HOLD",score,contrib

def update_weights(symbol, weights):
    ref=db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    if len(data)<2: return weights
    prev=data[-2]; curr=data[-1]
    if prev["signal"]=="HOLD": return weights
    actual_up = curr["price"]>prev["price"]
    lr=0.03
    contributions=prev.get("contributions",{})
    for key in weights:
        c=contributions.get(key,0)
        if c==0: continue
        if (c>0 and actual_up) or (c<0 and not actual_up): weights[key]*=(1+lr)
        else: weights[key]*=(1-lr)
        weights[key]=max(0.2,min(weights[key],3))
    return weights

def save_prediction(symbol, price, pred_price, signal, contrib):
    ref=db.reference(f"history/{symbol}")
    ref.push({"time":datetime.utcnow().isoformat(),"price":float(price),"predicted":float(pred_price),"signal":signal,"contributions":contrib})

def calculate_win_rate(symbol):
    ref=db.reference(f"history/{symbol}")
    data=list((ref.get() or {}).values())
    if len(data)<2: return 0
    wins,total=0,0
    for i in range(len(data)-1):
        curr,nxt=data[i],data[i+1]
        if curr["signal"]=="BUY" and nxt["price"]>curr["price"]: wins+=1
        elif curr["signal"]=="SELL" and nxt["price"]<curr["price"]: wins+=1
        total+=1
    return round((wins/total)*100,2)

st.title("🚀 Adaptive AI Crypto Bot")
coins_list = ["BTCUSDT","ETHUSDT","BNBUSDT","ADAUSDT","XAIUSD","SOLUSDT"]
symbols = st.multiselect("Select Coins", coins_list, default=["BTCUSDT","SOLUSDT"])
timeframe = st.selectbox("Select timeframe", ["1 Day","3 Days","5 Days","1 Month"])

usd_idr = "N/A"
try:
    url = "https://api.exchangerate.host/latest?base=USD&symbols=IDR"
    r = requests.get(url, timeout=5).json()
    usd_idr = round(r["rates"]["IDR"],2)
except: pass
st.markdown(f"**USDT → IDR Rate: {usd_idr}**")

results = {}
if st.button("Analyze"):
    results={}
    for symbol in symbols:
        prices, dates, errors = get_historical_data(symbol)
        new_prices, new_dates = get_hourly_data(symbol, hours=24)
        if len(new_prices)>0:
            prices=np.concatenate([prices,new_prices])
            dates=dates+new_dates
            try:
                ref=db.reference(f"historical/{symbol}")
                for i in range(len(new_prices)):
                    ref.push({"time":new_dates[i].isoformat(),"price":float(new_prices[i])})
            except: pass
        now=datetime.utcnow()
        if timeframe=="1 Day": cutoff=now-timedelta(days=1)
        elif timeframe=="3 Days": cutoff=now-timedelta(days=3)
        elif timeframe=="5 Days": cutoff=now-timedelta(days=5)
        else: cutoff=now-timedelta(days=30)
        filtered_prices=[p for d,p in zip(dates,prices) if d>=cutoff]
        filtered_dates=[d for d in dates if d>=cutoff]
        if len(filtered_prices)<2: continue
        weights=load_weights()
        rsi=calculate_rsi(filtered_prices)
        macd,signal_line=calculate_macd(filtered_prices)
        sentiment=get_sentiment()
        trade_signal, score, contrib = smart_signal(filtered_prices,rsi,macd,signal_line,sentiment,weights)
        remaining_hours=24-now.hour
        future_prices, upper, lower = hybrid_forecast(filtered_prices, filtered_dates, remaining_hours)
        future_dates=[filtered_dates[-1]+timedelta(hours=i+1) for i in range(remaining_hours)]
        save_prediction(symbol,filtered_prices[-1],future_prices[0],trade_signal,contrib)
        weights=update_weights(symbol,weights)
        save_weights(weights)
        results[symbol]={"prices":filtered_prices,"dates":filtered_dates,"future_prices":future_prices,"future_dates":future_dates,"signal":trade_signal,"score":score,"contrib":contrib,"winrate":calculate_win_rate(symbol)}
    if results:
        df_summary=pd.DataFrame([
            {"Coin":s,"Signal":results[s]["signal"],"Score":round(results[s]["score"],2),"WinRate(%)":results[s]["winrate"],"Change(%)":round((results[s]["future_prices"][0]-results[s]["prices"][-1])/results[s]["prices"][-1]*100,2)} 
            for s in results
        ])
        st.table(df_summary)
        for symbol,res in results.items():
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=res["dates"],y=res["prices"],mode='lines+markers',name='Actual',line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=[res["prices"][0]]+[res["future_dates"][0]],y=[res["prices"][0]]+[res["future_prices"][0]],mode='lines',name='Change',line=dict(color='green',dash='dot')))
            fig.add_trace(go.Scatter(x=res["future_dates"],y=res["future_prices"],mode='lines+markers',name='Forecast',line=dict(color='red',dash='dash')))
            st.plotly_chart(fig)
            st.metric(f"{symbol} Signal", res["signal"])
            st.metric(f"{symbol} Score", round(res["score"],2))
            st.metric(f"{symbol} Next Hour Price", round(res["future_prices"][0],2))
            st.metric(f"{symbol} WinRate(%)", res["winrate"])
