# app.py
import json, time
import streamlit as st
import requests, pandas as pd, numpy as np
from statsmodels.tsa.arima.model import ARIMA
from bs4 import BeautifulSoup
from textblob import TextBlob
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import LSTM, Dense

# ---------------- FIREBASE INIT ----------------
cred_dict = json.loads(st.secrets["FIREBASE_JSON"])
cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
cred = credentials.Certificate(cred_dict)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://infosharing-dantdkm-default-rtdb.firebaseio.com/"
    })
st.success("✅ Firebase initialized successfully")

# ---------------- CONFIG ----------------
coins = {
    "Bitcoin": {"symbol": "BTC", "cg_id": "bitcoin"},
    "Ethereum": {"symbol": "ETH", "cg_id": "ethereum"},
    "Solana": {"symbol": "SOL", "cg_id": "solana"},
    "Cardano": {"symbol": "ADA", "cg_id": "cardano"},
    "Arbitrum": {"symbol": "ARB", "cg_id": "arbitrum"},
    "Optimism": {"symbol": "OP", "cg_id": "optimism"}
}
HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------------- SAFE REQUEST ----------------
def safe_request(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200: return r.json()
    except Exception as e:
        print(f"Error: {e}")
    return None

# ---------------- PRICE SOURCES ----------------
@st.cache_data(ttl=900)
def get_price_cryptocompare(symbol, days):
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={days}"
    data = safe_request(url)
    if data and data.get("Response")=="Success" and "Data" in data and "Data" in data["Data"]:
        return [d.get("close",0) for d in data["Data"]["Data"] if d.get("close",0)>0]
    return []

@st.cache_data(ttl=900)
def get_price_coingecko(cg_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days={days}"
    data = safe_request(url)
    if data and "prices" in data:
        df = pd.DataFrame(data["prices"], columns=["t","p"])
        df['d'] = pd.to_datetime(df['t'], unit='ms').dt.date
        return df.groupby('d')['p'].last().tolist()
    return []

@st.cache_data(ttl=60)
def get_current(symbol):
    data = safe_request(f"https://min-api.cryptocompare.com/data/price?fsym={symbol}&tsyms=USD")
    if data and "USD" in data: return data["USD"]
    return 0

@st.cache_data(ttl=900)
def merge_prices(*sources):
    sources = [s for s in sources if len(s)>0]
    if not sources: return []
    min_len = min(len(s) for s in sources)
    trimmed = [s[-min_len:] for s in sources]
    return [np.mean([s[i] for s in trimmed]) for i in range(min_len)]

@st.cache_data(ttl=900)
def get_price(symbol, cg_id):
    cc = get_price_cryptocompare(symbol,30)
    cg = get_price_coingecko(cg_id,30)
    merged = merge_prices(cc,cg)
    if len(merged)<5:
        current = get_current(symbol)
        return [current]*30
    return merged

# ---------------- INDICATORS ----------------
@st.cache_data(ttl=300)
def rsi(prices, period=14):
    if len(prices)<period: return 50
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = -np.minimum(delta,0)
    avg_gain = np.mean(gain[:period])
    avg_loss = np.mean(loss[:period]) if np.mean(loss[:period])>0 else 0.0001
    rs = avg_gain/avg_loss
    return 100-(100/(1+rs))

@st.cache_data(ttl=300)
def macd(prices):
    if len(prices)<26: return 0
    s = pd.Series(prices)
    macd_line = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    signal_line = macd_line.ewm(span=9).mean()
    return macd_line.iloc[-1]-signal_line.iloc[-1]

@st.cache_data(ttl=300)
def momentum(prices):
    return prices[-1]-prices[-5] if len(prices)>5 else 0

def bollinger_bands(prices, period=20, k=2):
    if len(prices)<period: mean=np.mean(prices) if prices else 0; return mean, mean
    s = pd.Series(prices)
    sma = s.rolling(period).mean().iloc[-1]
    std = s.rolling(period).std().iloc[-1]
    return sma + k*std, sma - k*std

def stochastic_oscillator(prices, period=14):
    if len(prices)<period: return 50
    s = pd.Series(prices[-period:])
    low, high = s.min(), s.max()
    return 100*(s.iloc[-1]-low)/(high-low) if high!=low else 50

# ---------------- PREDICTIONS ----------------
def predict_linear(prices, steps=1):
    if len(prices)==0: return 0
    x,y=np.arange(len(prices)),np.array(prices)
    slope, intercept = np.polyfit(x,y,1)
    return slope*(len(prices)+steps)+intercept

def predict_ewma(prices, steps=1, span=10):
    if len(prices)==0: return 0
    return pd.Series(prices).ewm(span=span).mean().iloc[-1]

def predict_arima(prices, steps=1):
    if len(prices)==0: return 0
    try:
        model = ARIMA(prices, order=(2,1,2))
        return model.fit().forecast(steps)[-1]
    except: return prices[-1]

def predict_lstm(prices, steps=1):
    if len(prices)<10: return prices[-1] if prices else 0
    seq = np.array(prices[-30:]).reshape(-1,1)
    seq = (seq - seq.min())/(seq.max()-seq.min()+1e-8)
    X = seq[:-1].reshape(1, len(seq)-1, 1)
    y = seq[1:].reshape(1, len(seq)-1, 1)
    model = Sequential()
    model.add(LSTM(50, input_shape=(X.shape[1],1)))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X,y,epochs=5,verbose=0)
    pred = model.predict(X,verbose=0)[0,0]
    return pred*(seq.max()-seq.min())+seq.min()

def predict_combined(prices, steps=1):
    return np.mean([predict_linear(prices,steps),
                    predict_ewma(prices,steps),
                    predict_arima(prices,steps),
                    predict_lstm(prices,steps)])

# ---------------- NEWS & SENTIMENT ----------------
vader = SentimentIntensityAnalyzer()

@st.cache_data(ttl=600)
def get_crypto_news():
    url="https://www.coingecko.com/en/news"
    try:
        html=requests.get(url, headers=HEADERS, timeout=10).text
        soup=BeautifulSoup(html,'html.parser')
        headlines=[h.get_text(strip=True) for h in soup.find_all('h3')]
        return headlines[:20]
    except: return []

@st.cache_data(ttl=300)
def sentiment_analysis(texts):
    if not texts: return 0
    textblob_score = np.mean([TextBlob(t).sentiment.polarity for t in texts])
    vader_score = np.mean([vader.polarity_scores(t)['compound'] for t in texts])
    keywords = sum([1 if any(w in t.lower() for w in ["bull","rise","surge","gain","pump","moon","rally"]) else 0 for t in texts])
    neg_keywords = sum([1 if any(w in t.lower() for w in ["crash","drop","fall","hack","dump","plunge","bear"]) else 0 for t in texts])
    keyword_score = (keywords - neg_keywords)/len(texts) if texts else 0
    return np.mean([textblob_score, vader_score, keyword_score])

# ---------------- SIGNAL ----------------
def get_signal(current, est, sentiment=0):
    change_pct=(est-current)/current if current>0 else 0
    score = change_pct + sentiment*0.01
    if score>0.03: return "🚀 Strong Buy"
    elif score>0: return "🟢 Buy"
    elif score>-0.02: return "⚖️ Neutral"
    elif score>-0.05: return "🟠 Sell"
    else: return "🔴 Strong Sell"

# ---------------- UI ----------------
st.title("💥 INSANE PRO Crypto Dashboard")
selected_coins = st.multiselect("Select coins to analyze", list(coins.keys()), default=list(coins.keys()))
news = get_crypto_news()
sentiment_score = sentiment_analysis(news)
st.subheader("📰 Market News Sentiment")
st.write(f"Combined Sentiment Score: {sentiment_score:.3f}")
for h in news[:5]: st.write(f"- {h}")

if st.button("Analyze Market"):
    st.session_state.results=[]
    for name in selected_coins:
        c=coins[name]
        prices=get_price(c["symbol"],c["cg_id"])
        current=prices[-1] if prices else 0
        est1 = predict_combined(prices,1)
        est3 = (predict_combined(prices,3)+np.mean(prices[-3:]))/2
        est7 = (predict_combined(prices,7)+np.mean(prices[-7:]))/2
        upper, lower = bollinger_bands(prices)
        stoch = stochastic_oscillator(prices)
        volatility = np.std(prices)/np.mean(prices) if np.mean(prices)>0 else 0

        res = {"Coin":name,"Price":round(current,2),
               "24h %":round((est1-current)/current*100 if current>0 else 0,2),
               "3d %":round((est3-current)/current*100 if current>0 else 0,2),
               "7d %":round((est7-current)/current*100 if current>0 else 0,2),
               "Signal":get_signal(current,est1,sentiment_score),
               "Volatility":round(volatility*100,2),
               "Bollinger Upper":round(upper,2),
               "Bollinger Lower":round(lower,2),
               "Stochastic":round(stoch,2),
               "Prices":prices}
        st.session_state.results.append(res)

    # ---------- PUSH TO FIREBASE ----------
    ref=db.reference("/crypto_data")
    for r in st.session_state.results:
        ref.push({
            "coin": r["Coin"], "price":r["Price"],
            "24h_percent": r["24h %"], "3d_percent": r["3d %"], "7d_percent": r["7d %"],
            "signal": r["Signal"], "volatility": r["Volatility"],
            "bollinger_upper": r["Bollinger Upper"], "bollinger_lower": r["Bollinger Lower"],
            "stochastic": r["Stochastic"], "timestamp": time.time()
        })

# ---------------- DISPLAY ----------------
if 'results' in st.session_state and st.session_state.results:
    df = pd.DataFrame(st.session_state.results).sort_values(by="24h %", ascending=False)
    st.subheader("📊 Smart Market Table")
    st.dataframe(df.drop(columns=["Prices"]), use_container_width=True)

    if st.checkbox("Show Charts"):
        for c in st.session_state.results:
            st.subheader(c["Coin"])
            fig=go.Figure()
            fig.add_trace(go.Scatter(y=c["Prices"], mode='lines', name='Price'))
            upper, lower = bollinger_bands(c["Prices"])
            fig.add_trace(go.Scatter(y=[upper]*len(c["Prices"]), mode='lines', name='Bollinger Upper', line=dict(dash='dash')))
            fig.add_trace(go.Scatter(y=[lower]*len(c["Prices"]), mode='lines', name='Bollinger Lower', line=dict(dash='dash')))
            st.plotly_chart(fig, use_container_width=True)
