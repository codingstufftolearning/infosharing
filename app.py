# app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from bs4 import BeautifulSoup
from textblob import TextBlob

# ---------- CONFIG ----------
coins = {
    "Bitcoin": {"symbol": "BTC", "cg_id": "bitcoin"},
    "Ethereum": {"symbol": "ETH", "cg_id": "ethereum"},
    "Solana": {"symbol": "SOL", "cg_id": "solana"},
    "Cardano": {"symbol": "ADA", "cg_id": "cardano"},
    "Arbitrum": {"symbol": "ARB", "cg_id": "arbitrum"},
    "Optimism": {"symbol": "OP", "cg_id": "optimism"}
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ---------- SAFE REQUEST ----------
def safe_request(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"HTTP {r.status_code}: {url}")
    except Exception as e:
        print(f"Error: {e}")
    return None

# ---------- DATA SOURCES ----------
@st.cache_data(ttl=900)
def get_price_cryptocompare(symbol, days):
    url = f"https://min-api.cryptocompare.com/data/v2/histoday?fsym={symbol}&tsym=USD&limit={days}"
    data = safe_request(url)
    if data and isinstance(data, dict) and data.get("Response") == "Success" and "Data" in data and "Data" in data["Data"]:
        return [d.get("close",0) for d in data["Data"]["Data"] if isinstance(d, dict) and d.get("close",0)>0]
    return []

@st.cache_data(ttl=900)
def get_price_coingecko(cg_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart?vs_currency=usd&days={days}"
    data = safe_request(url)
    if data and isinstance(data, dict) and "prices" in data and isinstance(data["prices"], list) and len(data["prices"]) > 0:
        df = pd.DataFrame(data["prices"], columns=["t","p"])
        df['d'] = pd.to_datetime(df['t'], unit='ms').dt.date
        return df.groupby('d')['p'].last().tolist()
    return []

@st.cache_data(ttl=60)
def get_current(symbol):
    data = safe_request(f"https://min-api.cryptocompare.com/data/price?fsym={symbol}&tsyms=USD")
    if data and "USD" in data:
        return data["USD"]
    return 0

@st.cache_data(ttl=900)
def merge_prices(*sources):
    sources = [s for s in sources if len(s)>0]
    if not sources:
        return []
    min_len = min(len(s) for s in sources)
    trimmed = [s[-min_len:] for s in sources]
    merged = [np.mean([s[i] for s in trimmed]) for i in range(min_len)]
    return merged

@st.cache_data(ttl=900)
def get_price(symbol, cg_id):
    cc = get_price_cryptocompare(symbol,30)
    cg = get_price_coingecko(cg_id,30)
    merged = merge_prices(cc,cg)
    if len(merged)<5:
        current = get_current(symbol)
        return [current]*30
    return merged

# ---------- INDICATORS ----------
@st.cache_data(ttl=300)
def rsi(prices, period=14):
    if len(prices)<period:
        return 50
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = -np.minimum(delta,0)
    avg_gain = np.mean(gain[:period]) if len(gain[:period])>0 else 0
    avg_loss = np.mean(loss[:period]) if len(loss[:period])>0 else 0.0001
    rs = avg_gain/avg_loss
    return 100-(100/(1+rs))

@st.cache_data(ttl=300)
def macd(prices):
    if len(prices)<26:
        return 0
    s = pd.Series(prices)
    macd = s.ewm(span=12).mean()-s.ewm(span=26).mean()
    signal = macd.ewm(span=9).mean()
    return macd.iloc[-1]-signal.iloc[-1]

@st.cache_data(ttl=300)
def momentum(prices):
    return prices[-1]-prices[-5] if len(prices)>5 else 0

# ---------- PREDICTIONS ----------
def predict_linear(prices, steps=1):
    if len(prices)==0: return 0
    x = np.arange(len(prices))
    y = np.array(prices)
    slope, intercept = np.polyfit(x,y,1)
    return slope*(len(prices)+steps)+intercept

def predict_ewma(prices, steps=1, span=10):
    if len(prices)==0: return 0
    s = pd.Series(prices)
    return s.ewm(span=span).mean().iloc[-1]

def predict_arima(prices, steps=1):
    if len(prices)==0: return 0
    try:
        model = ARIMA(prices, order=(2,1,2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps)
        return forecast[-1]
    except:
        return prices[-1]

def predict_combined(prices, steps=1):
    return np.mean([
        predict_linear(prices, steps),
        predict_ewma(prices, steps),
        predict_arima(prices, steps)
    ])

# ---------- NEWS SYSTEM ----------
@st.cache_data(ttl=600)
def get_crypto_news():
    url = "https://www.coingecko.com/en/news"
    try:
        html = requests.get(url, headers=HEADERS, timeout=10).text
        soup = BeautifulSoup(html, 'html.parser')
        headlines = [h.get_text(strip=True) for h in soup.find_all('h3')]
        return headlines[:20]
    except:
        return []

@st.cache_data(ttl=300)
def keyword_sentiment(texts):
    positive = ["bull","rise","surge","gain","pump","all-time high","moon","rally","soar"]
    negative = ["crash","drop","fall","hack","dump","plunge","sell-off","bear"]
    score = 0
    for t in texts:
        t_lower = t.lower()
        for w in positive:
            if w in t_lower:
                score += 1
        for w in negative:
            if w in t_lower:
                score -= 1
    return score / len(texts) if texts else 0

@st.cache_data(ttl=300)
def polarity_sentiment(texts):
    if not texts:
        return 0
    scores = [TextBlob(t).sentiment.polarity for t in texts]
    return np.mean(scores)

@st.cache_data(ttl=300)
def combined_sentiment(texts):
    ks = keyword_sentiment(texts)
    ps = polarity_sentiment(texts)
    return (ks + ps)/2

# ---------- SIGNALS ----------
def get_signal(current, est, sentiment=0):
    change_pct = (est - current)/current if current>0 else 0
    combined_score = change_pct + sentiment*0.01
    if combined_score > 0.03:
        return "🚀 Strong Buy"
    elif combined_score > 0:
        return "🟢 Buy"
    elif combined_score > -0.02:
        return "⚖️ Neutral"
    elif combined_score > -0.05:
        return "🟠 Sell"
    else:
        return "🔴 Strong Sell"

# ---------- UI ----------
st.title("📊 PRO Smart Crypto Dashboard")

selected_coins = st.multiselect("Select coins to analyze", list(coins.keys()), default=list(coins.keys()))

news_headlines = get_crypto_news()
sentiment_score = combined_sentiment(news_headlines)
st.subheader("📰 Latest Crypto News Sentiment")
st.write(f"Market Sentiment Score: {sentiment_score}")
for h in news_headlines[:5]:
    st.write(f"- {h}")

if st.button("Analyze Market"):
    st.session_state.results=[]
    for name in selected_coins:
        c = coins[name]
        prices=get_price(c["symbol"],c["cg_id"])
        if len(prices)==0: prices=[0]
        current = prices[-1]

        est1 = predict_combined(prices, steps=1)
        est3 = (predict_combined(prices, steps=3) + np.mean(prices[-3:]))/2
        est7 = (predict_combined(prices, steps=7) + np.mean(prices[-7:]))/2

        upper, lower = bollinger_bands(prices)
        stoch = stochastic_oscillator(prices)
        volatility = np.std(prices)/np.mean(prices) if np.mean(prices)>0 else 0

        st.session_state.results.append({
            "Coin":name,
            "Price":round(current,2),
            "24h %":round((est1-current)/current*100 if current>0 else 0,2),
            "3d %":round((est3-current)/current*100 if current>0 else 0,2),
            "7d %":round((est7-current)/current*100 if current>0 else 0,2),
            "Signal":get_signal(current, est1, sentiment_score),
            "Volatility":round(volatility*100,2),
            "Bollinger Upper":round(upper,2),
            "Bollinger Lower":round(lower,2),
            "Stochastic":round(stoch,2),
            "Prices":prices
        })

if 'results' in st.session_state and st.session_state.results:
    df = pd.DataFrame(st.session_state.results).sort_values(by="24h %", ascending=False)
    st.subheader("📊 Smart Market Table")
    st.dataframe(df.drop(columns=["Prices"]), use_container_width=True)

    if st.checkbox("Show charts"):
        for coin in st.session_state.results:
            st.subheader(coin["Coin"])
            st.line_chart(pd.DataFrame(coin["Prices"], columns=["Price"]))
