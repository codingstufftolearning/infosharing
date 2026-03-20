import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import threading, websocket, time
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import firebase_admin
from firebase_admin import credentials, db

# =========================
# SMART AUTO REFRESH
# =========================
if 'loading' not in st.session_state:
    st.session_state.loading = False

if not st.session_state.loading:
    st_autorefresh(interval=300000, key='refresh')  # 5 minutes

# =========================
# FIREBASE INIT
# =========================
if not firebase_admin._apps:
    try:
        firebase_dict = dict(st.secrets['firebase'])
        firebase_dict['private_key'] = firebase_dict['private_key'].replace('\\n', '\n')
        cred = credentials.Certificate(firebase_dict)
        firebase_admin.initialize_app(cred, {'databaseURL': firebase_dict['databaseURL']})
    except Exception as e:
        st.error(f'Firebase Init Error: {e}')

# =========================
# GLOBALS
# =========================
ws_prices = {}
demo_trades = {}  # Stores active demo trades

# =========================
# WEBSOCKET LIVE PRICE
# =========================
def start_ws(symbol):
    def on_message(ws, message):
        try:
            data = eval(message)
            if 'c' in data:
                ws_prices[symbol] = float(data['c'])
        except:
            pass
    url = f'wss://stream.binance.com:9443/ws/{symbol.lower()}@ticker'
    ws = websocket.WebSocketApp(url, on_message=on_message)
    threading.Thread(target=ws.run_forever, daemon=True).start()

# =========================
# FETCH OHLC
# =========================
def get_ohlc_binance(symbol, interval, limit):
    try:
        url = 'https://api.binance.com/api/v3/klines'
        params = {'symbol': symbol, 'interval': interval, 'limit': limit}
        data = requests.get(url, params=params, timeout=5).json()
        d,o,h,l,c,v = [],[],[],[],[],[]
        for k in data:
            d.append(datetime.fromtimestamp(k[0]/1000))
            o.append(float(k[1])); h.append(float(k[2]))
            l.append(float(k[3])); c.append(float(k[4]))
            v.append(float(k[5]))
        return d,o,h,l,c,v
    except:
        return None

def get_ohlc(symbol, timeframe):
    mapping = {'15 Min':('15m',96),'Hourly':('1h',72),'Daily':('1d',180)}
    interval, limit = mapping[timeframe]
    data = get_ohlc_binance(symbol, interval, limit)
    return data

# =========================
# INDICATORS
# =========================
def rsi(prices):
    delta = np.diff(prices)
    gain = np.maximum(delta,0)
    loss = np.abs(np.minimum(delta,0))
    rs = pd.Series(gain).rolling(14).mean()/(pd.Series(loss).rolling(14).mean()+1e-9)
    return np.concatenate([[50],100-(100/(1+rs))])

def macd(prices):
    exp1 = pd.Series(prices).ewm(span=12, adjust=False).mean()
    exp2 = pd.Series(prices).ewm(span=26, adjust=False).mean()
    m = exp1 - exp2
    s = m.ewm(span=9, adjust=False).mean()
    return m.values, s.values

# =========================
# LSTM MODEL
# =========================
@st.cache_resource(ttl=1800)
def train_lstm(prices):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(np.array(prices).reshape(-1,1))
    X,y = [],[]
    for i in range(20,len(data)):
        X.append(data[i-20:i])
        y.append(data[i])
    X,y = np.array(X), np.array(y)
    model = Sequential([LSTM(50, input_shape=(20,1)), Dense(1)])
    model.compile('adam','mse')
    model.fit(X,y,epochs=3,verbose=0)
    return model, scaler

def lstm_predict(model,scaler,prices):
    data = scaler.transform(np.array(prices).reshape(-1,1))
    seq = data[-20:]
    pred = model.predict(seq.reshape(1,20,1), verbose=0)
    return scaler.inverse_transform(pred)[0][0]

# =========================
# SENTIMENT + EVENTS
# =========================
@st.cache_data(ttl=900)
def get_sentiment():
    try:
        data = requests.get('https://min-api.cryptocompare.com/data/v2/news/?lang=EN').json()
        scores = []
        for a in data.get('Data',[])[:5]:
            text = a['title']
            v = SentimentIntensityAnalyzer().polarity_scores(text)['compound']
            b = TextBlob(text).sentiment.polarity
            scores.append((v+b)/2)
        return np.mean(scores) if scores else 0
    except:
        return 0

def detect_spike(prices):
    change = (prices[-1]-prices[-5])/prices[-5]
    if change<-0.05: return '🔻 Sharp Drop'
    if change>0.05: return '🚀 Sharp Rise'
    return 'Normal'

def support_resistance(prices):
    return min(prices[-50:]), max(prices[-50:])

# =========================
# UI SETUP
# =========================
st.title('🚀 AI Crypto Demo Bot')
COINS = ['BTCUSDT','ETHUSDT','BNBUSDT','ADAUSDT','SOLUSDT']
timeframe = st.selectbox('Select Timeframe',['15 Min','Hourly','Daily'])
symbols = st.multiselect('Select Coins', COINS, default=['BTCUSDT','ETHUSDT'])

# =========================
# MAIN LOOP
# =========================
st.session_state.loading = True
for sym in symbols:
    if sym not in ws_prices:
        start_ws(sym)

    data = get_ohlc(sym, timeframe)
    if not data:
        continue

    fd,o,h,l,c,v = data
    c = np.array(c)

    r = rsi(c)
    m,sig = macd(c)
    sentiment = get_sentiment()
    model,scaler = train_lstm(c)
    pred = lstm_predict(model,scaler,c)
    sup,res = support_resistance(c)
    spike = detect_spike(c)
    conf = 100*(1-np.std(c)/c[-1])

    # =========================
    # COIN CARD
    # =========================
    with st.container():
        st.markdown(f"### {sym}")
        col1,col2 = st.columns([3,1])

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=fd, open=o, high=h, low=l, close=c))
            fig.add_trace(go.Scatter(x=fd, y=[pred]*len(fd), line=dict(color='yellow'), name='Prediction'))
            fig.add_hline(y=sup, line_dash='dash', line_color='green')
            fig.add_hline(y=res, line_dash='dash', line_color='red')
            fig.update_layout(dragmode='zoom')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write(f'Sentiment: {round(sentiment,2)}')
            st.write(f'Spike: {spike}')
            st.write(f'Confidence: {round(conf,2)}%')

        # =========================
        # DEMO PANEL
        # =========================
        with st.expander('🎮 Demo Trading Panel'):
            if sym not in demo_trades:
                demo_trades[sym] = []

            mode = st.selectbox('Mode',['Spot Trade','Future'], key=f'{sym}_mode')
            price_now = ws_prices.get(sym, c[-1])
            amount = st.number_input('Amount (coins)', min_value=0.0, step=0.01, key=f'{sym}_amt')
            tp_pct = st.number_input('Take Profit %', min_value=0.0, max_value=100.0, step=0.1, key=f'{sym}_tp')
            sl_pct = st.number_input('Stop Loss %', min_value=0.0, max_value=100.0, step=0.1, key=f'{sym}_sl')

            if st.button('Open Trade', key=f'{sym}_open'):
                trade = {'mode':mode, 'amount':amount, 'price':price_now, 'tp_pct':tp_pct, 'sl_pct':sl_pct, 'opened':datetime.now()}
                demo_trades[sym].append(trade)
                db.reference(f'trade_history/{sym}').push(trade)

            # Show active trades
            for t in demo_trades[sym]:
                cur_price = ws_prices.get(sym, c[-1])
                tp_price = t['price']*(1+t['tp_pct']/100)
                sl_price = t['price']*(1-t['sl_pct']/100)
                status = 'HOLD'
                if cur_price>=tp_price:
                    status='✅ TP Hit'
                elif cur_price<=sl_price:
                    status='❌ SL Hit'
                st.write(f"Mode: {t['mode']}, Open: {t['price']}, Current: {round(cur_price,2)}, TP: {round(tp_price,2)}, SL: {round(sl_price,2)}, Status: {status}")

st.session_state.loading = False
