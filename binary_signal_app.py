import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
import plotly.graph_objs as go
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

st.set_page_config(layout="wide", page_title="Binary Signal Bot")
st.title("📈 Binary Trading Signal Bot")

# ================= USER SETTINGS =================
symbols = ['EURUSD=X', 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X']
timeframe = '5m'
data_points = 500

# ================= FUNCTION DEFINITIONS =================
def fetch_data(symbol):
    df = yf.download(symbol, interval=timeframe, period='2d')
    df.dropna(inplace=True)
    df['EMA_9'] = EMAIndicator(df['Close'], window=9).ema_indicator()
    df['RSI'] = RSIIndicator(df['Close'], window=14).rsi()
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = BollingerBands(df['Close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['Volume'] = df['Volume'].fillna(0)
    return df

def generate_signal(df):
    latest = df.iloc[-1]
    previous = df.iloc[-2]
    signal = "HOLD"

    # Trend filter using EMA
    if latest['Close'] > latest['EMA_9']:
        trend = "UP"
    elif latest['Close'] < latest['EMA_9']:
        trend = "DOWN"
    else:
        trend = "NEUTRAL"

    # Signal conditions
    if trend == "UP" and latest['RSI'] < 70 and latest['MACD'] > latest['MACD_signal']:
        signal = "CALL"
    elif trend == "DOWN" and latest['RSI'] > 30 and latest['MACD'] < latest['MACD_signal']:
        signal = "PUT"

    return signal

def backtest(df):
    signals = []
    for i in range(1, len(df)):
        row = df.iloc[:i+1]
        signal = generate_signal(row)
        if signal in ["CALL", "PUT"]:
            if signal == "CALL" and df.iloc[i+1]['Close'] > df.iloc[i]['Close']:
                result = '✅'
            elif signal == "PUT" and df.iloc[i+1]['Close'] < df.iloc[i]['Close']:
                result = '✅'
            else:
                result = '❌'
            signals.append({"Time": df.index[i], "Signal": signal, "Result": result})
    return pd.DataFrame(signals)

# ================= MAIN APP =================
st.sidebar.header("Configuration")
selected_symbol = st.sidebar.selectbox("Choose Forex Pair", symbols)
data = fetch_data(selected_symbol)
signal = generate_signal(data)

tab1, tab2 = st.tabs(["📊 Live Signal", "📉 Backtest"])

with tab1:
    caption = f"🔔 {selected_symbol} Signal"
    st.subheader(caption)
    st.metric(label="Current Signal", value=signal)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=data.index, y=data['EMA_9'], mode='lines', name='EMA 9'))
    fig.update_layout(xaxis_rangeslider_visible=False, template='plotly_dark', height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("EUR/USD Recent Predictions"):
        st.write(data.tail(10))

    csv = data.to_csv().encode('utf-8')
    st.download_button("Download EUR/USD history", csv, f"{selected_symbol}_history.csv")

    countdown = 300 - (datetime.datetime.utcnow().minute % 5) * 60 - datetime.datetime.utcnow().second
    st.markdown(f"⏳ Next candle in **{countdown//60}m {countdown%60}s**")

with tab2:
    with st.spinner("Running backtest..."):
        bt_results = backtest(data)
        win_rate = bt_results['Result'].value_counts(normalize=True).get('✅', 0) * 100
        st.write(f"Backtest Win Rate: **{win_rate:.2f}%**")
        st.dataframe(bt_results.tail(10))
