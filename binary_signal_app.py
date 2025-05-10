# ✅ Updated Signal Thresholds:
# Confidence >= 65% and Model Accuracy >= 85%

import streamlit as st
import pandas as pd
import requests
import ta
import time
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os
import mplfinance as mpf
import matplotlib.pyplot as plt

# ─── AUTO REFRESH ──────────────────────────────────────
st_autorefresh(interval=1000, limit=None, key="refresh")

# ─── HARDCODED API KEYS (Testing Only) ────────────────
twelve_key = "4d5b1e81f9314e28a7ee285497d3b273"
telegram_token = "7557174507:AAFSmFW5nxJ-fLOPS-B_wi0uT5wkQ5-PEx8"
telegram_chat_id = "1278635048"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    payload = {"chat_id": telegram_chat_id, "text": message}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        st.warning(f"⚠️ Telegram error: {e}")

def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        payload = {"chat_id": telegram_chat_id, "caption": caption}
        files = {"photo": photo}
        try:
            requests.post(url, data=payload, files=files, timeout=10)
        except Exception as e:
            st.warning(f"⚠️ Telegram photo error: {e}")

# ─── SYMBOL LIST ───────────────────────────────────────
symbol_map = {
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
    "GBP/USD": "GBP/USD",
    "AUD/USD": "AUD/USD",
    "USD/CAD": "USD/CAD"
}
selected_symbols = st.multiselect("✅ Choose forex pairs for alerts:", list(symbol_map.keys()), default=["EUR/USD"])

# ─── FETCH DATA ────────────────────────────────────────
@st.cache_data(ttl=300)
def fetch_data(sym_key):
    sym = symbol_map[sym_key]
    url = f"https://api.twelvedata.com/time_series?symbol={sym}&interval=5min&outputsize=500&apikey={twelve_key}"
    r = requests.get(url, timeout=10)
    data = r.json()
    if data.get("status") != "ok" or "values" not in data:
        return None
    df = pd.DataFrame(data["values"])
    df.rename(columns={"datetime": "Datetime", "open": "Open", "high": "High",
                       "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").astype(float)
    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")
    df = df.sort_index()

    df["EMA9"] = ta.trend.ema_indicator(df["Close"], window=9)
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], window=21)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    df["Target"] = df["Close"].shift(-1) > df["Open"].shift(-1)
    df["Target"] = df["Target"].map({True: 1, False: 0})
    df.dropna(inplace=True)
    return df

# ─── MAIN LOOP ────────────────────────────────────────
st.title("🤖 Multi-Symbol Forex Signal Bot + Telegram Alerts + Charts")

for symbol in selected_symbols:
    st.subheader(f"📊 {symbol}")
    df = fetch_data(symbol)
    if df is None:
        st.warning(f"❌ Failed to fetch data for {symbol}")
        continue

    features = ["EMA9", "EMA21", "RSI", "MACD", "BB_upper", "BB_lower", "Close", "Open"]
    X = df[features]
    y = df["Target"]
    model_path = f"{symbol.replace('/', '')}_rf_model.pkl"

    if not os.path.exists(model_path):
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)

    df["ML_Signal"] = model.predict(X)
    df["Confidence"] = model.predict_proba(X)[:, 1]
    df["Actual"] = df["Target"].map({1: "CALL", 0: "PUT"})
    df["ML_Signal"] = df["ML_Signal"].map({1: "CALL", 0: "PUT"})
    df["Correct"] = df["ML_Signal"] == df["Actual"]
    ml_accuracy = df["Correct"].mean() * 100

    last_time = df.index[-1]
    last_signal = df.iloc[-1]["ML_Signal"]
    last_conf = df.iloc[-1]["Confidence"]
    last_actual = df.iloc[-1]["Actual"]
    last_outcome = int(last_signal == last_actual)

    if f"{symbol}_history" not in st.session_state:
        st.session_state[f"{symbol}_history"] = []

    if not st.session_state[f"{symbol}_history"] or st.session_state[f"{symbol}_history"][-1]['time'] != last_time:
        total = len(st.session_state[f"{symbol}_history"]) + 1
        correct = sum(i['outcome'] for i in st.session_state[f"{symbol}_history"]) + last_outcome
        session_acc = (correct / total) * 100
        st.session_state[f"{symbol}_history"].append({
            "time": last_time, "signal": last_signal, "actual": last_actual,
            "outcome": last_outcome, "accuracy": session_acc,
            "confidence": last_conf
        })

        # ✅ Signal Condition: Confidence ≥ 65% and Accuracy ≥ 85%
        if last_conf >= 0.65 and ml_accuracy >= 85:
            chart_path = f"chart_{symbol.replace('/', '')}.png"
            mpf.plot(df.tail(30), type='candle', style='charles', mav=(9, 21), savefig=chart_path)
            caption = (f"🔔 {symbol} Signal\n"
                       f"Signal: {last_signal}\n"
                       f"Confidence: {last_conf:.2%}\n"
                       f"Model Accuracy: {ml_accuracy:.2f}%\n"
                       f"Time: {last_time.strftime('%H:%M %d-%m-%Y')}")
            send_telegram_photo(chart_path, caption)
    else:
        session_acc = st.session_state[f"{symbol}_history"][-1]["accuracy"]

    acc_color = "green" if ml_accuracy >= 85 else "orange" if ml_accuracy >= 60 else "red"
    st.metric(f"{symbol} Accuracy", f"{ml_accuracy:.2f}%", delta=f"Session: {session_acc:.2f}%", delta_color="normal")

    history_df = pd.DataFrame(st.session_state[f"{symbol}_history"]).set_index("time")
    st.line_chart(history_df["accuracy"], height=150)

    if st.button(f"🕰️ Backtest {symbol}", key=symbol):
        backtest_df = df.copy()
        backtest_df["Pip_Return"] = (backtest_df["Close"] - backtest_df["Open"]) * backtest_df["ML_Signal"].map({"CALL": 1, "PUT": -1})
        backtest_df["Cumulative"] = backtest_df["Pip_Return"].cumsum()
        st.write(f"Signals: {len(backtest_df)} | Accuracy: {backtest_df['Correct'].mean()*100:.2f}% | Net Pips: {backtest_df['Cumulative'].iloc[-1]:.2f}")
        st.line_chart(backtest_df["Cumulative"], height=200)

        if st.checkbox(f"📩 Send {symbol} daily summary", key=f"summary_{symbol}"):
            msg = (f"📊 Daily Summary for {symbol}\n"
                   f"Signals: {len(backtest_df)}\n"
                   f"Accuracy: {backtest_df['Correct'].mean()*100:.2f}%\n"
                   f"Net Pips: {backtest_df['Cumulative'].iloc[-1]:.2f}")
            send_telegram_message(msg)

    with st.expander(f"📄 {symbol} Recent Predictions"):
        st.dataframe(df.tail(10))
        st.dataframe(history_df.tail(10))

    csv = history_df.to_csv().encode("utf-8")
    st.download_button(f"Download {symbol} history", csv, file_name=f"{symbol.replace('/', '')}_history.csv", mime="text/csv")

# ─── GLOBAL TIMER ─────────────────────────────────────
now = datetime.now(ZoneInfo("Asia/Kolkata"))
minute = (now.minute // 5) * 5
next_candle_time = now.replace(minute=minute, second=0, microsecond=0) + timedelta(minutes=5)
remaining = (next_candle_time - now).total_seconds()
minutes, seconds = divmod(int(remaining), 60)
st.metric("⏳ Next candle in", f"{minutes}m {seconds}s")
