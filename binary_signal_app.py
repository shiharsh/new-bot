# Optimized Multi-Symbol Forex Signal Bot with Telegram Alerts and ML
import streamlit as st
import pandas as pd
import requests
import ta
import joblib
import os
import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ─── CONFIGURATION ─────────────────────────────────────
twelve_key = "4d5b1e81f9314e28a7ee285497d3b273"
telegram_token = "7557174507:AAFSmFW5nxJ-fLOPS-B_wi0uT5wkQ5-PEx8"
telegram_chat_id = "1278635048"

# ─── AUTO REFRESH ──────────────────────────────────────
st_autorefresh(interval=60000, limit=None, key="refresh")

# ─── UTILS ─────────────────────────────────────────────
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{telegram_token}/sendMessage"
    try:
        requests.post(url, data={"chat_id": telegram_chat_id, "text": message}, timeout=10)
    except Exception as e:
        st.warning(f"⚠️ Telegram error: {e}")

def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{telegram_token}/sendPhoto"
    try:
        with open(photo_path, 'rb') as photo:
            requests.post(url, data={"chat_id": telegram_chat_id, "caption": caption}, files={"photo": photo}, timeout=10)
    except Exception as e:
        st.warning(f"⚠️ Telegram photo error: {e}")

def fetch_data(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=5min&outputsize=300&apikey={twelve_key}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("status") != "ok" or "values" not in data:
            return None
    except Exception as e:
        st.error(f"API fetch failed: {e}")
        return None

    df = pd.DataFrame(data["values"])
    df.rename(columns={"datetime": "Datetime", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)
    df = df.astype(float).sort_index()
    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")

    df["EMA9"] = ta.trend.ema_indicator(df["Close"], window=9)
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], window=21)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.MACD(df["Close"]).macd_diff()
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    df["Target"] = (df["Close"].shift(-1) > df["Open"].shift(-1)).astype(int)
    df.dropna(inplace=True)
    return df

def load_or_train_model(X, y, path):
    if os.path.exists(path):
        return joblib.load(path)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    model.fit(X_train, y_train)
    joblib.dump(model, path)
    return model

# ─── MAIN UI ───────────────────────────────────────────
st.title("🤖 Optimized Multi-Symbol Forex Signal Bot")
symbol_map = {"EUR/USD": "EUR/USD", "USD/JPY": "USD/JPY", "GBP/USD": "GBP/USD", "AUD/USD": "AUD/USD", "USD/CAD": "USD/CAD"}
selected = st.multiselect("✅ Choose forex pairs:", list(symbol_map.keys()), default=["EUR/USD"])
threshold = st.slider("🔎 Confidence Threshold", 0.5, 1.0, 0.70, 0.01)

for symbol in selected:
    st.subheader(f"📊 {symbol}")
    df = fetch_data(symbol_map[symbol])
    if df is None:
        st.warning(f"❌ Failed to fetch data for {symbol}")
        continue

    features = ["EMA9", "EMA21", "RSI", "MACD", "BB_upper", "BB_lower", "Close", "Open"]
    X, y = df[features], df["Target"]
    model_path = f"model_{symbol.replace('/', '')}.pkl"
    model = load_or_train_model(X, y, model_path)

    df["ML_Signal"] = model.predict(X)
    df["Confidence"] = model.predict_proba(X)[:, 1]
    df["ML_Signal"] = df["ML_Signal"].map({1: "CALL", 0: "PUT"})
    df["Actual"] = df["Target"].map({1: "CALL", 0: "PUT"})
    df["Correct"] = df["ML_Signal"] == df["Actual"]
    acc = df["Correct"].mean() * 100

    last_row = df.iloc[-1]
    if f"{symbol}_history" not in st.session_state:
        st.session_state[f"{symbol}_history"] = []
    if not st.session_state[f"{symbol}_history"] or st.session_state[f"{symbol}_history"][-1]['time'] != df.index[-1]:
        outcome = int(last_row["ML_Signal"] == last_row["Actual"])
        st.session_state[f"{symbol}_history"].append({
            "time": df.index[-1], "signal": last_row["ML_Signal"], "actual": last_row["Actual"],
            "outcome": outcome, "accuracy": acc, "confidence": last_row["Confidence"]
        })

        if last_row["Confidence"] >= threshold and acc >= 70:
            chart_path = f"chart_{symbol.replace('/', '')}.png"
            mpf.plot(df.tail(30), type='candle', style='charles', mav=(9, 21), savefig=chart_path)
            caption = (f"🔔 {symbol} Signal\n"
                       f"Signal: {last_row['ML_Signal']}\n"
                       f"Confidence: {last_row['Confidence']:.2%}\n"
                       f"Model Accuracy: {acc:.2f}%\n"
                       f"Time: {df.index[-1].strftime('%H:%M %d-%m-%Y')}")
            send_telegram_photo(chart_path, caption)

    history_df = pd.DataFrame(st.session_state[f"{symbol}_history"]).set_index("time")
    st.metric(f"{symbol} Accuracy", f"{acc:.2f}%", delta=f"Session: {history_df['accuracy'].iloc[-1]:.2f}%")
    st.line_chart(history_df['accuracy'], height=150)

    if st.button(f"🧪 Backtest {symbol}", key=symbol):
        backtest_df = df.copy()
        backtest_df["Pip_Return"] = (backtest_df["Close"] - backtest_df["Open"]) * backtest_df["ML_Signal"].map({"CALL": 1, "PUT": -1})
        backtest_df["Cumulative"] = backtest_df["Pip_Return"].cumsum()
        st.line_chart(backtest_df["Cumulative"], height=200)
        st.write(f"Signals: {len(backtest_df)} | Accuracy: {backtest_df['Correct'].mean()*100:.2f}% | Net Pips: {backtest_df['Cumulative'].iloc[-1]:.2f}")

        if st.checkbox(f"📤 Send summary for {symbol}", key=f"summary_{symbol}"):
            send_telegram_message(f"📊 Daily Summary for {symbol}\nSignals: {len(backtest_df)}\nAccuracy: {backtest_df['Correct'].mean()*100:.2f}%\nNet Pips: {backtest_df['Cumulative'].iloc[-1]:.2f}")

    with st.expander(f"🔍 Recent {symbol} Predictions"):
        st.dataframe(df.tail(10))
        st.dataframe(history_df.tail(10))

    csv = history_df.to_csv().encode("utf-8")
    st.download_button(f"⬇️ Download {symbol} history", csv, file_name=f"{symbol.replace('/', '')}_history.csv")

# ─── GLOBAL TIMER ─────────────────────────────────────
now = datetime.now(ZoneInfo("Asia/Kolkata"))
minute = (now.minute // 5) * 5
next_candle = now.replace(minute=minute, second=0, microsecond=0) + timedelta(minutes=5)
remaining = (next_candle - now).total_seconds()
mins, secs = divmod(int(remaining), 60)
st.metric("⏳ Next candle in", f"{mins}m {secs}s")
