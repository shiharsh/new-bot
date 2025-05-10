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

# Auto refresh every 1000 ms (1s)
st_autorefresh(interval=1000, limit=None, key="refresh")

# ─── HARDCODED CREDENTIALS ────────────────────────────
TELEGRAM_TOKEN = "7557174507:AAFSmFW5nxJ-fLOPS-B_wi0uT5wkQ5-PEx8"
TELEGRAM_CHAT_ID = "1278635048"  # or use "@Shiharsh" if it's a public channel
TWELVE_DATA_API_KEY = "4d5b1e81f9314e28a7ee285497d3b273"

# ─── SYMBOL LIST ───────────────────────────────────────
symbol_map = {
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
    "GBP/USD": "GBP/USD",
    "AUD/USD": "AUD/USD",
    "USD/CAD": "USD/CAD"
}
selected_symbols = st.multiselect("✅ Choose forex pairs:", list(symbol_map.keys()), default=["EUR/USD"])

# ─── UTILITY FUNCTIONS ─────────────────────────────────
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        st.warning(f"⚠️ Telegram Error: {e}")

def send_telegram_photo(photo_path, caption=""):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        files = {"photo": photo}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption}
        try:
            requests.post(url, data=data, files=files, timeout=10)
        except Exception as e:
            st.warning(f"⚠️ Telegram Photo Error: {e}")

@st.cache_data(ttl=300)
def fetch_data(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol_map[symbol]}&interval=5min&outputsize=500&apikey={TWELVE_DATA_API_KEY}"
    response = requests.get(url)
    data = response.json()
    if data.get("status") != "ok" or "values" not in data:
        return None

    df = pd.DataFrame(data["values"])
    df.rename(columns={"datetime": "Datetime", "open": "Open", "high": "High",
                       "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)
    df = df.astype(float).sort_index()
    df.index = df.index.tz_localize("UTC").tz_convert("Asia/Kolkata")

    # Indicators
    df["EMA9"] = ta.trend.ema_indicator(df["Close"], window=9)
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], window=21)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["Close"], window=20)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    df["Target"] = (df["Close"].shift(-1) > df["Open"].shift(-1)).astype(int)
    df.dropna(inplace=True)
    return df

# ─── APP HEADER ────────────────────────────────────────
st.title("🤖 Multi-Symbol Forex Signal Bot")
st.caption("With ML-based predictions, technical indicators, and Telegram alerts")

# ─── MAIN LOOP ─────────────────────────────────────────
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
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(*train_test_split(X, y, test_size=0.2, shuffle=False)[:2])
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)

    df["ML_Signal"] = model.predict(X)
    df["Confidence"] = model.predict_proba(X)[:, 1]
    df["ML_Signal"] = df["ML_Signal"].map({1: "CALL", 0: "PUT"})
    df["Actual"] = df["Target"].map({1: "CALL", 0: "PUT"})
    df["Correct"] = df["ML_Signal"] == df["Actual"]
    ml_acc = df["Correct"].mean() * 100

    # Last prediction
    last_time = df.index[-1]
    last_signal = df.iloc[-1]["ML_Signal"]
    last_conf = df.iloc[-1]["Confidence"]
    last_actual = df.iloc[-1]["Actual"]
    correct = last_signal == last_actual

    # Track session state
    key = f"{symbol}_history"
    if key not in st.session_state:
        st.session_state[key] = []

    if not st.session_state[key] or st.session_state[key][-1]['time'] != last_time:
        hist = st.session_state[key]
        session_acc = (sum(i['outcome'] for i in hist) + correct) / (len(hist) + 1) * 100
        st.session_state[key].append({
            "time": last_time,
            "signal": last_signal,
            "actual": last_actual,
            "outcome": int(correct),
            "accuracy": session_acc,
            "confidence": last_conf
        })

        # Send alert
        if last_conf >= 0.70 and ml_acc >= 70:
            chart_path = f"chart_{symbol.replace('/', '')}.png"
            mpf.plot(df.tail(30), type='candle', mav=(9, 21), style='charles', savefig=chart_path)
            caption = (f"🔔 {symbol} Signal\nSignal: {last_signal}\nConfidence: {last_conf:.2%}\n"
                       f"Accuracy: {ml_acc:.2f}%\nTime: {last_time:%H:%M %d-%m-%Y}")
            send_telegram_photo(chart_path, caption)

    # Display accuracy
    history_df = pd.DataFrame(st.session_state[key]).set_index("time")
    session_acc = history_df["accuracy"].iloc[-1]
    st.metric("Model Accuracy", f"{ml_acc:.2f}%", delta=f"Session: {session_acc:.2f}%")
    st.line_chart(history_df["accuracy"], height=150)

    # Backtest section
    if st.button(f"🕰️ Backtest {symbol}", key=symbol):
        back_df = df.copy()
        back_df["Pip_Return"] = (back_df["Close"] - back_df["Open"]) * back_df["ML_Signal"].map({"CALL": 1, "PUT": -1})
        back_df["Cumulative"] = back_df["Pip_Return"].cumsum()
        st.write(f"Signals: {len(back_df)} | Accuracy: {back_df['Correct'].mean()*100:.2f}% | Net Pips: {back_df['Cumulative'].iloc[-1]:.2f}")
        st.line_chart(back_df["Cumulative"], height=200)

        if st.checkbox(f"📩 Send daily summary {symbol}", key=f"summary_{symbol}"):
            send_telegram_message(
                f"📊 Daily Summary for {symbol}\nSignals: {len(back_df)}\n"
                f"Accuracy: {back_df['Correct'].mean()*100:.2f}%\nNet Pips: {back_df['Cumulative'].iloc[-1]:.2f}"
            )

    # Display data and download
    with st.expander(f"📄 {symbol} Data"):
        st.dataframe(df.tail(10))
        st.dataframe(history_df.tail(10))
    st.download_button("⬇️ Download History", history_df.to_csv().encode("utf-8"), file_name=f"{symbol.replace('/', '')}_history.csv")

# ─── TIMER UNTIL NEXT CANDLE ───────────────────────────
now = datetime.now(ZoneInfo("Asia/Kolkata"))
next_candle = (now + timedelta(minutes=5 - now.minute % 5)).replace(second=0, microsecond=0)
remaining = (next_candle - now).total_seconds()
mins, secs = divmod(int(remaining), 60)
st.metric("⏳ Next Candle In", f"{mins}m {secs}s")
