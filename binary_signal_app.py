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

# ─── AUTO REFRESH ──────────────────────────────────────
st_autorefresh(interval=1000, limit=None, key="refresh")

# ─── API KEY ───────────────────────────────────────────
twelve_key = "4d5b1e81f9314e28a7ee285497d3b273"

# ─── SYMBOL SELECT ─────────────────────────────────────
symbol_map = {
    "EUR/USD": "EUR/USD",
    "USD/JPY": "USD/JPY",
    "GBP/USD": "GBP/USD",
    "AUD/USD": "AUD/USD",
    "USD/CAD": "USD/CAD"
}
symbol = st.selectbox("Choose a forex pair:", list(symbol_map.keys()))

# ─── FETCH HISTORICAL DATA ─────────────────────────────
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

    # Add indicators
    df["EMA9"] = ta.trend.ema_indicator(df["Close"], window=9)
    df["EMA21"] = ta.trend.ema_indicator(df["Close"], window=21)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()

    # Target: 1 = next candle green (CALL), 0 = red (PUT)
    df["Target"] = df["Close"].shift(-1) > df["Open"].shift(-1)
    df["Target"] = df["Target"].map({True: 1, False: 0})

    df.dropna(inplace=True)
    return df

# ─── TITLE ─────────────────────────────────────────────
st.title("🤖 Binary Trading Signal Bot with ML + Alerts")

df = fetch_data(symbol)
if df is None:
    st.error("❌ Failed to fetch data.")
    st.stop()

features = ["EMA9", "EMA21", "RSI", "MACD", "BB_upper", "BB_lower", "Close", "Open"]
X = df[features]
y = df["Target"]

# ─── MODEL FILE PATH ───────────────────────────────────
model_path = f"{symbol.replace('/', '')}_rf_model.pkl"

# ─── RETRAIN BUTTON ────────────────────────────────────
if st.button("🔁 Retrain model now"):
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    st.success("✅ Model retrained successfully!")

# ─── LOAD OR TRAIN MODEL ───────────────────────────────
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)

# ─── PREDICTION & SIGNAL ───────────────────────────────
df["ML_Signal"] = model.predict(X)
df["ML_Signal"] = df["ML_Signal"].map({1: "CALL", 0: "PUT"})
df["Confidence"] = model.predict_proba(X)[:, 1]
df["Actual"] = df["Target"].map({1: "CALL", 0: "PUT"})
df["Correct"] = df["ML_Signal"] == df["Actual"]
ml_accuracy = df["Correct"].mean() * 100

# ─── SESSION TRACKING ──────────────────────────────────
if "ml_history" not in st.session_state:
    st.session_state.ml_history = []

last_time = df.index[-1]
last_signal = df.iloc[-1]["ML_Signal"]
last_actual = df.iloc[-1]["Actual"]
last_outcome = int(last_signal == last_actual)

if not st.session_state.ml_history or st.session_state.ml_history[-1]['time'] != last_time:
    total = len(st.session_state.ml_history) + 1
    correct = sum(i['outcome'] for i in st.session_state.ml_history) + last_outcome
    session_acc = (correct / total) * 100
    st.session_state.ml_history.append({
        "time": last_time, "signal": last_signal, "actual": last_actual,
        "outcome": last_outcome, "accuracy": session_acc
    })
else:
    session_acc = st.session_state.ml_history[-1]["accuracy"]

# ─── TIMER TO NEXT CANDLE ──────────────────────────────
now = datetime.now(ZoneInfo("Asia/Kolkata"))
minute = (now.minute // 5) * 5
next_candle_time = now.replace(minute=minute, second=0, microsecond=0) + timedelta(minutes=5)
remaining = (next_candle_time - now).total_seconds()
minutes, seconds = divmod(int(remaining), 60)

# ─── METRICS + COLOR CODES ─────────────────────────────
st.metric("⏳ Time to next candle", f"{minutes}m {seconds}s")
st.metric("📍 ML Latest Signal", last_signal)
st.metric("🧠 Confidence", f"{df.iloc[-1]['Confidence']:.2%}")

acc_color = "green" if ml_accuracy >= 70 else "orange" if ml_accuracy >= 50 else "red"
st.markdown(f"### <span style='color:{acc_color}'>📈 Model Accuracy: {ml_accuracy:.2f}%</span>", unsafe_allow_html=True)

acc2_color = "green" if session_acc >= 70 else "orange" if session_acc >= 50 else "red"
st.markdown(f"### <span style='color:{acc2_color}'>🧪 Session Accuracy: {session_acc:.2f}%</span>", unsafe_allow_html=True)

# ─── CHARTS ────────────────────────────────────────────
history_df = pd.DataFrame(st.session_state.ml_history).set_index("time")
st.line_chart(history_df["accuracy"], height=200)

with st.expander("📊 Show recent data & predictions"):
    st.dataframe(df.tail(10))
    st.dataframe(history_df.tail(10))

csv = history_df.to_csv().encode("utf-8")
st.download_button("Download session history", csv, file_name="ml_session_history.csv", mime="text/csv")
