import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import ta
import requests

# ─── Fetch historical data from Twelve Data API ───
def fetch_data():
    url = "https://api.twelvedata.com/time_series?symbol=EUR/USD&interval=5min&outputsize=500&apikey=4d5b1e81f9314e28a7ee285497d3b273"
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame(data["values"])
    df = df.rename(columns={
        "datetime": "Datetime",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume"
    })
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.set_index("Datetime").astype(float).sort_index()
    return df

df = fetch_data()

# ─── Add indicators ───
df["EMA9"]   = ta.trend.ema_indicator(df["Close"], window=9)
df["EMA21"]  = ta.trend.ema_indicator(df["Close"], window=21)
df["RSI"]    = ta.momentum.rsi(df["Close"], window=14)
macd        = ta.trend.MACD(df["Close"])
df["MACD"]   = macd.macd_diff()
bb          = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
df["BB_up"]  = bb.bollinger_hband()
df["BB_low"] = bb.bollinger_lband()
df["NextClose"] = df["Close"].shift(-1)
df.dropna(inplace=True)

# ─── Label & Train ───
df["Label"] = (df["NextClose"] > df["Close"]).astype(int)
features = ["EMA9", "EMA21", "RSI", "MACD", "BB_up", "BB_low", "Volume"]
X = df[features]
y = df["Label"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ─── Save model ───
joblib.dump(model, "rf_model.pkl")
print("✅ Model trained and saved to rf_model.pkl")
