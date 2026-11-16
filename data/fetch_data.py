import requests
import pandas as pd

url = (
    "https://query1.finance.yahoo.com/v8/finance/chart/"
    "%5EGSPC?range=30y&interval=1d"
)

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(
    url,
    headers=headers,
    timeout=30
)

response.raise_for_status()

data = response.json()["chart"]["result"][0]

timestamps = data["timestamp"]
quotes = data["indicators"]["quote"][0]

df = pd.DataFrame({
    "Date": pd.to_datetime(timestamps, unit="s").date,
    "Open": quotes["open"],
    "High": quotes["high"],
    "Low": quotes["low"],
    "Close": quotes["close"],
    "Volume": quotes["volume"]
})

df = df.dropna(subset=["Close"])
df.to_csv("SP500_daily.csv", index=False)

print(df.head())
print(df.tail())
print("数据总量：", len(df))