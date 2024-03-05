import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from datetime import datetime as dt
from datetime import timedelta as td

# Download historical data for Tesla with 1-minute intervals
end = dt.now()
start = end - td(days=7)
data = yf.download('TSLA', start=start, end=end, interval='1m')

# Calculate the RSI
rsi = RSIIndicator(data['Close']).rsi()

# Create a DataFrame to hold the data and the RSI
df = pd.DataFrame({
    'Close': data['Close'],
    'RSI': rsi
})

# Define a simple momentum strategy: buy when the RSI is below 30 (oversold) and sell when the RSI is above 70 (overbought)
df['Buy_Signal'] = (df['RSI'] < 30)
df['Sell_Signal'] = (df['RSI'] > 70)

print(df)
