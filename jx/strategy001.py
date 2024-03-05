import yfinance as yf
import pandas as pd

# Download historical data
data = yf.download('TSLA', start='2020-01-01', end='2022-12-31')

# Calculate moving averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Create signals
data['Buy_Signal'] = (data['MA50'] > data['MA200']) & (data['MA50'].shift(1) < data['MA200'].shift(1))
data['Sell_Signal'] = (data['MA50'] < data['MA200']) & (data['MA50'].shift(1) > data['MA200'].shift(1))

# Backtest strategy
capital = 10000  # starting capital
shares = 0  # number of shares owned
for i in range(200, len(data)):
    if data['Buy_Signal'].iloc[i]:
        shares += capital / data['Close'].iloc[i]  # buy as many shares as possible
        capital = 0  # use all capital to buy shares
    elif data['Sell_Signal'].iloc[i]:
        capital += shares * data['Close'].iloc[i]  # sell all shares
        shares = 0  # sell all shares

# Calculate final portfolio value
portfolio_value = capital + shares * data['Close'].iloc[-1]
print(f'Final portfolio value: {portfolio_value}')
