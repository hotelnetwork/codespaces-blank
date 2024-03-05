import pandas as pd
import yfinance as yf


# Download historical data
# Get user input for end date using fzf with end date starting at day, 2 days, 5 days, 10 days, 1 month, 3 months, 6 months, 1 year, 2 years, 5 years, 10 years, and max.

data = yf.download('TSLA', start='2020-01-01', end='2024-02-29')

# Calculate moving averages
data['MA50'] = data['Close'].rolling(window=50).mean()
data['MA200'] = data['Close'].rolling(window=200).mean()

# Create signals
data['Buy_Signal'] = (data['MA50'] > data['MA200']) & (
    data['MA50'].shift(1) < data['MA200'].shift(1))
data['Sell_Signal'] = (data['MA50'] < data['MA200']) & (
    data['MA50'].shift(1) > data['MA200'].shift(1))

# Backtest strategy
initial_capital = 10000  # starting capital
capital = initial_capital
shares = 0  # number of shares owned
for i in range(200, len(data)):
    if data['Buy_Signal'].iloc[i]:
        # buy as many shares as possible
        shares += capital / data['Close'].iloc[i]
        capital = 0  # use all capital to buy shares
    elif data['Sell_Signal'].iloc[i]:
        capital += shares * data['Close'].iloc[i]  # sell all shares
        shares = 0  # sell all shares

# Calculate final portfolio value
final_portfolio_value = capital + shares * data['Close'].iloc[-1]

# Calculate annualized return
num_days = (data.index[-1] - data.index[200]).days
annualized_return = (final_portfolio_value /
                     initial_capital) ** (365.0 / num_days) - 1

print(f'Annualized return: {annualized_return * 100}%')
