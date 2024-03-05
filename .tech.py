import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class StockAnalyzer:
    def __init__(self, ticker):
        self.ticker = ticker
        self.data = yf.download(self.ticker)

    def _calculate_momentum(self, days):
        return self.data['Close'].diff(days)

    def _calculate_std(self, days):
        return self.data['Close'].rolling(window=days).std()

    def _calculate_vwap(self, days):
        vwap = (self.data['Volume'] * self.data['Close']).rolling(window=days).sum() / self.data['Volume'].rolling(window=days).sum()
        return vwap

    def calculate_indicators(self):
        indicators = {}
        for days in [10, 20, 60]:
            indicators[f'momentum_{days}'] = self._calculate_momentum(days)
            indicators[f'std_{days}'] = self._calculate_std(days)
            indicators[f'vwap_{days}'] = self._calculate_vwap(days)
        return indicators

    def plot_indicators(self, indicators):
        plt.figure(figsize=(14, 7))
        for name, values in indicators.items():
            plt.plot(values, label=name)
        plt.legend()
        plt.title(f'Indicators for {self.ticker}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)
        plt.savefig(f'tech/{self.ticker}.png')
        plt.close()

# Create the tech directory if it doesn't exist
if not os.path.exists('tech'):
    os.makedirs('tech')

# List of tickers to analyze
tickers = ['^DJI', '^IXIC', 'TSLA', 'AAPL', 'AMZN', 'GOOGL', 'GOOG']

# Calculate and plot indicators for each ticker
for ticker in tickers:
    analyzer = StockAnalyzer(ticker)
    indicators = analyzer.calculate_indicators()
    analyzer.plot_indicators(indicators)