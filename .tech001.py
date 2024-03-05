import os
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

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
        fig = go.Figure()
        for name, values in indicators.items():
            fig.add_trace(go.Scatter(x=values.index, y=values, mode='lines', name=name))
        fig.update_layout(title=f'Indicators for {self.ticker}', xaxis_title='Date', yaxis_title='Value')
        fig.write_html(f'tech/{self.ticker}.html')

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

# Create an HTML file with a Bootstrap form to select the ticker pages
with open('tech/index.html', 'w') as f:
    f.write('<!DOCTYPE html>\n')
    f.write('<html lang="en">\n')
    f.write('<head>\n')
    f.write('  <meta charset="UTF-8">\n')
    f.write('  <meta name="viewport" content="width=device-width, initial-scale=1">\n')
    f.write('  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css">\n')
    f.write('  <title>Select Ticker</title>\n')
    f.write('</head>\n')
    f.write('<body>\n')
    f.write('  <div class="container">\n')
    f.write('    <h1>Select Ticker</h1>\n')
    f.write('    <form>\n')
    f.write('      <div class="form-group">\n')
    f.write('        <label for="tickerSelect">Choose a ticker:</label>\n')
    f.write('        <select class="form-control" id="tickerSelect" onchange="location = this.value;">\n')
    for ticker in tickers:
        f.write(f'          <option value="{ticker}.html">{ticker}</option>\n')
    f.write('        </select>\n')
    f.write('      </div>\n')
    f.write('    </form>\n')
    f.write('  </div>\n')
    f.write('</body>\n')
    f.write('</html>\n')