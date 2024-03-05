import pandas as pd
import yfinance as yf
from ta.momentum import StochasticOscillator
from datetime import datetime, timedelta
import plotly.graph_objects as go


class MomentumStrategy:
    def __init__(self, symbol, stoch_low=20, stoch_high=80):
        self.symbol = symbol
        self.stoch_low = stoch_low
        self.stoch_high = stoch_high

    @staticmethod
    def _calculate_std_dev(data):
        return data['Close'].rolling(window=20).std()

    def calculate_signals(self, start, end, interval):
        # Download historical data with specified interval
        data = yf.download(self.symbol, start=start, end=end, interval=interval)

        # Calculate the Stochastic Oscillator
        stoch = StochasticOscillator(data['High'], data['Low'], data['Close']).stoch()

        # Calculate the Standard Deviation
        std_dev = self._calculate_std_dev(data)

        # Create a DataFrame to hold the data, the Stochastic Oscillator, and the Standard Deviation
        df = pd.DataFrame({
            'Close': data['Close'],
            'Stoch': stoch,
            'Std_Dev': std_dev
        })

        # Define the momentum strategy: buy when the Stochastic Oscillator is below stoch_low and sell when it is above stoch_high
        df['Buy_Signal'] = (df['Stoch'] < self.stoch_low)
        df['Sell_Signal'] = (df['Stoch'] > self.stoch_high)

        return df


class Backtest:
    def __init__(self, strategy):
        self.strategy = strategy

    def run(self, start, end, interval):
        signals = self.strategy.calculate_signals(start, end, interval)

        # Plot closing prices and signals using Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=signals.index, y=signals['Close'], mode='lines', name='Close'))
        fig.add_trace(go.Scatter(x=signals[signals['Buy_Signal']].index, y=signals[signals['Buy_Signal']]
                      ['Close'], mode='markers', name='Buy', marker=dict(color='green', size=10, symbol='circle')))
        fig.add_trace(go.Scatter(x=signals[signals['Sell_Signal']].index, y=signals[signals['Sell_Signal']]
                      ['Close'], mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='circle')))
        # Define dropdown menu
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(
                            args=["type", "scatter"],
                            label="Line",
                            method="restyle"
                        ),
                        dict(
                            args=["type", "bar"],
                            label="Bar",
                            method="restyle"
                        ),
                        dict(
                            args= ["type", "candlestick"],
                            label="Candlestick",
                            method="restyle"
                        )
                    ]),
                    direction="down",
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )
        layout = go.Layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                            label="1m",
                            step="month",
                            stepmode="backward"),
                        dict(count=6,
                            label="6m",
                            step="month",
                            stepmode="backward"),
                        dict(count=1,
                            label="YTD",
                            step="year",
                            stepmode="todate"),
                        dict(count=1,
                            label="1y",
                            step="year",
                            stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(
                    visible=True
                ),
                type="date"
            )
        )
        fig.show()
        fig.write_html('templates/backtest' + datetime.now().strftime("%Y%m%d%H%M%S") + '.html')
        self.generate_report(signals, 'trades/trades' + datetime.now().strftime('%Y%m%d%H%M%S') + interval + '.csv')
    

    def generate_report(self, signals, filename):
        # Generate a report based on the signals and save it to a CSV file
        signals.to_csv(filename)

class BacktestWithCapital(Backtest):
    def __init__(self, strategy, capital=10000):
        super().__init__(strategy)
        self.capital = capital

    def run(self, start, end, interval):
        signals = self.strategy.calculate_signals(start, end, interval)

        # Calculate returns
        signals['Return'] = signals['Close'].pct_change()
        signals['Cumulative Return'] = (1 + signals['Return']).cumprod()

        # Calculate portfolio value
        signals['Portfolio Value'] = self.capital * signals['Cumulative Return']

        # Plot portfolio value
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=signals.index, y=signals['Portfolio Value'], mode='lines', name='Portfolio Value'))
        fig1.show()
        fig1.write_html('templates/backtest_portfolio' + datetime.now().strftime("%Y%m%d%H%M%S") + '.html')
        fig1.write_image('static/images/backtest_portfolio' + datetime.now().strftime("%Y%m%d%H%M%S") + '.png')

        # Plot ticker price with buy/sell signals
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=signals.index, y=signals['Close'], mode='lines', name='Price'))
        fig2.add_trace(go.Scatter(x=signals[signals['Buy_Signal']].index, y=signals[signals['Buy_Signal']]['Close'], mode='markers', name='Buy', marker=dict(color='green', size=10, symbol='circle')))
        fig2.add_trace(go.Scatter(x=signals[signals['Sell_Signal']].index, y=signals[signals['Sell_Signal']]['Close'], mode='markers', name='Sell', marker=dict(color='red', size=10, symbol='circle')))
        fig2.show()
        fig2.write_html('templates/backtest_ticker' + datetime.now().strftime("%Y%m%d%H%M%S") + '.html')
        fig2.write_image('static/images/backtest_ticker' + datetime.now().strftime("%Y%m%d%H%M%S") + '.png')

        return signals

# Use the classes to calculate and backtest signals for Tesla
strategy = MomentumStrategy('TSLA')
# backtest = BacktestWithCapital(strategy)
backtest = Backtest(strategy)
end = datetime.now()

backtest.run(end - timedelta(days=59), end, '15m')