from flask import Flask, render_template
from strategy007 import Backtest, MomentumStrategy
from datetime import datetime, timedelta

app = Flask(__name__)

@app.route('/')
def home():
    # Use the classes to calculate and backtest signals for Tesla
    strategy = MomentumStrategy('TSLA')
    backtest = Backtest(strategy)
    end = datetime.now()
    signals = backtest.run(end - timedelta(days=59), end, '15m')

    # Pass the signals DataFrame to the template
    return render_template('home.html', signals=signals.to_html())

if __name__ == '__main__':
    app.run(debug=True)
