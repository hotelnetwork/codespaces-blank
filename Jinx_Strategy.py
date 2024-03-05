import pandas as pd
import yfinance as yf
from ta.trend import MACD
from ta.momentum import StochasticOscillator, RSIIndicator
from ta.volatility import BollingerBands

# Download h<?php
// Get all template files
$files = glob('../templates/*.php');

// Function to get the base name of the file without the extension
function getFileName($file) {
    return basename($file, '.php');
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>Menu</title>
    <style>
        /* Add some basic styling to the menu */
        .menu {
            list-style-type: none;
            padding: 0;
        }
        .menu li {
            padding: 10px;
            margin-bottom: 10px;
            background-color: #f0f0f0;
        }
        .menu li a {
            text-decoration: none;
            color: black;
        }
        .menu li a:hover {
            color: white;
            background-color: #000;
        }
    </style>
</head>
<body>
    <ul class="menu">
        <?php foreach ($files as $file): ?>
            <li><a href="<?php echo '../' . $file; ?>"><?php echo getFileName($file); ?></a></li>
        <?php endforeach; ?>
    </ul>
</body>
</html>istorical data for desired ticker symbol
data = yf.download('AAPL', start='2020-01-01', end='2022-12-31')

# Calculate MACD
macd = MACD(data['Close'])
data['MACD'] = macd.macd_diff()

# Calculate RSI
rsi = RSIIndicator(data['Close'])
data['RSI'] = rsi.rsi()

# Calculate Stochastics
stoch = StochasticOscillator(data['High'], data['Low'], data['Close'])
data['%K'] = stoch.stoch()
data['%D'] = stoch.stoch_signal()

# Calculate Bollinger Bands
bb = BollingerBands(data['Close'])
data['BB_High'] = bb.bollinger_hband()
data['BB_Low'] = bb.bollinger_lband()

# Create signals based on the indicators
data['Buy_Signal'] = (data['MACD'] > 0) & (data['RSI'] < 30) & (data['%K'] < 20) & (data['Close'] < data['BB_Low'])
data['Sell_Signal'] = (data['MACD'] < 0) & (data['RSI'] > 70) & (data['%K'] > 80) & (data['Close'] > data['BB_High'])

# print buy if true and sell if false.


print(data)