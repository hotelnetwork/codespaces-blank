from datetime import datetime as dt
import os
from os import system as jinx, getenv as env, environ as envs, listdir as ls, remove as rm, path, mkdir, chdir, getcwd, rename as mv, rmdir as rmdir
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

# Time the script
start = dt.now()
print("Start time: ", start)

# jinx(["python", "index.py"])
tickers = ["TSLA", "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "BTC-USD", "ETH-USD", "NVDA", "NFLX", "PYPL", "ADBE", "CRM", "INTC", "CSCO", "AVGO", "QCOM", "TXN", "IBM", "MU", "AMD", "LRCX", "ADI", "ADI", "MCHP", "FISV", "INTU", "NOW", "AMAT", "ADSK", "CTSH", "XLNX", "KLAC", "CDNS", "ANSS", "SNPS", "VRSN", "CDW", "SWKS", "NTAP", "WDC", "STX", "KEYS", "FTNT", "CTXS", "AKAM", "FFIV", "TER", "QRVO", "LSCC", "MXIM", "GRMN", "ZBRA", "CDK", "SSNC", "NLOK", "JKHY", "ANET", "BR", "TYL", "GIB",
           "GPN", "PAYC", "FLT", "WEX", "FIS", "VNT", "VRSK", "SNX", "EPAM", "LDOS", "SAIC", "IT", "LDOS", "SAIC", "IT", "XRP-USD", "LTC-USD", "BCH-USD", "LINK-USD", "ADA-USD", "XLM-USD", "USDT-USD", "DOGE-USD", "WBTC-USD", "UNI3-USD", "AAVE-USD", "SNX-USD", "COMP-USD", "MKR-USD", "YFI-USD", "UMA-USD", "SUSHI-USD", "CRV-USD", "REN-USD", "BAL-USD", "KNC-USD", "BNT-USD", "GRT-USD", "LRC-USD", "OCEAN-USD", "BAND-USD", "RLC-USD", "NMR-USD", "REP-USD", "MLN-USD", "FIL-USD", "LPT-USD", "CVC-USD", "NU-USD"]
os.environ['INTERVALS'] = '1m,5m,15m,30m,1h,4h,1d,1wk,1mo,3mo'
intervals = os.getenv('INTERVALS').split(',')
os.environ['PERIODS'] = '1m:7d,5m:60d,15m:60d,30m:60d,1h:730d,4h:max,1d:max,1wk:max,1mo:max,3mo:max'

# set period based on interval using switch case
periods = {
    '1m': '7d',
    '5m': '60d',
    '15m': '60d',
    '30m': '60d',
    '1h': '730d',
    '4h': 'max',
    '1d': 'max',
    '1wk': 'max',
    '1mo': 'max',
    '3mo': 'max'
}

tkr = tickers[7]
itl = intervals[7]
prd = periods[itl]
data = yf.download(tkr, period=prd, interval=itl)

# Use only 'Close' column
data = data['Close'].values
data = data.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_data = data[:int(len(data)*0.8)]
test_data = data[int(len(data)*0.8):]

# Create the training data
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile and fit the model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# Test the model
x_test = []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# Apply inverse transformation to train_data
train_data_transformed = scaler.inverse_transform(train_data)

# Print the last 60 predictions
print("Last 60 Predictions", predictions[-60:])
print("Last 60 Actual", train_data_transformed[-60:])
print("Last Close: ", data[-1])
print("Last Prediction: ", predictions[-1])
print("Last Actual: ", train_data_transformed[-1])
print("Time taken: ", dt.now() - start)


# Plot the results
plt.figure(figsize=(10,5))
# Use tkr in title.
plt.plot(data, color='blue', label='Actual ' + tkr + ' Stock Price')
# plt.plot(data[:len(train_data)], color='green', label='Training ' + tkr + ' Stock Price')
# plt.plot(data[:len(train_data_transformed)], color='green', label='Training ' + tkr + ' Stock Price')
plt.plot(range(len(y_train)+60, len(y_train)+60+len(predictions)), predictions, color='red', label='Predicted ' + tkr + ' Stock Price')
plt.title(tkr + ' Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(tkr + ' Stock Price')
plt.autoscale(tight=True)
plt.annotate('Last Close: ' + str(data[-1]), xy=(1, 0), xycoords='axes fraction', fontsize=8, xytext=(-5, 5), textcoords='offset points', ha='right', va='bottom')
# plt.annotate('Last Prediction: ' + str(predictions[-1]), xy=(1, 0), xycoords='axes fraction', fontsize=8, xytext(-5, 15), textcoords='offset points', ha='right', va='bottom')
# plt.annotate('Time taken: ' + str(dt.now() - start), xy=(1, 0), xycoords='axes fraction', fontsize=8, xytext(-5, 25), textcoords='offset points', ha='right', va='bottom') 

plt.legend()
plt.show()

# Save Plot to file using name of ticker and timestamp format dt.now().timestamp() to avoid overwriting
# If any of the directories do not exist, create them
if not path.exists('static/images/' + tkr + '/'):
    mkdir('static/images/' + tkr + '/')
if not path.exists('static/images/' + tkr + '/' + itl+ '/'):
    mkdir('static/images/' + tkr + '/' + itl+ '/')
if not path.exists('static/images/' + tkr + '/' + itl+ '/' + prd + '/'):
    mkdir('static/images/' + tkr + '/' + itl+ '/' + prd + '/')

plt.savefig('static/images/' + tkr + '/' + itl+ '/' + prd + '/' + str(dt.now().timestamp()) + '.png')

# Print the time taken
print("Time taken: ", dt.now() - start)