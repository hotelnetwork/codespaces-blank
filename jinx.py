from flask import request, jsonify
import plotly.graph_objects as go
import plotly.express as px
import io
from contextlib import redirect_stdout
import torch.nn as nn
import torch
from sklearn.preprocessing import MinMaxScaler
import sys
from scipy.stats import beta, norm, binom, poisson, laplace, uniform, expon, gamma, t, chi2, f, multivariate_normal
import matplotlib.pyplot as plt
import numpy as np
from subprocess import call as jinx
import concurrent.futures
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
import os
from datetime import datetime
import yfinance as y
import warnings as w
from flask import Flask as F, render_template
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

w.filterwarnings("ignore")
w.filterwarnings("ignore", category=DeprecationWarning)
w.filterwarnings("ignore", category=FutureWarning)

if sys.argv[0] == 'jinx.py':
    from index import clf

# Use a Template to render the dataframe as HTML


# Find p(movement in 1 direction with less than (1, 2, std.dev) percent revert in the opposite, given tesla 1m data)

# Set Date and Time to Pacific Standard Time
os.environ['TZ'] = 'PST8PDT'
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Disable GPU

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

# Get tkr from Ticker in search bar If it is set to None, then use the first ticker in the list
tkr = tickers[0]
itl = intervals[0]
# prd uses itl as the index to get the period from the periods environment variable
prd = periods[itl]

# Prior parameters
alpha_prior = 1
beta_prior = 1

# Data: 10 successes out of 30 trials
successes = 10  # number of successes
trials = 30

# Posterior parameters
# Set formula RHS to Environment Variable
alpha_post = alpha_prior + successes
beta_post = beta_prior + trials - successes

# Plotting
x = np.linspace(0, 1, 1002)[1:-1]
plt.plot(x, beta.pdf(x, alpha_prior, beta_prior), label='Prior')
plt.plot(x, beta.pdf(x, alpha_post, beta_post), label='Posterior')
plt.legend()
plt.show()

# Save the plot to file in the static folder with a random hash
# plt.savefig('static/images/plot.png')
plt.savefig('static/images/plts/plot' + str(np.random.randint(10000)) + '.png')

# Print the mean of the posterior
if sys.argv[0] == 'jinx.py':
    print("Beta: Mean: ", beta.mean(alpha_post, beta_post))
    # Print the variance of the posterior
    print("Beta: Var: ", beta.var(alpha_post, beta_post))
    # Print the 95% credible interval
    print("Beta: Interval: ", beta.interval(0.95, alpha_post, beta_post))

# Set CSRFProtect Secret Key to Random String
os.environ.update({'SECRET_KEY': str(np.random.random())})


# Load data
# data = pd.read_csv('/workspaces/codespaces-blank/python/tickers/TSLA/index0002.csv')

# Load the data from yfinance of the 1m interval of the last 7 days
data = y.download(tkr, period=prd, interval=itl)
# Convert the data to a Pandas DataFrame
data = pd.DataFrame(data)

# Print the data
if sys.argv[0] == 'jinx.py':
    print(data.head())

# Print the shape of the data
if sys.argv[0] == 'jinx.py':
    print(data.shape)

# Preprocess data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Create training and test datasets
train_data = data_scaled[:int(len(data_scaled)*0.8)]
test_data = data_scaled[int(len(data_scaled)*0.8):]

# Run training and test data
train_data = np.expand_dims(train_data, axis=1)
test_data = np.expand_dims(test_data, axis=1)

# Print the shapes of the data
# print(train_data.shape)
# print(test_data.shape)

# Import libraries

# Define model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(1, 50)
        self.l2 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.l1(x))
        x = self.l2(x)
        return x


model = Net()

# Train model
criterion = nn.MSELoss()
# Print the model and criterion
if sys.argv[0] == 'jinx.py':
    print("model: ", model)
    print("criterion: ", criterion)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    inputs = torch.from_numpy(train_data).float()
    targets = torch.from_numpy(train_data).float()

    optimizer.zero_grad()
    outputs = model(inputs)

    # Print the model and criterion
    if sys.argv[0] == 'jinx.py':
        print("Outputs", outputs)

    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        if sys.argv[0] == 'jinx.py':
            print(f'Epoch {epoch+1}/{100}, Loss: {loss.item()}')

# Test model
model.eval()
with torch.no_grad():
    inputs = torch.from_numpy(test_data).float()
    targets = torch.from_numpy(test_data).float()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    if sys.argv[0] == 'jinx.py':
        print(f'Test Loss: {loss.item()}')

# Create training and test labels
train_labels = data['Open'][:int(len(data)*0.8)]
test_labels = data['Open'][int(len(data)*0.8):]

# Print the shapes of the data
if sys.argv[0] == 'jinx.py':
    print(train_data.shape, test_data.shape,
          train_labels.shape, test_labels.shape)

# Print Jinx Data
if sys.argv[0] == 'jinx.py':
    print("Clf:", clf)

# Save the data to file
np.save('train_data.npy', train_data)
np.save('test_data.npy', test_data)
np.save('train_labels.npy', train_labels)
np.save('test_labels.npy', test_labels)

# Check what file called this file.
if sys.argv[0] == 'jinx.py':

    # Print the shapes of the data
    print(train_data.shape, test_data.shape,
          train_labels.shape, test_labels.shape)
    print('This file was called directly')

    # Load the data from file
    test_data = np.load('test_data.npy')
    test_labels = np.load('test_labels.npy')
    train_data = np.load('train_data.npy')
    train_labels = np.load('train_labels.npy')


class Option:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def calculate_price(self):
        d1 = (math.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2)
              * self.T) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)

        call_price = self.S * norm.cdf(d1) - self.K * math.exp(-self.r * self.T) * norm.cdf(d2)
        put_price = self.K * math.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

        return call_price, put_price


# Convert the data to a pandas dataframe
df = pd.DataFrame(data)

# Drop the columns that are not needed
# df = df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)

# Convert the Date column to a datetime object
df['date'] = pd.to_datetime(df.index)

# Convert the Date column to a datetime object
# df['date'] = pd.to_datetime(df['date'])

# Set the Date column as the index
# df = df.set_index('date')

# Convert the dataframe to a numpy array
X = df.to_numpy()

# Split the data into training and testing sets
X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]

x_train = []
y_train = []

# Define the training loop
for i in range(60, len(X_train)):
    x_train.append(X_train[i-60:i, 0])
    y_train.append(X_train[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data to fit the model
x_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
x_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the model
model = keras.Sequential([
    keras.layers.LSTM(128, input_shape=(X_train.shape[1], 1)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])
