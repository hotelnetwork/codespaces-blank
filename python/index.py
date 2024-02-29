import tensorflow as tf
import pandas as pd
import os
import datetime as dt
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load data using yfinance
import yfinance as yf

# Array of Tickers
tickers = ['TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'FB', 'NFLX', 'NVDA', 'PYPL', 'SQ', 'PLTR', 'GME', 'BB']

## # Download data for each ticker
## data = yf.download(tickers, start='2010-01-01', end='2023-02-19')
## 
## # Calculate the returns
## returns = data.pct_change()
## 
## # Calculate the covariance matrix
## cov_matrix = returns.cov()
## 
## # Calculate the correlation matrix
## corr_matrix = returns.corr()
## 
## # Calculate the mean returns
## mean_returns = returns.mean()
## 
## # Calculate the portfolio weights
## weights = np.random.random(len(tickers))
## weights /= np.sum(weights)
## 
## # Save calculations for each company to file as well as downloaded data for each company in 2 separate folders per company.
## 
## # Create a folder for each company
## 
# Download the data for each company
for ticker in tickers:
    # End date is always today
    end_date = dt.date.today()
    # Start date is 1 year ago
    start_date = end_date - dt.timedelta(days=365)
    # All avaliable intervals
    _interval = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
    # Download the data for the company
    data = yf.download(ticker, start=start_date, end=end_date, group_by='ticker', progress=False, auto_adjust=True, actions=False, interval=_interval)
    # If the folder doesn't exist, create it. If it does, increment that one name by 1 and create it.
    if not os.path.exists(ticker):
        os.mkdir(ticker)
    else:
        # Increment the folder name by 1
        ticker = ticker + '1'
        # Create the folder
        os.mkdir(ticker)
    data.to_csv(f'tickers/{ticker + "_data"}.csv')
## 
## # Create a file for each company
##     with open(f'{ticker}/calculations.txt', 'w') as f:
##         f.write(f'Tickers: {tickers}\n')
##         f.write(f'Weights: {weights}\n')
##         f.write(f'Mean Returns: {mean_returns}\n')
##         f.write(f'Covariance Matrix: {cov_matrix}\n')
##         f.write(f'Correlation Matrix: {corr_matrix}\n')
##         f.write(f'Returns: {returns}\n')
##         f.write(f'Data: {data}\n')

# Save the data to file
### data.to_csv('data.csv')

# Get data from tsla file to save as data converted to numpy

## data = pd.read_csv('tickers/TSLA_data.csv')
## # Drop the date column
## data = data.drop('Date', axis=1)
## # Convert the data to a numpy array
## data = data.to_numpy()
## 
## # Split the data into training and testing sets
## # Load the data
## x = data[:, 0]  # The dates
## y = data[:, 1]  # The stock prices
## 
## # Split the data into training and testing sets
## 
## # Split the data into training and testing sets
## x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
## 
## # Create the model
## model = tf.keras.Sequential()
## model.add(tf.keras.layers.Dense(1, input_shape=(1,)))
## model.compile(optimizer='sgd', loss='mse')
## 
## # Train the model
## model.fit(x_train, y_train, epochs=100)
## 
## # Make predictions
## y_pred = model.predict(x_test)
## 
## # Plot the results
## plt.plot(x_test, y_test, 'b')
## plt.plot(x_test, y_pred, 'r')
## plt.show()
## 
## # Save the model
## # model.save('model.h5')
## model.save(filepath='model.keras', overwrite=True) # , include_optimizer=False)
## 
## # Load the model
## model = tf.keras.models.load_model('model.keras')