import tensorflow as tf
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Load data from yfinance and convert
# import yfinance as yf
# 
# data = yf(tickers='AAPL', period='1d', interval='1m')
# data.reset_index(inplace=True)
# data.rename(columns={'index': 'Date'}, inplace=True)
# data.Date = pd.to_datetime(data.Date)
# data.set_index('Date', inplace=True)
# Save the data to a CSV file
# data.to_csv('stock_data.csv')
# 

# Load the data
# data = pd.read_csv('stock_data.csv')

# Load the data
data = np.loadtxt('stock_data.csv', delimiter=',')
x = data[:, 0]  # The dates
y = data[:, 1]  # The stock prices

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Create the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=100)

# Evaluate the model
model.evaluate(x_test, y_test)

# Make predictions
y_pred = model.predict(x_test)

# Plot the predictions
plt.plot(x_test, y_test, label='Actual')
plt.plot(x_test, y_pred, label='Predicted')
plt.legend()
plt.show()
