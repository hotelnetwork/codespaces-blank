# Tensor Stock price prediction
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Use array of tickers
tickers = ['TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'META', 'GOOGL', 'NVDA', 'NFLX', 'FB', 'MSFT', 'INTC', 'CSCO', 'CMCSA', 'ADBE', 'AVGO', 'TXN', 'QCOM', 'CHTR', 'SBUX', 'ADP', 'AMD']

# Get historical data for all tickers via for loop.
for ticker in tickers:
    data = yf.download(ticker)
    # Save to dir tickers/ticker[i]
    # Create directory tickers if necessary.
    if not os.path.exists('tickers'):
        os.makedirs('tickers')
    # Save data to file.
    data.to_csv(f'tickers/{ticker}.csv')



# Load data from file in tickers on file from TSLA.
data = pd.read_csv('tickers/TSLA.csv')

# Print the first 5 rows of the data
print(data.head())

# Print the last 5 rows of the data
print(data.tail())

# Print the shape of the data (number of rows and columns)
print(data.shape)

# Print the data types of each column
print(data.dtypes)

# Print the summary statistics of the data
print(data.describe())

# Print the correlation matrix of the data
print(data.corr())

# Print the correlation between the "Open" and "Close" columns
print(data["Open"].corr(data["Close"]))

# Print the correlation between the "High" and "Low" columns
print(data["High"].corr(data["Low"]))

# Print the correlation between the "Volume" and "Close" columns
print(data["Volume"].corr(data["Close"]))

# Print the correlation between the "Open" and "Volume" columns
print(data["Open"].corr(data["Volume"]))

# Print the correlation between the "High" and "Volume" columns
print(data["High"].corr(data["Volume"]))

# Print the correlation between the "Low" and "Volume" columns
print(data["Low"].corr(data["Volume"],method='pearson'))

# Print the correlation between the "Close" and "Volume" columns
print(data["Close"].corr(data["Volume"],method='pearson'))

# Print the correlation between the "Open" and "Volume" columns
print(data["Open"].corr(data["Volume"],method='pearson'))

# Print the correlation between the "High" and "Volume" columns
print(data["High"].corr(data["Volume"],method='pearson'))

# Load the historical stock price data for the ticker symbol described by the user as a supplied arg to the script.
data = yf.download(tickers[0])
# data = yf.download('HOLO')

# Drop the date column
data = data.drop('Date', axis=1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(train_data.shape[1], 1)))
model.add(LSTM(units=100))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(train_data, train_data, epochs=100, batch_size=32)

# Evaluate the model
model.evaluate(test_data, test_data)

# Make predictions
predictions = model.predict(test_data)

# Denormalize the predictions
predictions = scaler.inverse_transform(predictions)

# Plot the predictions
plt.plot(test_data, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()

# Save the model
model.save('lstm_model.keras')

# Load the model
model = tf.keras.models.load_model('lstm_model.keras')