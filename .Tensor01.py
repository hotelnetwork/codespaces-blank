import numpy as np
import time as tm
import datetime as dt
import tensorflow as tf
from os import path, mkdir
from datetime import datetime as d

# Data preparation
from yahoo_fin import stock_info as yf
from sklearn.preprocessing import MinMaxScaler
from collections import deque

# AI
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

# Graphics library
import matplotlib.pyplot as plt
import pyttsx3

# Time the script
# start = d.now()

# SETTINGS

# Window size or the sequence length, 7 (1 week)
N_STEPS = 7

# Lookup steps, 1 is the next day, 3 = after tomorrow
LOOKUP_STEPS = [1, 2, 3]

tickers = ['TSLA', 'AAPL', 'GOOG', 'GOOGL', 'AMZN', 'MSFT', 'META', 'BTC-USD', 'ETH-USD']

# Stock ticker, GOOGL
tkr = tickers[0]

# Current date
date_now = tm.strftime('%Y-%m-%d')
# Get user input for the period
yrs = 10
prd = (dt.date.today() - dt.timedelta(days=yrs*365)).strftime('%Y-%m-%d')

itl = '1d'

# LOAD DATA 
# from yahoo_fin 
# for 1104 bars with interval = 1d (one day)
init_df = yf.get_data(
    tkr, 
    start_date=prd, 
    end_date=date_now, 
    interval=itl)

# remove columns which our neural network will not use
init_df = init_df.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
# create the column 'date' based on index column
init_df['date'] = init_df.index

# Let's preliminary see our data on the graphic
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(init_df['close'][-200:])
plt.xlabel("days")
plt.ylabel("price")
plt.legend([f'Actual price for {tkr}'])
plt.show()

# Scale data for ML engine
scaler = MinMaxScaler()
init_df['close'] = scaler.fit_transform(np.expand_dims(init_df['close'].values, axis=1))

def PrepareData(days):
  df = init_df.copy()
  df['future'] = df['close'].shift(-days)
  last_sequence = np.array(df[['close']].tail(days))
  df.dropna(inplace=True)
  sequence_data = []
  sequences = deque(maxlen=N_STEPS)

  for entry, target in zip(df[['close'] + ['date']].values, df['future'].values):
      sequences.append(entry)
      if len(sequences) == N_STEPS:
          sequence_data.append([np.array(sequences), target])

  last_sequence = list([s[:len(['close'])] for s in sequences]) + list(last_sequence)
  last_sequence = np.array(last_sequence).astype(np.float32)

  # construct the X's and Y's
  X, Y = [], []
  for seq, target in sequence_data:
      X.append(seq)
      Y.append(target)

  # convert to numpy arrays
  X = np.array(X)
  Y = np.array(Y)

  return df, last_sequence, X, Y

def data_generator(x_train, y_train, batch_size):
  while True:
    for i in range(0, len(x_train), batch_size):
      yield x_train[i:i+batch_size], y_train[i:i+batch_size]

# Create the generator
# generator = data_generator(x_train, y_train, batch_size)

# Train the model
# model.fit_generator(generator, steps_per_epoch=len(x_train)//batch_size, epochs=num_epochs)

def GetTrainedModel(x_train, y_train):
  model = Sequential()
  model.add(LSTM(60, return_sequences=True, input_shape=(N_STEPS, len(['close']))))
  model.add(Dropout(0.3))
  model.add(LSTM(120, return_sequences=False))
  model.add(Dropout(0.3))
  model.add(Dense(20))
  model.add(Dense(1))

  BATCH_SIZE = 5
  EPOCHS = 25

  model.compile(loss='mean_squared_error', optimizer='adam')

  model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)

  model.summary()

  return model

# GET PREDICTIONS
predictions = []

for step in LOOKUP_STEPS:
  df, last_sequence, x_train, y_train = PrepareData(step)
  x_train = x_train[:, :, :len(['close'])].astype(np.float32)

  model = GetTrainedModel(x_train, y_train)

  last_sequence = last_sequence[-N_STEPS:]
  last_sequence = np.expand_dims(last_sequence, axis=0)
  prediction = model.predict(last_sequence)
  predicted_price = scaler.inverse_transform(prediction)[0][0]

  predictions.append(round(float(predicted_price), 2))
# Execute model for the whole history range
copy_df = init_df.copy()
y_predicted = model.predict(x_train)
y_predicted_transformed = np.squeeze(scaler.inverse_transform(y_predicted))
first_seq = scaler.inverse_transform(np.expand_dims(y_train[:6], axis=1))
last_seq = scaler.inverse_transform(np.expand_dims(y_train[-3:], axis=1))
y_predicted_transformed = np.append(first_seq, y_predicted_transformed)
y_predicted_transformed = np.append(y_predicted_transformed, last_seq)
copy_df[f'predicted_close'] = y_predicted_transformed

if len(predictions) < 3:
    # If the model did not predict the next 3 days, fill the rest with 0
    for i in range(3 - len(predictions)):
        predictions.append(0)

# Add predicted results to the table
date_now = dt.date.today()
date_tomorrow = dt.date.today() + dt.timedelta(days=1)
date_after_tomorrow = dt.date.today() + dt.timedelta(days=2)

# Add the predicted results to the table for the next 3 days
try:
    copy_df.loc[date_now] = [predictions[0], f'{date_now}', 0, 0]
    copy_df.loc[date_tomorrow] = [predictions[1], f'{date_tomorrow}', 0, 0]
    copy_df.loc[date_after_tomorrow] = [predictions[2], f'{date_after_tomorrow}', 0, 0]
except:
    copy_df.loc[date_now] = [predictions[0], f'{date_now}', 0]
    copy_df.loc[date_tomorrow] = [predictions[1], f'{date_tomorrow}', 0]
    copy_df.loc[date_after_tomorrow] = [predictions[2], f'{date_after_tomorrow}', 0]

# Result chart
plt.style.use(style='ggplot')
plt.figure(figsize=(16,10))
plt.plot(copy_df['close'][-150:].head(147))
plt.plot(copy_df['predicted_close'][-150:].head(147), linewidth=1, linestyle='dashed')
plt.plot(copy_df['close'][-150:].tail(4))
plt.xlabel('days')
plt.ylabel('price')
plt.legend([f'Actual price for {tkr}', 
            f'Predicted price for {tkr}',
            f'Predicted price for future 3 days'])
plt.show()

# Print the last 60 predictions
print("Last 60 Predictions", predictions)

# Print the last 60 actuals
print("Last 60 Actuals", copy_df['close'].tail(60))

from pandas import Series as ser

# Print the last close
# print("Last Close: ", copy_df['close'][-1])
print("Last Close: ", ser.iloc[-1])

# Print the last prediction
print("Last Prediction: ", predictions[-1])

# Print the last actual
# print("Last Actual: ", copy_df['close'][-1])

# Save Plot to file using name of ticker and timestamp format dt.now().timestamp() to avoid overwriting
# If any of the directories do not exist, create them
if not path.exists('static/images/' + tkr + '/'):
    mkdir('static/images/' + tkr + '/')
if not path.exists('static/images/' + tkr + '/' + itl+ '/'):
    mkdir('static/images/' + tkr + '/' + itl+ '/')
if not path.exists('static/images/' + tkr + '/' + itl+ '/' + prd + '/'):
    mkdir('static/images/' + tkr + '/' + itl+ '/' + prd + '/')

plt.savefig('static/images/' + tkr + '/' + itl+ '/' + prd + '/' + str(d.now().timestamp()) + '.png')

# Print the time taken
# print("Time taken: ", d.datetime.now() - start)

# Initialize the speech engine
engine = pyttsx3.init()

# Say something
engine.say("The program has finished executing.")

# Flush the say() commands and wait for them to finish
engine.runAndWait()