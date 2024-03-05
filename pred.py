from datetime import datetime as dt
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
import matplotlib.pyplot as plt

class StockPredictor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.model = None

    def download_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)

    def calculate_technical_indicators(self):
        self.data['momentum_60'] = self.data['Close'] - self.data['Close'].shift(60)
        self.data['momentum_20'] = self.data['Close'] - self.data['Close'].shift(20)
        self.data['momentum_10'] = self.data['Close'] - self.data['Close'].shift(10)
        self.data['std_60'] = self.data['Close'].rolling(window=60).std()
        self.data['std_20'] = self.data['Close'].rolling(window=20).std()
        self.data['std_10'] = self.data['Close'].rolling(window=10).std()
        self.data['vwap_60'] = (self.data['Volume'] * self.data['Close']).rolling(window=60).sum() / self.data['Volume'].rolling(window=60).sum()
        self.data['vwap_20'] = (self.data['Volume'] * self.data['Close']).rolling(window=20).sum() / self.data['Volume'].rolling(window=20).sum()
        self.data['vwap_10'] = (self.data['Volume'] * self.data['Close']).rolling(window=10).sum() / self.data['Volume'].rolling(window=10).sum()
        self.data = self.data.dropna()

    def build_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(5)  # Predicting 5 outputs
        ])
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, epochs):
        X = self.data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
        y = self.data[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.model.fit(X, y, epochs=epochs, verbose=False)

    def predict_next_day(self):
        X_last = self.data.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1).iloc[-1]
        next_day_prediction = self.model.predict([X_last])[0]
        return next_day_prediction

# Usage:
predictor = StockPredictor('TSLA', '2010-01-01', dt.now().strftime('%Y-%m-%d'))
predictor.download_data()
predictor.calculate_technical_indicators()
predictor.build_model()
predictor.train_model(1000)
print("Predicted values for TSLA for the next day: ", predictor.predict_next_day())

# Print each of the Calculated Technical Indicator Momentum
print(predictor.data['momentum_60'])
print(predictor.data['momentum_20'])
print(predictor.data['momentum_10'])

# Print each of the Calculated Technical Indicator Standard Deviation
print(predictor.data['std_60'])
print(predictor.data['std_20'])
print(predictor.data['std_10'])

