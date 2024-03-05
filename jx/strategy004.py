from flask import Flask, render_template
from flask_sse import sse
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import json

app = Flask(__name__)
app.config["REDIS_URL"] = "redis://localhost"
app.register_blueprint(sse, url_prefix='/stream')


def predict_stock(symbol):
    # Fetch stock data
    df = yf.download(symbol, start='2020-01-01', end='2022-12-31')

    # Preprocess data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Split data into 70% training and 30% testing
    training_data_len = int(np.ceil(len(scaled_data) * .7))

    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]

    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

        # Convert x_train and y_train to numpy arrays
        x_train, y_train = np.array(x_train), np.array(y_train)

        # Reshape the data into the shape accepted by the LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True,
                  input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(x_train, y_train, batch_size=1, epochs=1)

        # Test data set
        test_data = scaled_data[training_data_len - 60:, :]

        # Create the x_test and y_test data sets
        x_test = []
        y_test = df['Close'][training_data_len:]
        for i in range(60, len(test_data)):
            x_test.append(
                test_data[i-60:i, 0])

            # Convert x_test to a numpy array
            x_test = np.array(x_test)

            # Reshape the data into the shape accepted by the LSTM
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Getting the models predicted price values
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)  # Undo scaling

            # Calculate RMSE
            rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

            # Plot the data
            train = df[:training_data_len]
            valid = df[training_data_len:]
            valid['Predictions'] = predictions

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train.index, y=train['Close'], mode='lines', name='Train'))
            fig.add_trace(go.Scatter(
                x=valid.index, y=valid['Close'], mode='lines', name='Actual Value'))
            fig.add_trace(go.Scatter(
                x=valid.index, y=valid['Predictions'], mode='lines', name='Predicted Value'))
            fig.update_layout(showlegend=True)

            # Convert the figure to JSON
            fig_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

            return fig_json, rmse
