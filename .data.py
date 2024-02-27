from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

app = Flask(__name__)

# Load the data
data = pd.read_csv('stock_data.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data.drop('stock_price', axis=1), data['stock_price'], test_size=0.2)

# Create the model
model = keras.Sequential([
  layers.Dense(units=128, activation='relu', input_shape=(X_train.shape[1],)),
  layers.Dense(units=64, activation='relu'),
  layers.Dense(units=1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100)

# Evaluate the model
model.evaluate(X_test, y_test)

# Create a route for the home page
@app.route('/')
def home():
    return render_template('home.html')

# Create a route for the predict page
@app.route('/predict', methods=['POST'])
def predict():
    # Get the user's input
    ticker = request.form['ticker']
    days = int(request.form['days'])

    # Make a prediction
    prediction = model.predict([[ticker, days]])

    # Return the prediction
    return render_template('predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run()
