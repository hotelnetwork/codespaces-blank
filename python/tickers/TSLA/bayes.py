# Bayes Formula for 2 Sets of Data Points.
import math as m
from scipy.stats import norm
from scipy.stats import binom
from scipy.stats import poisson
from scipy.stats import expon
from scipy.stats import uniform
from numpy import random as rnd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read inputs from file index0002.csv


# Use Open and Close From index0002.csv as Inputs
input_data = pd.read_csv("python/tickers/TSLA/index0002.csv")

# Bayes Formulaimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in the stock data
data = pd.read_csv('stock_data.csv')

# Calculate the sine of the closing price
sine_data = np.sin(data['Close'])

# Plot the sine of the closing price
plt.plot(data['Date'], sine_data)
plt.show()

# Calculate the moving average of the sine of the closing price
sma = pd.Series(sine_data).rolling(window=10).mean()

# Plot the moving average of the sine of the closing price
plt.plot(data['Date'], sma)
plt.show()

# Calculate the difference between the sine of the closing price and the moving average
diff = sine_data - sma

# Plot the difference between the sine of the closing price and the moving average
plt.plot(data['Date'], diff)
plt.show()

# Calculate the cumulative sum of the difference between the sine of the closing price and the moving average
cumsum = diff.cumsum()

# Plot the cumulative sum of the difference between the sine of the closing price and the moving average
plt.plot(data['Date'], cumsum)
plt.show()

# Calculate the maximum value of the cumulative sum
max_cumsum = cumsum.max()

# Calculate the minimum value of the cumulative sum
min_cumsum = cumsum.min()

# Calculate the range of the cumulative sum
range_cumsum = max_cumsum - min_cumsum

# Print the range of the cumulative sum
print(range_cumsum)

# If the range of the cumulative sum is greater than or equal to 1, then the sine squeeze theorem is true
if range_cumsum >= 1:
    print('The sine squeeze theorem is true.')
else:
    print('The sine squeeze theorem is false.')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in the stock data
data = pd.read_csv('stock_data.csv')

# Calculate the sine of the closing price
sine_data = np.sin(data['Close'])

# Plot the sine of the closing price
plt.plot(data['Date'], sine_data)
plt.show()

# Calculate the moving average of the sine of the closing price
sma = pd.Series(sine_data).rolling(window=10).mean()

# Plot the moving average of the sine of the closing price
plt.plot(data['Date'], sma)
plt.show()

# Calculate the difference between the sine of the closing price and the moving average
diff = sine_data - sma

# Plot the difference between the sine of the closing price and the moving average
plt.plot(data['Date'], diff)
plt.show()

# Calculate the cumulative sum of the difference between the sine of the closing price and the moving average
cumsum = diff.cumsum()

# Plot the cumulative sum of the difference between the sine of the closing price and the moving average
plt.plot(data['Date'], cumsum)
plt.show()

# Calculate the maximum value of the cumulative sum
max_cumsum = cumsum.max()

# Calculate the minimum value of the cumulative sum
min_cumsum = cumsum.min()

# Calculate the range of the cumulative sum
range_cumsum = max_cumsum - min_cumsum

# Print the range of the cumulative sum
print(range_cumsum)

# If the range of the cumulative sum is greater than or equal to 1, then the sine squeeze theorem is true
if range_cumsum >= 1:
    print('The sine squeeze theorem is true.')
else:
    print('The sine squeeze theorem is false.')

w = pd.read_csv("python/tickers/TSLA/index0002.csv", index_col=0)



def bayes(prior, likelihood):
    # Open File
    # Read Inputs from file index0002.csv
    # Use Open and Close From index0002.csv as Inputs
    w = pd.read_csv("python/tickers/TSLA/index0002.csv")
    # Bayes Formula for 2 Sets of Data Points.

    return prior * likelihood / (prior * likelihood + (1 - prior) * (1 - likelihood))

# Bayes on input_data

def output_bayes():
    print("Bayes Formula for 2 Sets of Data Points.")
    print("Bayes Formula: P(A|B) = P(B|A) * P(A) / P(B)")
    print("P(A|B) = Probability of A given B")
    print("P(B|A) = Probability of B given A")
    print("P(A) = Probability of A")
    print("P(B) = Probability of B")
    print("P(A|B) = P(B|A) * P(A) / P(B)")

# Main
if __name__ == "__main__":
    output_bayes()

# Bayes on input_data
    
# Function to calculate bayes on input_data
    
