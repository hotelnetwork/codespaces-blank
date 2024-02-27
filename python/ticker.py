import yfinance as yf
import os

# Intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo

# Set the ticker symbol
ticker = ['TSLA','GOOGL', 'GOOG', 'AMZN', 'MSFT', 'AAPL', 'FB', 'NFLX', 'NVDA']
# Array of Intervals for yFinance
intervals = ['1m','2m','5m','15m','30m','60m','90m','1h','1d','5d','1wk','1mo','3mo']
intv = '1m'
# Set the start and end dates for the data depending on interval using an if.
if intv == '1m':
    # Start date is always 7 days before end date and end date is always today.
    start_date = '2024-02-10'
    end_date = '2024-02-17'
elif intv == '2m':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
elif intv == '5m':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
elif intv == '15m':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
elif intv == '30m':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
elif intv == '60m':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
elif intv == '90m':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
elif intv == '1h':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
elif intv == '1d':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
elif intv == '5d':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
elif intv == '1wk':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
elif intv == '1mo':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
elif intv == '3mo':
    start_date = '2022-01-01'
    end_date = '2022-01-01'
else:
    print('Invalid interval')
    exit()

# Get the historical stock price data for the ticker symbol
data = yf.download(ticker[0], start=start_date, end=end_date, interval=intv, progress=False, auto_adjust=True, prepost=True, threads=True, proxy=None)

# Get number of files in folder and add the number to the end of index to name new file make the folder if necessary
if not os.path.exists(f'tickers/{ticker[0]}'):
    os.makedirs(f'tickers/{ticker[0]}')
# Get list of files in folder
files = os.listdir(f'tickers/{ticker[0]}')
# Format index to always be four digits.
index = len(files) + 1

# Save the data to a file named tickers/ticker/index.csv autoincrementing from 1 up.
data.to_csv(f'tickers/{ticker[0]}/index{index:04}.csv')

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

