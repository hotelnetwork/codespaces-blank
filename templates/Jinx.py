import yfinance as yf
import sqlite3
import mysql.connector
import pandas as pd

class Ticker:
    def __init__(self, ticker):
        self.ticker = ticker

    def download_data(self, period='7d', interval='1m'):
        self.data = yf.download(self.ticker, period=period, interval=interval)
        return self.data

    def save_to_sqlite(self, db_name):
        conn = sqlite3.connect(db_name)
        self.data.to_sql(self.ticker, conn, if_exists='replace')
        conn.close()

    def save_to_mysql(self, host, user, password, db):
        conn = mysql.connector.connect(host=host, user=user, password=password, database=db)
        self.data.to_sql(self.ticker, conn, if_exists='replace')
        conn.close()

    def save_to_csv(self, file_name):
        self.data.to_csv(file_name)

    def save_to_excel(self, file_name):
        self.data.to_excel(file_name)

    def save_to_json(self, file_name):
        self.data.to_json(file_name)

    def save_to_html(self, file_name):
        self.data.to_html(file_name)

    def load_from_sqlite(self, db_name, table_name=None):
        conn = sqlite3.connect(db_name)
        table_name = table_name or self.ticker
        query = f"SELECT * FROM {table_name}"
        self.data = pd.read_sql_query(query, conn)
        conn.close()
        return self.data

    def load_from_mysql(self, host, user, password, db, table_name=None):
        conn = mysql.connector.connect(host=host, user=user, password=password, database=db)
        table_name = table_name or self.ticker
        query = f"SELECT * FROM {table_name}"
        self.data = pd.read_sql_query(query, conn)
        conn.close()
        return self.data
    
    def load_from_csv(self, file_name):
        self.data = pd.read_csv(file_name)
        return self.data

    def load_from_excel(self, file_name):
        self.data = pd.read_excel(file_name)
        return self.data

    def load_from_json(self, file_name):
        self.data = pd.read_json(file_name)
        return self.data
    
    def load_from_html(self, file_name):
        self.data = pd.read_html(file_name)
        return self.data

    def get_data(self):
        return self.data

    def get_ticker(self):
        return self.ticker
    
    def get_open(self):
        return self.data['Open']
    
    def get_high(self):
        return self.data['High']
    
    def get_low(self):
        return self.data['Low']
    
    def get_close(self):
        return self.data['Close']
    
    def get_volume(self):
        return self.data['Volume']
    
    def get_date(self):
        return self.data.index
    
    def get_open_at(self, date):
        return self.data.loc[date, 'Open']
    
    def get_high_at(self, date):
        return self.data.loc[date, 'High']
    
    def get_low_at(self, date):
        return self.data.loc[date, 'Low']
    
    def get_close_at(self, date):
        return self.data.loc[date, 'Close']
    
    def get_volume_at(self, date):
        return self.data.loc[date, 'Volume']
    
    def get_date_at(self, date):
        return self.data.loc[date, 'Date']
    
    def get_open_between(self, start_date, end_date):
        return self.data.loc[start_date:end_date, 'Open']
    
    def get_high_between(self, start_date, end_date):
        return self.data.loc[start_date:end_date, 'High']
    
    def get_low_between(self, start_date, end_date):
        return self.data.loc[start_date:end_date, 'Low']
    
    def get_close_between(self, start_date, end_date):
        return self.data.loc[start_date:end_date, 'Close']
    
    def get_volume_between(self, start_date, end_date):
        return self.data.loc[start_date:end_date, 'Volume']
    
    def get_date_between(self, start_date, end_date):
        return self.data.loc[start_date:end_date, 'Date']
    
    def get_open_after(self, date):
        return self.data.loc[date:, 'Open']
    
    def get_high_after(self, date):
        return self.data.loc[date:, 'High']
    
    def get_low_after(self, date):
        return self.data.loc[date:, 'Low']
    
    def get_close_after(self, date):
        return self.data.loc[date:, 'Close']
    
    def get_volume_after(self, date):
        return self.data.loc[date:, 'Volume']
    
    def get_date_after(self, date):
        return self.data.loc[date:, 'Date']
    
    def get_open_before(self, date):
        return self.data.loc[:date, 'Open']
    
    def get_high_before(self, date):
        return self.data.loc[:date, 'High']
    
    def get_low_before(self, date):
        return self.data.loc[:date, 'Low']
    
    def get_close_before(self, date):
        return self.data.loc[:date, 'Close']
    
    def get_volume_before(self, date):
        return self.data.loc[:date, 'Volume']
    
    def get_date_before(self, date):
        return self.data.loc[:date, 'Date']
    
    def get_open_on(self, date):
        return self.data.loc[date, 'Open']
    
    def get_high_on(self, date):
        return self.data.loc[date, 'High']
    
    def get_low_on(self, date):
        return self.data.loc[date, 'Low']
    
    def get_close_on(self, date):
        return self.data.loc[date, 'Close']
    
    def get_volume_on(self, date):
        return self.data.loc[date, 'Volume']
    
    def get_date_on(self, date):
        return self.data.loc[date, 'Date']
    
    def get_open_in(self, period):
        return self.data.loc[period, 'Open']
    
    def get_high_in(self, period):
        return self.data.loc[period, 'High']
    
    def get_low_in(self, period):
        return self.data.loc[period, 'Low']
    
    def get_close_in(self, period):
        return self.data.loc[period, 'Close']
    
    def get_volume_in(self, period):
        return self.data.loc[period, 'Volume']
    
    def get_date_in(self, period):
        return self.data.loc[period, 'Date']
    
    def get_open_between_in(self, start_date, end_date, period):
        return self.data.loc[start_date:end_date, period]['Open']
    
    def get_high_between_in(self, start_date, end_date, period):
        return self.data.loc[start_date:end_date, period]['High']
    
    def get_low_between_in(self, start_date, end_date, period):
        return self.data.loc[start_date:end_date, period]['Low']
    
    def get_close_between_in(self, start_date, end_date, period):
        return self.data.loc[start_date:end_date, period]['Close']
    
    def get_volume_between_in(self, start_date, end_date, period):
        return self.data.loc[start_date:end_date, period]['Volume']
    
    def get_date_between_in(self, start_date, end_date, period):
        return self.data.loc[start_date:end_date, period]['Date']
    
    def get_open_after_in(self, date, period):
        return self.data.loc[date:, period]['Open']
    
    def get_high_after_in(self, date, period):
        return self.data.loc[date:, period]['High']
    
    def get_low_after_in(self, date, period):
        return self.data.loc[date:, period]['Low']
    
    def get_close_after_in(self, date, period):
        return self.data.loc[date:, period]['Close']
    
    def get_volume_after_in(self, date, period):
        return self.data.loc[date:, period]['Volume']
    
    def update_mysql(self, db_config, table_name, column_values, condition):
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        set_clause = ', '.join(f"{col} = '{val}'" for col, val in column_values.items())
        query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"

        cursor.execute(query)
        conn.commit()

        cursor.close()
        conn.close()

    def update_sqlite(self, db_name, table_name, column_values, condition):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        set_clause = ', '.join(f"{col} = '{val}'" for col, val in column_values.items())
        query = f"UPDATE {table_name} SET {set_clause} WHERE {condition}"

        cursor.execute(query)
        conn.commit()

        cursor.close()
        conn.close()

    def update_csv(self, file_name, column_values, condition):
        data = pd.read_csv(file_name)
        data.loc[data[condition] == True, column_values.keys()] = column_values.values()
        data.to_csv(file_name, index=False)

    def update_excel(self, file_name, column_values, condition):
        data = pd.read_excel(file_name)
        data.loc[data[condition] == True, column_values.keys()] = column_values.values()
        data.to_excel(file_name, index=False)

    def update_json(self, file_name, column_values, condition):
        data = pd.read_json(file_name)
        data.loc[data[condition] == True, column_values.keys()] = column_values.values()
        data.to_json(file_name, index=False)

    def update_html(self, file_name, column_values, condition):
        data = pd.read_html(file_name)
        data.loc[data[condition] == True, column_values.keys()] = column_values.values()
        data.to_html(file_name, index=False)

    def delete_mysql(self, db_config, table_name, condition):
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        query = f"DELETE FROM {table_name} WHERE {condition}"

        cursor.execute(query)
        conn.commit()

        cursor.close()
        conn.close()

    def delete_sqlite(self, db_name, table_name, condition):
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        query = f"DELETE FROM {table_name} WHERE {condition}"

        cursor.execute(query)
        conn.commit()

        cursor.close()
        conn.close()

    def delete_csv(self, file_name, condition):
        data = pd.read_csv(file_name)
        data = data[data[condition] == False]
        data.to_csv(file_name, index=False)

    def delete_excel(self, file_name, condition):
        data = pd.read_excel(file_name)
        data = data[data[condition] == False]
        data.to_excel(file_name, index=False)

    def delete_json(self, file_name, condition):
        data = pd.read_json(file_name)
        data = data[data[condition] == False]
        data.to_json(file_name, index=False)

    def delete_html(self, file_name, condition):
        data = pd.read_html(file_name)
        data = data[data[condition] == False]
        data.to_html(file_name, index=False)

    def __repr__(self):
        return f"{self.data.head()}"

    def __str__(self):
        return f"{self.data.head()}"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index]

    def __setitem__(self, index, value):
        self.data.iloc[index] = value

    def __delitem__(self, index):
        self.data.drop(index, inplace=True)

    def __iter__(self):
        return iter(self.data)
    
    def __next__(self):
        return next(self.data)
    
    def __reversed__(self):
        return reversed(self.data)
    
    def __contains__(self, item):
        return item in self.data
    
    def __add__(self, other):
        return self.data + other
    
    def __sub__(self, other):
        return self.data - other
    
    def __mul__(self, other):
        return self.data * other
    
    def __truediv__(self, other):
        return self.data / other
    
    def __floordiv__(self, other):
        return self.data // other
    
    def __mod__(self, other):
        return self.data % other
    
    def __pow__(self, other):
        return self.data ** other

    def __eq__(self, other):
        return self.data == other
    


# Path: templates/Jinx.py

if __name__ == "__main__":
    ticker = Ticker('AAPL')
    ticker.download_data()
    ticker.save_to_sqlite(db_name=ticker.ticker + '.db')
    ticker.save_to_mysql(table_name=ticker.ticker)
    ticker.save_to_csv(ticker.ticker + '.csv')
    ticker.save_to_excel(ticker.ticker + '.xlsx')
    ticker.save_to_json(ticker.ticker + '.json')
    ticker.save_to_html(ticker.ticker + '.html')