import sqlite3
import argparse
import os
import numpy as np
import torch
import sys


class StockDatabase:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute(
            '''CREATE TABLE IF NOT EXISTS stocks (date text, trans text, symbol text, qty real, price real)''')

    def insert_stock(self, date, trans, symbol, qty, price):
        self.cursor.execute("INSERT INTO stocks VALUES (?, ?, ?, ?, ?)",
                            (date, trans, symbol, qty, price))
        self.conn.commit()

    def get_all_stocks(self):
        self.cursor.execute("SELECT * FROM stocks")
        return self.cursor.fetchall()

    def update_stock(self, symbol, price):
        self.cursor.execute(
            "UPDATE stocks SET price = ? WHERE symbol = ?", (price, symbol))
        self.conn.commit()

    def delete_stock(self, symbol):
        self.cursor.execute(
            "DELETE FROM stocks WHERE symbol = ?", (symbol,))
        self.conn.commit()

    def close(self):
        self.conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['insert', 'get_all', 'update', 'delete', 'close'])
    parser.add_argument('--date')
    parser.add_argument('--trans')
    parser.add_argument('--symbol')
    parser.add_argument('--qty', type=float)
    parser.add_argument('--price', type=float)
    args = parser.parse_args()

    db = StockDatabase('example.db')

    if args.action == 'insert':
        db.insert_stock(args.date, args.trans, args.symbol, args.qty, args.price)
    elif args.action == 'get_all':
        print(db.get_all_stocks())
    elif args.action == 'update':
        db.update_stock(args.symbol, args.price)
    elif args.action == 'delete':
        db.delete_stock(args.symbol)
    elif args.action == 'close':
        db.close()


if __name__ == "__main__":
    main()
