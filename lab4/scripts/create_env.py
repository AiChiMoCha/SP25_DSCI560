import mysql.connector
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np


# connect to MySQL database
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="xxxxxx",
        password="xxxxx",
        database="stock_trading"
    )


#
def create_transaction(conn, ticker, action, date, price, quantity, status):
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO transactions (ticker, action, date, price, quantity, status) 
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (ticker, action, date, float(price), int(quantity), status)
    )
    conn.commit()
    cursor.close()


# get stock data
def fetch_stock_data(ticker, date):
    stock_data = yf.download(ticker, start=date, end=date + timedelta(days=1))
    if stock_data.empty:
        print(f"No data found for {ticker} on {date}.")
        return None
    return stock_data.iloc[0]['Close']


# do transaction
def execute_trade(conn, predicted_signals, initial_cash):
    cash = float(initial_cash)
    portfolio = {}
    total_spent = 0

    for signal in predicted_signals:
        if not signal["signal_valid"]:
            continue

        ticker = signal["ticker"]
        action = signal["action"]
        signal_date = signal["signal_date"].date()
        price = fetch_stock_data(ticker, signal_date)
        signal_strength = signal["signal_strength"]
        if price is None:
            continue

        price = float(price.iloc[0])

        if action == "BUY":
            amount_to_invest = initial_cash * signal_strength
            shares = int(amount_to_invest / price)
            total_cost = shares * price

            total_cost = float(total_cost)

            if total_cost > cash:
                print(f"Skipping {ticker}: Not enough funds.")
                continue
            cash -= total_cost
            total_spent += total_cost

            create_transaction(conn, ticker, "BUY", signal_date, price, shares, "OPEN")
            portfolio[ticker] = {"buy_date": signal_date, "buy_price": price, "shares": shares}

        elif action == "SELL" and ticker in portfolio:
            sell_date = signal_date
            shares = portfolio[ticker]["shares"]
            sell_price = price
            amount_to_receive = shares * sell_price

            create_transaction(conn, ticker, "SELL", sell_date, sell_price, shares, "CLOSED")
            cash += amount_to_receive
            print(f"Sold {shares} shares of {ticker} at {sell_price:.2f} on {sell_date}.")
            del portfolio[ticker]

    return cash, portfolio, total_spent


# calculate return of portfolio
def calculate_portfolio_returns(conn, portfolio, initial_cash):
    total_profit = 0
    total_value = 0

    for ticker, data in portfolio.items():
        buy_price = data["buy_price"]
        shares = data["shares"]
        buy_date = data["buy_date"]

        sell_date = datetime.now()
        sell_price = fetch_stock_data(ticker, sell_date)

        if sell_price is None:
            print(f"No data for {ticker} on {sell_date}. Skipping.")
            continue

        profit = (sell_price - buy_price) * shares
        total_profit += profit

        total_value += sell_price * shares

    final_cash = initial_cash + total_profit
    portfolio_return_percent = (final_cash - initial_cash) / initial_cash * 100

    return total_profit, portfolio_return_percent, total_value


def main():
    # example inputs
    predicted_signals = [
        {'ticker': 'AAPL', 'action': 'BUY', 'signal_date': datetime(2025, 1, 21), 'ref_price': 222.64,
         'signal_valid': True, 'signal_strength': 0.2},
        {'ticker': 'AAPL', 'action': 'SELL', 'signal_date': datetime(2025, 2, 12), 'ref_price': 230.00,
         'signal_valid': True, 'signal_strength': 0.2},
        {'ticker': 'GOOGL', 'action': 'BUY', 'signal_date': datetime(2025, 1, 21), 'ref_price': 198.05,
         'signal_valid': True, 'signal_strength': 0.15},
        {'ticker': 'GOOGL', 'action': 'SELL', 'signal_date': datetime(2025, 2, 13), 'ref_price': 210.50,
         'signal_valid': True, 'signal_strength': 0.15}
    ]
    initial_cash = 10000  # initial funds
    conn = connect_db()

    cash, portfolio, total_spent = execute_trade(conn, predicted_signals, initial_cash)

    total_profit, portfolio_return_percent, total_value = calculate_portfolio_returns(conn, portfolio, initial_cash)

    print(f"\nRemaining cash: ${cash:.2f}")
    print(f"Total spent: ${total_spent:.2f}")
    print(f"Portfolio after trades: {portfolio}")
    print(f"Total profit: ${total_profit:.2f}")
    print(f"Portfolio return: {portfolio_return_percent:.2f}%")
    print(f"Total portfolio value (cash + stocks): ${total_value + cash:.2f}")

    conn.close()

if __name__ == "__main__":
    main()