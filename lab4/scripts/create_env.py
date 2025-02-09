import mysql.connector
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np


# 连接 MySQL 数据库
def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="xxxxxx",  # 替换为实际用户名
        password="xxxxx",  # 替换为实际密码
        database="stock_trading"
    )


# 创建交易记录
def create_transaction(conn, ticker, action, date, price, quantity, status):
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO transactions (ticker, action, date, price, quantity, status) 
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (ticker, action, date, float(price), int(quantity), status)  # 确保 price 和 quantity 是标量
    )
    conn.commit()
    cursor.close()


# 获取股票数据
def fetch_stock_data(ticker, date):
    stock_data = yf.download(ticker, start=date, end=date + timedelta(days=1))
    if stock_data.empty:
        print(f"No data found for {ticker} on {date}.")
        return None
    return stock_data.iloc[0]['Close']


# 执行交易
def execute_trade(conn, predicted_signals, initial_cash):
    cash = float(initial_cash)  # 确保 cash 是标量值
    portfolio = {}  # 用于跟踪当前持仓
    total_spent = 0

    for signal in predicted_signals:
        if not signal["signal_valid"]:
            continue

        ticker = signal["ticker"]
        action = signal["action"]
        signal_date = signal["signal_date"].date()  # 直接使用 signal_date 作为交易日期
        price = fetch_stock_data(ticker, signal_date)
        signal_strength = signal["signal_strength"]
        if price is None:
            continue

        # 确保 price 是单一的数值
        price = float(price.iloc[0])

        if action == "BUY":
            # 根据 signal_strength 分配资金
            amount_to_invest = initial_cash * signal_strength
            shares = int(amount_to_invest / price)  # 计算买入数量，确保 shares 是整数
            total_cost = shares * price

            # 确保 total_cost 是浮动数值
            total_cost = float(total_cost)

            if total_cost > cash:  # 直接比较浮动数值，不使用 Series
                print(f"Skipping {ticker}: Not enough funds.")
                continue
            cash -= total_cost
            total_spent += total_cost

            # 保存买入记录到数据库
            create_transaction(conn, ticker, "BUY", signal_date, price, shares, "OPEN")
            portfolio[ticker] = {"buy_date": signal_date, "buy_price": price, "shares": shares}

        elif action == "SELL" and ticker in portfolio:
            sell_date = signal_date  # 卖出日期与信号日期相同
            shares = portfolio[ticker]["shares"]
            sell_price = price
            amount_to_receive = shares * sell_price

            # 保存卖出记录到数据库
            create_transaction(conn, ticker, "SELL", sell_date, sell_price, shares, "CLOSED")
            cash += amount_to_receive
            print(f"Sold {shares} shares of {ticker} at {sell_price:.2f} on {sell_date}.")
            del portfolio[ticker]  # 卖出后清空该股票的持仓

    return cash, portfolio, total_spent


# 计算投资组合收益
def calculate_portfolio_returns(conn, portfolio, initial_cash):
    total_profit = 0
    total_value = 0  # 计算总的股票市值

    # 计算每个股票的收益
    for ticker, data in portfolio.items():
        buy_price = data["buy_price"]
        shares = data["shares"]
        buy_date = data["buy_date"]

        # 获取卖出价格（假设已经有一个卖出信号）
        sell_date = datetime.now()  # 可以替换为实际的卖出日期
        sell_price = fetch_stock_data(ticker, sell_date)

        if sell_price is None:
            print(f"No data for {ticker} on {sell_date}. Skipping.")
            continue

        # 计算该股票的收益
        profit = (sell_price - buy_price) * shares
        total_profit += profit

        # 计算该股票的总市值
        total_value += sell_price * shares

    # 计算总的百分比收益
    final_cash = initial_cash + total_profit
    portfolio_return_percent = (final_cash - initial_cash) / initial_cash * 100

    return total_profit, portfolio_return_percent, total_value


# 在主程序中计算收益
def main():
    predicted_signals = [
        {'ticker': 'AAPL', 'action': 'BUY', 'signal_date': datetime(2025, 1, 21), 'ref_price': 222.64,
         'signal_valid': True, 'signal_strength': 0.2},  # 20%资金买入
        {'ticker': 'AAPL', 'action': 'SELL', 'signal_date': datetime(2025, 2, 12), 'ref_price': 230.00,
         'signal_valid': True, 'signal_strength': 0.2},  # 20%资金卖出
        {'ticker': 'GOOGL', 'action': 'BUY', 'signal_date': datetime(2025, 1, 21), 'ref_price': 198.05,
         'signal_valid': True, 'signal_strength': 0.15},  # 15%资金买入
        {'ticker': 'GOOGL', 'action': 'SELL', 'signal_date': datetime(2025, 2, 13), 'ref_price': 210.50,
         'signal_valid': True, 'signal_strength': 0.15}  # 15%资金卖出
    ]
    initial_cash = 10000  # 初始资金
    conn = connect_db()

    # 执行交易
    cash, portfolio, total_spent = execute_trade(conn, predicted_signals, initial_cash)

    # 计算收益
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