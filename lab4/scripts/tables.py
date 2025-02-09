import mysql.connector
import pandas as pd
import matplotlib.pyplot as plt

# MySQL 数据库连接配置
db_config = {
    'user': 'DSCI560',
    'password': '560560',
    'host': '172.16.161.128',
    'database': 'stock_data'
}

# 你要绘制的股票代码列表
stock_list = ['DIS', 'ET', 'CAE', 'PANW', 'PYPL', 'VST', 'USAC', 'AM', 'TEM',
              'TSLA', 'NFLX', 'ENLC', 'NGL', 'PTEN', 'FEI', 'T', 'LGF/A',
              'AB', 'NVDA', 'AVGO', 'GOOGL', 'MU', 'BAX', 'ETRN',
              'NGD', 'XLE', 'AGI', 'MAG', 'AMZN', 'V', 'SPY', 'AAPL',
              'HIL', 'HL', 'CRM', 'RBLX', 'MSFT', 'WBD', 'VSAT']

def fetch_stock_data(symbol):
    """ 从 MySQL 读取指定股票的历史数据 """
    conn = mysql.connector.connect(**db_config)
    query = f"""
        SELECT date, close_price FROM stocks 
        WHERE symbol = '{symbol}' 
        ORDER BY date;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def plot_stock(symbol):
    """ 绘制股票时间序列图 """
    df = fetch_stock_data(symbol)
    
    if df.empty:
        print(f"⚠️ No data found for {symbol}")
        return
    
    df['date'] = pd.to_datetime(df['date'])
    
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['close_price'], label=symbol, linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("Close Price (USD)")
    plt.title(f"Stock Price Time Series for {symbol}")
    plt.legend()
    plt.grid(True)
    plt.show()

# 🚀 选择一只股票（比如 AAPL）绘制图表
plot_stock("AAPL")
