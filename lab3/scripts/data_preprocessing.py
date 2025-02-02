import mysql.connector
import pandas as pd

# MySQL database connection configuration
db_config = {
    'user': 'DSCI560',
    'password': '560560',
    'host': '172.16.161.128',
    'database': 'stock_data'
}

def create_preprocessed_table(cursor):
    """
    Create a new table 'stocks_preprocessed' to store the transformed data.
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stocks_preprocessed (
            symbol VARCHAR(10),
            date DATE,
            open_price DOUBLE,
            high_price DOUBLE,
            low_price DOUBLE,
            close_price DOUBLE,
            volume BIGINT,
            daily_return DOUBLE,
            PRIMARY KEY (symbol, date)
        )
    """)

def fetch_raw_data():
    """
    Fetch the original stocks data from the 'stocks' table.
    Return it as a pandas DataFrame.
    """
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    query = """
        SELECT symbol, date, open_price, high_price, low_price, close_price, volume
        FROM stocks
        ORDER BY symbol, date
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    if not rows:
        print("No data found in 'stocks' table.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df

def handle_missing_values(df):
    """
    Use transform to forward fill and backward fill the missing values for each symbol.
    This avoids multi-index issues caused by groupby.apply.
    """
    # Make sure date is datetime
    df["date"] = pd.to_datetime(df["date"])
    # Sort for consistency
    df.sort_values(by=["symbol", "date"], inplace=True)

    cols_to_fill = ["open_price", "high_price", "low_price", "close_price", "volume"]

    # groupby + transform ensures the result aligns with df's original index
    df[cols_to_fill] = df.groupby("symbol")[cols_to_fill].transform(
        lambda group: group.ffill().bfill()
    )

    return df

def calculate_metrics(df):
    """
    Calculate daily returns using groupby + transform to avoid index misalignment.
    daily_return = (close[t] - close[t-1]) / close[t-1] * 100
    """
    df.sort_values(by=["symbol", "date"], inplace=True)

    # Use transform so the returned series aligns with original df
    df["daily_return"] = df.groupby("symbol")["close_price"].transform(
        lambda x: x.diff() / x.shift() * 100
    )

    return df

def store_preprocessed_data(df):
    """
    Write the preprocessed data into a new table 'stocks_preprocessed'.
    """
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    create_preprocessed_table(cursor)

    insert_query = """
        INSERT INTO stocks_preprocessed
        (symbol, date, open_price, high_price, low_price, close_price, volume, daily_return)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            open_price=VALUES(open_price),
            high_price=VALUES(high_price),
            low_price=VALUES(low_price),
            close_price=VALUES(close_price),
            volume=VALUES(volume),
            daily_return=VALUES(daily_return)
    """

    for _, row in df.iterrows():
        cursor.execute(insert_query, (
            row["symbol"],
            row["date"].date(),
            float(row["open_price"]) if not pd.isna(row["open_price"]) else None,
            float(row["high_price"]) if not pd.isna(row["high_price"]) else None,
            float(row["low_price"])  if not pd.isna(row["low_price"])  else None,
            float(row["close_price"]) if not pd.isna(row["close_price"]) else None,
            int(row["volume"]) if not pd.isna(row["volume"]) else None,
            float(row["daily_return"]) if not pd.isna(row["daily_return"]) else None
        ))

    conn.commit()
    cursor.close()
    conn.close()

def main():
    # 1. 读取原始数据
    df = fetch_raw_data()
    if df.empty:
        return

    # 2. 缺失值处理
    df_filled = handle_missing_values(df)

    # 3. 计算指标 (日收益率)
    df_transformed = calculate_metrics(df_filled)

    # 4. 存储预处理后的数据
    store_preprocessed_data(df_transformed)

    print("Data preprocessing complete. Table 'stocks_preprocessed' updated.")

if __name__ == "__main__":
    main()
