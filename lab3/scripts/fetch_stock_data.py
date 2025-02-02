import mysql.connector
import yfinance as yf
import pandas as pd

# MySQL database connection configuration
db_config = {
    'user': 'DSCI560',
    'password': '560560',
    'host': '172.16.161.128',
    'database': 'stock_data'
}

def fetch_stock_data(symbols, start_date, end_date):
    """Fetch historical stock data along with company name & sector and store it in MySQL."""
    
    # Connect to MySQL
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    for symbol in symbols:
        try:
            print(f"Fetching data for {symbol}...")
            stock = yf.Ticker(symbol)
            stock_info = stock.info  # Get company details
            hist = stock.history(start=start_date, end=end_date)

            # ✅ Check if data is available
            if hist is None or hist.empty:
                print(f"⚠️ No data found for {symbol}. Skipping...")
                continue

            # Extract company details
            company_name = stock_info.get("shortName", symbol)
            sector = stock_info.get("sector", "Unknown")

            # Insert stock data into MySQL
            for index, row in hist.iterrows():
                cursor.execute("""
                    INSERT INTO stocks (symbol, company_name, sector, date, open_price, high_price, low_price, close_price, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                    open_price=VALUES(open_price), high_price=VALUES(high_price), 
                    low_price=VALUES(low_price), close_price=VALUES(close_price), volume=VALUES(volume),
                    company_name=VALUES(company_name), sector=VALUES(sector)
                """, (symbol, company_name, sector, index.date(), 
                      float(row['Open']) if pd.notna(row['Open']) else None, 
                      float(row['High']) if pd.notna(row['High']) else None, 
                      float(row['Low']) if pd.notna(row['Low']) else None, 
                      float(row['Close']) if pd.notna(row['Close']) else None, 
                      int(row['Volume']) if pd.notna(row['Volume']) else None))
            
            print(f"✅ Successfully stored data for {symbol}.")

        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")

    # Commit and close MySQL connection
    conn.commit()
    cursor.close()
    conn.close()
    print("🎉 All stock data has been processed.")

if __name__ == "__main__":
    stock_list = ["AAPL", "GOOGL", "MSFT","ASML"]  # Modify as needed
    fetch_stock_data(stock_list, "2024-01-01", "2024-02-01")
