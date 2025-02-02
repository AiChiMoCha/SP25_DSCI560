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

def create_portfolio():
    """Create a new portfolio with user input."""
    portfolio_name = input("Enter portfolio name: ").strip()

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    try:
        cursor.execute("INSERT INTO portfolios (portfolio_name) VALUES (%s)", (portfolio_name,))
        conn.commit()
        print(f"✅ Portfolio '{portfolio_name}' created successfully.")
    except mysql.connector.IntegrityError:
        print(f"⚠️ Portfolio '{portfolio_name}' already exists.")

    cursor.close()
    conn.close()

def add_stock_to_portfolio():
    """Add a stock to a portfolio with user input and validation."""
    portfolio_name = input("Enter portfolio name: ").strip()
    stock_symbol = input("Enter stock symbol: ").strip().upper()

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Validate stock exists in Yahoo Finance
    stock = yf.Ticker(stock_symbol)
    try:
        stock_name = stock.info.get("shortName", None)
    except:
        stock_name = None

    if not stock_name:
        print(f"❌ Invalid stock symbol: {stock_symbol}. Cannot add to portfolio.")
        cursor.close()
        conn.close()
        return

    # Check if portfolio exists
    cursor.execute("SELECT id FROM portfolios WHERE portfolio_name = %s", (portfolio_name,))
    result = cursor.fetchone()
    
    if not result:
        print(f"⚠️ Portfolio '{portfolio_name}' does not exist. Please create it first.")
        cursor.close()
        conn.close()
        return

    portfolio_id = result[0]

    # Check if stock is already in portfolio
    cursor.execute(
        "SELECT id FROM portfolio_stocks WHERE portfolio_id = %s AND stock_symbol = %s",
        (portfolio_id, stock_symbol)
    )
    if cursor.fetchone():
        print(f"⚠️ Stock '{stock_symbol}' is already in portfolio '{portfolio_name}'.")
        cursor.close()
        conn.close()
        return

    # Add stock to portfolio
    cursor.execute(
        "INSERT INTO portfolio_stocks (portfolio_id, stock_symbol) VALUES (%s, %s)",
        (portfolio_id, stock_symbol)
    )
    conn.commit()
    print(f"✅ Stock '{stock_symbol}' added to portfolio '{portfolio_name}'.")

    cursor.close()
    conn.close()

def remove_stock_from_portfolio():
    """Remove a stock from a portfolio."""
    portfolio_name = input("Enter portfolio name: ").strip()
    stock_symbol = input("Enter stock symbol to remove: ").strip().upper()

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("""
        DELETE ps FROM portfolio_stocks ps
        JOIN portfolios p ON ps.portfolio_id = p.id
        WHERE p.portfolio_name = %s AND ps.stock_symbol = %s
    """, (portfolio_name, stock_symbol))
    
    if cursor.rowcount > 0:
        conn.commit()
        print(f"✅ Stock '{stock_symbol}' removed from portfolio '{portfolio_name}'.")
    else:
        print(f"⚠️ Stock '{stock_symbol}' was not found in portfolio '{portfolio_name}'.")

    cursor.close()
    conn.close()

def list_portfolios():
    """Display all portfolios along with their stocks."""
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    cursor.execute("SELECT id, portfolio_name, created_at FROM portfolios")
    portfolios = cursor.fetchall()

    if not portfolios:
        print("⚠️ No portfolios found.")
        cursor.close()
        conn.close()
        return

    for portfolio_id, name, created_at in portfolios:
        print(f"\n📌 Portfolio: {name} (Created: {created_at})")
        cursor.execute("SELECT stock_symbol FROM portfolio_stocks WHERE portfolio_id = %s", (portfolio_id,))
        stocks = cursor.fetchall()

        if stocks:
            print("   Stocks:", ", ".join(stock[0] for stock in stocks))
        else:
            print("   No stocks in this portfolio.")

    cursor.close()
    conn.close()

def fetch_portfolio_data():
    """Fetch stock price data for all stocks in a portfolio within a given date range.
       If both start_date and end_date are empty, fetch all available data.
    """
    portfolio_name = input("Enter portfolio name: ").strip()
    start_date = input("Enter start date (YYYY-MM-DD): ").strip()
    end_date = input("Enter end date (YYYY-MM-DD): ").strip()

    if not portfolio_name:
        print("⚠️ Portfolio name cannot be empty.")
        return

    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # Check if portfolio exists
    cursor.execute("SELECT id FROM portfolios WHERE portfolio_name = %s", (portfolio_name,))
    result = cursor.fetchone()
    if not result:
        print(f"⚠️ Portfolio '{portfolio_name}' does not exist.")
        cursor.close()
        conn.close()
        return
    portfolio_id = result[0]

    # Get stock list
    cursor.execute("SELECT stock_symbol FROM portfolio_stocks WHERE portfolio_id = %s", (portfolio_id,))
    stocks = [row[0] for row in cursor.fetchall()]

    if not stocks:
        print(f"⚠️ No stocks in portfolio '{portfolio_name}'.")
        cursor.close()
        conn.close()
        return

    def safe_float(val):
        if pd.isna(val):
            return None
        return float(val)

    print(f"\nFetching stock data for portfolio '{portfolio_name}' ({start_date or 'ALL'} - {end_date or 'ALL'})...")

    for symbol in stocks:
        # If both start_date and end_date are empty, fetch all data
        if not start_date and not end_date:
            stock_data = yf.download(symbol, period="max")
        else:
            s_date = start_date if start_date else None
            e_date = end_date if end_date else None
            stock_data = yf.download(symbol, start=s_date, end=e_date)

        # Print columns for debugging
        print(f"\nDownloaded columns for {symbol}:", stock_data.columns)
        if not stock_data.empty:
            print(stock_data.head(5))

        # If columns are multi-indexed, drop one level
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(level=1)
            print("Flattened columns:", stock_data.columns)

        if stock_data.empty:
            print(f"⚠️ No data found for {symbol}. Skipping...")
            continue

        # Ensure required columns exist
        needed_cols = ["Open", "High", "Low", "Close", "Volume"]
        for needed in needed_cols:
            if needed not in stock_data.columns:
                print(f"⚠️ Column '{needed}' missing for {symbol}. Skipping...")
                break
        else:
            # Only proceed if none of the needed columns are missing
            for index, row in stock_data.iterrows():
                open_price = safe_float(row["Open"])
                high_price = safe_float(row["High"])
                low_price = safe_float(row["Low"])
                close_price = safe_float(row["Close"])
                volume = None
                if not pd.isna(row["Volume"]):
                    volume = int(row["Volume"])

                cursor.execute("""
                    INSERT INTO stocks (symbol, date, open_price, high_price, low_price, close_price, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                        open_price=VALUES(open_price),
                        high_price=VALUES(high_price),
                        low_price=VALUES(low_price),
                        close_price=VALUES(close_price),
                        volume=VALUES(volume)
                """, (
                    symbol,
                    index.date(),
                    open_price,
                    high_price,
                    low_price,
                    close_price,
                    volume
                ))

            print(f"✅ Data for {symbol} fetched successfully.")

    conn.commit()
    cursor.close()
    conn.close()
    print("🎉 All stock data has been processed.")

def main():
    while True:
        print("\n📌 Portfolio Management System")
        print("1️⃣ Create Portfolio")
        print("2️⃣ Add Stock to Portfolio")
        print("3️⃣ Remove Stock from Portfolio")
        print("4️⃣ List Portfolios")
        print("5️⃣ Fetch Portfolio Data")
        print("0️⃣ Exit")
        
        choice = input("Select an option: ").strip()

        if choice == "1":
            create_portfolio()
        elif choice == "2":
            add_stock_to_portfolio()
        elif choice == "3":
            remove_stock_from_portfolio()
        elif choice == "4":
            list_portfolios()
        elif choice == "5":
            fetch_portfolio_data()
        elif choice == "0":
            print("👋 Exiting Portfolio Management System.")
            break
        else:
            print("❌ Invalid option. Please try again.")

if __name__ == "__main__":
    main()
