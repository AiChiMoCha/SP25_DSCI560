import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import csv
import os

def get_gold_price_data(ticker="GLD"):
    """
    Get minute-by-minute gold price data for the current day from Yahoo Finance
    
    Args:
        ticker (str): The ticker symbol for gold ETF (default is "GLD")
        
    Returns:
        DataFrame with timestamp and price data
    """
    # Calculate timestamps (Yahoo Finance uses unix timestamps in seconds)
    end_time = int(time.time())
    # Start from beginning of the current day
    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    start_time = int(today.timestamp())
    
    # Yahoo Finance API URL for historical data
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    
    # Parameters for minute-by-minute data
    params = {
        "period1": start_time,
        "period2": end_time,
        "interval": "1m",        # 1-minute intervals
        "includePrePost": "true"  # Include pre-market and post-market data
    }
    
    # Set headers to mimic a browser
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        # Check if we got valid data
        if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
            print("No data available or invalid response from Yahoo Finance")
            return None
        
        # Extract timestamp and price data
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        prices = result["indicators"]["quote"][0]["close"]  # Closing prices
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": [datetime.fromtimestamp(ts) for ts in timestamps],
            "price": prices
        })
        
        # Remove any NaN values
        df = df.dropna()
        
        return df
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def save_to_csv(df, filename="gold_prices_minute.csv"):
    """Save the price data to a CSV file"""
    if df is not None and not df.empty:
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    else:
        print("No data to save")

def main():
    print("Fetching minute-by-minute gold price data for today...")
    df = get_gold_price_data()
    
    if df is not None and not df.empty:
        print(f"Retrieved {len(df)} data points")
        print("\nSample data:")
        print(df.head())
        
        # Save to CSV
        save_to_csv(df)
    else:
        print("Failed to retrieve gold price data")

if __name__ == "__main__":
    main()