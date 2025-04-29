import os
from dotenv import load_dotenv
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import logging
import matplotlib.pyplot as plt
import os.path

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("gold_data.log"), logging.StreamHandler()]
)
logger = logging.getLogger("GoldDataModule")

class GoldDataCollector:
    """Module for collecting real-time gold price data and related information."""
    
    def __init__(self, history_file="gold_history.json"):
        """
        Initialize the data collector, read API keys from environment variables.
        
        Parameters:
            history_file (str): Path to the history data file
        """
        self.api_keys = {
            'alpha_vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'fred': os.getenv('FRED_API_KEY'),
            'news_api': os.getenv('NEWS_API_KEY')
        }
        self.data_cache = {}
        self.history_file = history_file
        self.history_data = self.load_history()
        logger.info("Gold data collector initialized")
    
    def load_history(self):
        """Load history data file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                logger.info(f"Loaded history data with {len(history)} records")
                return history
            except Exception as e:
                logger.error(f"Error loading history data: {str(e)}")
                return []
        else:
            logger.info("History data file not found, will create new file")
            return []
    
    def save_to_history(self):
        """Save current data to history"""
        # Add timestamp
        self.data_cache["record_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add current data to history
        self.history_data.append(self.data_cache.copy())
        
        # Keep only the last 30 days of data
        if len(self.history_data) > 30:
            self.history_data = self.history_data[-30:]
        
        # Save to file
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Data saved to history, now with {len(self.history_data)} records")
            return True
        except Exception as e:
            logger.error(f"Error saving history data: {str(e)}")
            return False
    
    def get_spot_gold_price(self):
        """Get current gold spot price (XAU/USD)."""
        try:
            # Use Yahoo Finance API to get gold price
            url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers)
            data = response.json()
            
            if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
                result = data["chart"]["result"][0]
                
                # Get latest price
                latest_price = result["meta"]["regularMarketPrice"]
                timestamp = result["meta"]["regularMarketTime"]
                
                # Convert timestamp to datetime
                dt = datetime.fromtimestamp(timestamp)
                
                result = {
                    "price": latest_price,
                    "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Yahoo Finance"
                }
                
                self.data_cache["spot_gold"] = result
                logger.info(f"Got gold spot price: ${latest_price}")
                return result
            else:
                logger.error(f"Error getting gold spot price: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Error while getting gold spot price: {str(e)}")
            return None
    
    def get_futures_data(self, contract="GC=F"):
        """
        Get gold futures data.
        
        Parameters:
            contract (str): Futures contract code (default: "GC=F" for COMEX Gold)
        """
        try:
            # Use Yahoo Finance API
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{contract}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers)
            data = response.json()
            
            if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
                result = data["chart"]["result"][0]
                
                # Get latest price
                latest_price = result["meta"]["regularMarketPrice"]
                timestamp = result["meta"]["regularMarketTime"]
                
                # Convert timestamp to datetime
                dt = datetime.fromtimestamp(timestamp)
                
                futures_data = {
                    "contract": contract,
                    "price": latest_price,
                    "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Yahoo Finance"
                }
                
                self.data_cache[f"futures_{contract}"] = futures_data
                logger.info(f"Got {contract} futures data: ${latest_price}")
                return futures_data
            else:
                logger.error(f"Error getting futures data: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting futures data: {str(e)}")
            return None
    
    def get_economic_indicators(self, indicator="DXY"):
        """
        Get economic indicators that affect gold prices.
        
        Parameters:
            indicator (str): Indicator code
        """
        try:
            # Get DXY (US Dollar Index)
            if indicator == "DXY":
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
                
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                
                response = requests.get(url, headers=headers)
                data = response.json()
                
                if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
                    result = data["chart"]["result"][0]
                    
                    # Get latest value
                    latest_value = result["meta"]["regularMarketPrice"]
                    timestamp = result["meta"]["regularMarketTime"]
                    
                    # Convert timestamp to datetime
                    dt = datetime.fromtimestamp(timestamp)
                    
                    indicator_data = {
                        "indicator": "US Dollar Index (DXY)",
                        "value": latest_value,
                        "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "source": "Yahoo Finance"
                    }
                    
                    self.data_cache[f"indicator_{indicator}"] = indicator_data
                    logger.info(f"Got {indicator} data: {latest_value}")
                    return indicator_data
            
            # Other indicators need different APIs or data sources
            logger.error(f"Indicator {indicator} not yet implemented")
            return None
                
        except Exception as e:
            logger.error(f"Error getting {indicator} data: {str(e)}")
            return None
    
    def get_gold_etf_holdings(self):
        """Get gold ETF holdings data, using Yahoo Finance for GLD price and volume."""
        try:
            # Use Yahoo Finance API to get GLD data
            url = "https://query1.finance.yahoo.com/v8/finance/chart/GLD"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers)
            data = response.json()
            
            if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
                result = data["chart"]["result"][0]
                
                # Get latest price and volume
                latest_price = result["meta"]["regularMarketPrice"]
                timestamp = result["meta"]["regularMarketTime"]
                
                # Try to get volume
                volume = None
                if "indicators" in result and "quote" in result["indicators"] and result["indicators"]["quote"]:
                    quotes = result["indicators"]["quote"][0]
                    if "volume" in quotes and quotes["volume"]:
                        volume = quotes["volume"][-1]  # Get latest volume
                
                # Convert timestamp to datetime
                dt = datetime.fromtimestamp(timestamp)
                
                # Note: We can't directly get ETF holdings, only price and volume
                # Holdings typically need to be retrieved from the ETF provider's website
                etf_data = {
                    "etf": "SPDR Gold Trust (GLD)",
                    "price": latest_price,
                    "volume": volume,
                    "date": dt.strftime("%Y-%m-%d"),
                    "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Yahoo Finance"
                }
                
                self.data_cache["gold_etf"] = etf_data
                logger.info(f"Got gold ETF data: GLD price ${latest_price}")
                return etf_data
            else:
                logger.error(f"Error getting ETF data: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting gold ETF data: {str(e)}")
            return None
    
    def get_technical_indicators(self, timeframe="1d", symbol="GLD", limit=100):
        """
        Calculate technical indicators for gold prices.
        
        Parameters:
            timeframe (str): Time interval between data points
            symbol (str): Stock symbol, default is GLD (SPDR Gold Trust)
            limit (int): Number of historical data points to retrieve
        """
        try:
            # Use Yahoo Finance API to get historical data
            # Map timeframe to Yahoo Finance interval
            interval_map = {
                "1h": "1h",
                "1d": "1d",
                "1w": "1wk"
            }
            
            yf_interval = interval_map.get(timeframe, "1d")
            
            # Calculate range
            range_map = {
                "1h": "5d",   # 5 days of hourly data
                "1d": "1y",   # 1 year of daily data
                "1w": "5y"    # 5 years of weekly data
            }
            
            yf_range = range_map.get(timeframe, "1y")
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={yf_interval}&range={yf_range}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers)
            data = response.json()
            
            if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
                result = data["chart"]["result"][0]
                
                # Extract timestamps and price data
                timestamps = result["timestamp"]
                
                # Extract OHLC and volume data
                quote = result["indicators"]["quote"][0]
                opens = quote.get("open", [])
                highs = quote.get("high", [])
                lows = quote.get("low", [])
                closes = quote.get("close", [])
                volumes = quote.get("volume", [])
                
                # Create DataFrame
                df = pd.DataFrame({
                    "timestamp": timestamps,
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes
                })
                
                # Handle potential NaN values
                df = df.dropna()
                
                # Convert timestamps to datetimes
                df["datetime"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x))
                df.set_index("datetime", inplace=True)
                
                # Calculate indicators
                # 1. Moving averages
                df['SMA_20'] = df['close'].rolling(window=20).mean()
                df['SMA_50'] = df['close'].rolling(window=50).mean()
                df['SMA_200'] = df['close'].rolling(window=200).mean()
                
                # 2. RSI (14 periods)
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                df['RSI'] = 100 - (100 / (1 + rs))
                
                # 3. MACD
                df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
                df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = df['EMA_12'] - df['EMA_26']
                df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                
                # Get latest values
                latest = df.iloc[-1].to_dict()
                prev = df.iloc[-2].to_dict() if len(df) > 1 else {}
                
                tech_indicators = {
                    "price": latest['close'],
                    "change": latest['close'] - prev.get('close', latest['close']),
                    "change_pct": ((latest['close'] - prev.get('close', latest['close'])) / prev.get('close', latest['close'])) * 100 if prev else 0,
                    "sma_20": latest['SMA_20'],
                    "sma_50": latest['SMA_50'],
                    "sma_200": latest['SMA_200'],
                    "rsi": latest['RSI'],
                    "macd": latest['MACD'],
                    "macd_signal": latest['MACD_signal'],
                    "volume": latest['volume'],
                    "timeframe": timeframe,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Yahoo Finance + Calculation"
                }
                
                # Add trend signals
                tech_indicators["trend_sma"] = "Bullish" if latest['close'] > latest['SMA_50'] else "Bearish"
                tech_indicators["trend_macd"] = "Bullish" if latest['MACD'] > latest['MACD_signal'] else "Bearish"
                tech_indicators["trend_rsi"] = "Overbought" if latest['RSI'] > 70 else ("Oversold" if latest['RSI'] < 30 else "Neutral")
                
                self.data_cache[f"technical_{timeframe}"] = tech_indicators
                logger.info(f"Calculated technical indicators for {timeframe} timeframe")
                return tech_indicators
            else:
                logger.error(f"Error getting time series data: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return None
    
    # NEW FUNCTION: Get historical gold price data and save to CSV
    def get_historical_gold_data(self, symbol="GC=F", period1=None, period2=None, interval="1d", csv_filename="gold_historical_data.csv"):
        """
        Get historical gold price data from Yahoo Finance and save to CSV.
        
        Parameters:
            symbol (str): Symbol to get data for. Default is "GC=F" for gold futures
                          Other options include "GLD" for the SPDR Gold Trust ETF
            period1 (str or datetime): Start date in YYYY-MM-DD format or as datetime object. Default is 1 year ago
            period2 (str or datetime): End date in YYYY-MM-DD format or as datetime object. Default is today
            interval (str): Data interval. Options: "1d", "1wk", "1mo" for daily, weekly, monthly
            csv_filename (str): Name of CSV file to save data to
            
        Returns:
            DataFrame: Historical price data or None if there was an error
        """
        try:
            # Set default dates if not provided
            if period2 is None:
                period2 = datetime.now()
            elif isinstance(period2, str):
                period2 = datetime.strptime(period2, "%Y-%m-%d")
                
            if period1 is None:
                period1 = period2 - timedelta(days=365)  # Default to 1 year of data
            elif isinstance(period1, str):
                period1 = datetime.strptime(period1, "%Y-%m-%d")
            
            # Convert dates to Unix timestamps (seconds since epoch)
            period1_unix = int(period1.timestamp())
            period2_unix = int(period2.timestamp())
            
            # Construct URL for Yahoo Finance API
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                "period1": period1_unix,
                "period2": period2_unix,
                "interval": interval,
                "includePrePost": "false",
                "events": "div,split"
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            logger.info(f"Getting historical data for {symbol} from {period1.strftime('%Y-%m-%d')} to {period2.strftime('%Y-%m-%d')}")
            response = requests.get(url, params=params, headers=headers)
            data = response.json()
            
            if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
                result = data["chart"]["result"][0]
                
                # Extract timestamps and price data
                timestamps = result["timestamp"]
                
                # Extract OHLC data
                quote = result["indicators"]["quote"][0]
                opens = quote.get("open", [])
                highs = quote.get("high", [])
                lows = quote.get("low", [])
                closes = quote.get("close", [])
                volumes = quote.get("volume", [])
                
                # Create DataFrame
                df = pd.DataFrame({
                    "timestamp": timestamps,
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes
                })
                
                # Handle potential NaN values
                df = df.dropna()
                
                # Convert timestamps to dates
                df["date"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
                
                # Reorder columns to put date first
                df = df[["date", "open", "high", "low", "close", "volume", "timestamp"]]
                
                # Save to CSV
                df.to_csv(csv_filename, index=False)
                logger.info(f"Historical data saved to {csv_filename} with {len(df)} records")
                
                return df
            else:
                logger.error(f"Error getting historical data: {data}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching historical gold data: {str(e)}")
            return None
    
    # Function to get multiple timeframes of historical data and save to CSV
    def get_multiple_timeframe_history(self, symbol="GC=F", save_dir="."):
        """
        Get historical gold price data for multiple timeframes and save to separate CSV files.
        
        Parameters:
            symbol (str): Symbol to get data for. Default is "GC=F" for gold futures
            save_dir (str): Directory to save CSV files
            
        Returns:
            dict: Dictionary containing DataFrames for each timeframe
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        timeframes = {
            "1y_daily": {"period1": datetime.now() - timedelta(days=365), "period2": datetime.now(), "interval": "1d"},
            "5y_weekly": {"period1": datetime.now() - timedelta(days=5*365), "period2": datetime.now(), "interval": "1wk"},
            "10y_monthly": {"period1": datetime.now() - timedelta(days=10*365), "period2": datetime.now(), "interval": "1mo"}
        }
        
        results = {}
        
        for tf_name, tf_params in timeframes.items():
            filename = os.path.join(save_dir, f"gold_{symbol}_{tf_name}.csv")
            logger.info(f"Getting {tf_name} data for {symbol}")
            
            df = self.get_historical_gold_data(
                symbol=symbol,
                period1=tf_params["period1"],
                period2=tf_params["period2"],
                interval=tf_params["interval"],
                csv_filename=filename
            )
            
            results[tf_name] = df
            
        return results
    
    # Add a function to get historical data for multiple gold-related symbols
    def get_gold_market_data(self, save_dir="gold_data"):
        """
        Get historical data for multiple gold-related symbols and save to CSV files.
        
        Parameters:
            save_dir (str): Directory to save CSV files
            
        Returns:
            dict: Dictionary containing DataFrames for each symbol
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        symbols = {
            "GC=F": "Gold Futures",
            "GLD": "SPDR Gold Trust ETF",
            "IAU": "iShares Gold Trust",
            "SGOL": "Aberdeen Standard Physical Gold Shares ETF",
            "DGP": "DB Gold Double Long ETN",
            "GDX": "VanEck Gold Miners ETF",
            "GDXJ": "VanEck Junior Gold Miners ETF"
        }
        
        today = datetime.now()
        one_year_ago = today - timedelta(days=365)
        
        results = {}
        
        for symbol, description in symbols.items():
            filename = os.path.join(save_dir, f"{symbol}_daily_1y.csv")
            logger.info(f"Getting 1-year daily data for {symbol} ({description})")
            
            df = self.get_historical_gold_data(
                symbol=symbol,
                period1=one_year_ago,
                period2=today,
                interval="1d",
                csv_filename=filename
            )
            
            if df is not None:
                # Add a column for the symbol description
                df["symbol"] = symbol
                df["description"] = description
                results[symbol] = df
                
        return results
    
    def get_all_data(self):
        """Collect all available data at once."""
        data = {}
        
        # Core price data
        data["spot_gold"] = self.get_spot_gold_price()
        data["futures"] = self.get_futures_data()
        
        # Economic data
        data["dxy"] = self.get_economic_indicators("DXY")
        
        # Market data
        data["etf_holdings"] = self.get_gold_etf_holdings()
        data["technical"] = self.get_technical_indicators(symbol="GLD")
        
        # Add timestamp
        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Update cache
        self.data_cache.update(data)
        
        return data
    
    def compare_with_history(self, days=7):
        """
        Compare current data with historical data
        
        Parameters:
            days (int): Number of historical days to compare with
        
        Returns:
            dict: Dictionary containing comparison results
        """
        if not self.history_data:
            return {"error": "No historical data available for comparison"}
        
        # Ensure we have current data
        if not self.data_cache:
            self.get_all_data()
        
        # Get recent history for specified days
        recent_history = self.history_data[-days:] if len(self.history_data) > days else self.history_data
        
        comparison = {}
        
        # Compare gold spot prices
        if "spot_gold" in self.data_cache:
            current_price = self.data_cache["spot_gold"].get("price")
            if current_price:
                price_history = []
                for record in recent_history:
                    if "spot_gold" in record and "price" in record["spot_gold"]:
                        price_history.append({
                            "price": record["spot_gold"]["price"],
                            "timestamp": record.get("record_time", record["spot_gold"].get("timestamp")),
                        })
                
                if price_history:
                    # Calculate changes
                    first_price = price_history[0]["price"]
                    price_change = current_price - first_price
                    price_change_pct = (price_change / first_price) * 100 if first_price else 0
                    
                    comparison["spot_gold"] = {
                        "current": current_price,
                        "history": price_history,
                        "change": price_change,
                        "change_pct": price_change_pct,
                        "trend": "Rising" if price_change > 0 else "Falling" if price_change < 0 else "Stable"
                    }
        
        # Compare gold ETF prices
        if "gold_etf" in self.data_cache:
            current_etf_price = self.data_cache["gold_etf"].get("price")
            if current_etf_price:
                etf_price_history = []
                for record in recent_history:
                    if "gold_etf" in record and "price" in record["gold_etf"]:
                        etf_price_history.append({
                            "price": record["gold_etf"]["price"],
                            "timestamp": record.get("record_time", record["gold_etf"].get("timestamp")),
                        })
                
                if etf_price_history:
                    # Calculate changes
                    first_etf_price = etf_price_history[0]["price"]
                    etf_price_change = current_etf_price - first_etf_price
                    etf_price_change_pct = (etf_price_change / first_etf_price) * 100 if first_etf_price else 0
                    
                    comparison["gold_etf"] = {
                        "current": current_etf_price,
                        "history": etf_price_history,
                        "change": etf_price_change,
                        "change_pct": etf_price_change_pct,
                        "trend": "Rising" if etf_price_change > 0 else "Falling" if etf_price_change < 0 else "Stable"
                    }
        
        # Compare US Dollar Index
        if "indicator_DXY" in self.data_cache:
            current_dxy = self.data_cache["indicator_DXY"].get("value")
            if current_dxy:
                dxy_history = []
                for record in recent_history:
                    if "indicator_DXY" in record and "value" in record["indicator_DXY"]:
                        dxy_history.append({
                            "value": record["indicator_DXY"]["value"],
                            "timestamp": record.get("record_time", record["indicator_DXY"].get("timestamp")),
                        })
                
                if dxy_history:
                    # Calculate changes
                    first_dxy = dxy_history[0]["value"]
                    dxy_change = current_dxy - first_dxy
                    dxy_change_pct = (dxy_change / first_dxy) * 100 if first_dxy else 0
                    
                    comparison["dxy"] = {
                        "current": current_dxy,
                        "history": dxy_history,
                        "change": dxy_change,
                        "change_pct": dxy_change_pct,
                        "trend": "Rising" if dxy_change > 0 else "Falling" if dxy_change < 0 else "Stable"
                    }
        
        return comparison
    
    def generate_price_chart(self, days=30, save_path="gold_price_chart.png"):
        """
        Generate a chart of gold price history
        
        Parameters:
            days (int): Number of historical days to show
            save_path (str): Path to save the chart
        
        Returns:
            bool: Whether the chart was successfully generated
        """
        try:
            if not self.history_data:
                logger.error("Not enough historical data to generate chart")
                return False
                
            # Ensure we have current data
            if not self.data_cache:
                self.get_all_data()
            
            # Get recent history for specified days
            available_days = min(days, len(self.history_data))
            recent_history = self.history_data[-available_days:]
            
            # Prepare data
            dates = []
            spot_prices = []
            etf_prices = []
            dxy_values = []
            
            # Add historical data
            for record in recent_history:
                # Get date
                record_time = record.get("record_time")
                if not record_time and "spot_gold" in record:
                    record_time = record["spot_gold"].get("timestamp")
                
                if record_time:
                    dates.append(record_time)
                    
                    # Gold spot price
                    if "spot_gold" in record and "price" in record["spot_gold"]:
                        spot_prices.append(record["spot_gold"]["price"])
                    else:
                        spot_prices.append(None)
                    
                    # GLD ETF price
                    if "gold_etf" in record and "price" in record["gold_etf"]:
                        etf_prices.append(record["gold_etf"]["price"])
                    else:
                        etf_prices.append(None)
                    
                    # US Dollar Index
                    if "indicator_DXY" in record and "value" in record["indicator_DXY"]:
                        dxy_values.append(record["indicator_DXY"]["value"])
                    else:
                        dxy_values.append(None)
            
            # Add current data
            if self.data_cache:
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                dates.append(current_time)
                
                # Gold spot price
                if "spot_gold" in self.data_cache and "price" in self.data_cache["spot_gold"]:
                    spot_prices.append(self.data_cache["spot_gold"]["price"])
                else:
                    spot_prices.append(None)
                
                # GLD ETF price
                if "gold_etf" in self.data_cache and "price" in self.data_cache["gold_etf"]:
                    etf_prices.append(self.data_cache["gold_etf"]["price"])
                else:
                    etf_prices.append(None)
                
                # US Dollar Index
                if "indicator_DXY" in self.data_cache and "value" in self.data_cache["indicator_DXY"]:
                    dxy_values.append(self.data_cache["indicator_DXY"]["value"])
                else:
                    dxy_values.append(None)
            
            # Create chart
            plt.figure(figsize=(12, 8))
            
            # Create first Y-axis (gold price)
            ax1 = plt.gca()
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Gold Price (USD)', color='gold')
            
            # Plot gold spot price
            if any(spot_prices):
                ax1.plot(dates, spot_prices, 'o-', color='gold', label='Gold Spot Price')
            
            # Plot GLD ETF price
            if any(etf_prices):
                ax1.plot(dates, etf_prices, 's-', color='orange', label='GLD ETF Price')
            
            ax1.tick_params(axis='y', labelcolor='gold')
            
            # Create second Y-axis (US Dollar Index)
            if any(dxy_values):
                ax2 = ax1.twinx()
                ax2.set_ylabel('US Dollar Index', color='blue')
                ax2.plot(dates, dxy_values, '^-', color='blue', label='US Dollar Index')
                ax2.tick_params(axis='y', labelcolor='blue')
            
            # Add title and legend
            plt.title('Gold Price and US Dollar Index Trends')
            
            # Merge legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            if any(dxy_values):
                lines2, labels2 = ax2.get_legend_handles_labels()
                plt.legend(lines1 + lines2, labels1 + labels2, loc='best')
            else:
                plt.legend(loc='best')
            
            # Rotate date labels
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save chart
            plt.savefig(save_path)
            logger.info(f"Price chart saved to {save_path}")
            
            # Close figure
            plt.close()
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating price chart: {str(e)}")
            return False
    
    def export_data_to_json(self, filename="gold_data.json"):
        """Export current data to a JSON file."""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.data_cache, f, indent=4, ensure_ascii=False)
            logger.info(f"Data exported to {filename}")
            return True
        except Exception as e:
            logger.error(f"Error exporting data: {str(e)}")
            return False
            
    # New function to create and save consolidated CSV of historical data
    def create_consolidated_historical_dataset(self, symbols=None, start_date=None, end_date=None, 
                                               interval="1d", output_file="consolidated_gold_data.csv"):
        """
        Create a consolidated dataset of historical data for multiple symbols.
        
        Parameters:
            symbols (list): List of symbols to include. Default is GC=F and GLD
            start_date (str or datetime): Start date. Default is 5 years ago
            end_date (str or datetime): End date. Default is today
            interval (str): Data interval ('1d', '1wk', '1mo')
            output_file (str): Output CSV filename
            
        Returns:
            DataFrame: Consolidated dataset
        """
        if symbols is None:
            symbols = ["GC=F", "GLD", "DX-Y.NYB"]  # Gold futures, Gold ETF, USD Index
            
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        if start_date is None:
            start_date = end_date - timedelta(days=5*365)  # 5 years of data
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Dictionary to store dataframes for each symbol
        dfs = {}
        
        # Get data for each symbol
        for symbol in symbols:
            logger.info(f"Getting historical data for {symbol}")
            
            # Create a temporary file for each symbol
            temp_file = f"temp_{symbol.replace('=', '_').replace('-', '_').replace('.', '_')}.csv"
            
            df = self.get_historical_gold_data(
                symbol=symbol,
                period1=start_date,
                period2=end_date,
                interval=interval,
                csv_filename=temp_file
            )
            
            if df is not None:
                # Store dataframe with symbol as key
                dfs[symbol] = df
                
                # Add symbol column
                df['symbol'] = symbol
                
                # Add symbol description
                symbol_descriptions = {
                    "GC=F": "Gold Futures",
                    "GLD": "SPDR Gold Trust ETF",
                    "IAU": "iShares Gold Trust",
                    "DX-Y.NYB": "US Dollar Index",
                    "XAUUSD=X": "Gold Spot Price"
                }
                
                df['description'] = symbol_descriptions.get(symbol, symbol)
        
        if not dfs:
            logger.error("No data retrieved for any symbol")
            return None
            
        # Create consolidated dataframe with date as index
        consolidated_df = None
        
        for symbol, df in dfs.items():
            # Set index to date column
            df.set_index('date', inplace=True)
            
            # Create column names with symbol prefix
            price_columns = {
                'open': f'{symbol}_open',
                'high': f'{symbol}_high',
                'low': f'{symbol}_low',
                'close': f'{symbol}_close',
                'volume': f'{symbol}_volume'
            }
            
            # Rename columns
            df = df.rename(columns=price_columns)
            
            # Keep only price columns
            df = df[[f'{symbol}_open', f'{symbol}_high', f'{symbol}_low', f'{symbol}_close', f'{symbol}_volume']]
            
            if consolidated_df is None:
                consolidated_df = df
            else:
                # Join with existing consolidated dataframe
                consolidated_df = consolidated_df.join(df, how='outer')
        
        # Reset index to make date a column again
        consolidated_df.reset_index(inplace=True)
        consolidated_df.rename(columns={'index': 'date'}, inplace=True)
        
        # Fill missing values with forward fill then backward fill
        consolidated_df = consolidated_df.fillna(method='ffill').fillna(method='bfill')
        
        # Save to CSV
        consolidated_df.to_csv(output_file, index=False)
        logger.info(f"Consolidated historical data saved to {output_file} with {len(consolidated_df)} records")
        
        # Delete temporary files
        for symbol in symbols:
            temp_file = f"temp_{symbol.replace('=', '_').replace('-', '_').replace('.', '_')}.csv"
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return consolidated_df


# Example usage
if __name__ == "__main__":
    collector = GoldDataCollector()
    
    # Example 1: Get one year of daily gold futures (GC=F) data
    futures_data = collector.get_historical_gold_data(
        symbol="GC=F",
        period1="2024-04-25",  # One year ago
        period2="2025-04-25",  # Today
        interval="1d",
        csv_filename="gold_futures_daily_1y.csv"
    )
    
    # Example 2: Get multiple timeframes for gold ETF (GLD)
    multiple_tf_data = collector.get_multiple_timeframe_history(
        symbol="GLD",
        save_dir="gold_etf_data"
    )
    
    # Example 3: Get data for multiple gold-related symbols
    gold_market_data = collector.get_gold_market_data(
        save_dir="gold_market_data"
    )
    
    # Example 4: Create a consolidated dataset of gold-related assets
    consolidated_data = collector.create_consolidated_historical_dataset(
        symbols=["GC=F", "GLD", "DX-Y.NYB", "XAUUSD=X"],
        start_date="2020-04-25",
        end_date="2025-04-25",
        interval="1d",
        output_file="gold_market_analysis_dataset.csv"
    )
    
    print("Historical gold price data has been saved to CSV files.")