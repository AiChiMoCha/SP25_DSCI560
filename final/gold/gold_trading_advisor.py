import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from openai import OpenAI
import io
import base64

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("gold_chatbot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GoldChatBot")

class GoldDataLoader:
    """Load and process gold data from saved CSV files"""
    
    def __init__(self, data_dir="gold_data_unified"):
        self.data_dir = data_dir
        self.data_cache = {}
        logger.info(f"Initializing Gold Data Loader with data directory: {data_dir}")
        
    def load_latest_file(self, prefix, exact_name=None):
        """
        Load the latest file with the given prefix or exact name
        
        Args:
            prefix (str): Prefix of files to search for
            exact_name (str, optional): Exact filename to load
            
        Returns:
            DataFrame or None: Loaded data
        """
        try:
            if exact_name and os.path.exists(os.path.join(self.data_dir, exact_name)):
                file_path = os.path.join(self.data_dir, exact_name)
                df = pd.read_csv(file_path)
                logger.info(f"Loaded exact file: {file_path}")
                return df
            
            # Find all files with the given prefix
            files = [f for f in os.listdir(self.data_dir) if f.startswith(prefix) and f.endswith('.csv')]
            
            if not files:
                logger.warning(f"No files found with prefix: {prefix}")
                return None
                
            # Sort files by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.data_dir, x)), reverse=True)
            
            # Load the newest file
            newest_file = os.path.join(self.data_dir, files[0])
            df = pd.read_csv(newest_file)
            logger.info(f"Loaded latest file: {newest_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data file: {str(e)}")
            return None
    
    def load_all_data(self):
        """
        Load all the necessary data files
        
        Returns:
            dict: Dictionary containing all loaded data
        """
        try:
            logger.info("Loading all gold data...")
            
            # Load integrated dataset (main dataset)
            integrated_df = self.load_latest_file("analysis_integrated_dataset", "gold_integrated_dataset.csv")
            self.data_cache["integrated_data"] = integrated_df
            
            # Load gold price data
            gold_price_df = self.load_latest_file("price_GC=F_1d", "GC=F_1d_data.csv")
            self.data_cache["gold_price"] = gold_price_df
            
            # Load gold ETF data
            gold_etf_df = self.load_latest_file("price_GLD_1d", "GLD_1d_data.csv")
            self.data_cache["gold_etf"] = gold_etf_df
            
            # Load dollar index data
            dollar_index_df = self.load_latest_file("macro_dollar_index")
            self.data_cache["dollar_index"] = dollar_index_df
            
            # Load fear and greed data
            fear_greed_df = self.load_latest_file("sentiment_fear_greed", "fear_greed_cumulative.csv")
            self.data_cache["fear_greed"] = fear_greed_df
            
            # Load macroeconomic data
            macro_df = self.load_latest_file("macro_full_dataset", "macro_full_dataset.csv")
            self.data_cache["macro_data"] = macro_df
            
            logger.info("All data loaded successfully")
            return self.data_cache
            
        except Exception as e:
            logger.error(f"Error loading all data: {str(e)}")
            return {}
    
    def get_current_gold_price(self):
        """
        Get the most recent gold price
        
        Returns:
            float: Current gold price
        """
        try:
            if "gold_price" not in self.data_cache or self.data_cache["gold_price"] is None:
                gold_df = self.load_latest_file("price_GC=F_1d", "GC=F_1d_data.csv")
                self.data_cache["gold_price"] = gold_df
            else:
                gold_df = self.data_cache["gold_price"]
                
            if gold_df is not None and not gold_df.empty:
                # Sort by date descending to get the most recent price
                gold_df['date'] = pd.to_datetime(gold_df['date'])
                gold_df = gold_df.sort_values('date', ascending=False)
                
                # Get the most recent closing price
                latest_price = gold_df.iloc[0]['close']
                latest_date = gold_df.iloc[0]['date']
                
                return {
                    "price": latest_price,
                    "date": latest_date.strftime("%Y-%m-%d"),
                    "source": "Yahoo Finance (via saved data)"
                }
            else:
                logger.warning("No gold price data available")
                return None
                
        except Exception as e:
            logger.error(f"Error getting current gold price: {str(e)}")
            return None
    
    def generate_price_chart(self, days=90):
        """
        Generate a gold price chart for the specified number of days
        
        Args:
            days (int): Number of days to include in the chart
            
        Returns:
            str: Base64 encoded image
        """
        try:
            if "gold_price" not in self.data_cache or self.data_cache["gold_price"] is None:
                gold_df = self.load_latest_file("price_GC=F_1d", "GC=F_1d_data.csv")
                self.data_cache["gold_price"] = gold_df
            else:
                gold_df = self.data_cache["gold_price"]
                
            if gold_df is None or gold_df.empty:
                logger.warning("No gold price data available for chart generation")
                return None
                
            # Convert date column to datetime
            gold_df['date'] = pd.to_datetime(gold_df['date'])
            
            # Get data for the specified number of days
            end_date = gold_df['date'].max()
            start_date = end_date - timedelta(days=days)
            
            # Filter data
            filtered_df = gold_df[gold_df['date'] >= start_date]
            
            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(filtered_df['date'], filtered_df['close'], label='Gold Price (USD/oz)')
            
            # Add moving averages
            if len(filtered_df) > 20:
                plt.plot(filtered_df['date'], filtered_df['ma_20'], label='20-day MA', linestyle='--')
            if len(filtered_df) > 50:
                plt.plot(filtered_df['date'], filtered_df['ma_50'], label='50-day MA', linestyle='-.')
                
            plt.title(f'Gold Price Last {days} Days')
            plt.xlabel('Date')
            plt.ylabel('Price (USD/oz)')
            plt.grid(True)
            plt.legend()
            
            # Convert plot to base64 encoded image
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
            
            return img_str
            
        except Exception as e:
            logger.error(f"Error generating price chart: {str(e)}")
            return None
    
    def get_market_indicators(self):
        """
        Get current market indicators relevant to gold
        
        Returns:
            dict: Market indicators
        """
        indicators = {}
        
        try:
            # Get dollar index
            if "dollar_index" in self.data_cache and self.data_cache["dollar_index"] is not None:
                dollar_df = self.data_cache["dollar_index"]
                if not dollar_df.empty:
                    dollar_df = dollar_df.sort_values('date', ascending=False)
                    indicators["dollar_index"] = {
                        "value": dollar_df.iloc[0]['dxy_value'],
                        "date": dollar_df.iloc[0]['date']
                    }
            
            # Get fear and greed index
            if "fear_greed" in self.data_cache and self.data_cache["fear_greed"] is not None:
                fg_df = self.data_cache["fear_greed"]
                if not fg_df.empty:
                    fg_df = fg_df.sort_values('timestamp', ascending=False)
                    indicators["fear_greed"] = {
                        "value": fg_df.iloc[0]['value'],
                        "classification": fg_df.iloc[0]['classification'],
                        "date": fg_df.iloc[0]['timestamp']
                    }
            
            # Get real interest rate
            if "macro_data" in self.data_cache and self.data_cache["macro_data"] is not None:
                macro_df = self.data_cache["macro_data"]
                if not macro_df.empty and "real_interest_rate" in macro_df.columns:
                    macro_df = macro_df.sort_values('date', ascending=False)
                    indicators["real_interest_rate"] = {
                        "value": macro_df.iloc[0]['real_interest_rate'],
                        "date": macro_df.iloc[0]['date']
                    }
                    
                # Get inflation data
                if "cpi_yoy_change" in macro_df.columns:
                    indicators["inflation"] = {
                        "value": macro_df.iloc[0]['cpi_yoy_change'],
                        "date": macro_df.iloc[0]['date']
                    }
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error getting market indicators: {str(e)}")
            return indicators
    
    def analyze_gold_trend(self, days=90):
        """
        Analyze recent gold price trends
        
        Args:
            days (int): Number of days to analyze
            
        Returns:
            dict: Analysis results
        """
        try:
            if "gold_price" not in self.data_cache or self.data_cache["gold_price"] is None:
                gold_df = self.load_latest_file("price_GC=F_1d", "GC=F_1d_data.csv")
                self.data_cache["gold_price"] = gold_df
            else:
                gold_df = self.data_cache["gold_price"]
                
            if gold_df is None or gold_df.empty:
                logger.warning("No gold price data available for trend analysis")
                return None
                
            # Convert date column to datetime
            gold_df['date'] = pd.to_datetime(gold_df['date'])
            
            # Get data for the specified number of days
            end_date = gold_df['date'].max()
            start_date = end_date - timedelta(days=days)
            
            # Filter data
            filtered_df = gold_df[gold_df['date'] >= start_date]
            
            if filtered_df.empty:
                return None
                
            # Calculate key metrics
            start_price = filtered_df.iloc[0]['close']
            end_price = filtered_df.iloc[-1]['close']
            price_change = end_price - start_price
            percent_change = (price_change / start_price) * 100
            
            # Get highs and lows
            high_price = filtered_df['high'].max()
            low_price = filtered_df['low'].min()
            
            # Get volatility
            volatility = filtered_df['close'].pct_change().std() * 100
            
            # Determine trend
            if percent_change > 5:
                trend = "Strong Uptrend"
            elif percent_change > 2:
                trend = "Uptrend"
            elif percent_change < -5:
                trend = "Strong Downtrend"
            elif percent_change < -2:
                trend = "Downtrend"
            else:
                trend = "Sideways/Neutral"
                
            # Check momentum using RSI if available
            momentum = "Unknown"
            if 'rsi' in filtered_df.columns:
                latest_rsi = filtered_df.iloc[-1]['rsi']
                if not pd.isna(latest_rsi):
                    if latest_rsi > 70:
                        momentum = "Overbought"
                    elif latest_rsi < 30:
                        momentum = "Oversold"
                    elif latest_rsi > 50:
                        momentum = "Positive"
                    else:
                        momentum = "Negative"
            
            return {
                "period": f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}",
                "start_price": start_price,
                "end_price": end_price,
                "price_change": price_change,
                "percent_change": percent_change,
                "high": high_price,
                "low": low_price,
                "volatility": volatility,
                "trend": trend,
                "momentum": momentum
            }
            
        except Exception as e:
            logger.error(f"Error analyzing gold trend: {str(e)}")
            return None


class GPT4OGoldAdvisor:
    """Gold investment advisor powered by GPT-4o"""
    
    def __init__(self, api_key):
        """
        Initialize the GPT-4o advisor
        
        Args:
            api_key (str): OpenAI API key
        """
        self.client = OpenAI(api_key=api_key)
        self.system_prompt = """
        You are a specialized gold investment advisor chatbot. Your role is to provide information
        and advice about gold investments based on current market data and trends.

        When providing advice:
        1. Always base your recommendations on the latest data provided
        2. Consider both fundamental factors (inflation, interest rates, etc.) and technical analysis
        3. Explain your reasoning clearly and concisely
        4. Acknowledge uncertainty and avoid making definitive price predictions
        5. Provide balanced perspective on both potential upsides and downsides
        6. Keep in mind that all investments carry risk

        You have access to:
        - Current gold prices
        - Historical price trends
        - Price charts
        - Market indicators (dollar index, fear & greed index, interest rates)
        - Technical indicators (RSI, moving averages, etc.)
        
        Remember that you're not a financial advisor, and users should consult with financial professionals
        before making investment decisions.
        """
        self.conversation_history = []
        logger.info("GPT-4o Gold Advisor initialized")
        
    def generate_response(self, user_message, gold_data):
        """
        Generate a response using GPT-4o based on user message and gold data
        
        Args:
            user_message (str): User message
            gold_data (dict): Gold market data
            
        Returns:
            str: Generated response
        """
        try:
            # Format gold data for inclusion in prompt
            current_price = gold_data.get("current_price", {})
            market_indicators = gold_data.get("market_indicators", {})
            trend_analysis = gold_data.get("trend_analysis", {})
            
            data_context = json.dumps({
                "current_gold_price": current_price,
                "market_indicators": market_indicators,
                "gold_trend_analysis": trend_analysis,
                "current_date": datetime.now().strftime("%Y-%m-%d")
            }, indent=2)
            
            # Build messages array
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Here is the current gold market data for your reference:\n```json\n{data_context}\n```"}
            ]
            
            # Add conversation history for context
            for msg in self.conversation_history[-5:]:  # Limit to last 5 messages
                messages.append(msg)
                
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Call GPT-4o API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            response_text = response.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating GPT-4o response: {str(e)}")
            return "I'm sorry, I encountered an error processing your request. Please try again later."


class GoldChatBot:
    """Main class for the gold purchase chatbot"""
    
    def __init__(self, openai_api_key, data_dir="gold_data_unified"):
        """
        Initialize the gold chatbot
        
        Args:
            openai_api_key (str): OpenAI API key
            data_dir (str): Directory containing gold data files
        """
        self.data_loader = GoldDataLoader(data_dir)
        self.advisor = GPT4OGoldAdvisor(openai_api_key)
        
        # Load data
        self.data = self.data_loader.load_all_data()
        logger.info("Gold ChatBot initialized and data loaded")
        
    def process_message(self, user_message):
        """
        Process a user message and generate a response
        
        Args:
            user_message (str): User message
            
        Returns:
            dict: Response with text and optional chart
        """
        try:
            # Get latest gold data
            current_price = self.data_loader.get_current_gold_price()
            market_indicators = self.data_loader.get_market_indicators()
            trend_analysis = self.data_loader.analyze_gold_trend()
            
            gold_data = {
                "current_price": current_price,
                "market_indicators": market_indicators,
                "trend_analysis": trend_analysis
            }
            
            # Check if user wants a chart
            include_chart = False
            if "chart" in user_message.lower() or "graph" in user_message.lower() or "visual" in user_message.lower():
                include_chart = True
                
            # Generate text response
            response_text = self.advisor.generate_response(user_message, gold_data)
            
            response = {
                "text": response_text
            }
            
            # Add chart if requested
            if include_chart:
                chart_data = self.data_loader.generate_price_chart()
                if chart_data:
                    response["chart"] = chart_data
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {"text": "I apologize, but I encountered an error processing your request. Please try again."}
    
    def run_cli(self):
        """Run the chatbot in CLI mode"""
        print("🥇 Gold Investment Advisor Chatbot 🥇")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("-" * 50)
        
        current_price = self.data_loader.get_current_gold_price()
        if current_price:
            print(f"Current Gold Price: ${current_price['price']:.2f} (as of {current_price['date']})")
        
        print("-" * 50)
        
        while True:
            user_input = input("You: ")
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("Thank you for using the Gold Investment Advisor! Goodbye.")
                break
                
            response = self.process_message(user_input)
            print("\nAdvisor:", response["text"])
            
            if "chart" in response:
                print("\n[A gold price chart would be displayed here in a GUI application]")
                
            print("-" * 50)


# Example usage
if __name__ == "__main__":
    # Replace with your actual OpenAI API key
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")
    
    # Create and run the chatbot
    chatbot = GoldChatBot(OPENAI_API_KEY)
    chatbot.run_cli()