import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import logging
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("gold_macro_data.log"), logging.StreamHandler()]
)
logger = logging.getLogger("GoldMacroDataCollector")

class GoldMacroDataCollector:
    """用于收集黄金市场相关的宏观经济数据的工具"""
    
    def __init__(self, data_dir="gold_macro_data"):
        """
        初始化数据收集器
        
        参数:
            data_dir (str): 数据保存目录
        """
        self.data_dir = data_dir
        
        # 确保数据目录存在
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            
        # 初始化API密钥
        self.api_keys = {
            'fred': os.getenv('FRED_API_KEY')
        }
        
        # 检查API密钥
        missing_keys = [key for key, value in self.api_keys.items() if not value]
        if missing_keys:
            logger.warning(f"以下API密钥未设置: {', '.join(missing_keys)}")
            
        logger.info("黄金市场宏观数据收集器已初始化")
    
    def get_timestamp(self):
        """获取当前时间戳，格式为YYYY-MM-DD_HH-MM-SS"""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    def get_fred_data(self, series_id, start_date=None, end_date=None):
        """
        从FRED获取经济数据
        
        参数:
            series_id (str): FRED数据系列ID
            start_date (str): 开始日期，格式为YYYY-MM-DD
            end_date (str): 结束日期，格式为YYYY-MM-DD
            
        返回:
            DataFrame: 包含时间序列数据的DataFrame
        """
        if not self.api_keys['fred']:
            logger.error("FRED API密钥未设置")
            return None
            
        # 设置默认日期
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        if start_date is None:
            # 默认获取过去1年的数据
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            
        try:
            # 构建API URL
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_keys['fred'],
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if "observations" in data:
                observations = data["observations"]
                
                # 创建DataFrame
                df = pd.DataFrame(observations)
                
                # 转换日期和值
                df["date"] = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                
                # 添加系列ID
                df["series_id"] = series_id
                
                # 添加系列描述（如果需要更多信息可以再调用series API）
                series_descriptions = {
                    "DGS10": "10-Year Treasury Constant Maturity Rate",
                    "DFII10": "10-Year Treasury Inflation-Indexed Security, Constant Maturity",
                    "CPIAUCSL": "Consumer Price Index for All Urban Consumers: All Items",
                    "PCEPI": "Personal Consumption Expenditures: Chain-type Price Index",
                    "PCEPILFE": "Personal Consumption Expenditures Excluding Food and Energy (Core PCE)",
                    "FEDFUNDS": "Federal Funds Effective Rate",
                    "WALCL": "Federal Reserve Total Assets",
                    "DXY": "U.S. Dollar Index",
                    "T10YIE": "10-Year Breakeven Inflation Rate"
                }
                
                df["description"] = series_descriptions.get(series_id, series_id)
                
                return df
            else:
                logger.error(f"从FRED获取{series_id}数据时出错: {data}")
                return None
                
        except Exception as e:
            logger.error(f"从FRED获取{series_id}数据时发生错误: {str(e)}")
            return None
    
    def get_dollar_index(self):
        """
        获取美元指数(DXY)数据
        
        返回:
            DataFrame: 美元指数数据
        """
        try:
            # 使用Yahoo Finance API获取最新美元指数数据
            url = "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            # 获取过去30天的数据
            end_time = int(time.time())
            start_time = end_time - (30 * 24 * 60 * 60)  # 30天前
            
            params = {
                "period1": start_time,
                "period2": end_time,
                "interval": "1d"
            }
            
            response = requests.get(url, headers=headers, params=params)
            data = response.json()
            
            if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
                result = data["chart"]["result"][0]
                
                # 提取时间戳和价格数据
                timestamps = result["timestamp"]
                
                # 提取指数数据
                quote = result["indicators"]["quote"][0]
                closes = quote.get("close", [])
                
                # 创建DataFrame
                df = pd.DataFrame({
                    "timestamp": timestamps,
                    "dxy_value": closes
                })
                
                # 处理可能的NaN值
                df = df.dropna()
                
                # 将时间戳转换为日期时间
                df["date"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
                df["datetime"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
                
                # 添加描述
                df["description"] = "U.S. Dollar Index (DXY)"
                
                # 保存到CSV
                filename = os.path.join(self.data_dir, f"dollar_index_{self.get_timestamp()}.csv")
                df.to_csv(filename, index=False)
                logger.info(f"美元指数数据已保存到 {filename}")
                
                return df
            else:
                logger.error(f"获取美元指数数据时出错: {data}")
                return None
                
        except Exception as e:
            logger.error(f"获取美元指数数据时发生错误: {str(e)}")
            return None
    
    def get_real_interest_rates(self):
        """
        获取美国实际利率数据（10年期TIPS收益率）
        
        返回:
            DataFrame: 实际利率数据
        """
        df = self.get_fred_data("DFII10")
        
        if df is not None:
            # 保存到CSV
            filename = os.path.join(self.data_dir, f"real_interest_rate_{self.get_timestamp()}.csv")
            df.to_csv(filename, index=False)
            logger.info(f"实际利率数据已保存到 {filename}")
            
        return df
    
    def get_inflation_data(self):
        """
        获取通胀数据（CPI和核心PCE）
        
        返回:
            dict: 包含各类通胀指标的DataFrame
        """
        results = {}
        
        # 获取CPI数据
        cpi_df = self.get_fred_data("CPIAUCSL")
        if cpi_df is not None:
            filename = os.path.join(self.data_dir, f"cpi_{self.get_timestamp()}.csv")
            cpi_df.to_csv(filename, index=False)
            logger.info(f"CPI数据已保存到 {filename}")
            results["cpi"] = cpi_df
            
        # 获取PCE数据
        pce_df = self.get_fred_data("PCEPI")
        if pce_df is not None:
            filename = os.path.join(self.data_dir, f"pce_{self.get_timestamp()}.csv")
            pce_df.to_csv(filename, index=False)
            logger.info(f"PCE数据已保存到 {filename}")
            results["pce"] = pce_df
            
        # 获取核心PCE数据（美联储关注的指标）
        core_pce_df = self.get_fred_data("PCEPILFE")
        if core_pce_df is not None:
            filename = os.path.join(self.data_dir, f"core_pce_{self.get_timestamp()}.csv")
            core_pce_df.to_csv(filename, index=False)
            logger.info(f"核心PCE数据已保存到 {filename}")
            results["core_pce"] = core_pce_df
            
        return results
    
    def get_fed_policy_data(self):
        """
        获取美联储政策数据（联邦基金利率和资产负债表）
        
        返回:
            dict: 包含联邦基金利率和资产负债表的DataFrame
        """
        results = {}
        
        # 获取联邦基金利率
        fed_funds_df = self.get_fred_data("FEDFUNDS")
        if fed_funds_df is not None:
            filename = os.path.join(self.data_dir, f"fed_funds_rate_{self.get_timestamp()}.csv")
            fed_funds_df.to_csv(filename, index=False)
            logger.info(f"联邦基金利率数据已保存到 {filename}")
            results["fed_funds"] = fed_funds_df
            
        # 获取美联储资产负债表
        fed_assets_df = self.get_fred_data("WALCL")
        if fed_assets_df is not None:
            filename = os.path.join(self.data_dir, f"fed_assets_{self.get_timestamp()}.csv")
            fed_assets_df.to_csv(filename, index=False)
            logger.info(f"美联储资产负债表数据已保存到 {filename}")
            results["fed_assets"] = fed_assets_df
            
        return results
    
    def get_all_macro_data(self):
        """
        获取所有宏观经济数据并保存到CSV
        
        返回:
            dict: 包含所有宏观经济数据的字典
        """
        logger.info("开始收集宏观经济数据...")
        
        macro_data = {}
        
        # 获取美元指数数据
        macro_data["dollar_index"] = self.get_dollar_index()
        
        # 获取实际利率数据
        macro_data["real_interest_rate"] = self.get_real_interest_rates()
        
        # 获取通胀数据
        macro_data["inflation"] = self.get_inflation_data()
        
        # 获取美联储政策数据
        macro_data["fed_policy"] = self.get_fed_policy_data()
        
        # 将主要宏观数据合并到一个综合CSV文件
        self.create_combined_macro_csv(macro_data)
        
        logger.info("宏观经济数据收集完成")
        
        return macro_data
    
    def create_combined_macro_csv(self, macro_data):
        """
        创建综合宏观数据CSV文件，用于分析
        
        参数:
            macro_data (dict): 宏观经济数据字典
        """
        try:
            combined_df = None
            
            # 准备美元指数数据
            if "dollar_index" in macro_data and macro_data["dollar_index"] is not None:
                dxy_df = macro_data["dollar_index"].copy()
                dxy_df = dxy_df[["date", "dxy_value"]]
                dxy_df.set_index("date", inplace=True)
                
                if combined_df is None:
                    combined_df = dxy_df
                else:
                    combined_df = combined_df.join(dxy_df, how="outer")
            
            # 准备实际利率数据
            if "real_interest_rate" in macro_data and macro_data["real_interest_rate"] is not None:
                rate_df = macro_data["real_interest_rate"].copy()
                rate_df["date"] = rate_df["date"].dt.strftime("%Y-%m-%d")
                rate_df = rate_df[["date", "value"]]
                rate_df.columns = ["date", "real_interest_rate"]
                rate_df.set_index("date", inplace=True)
                
                if combined_df is None:
                    combined_df = rate_df
                else:
                    combined_df = combined_df.join(rate_df, how="outer")
            
            # 准备通胀数据
            if "inflation" in macro_data:
                # CPI
                if "cpi" in macro_data["inflation"] and macro_data["inflation"]["cpi"] is not None:
                    cpi_df = macro_data["inflation"]["cpi"].copy()
                    cpi_df["date"] = cpi_df["date"].dt.strftime("%Y-%m-%d")
                    cpi_df = cpi_df[["date", "value"]]
                    cpi_df.columns = ["date", "cpi"]
                    cpi_df.set_index("date", inplace=True)
                    
                    if combined_df is None:
                        combined_df = cpi_df
                    else:
                        combined_df = combined_df.join(cpi_df, how="outer")
                
                # 核心PCE
                if "core_pce" in macro_data["inflation"] and macro_data["inflation"]["core_pce"] is not None:
                    pce_df = macro_data["inflation"]["core_pce"].copy()
                    pce_df["date"] = pce_df["date"].dt.strftime("%Y-%m-%d")
                    pce_df = pce_df[["date", "value"]]
                    pce_df.columns = ["date", "core_pce"]
                    pce_df.set_index("date", inplace=True)
                    
                    if combined_df is None:
                        combined_df = pce_df
                    else:
                        combined_df = combined_df.join(pce_df, how="outer")
            
            # 准备美联储政策数据
            if "fed_policy" in macro_data:
                # 联邦基金利率
                if "fed_funds" in macro_data["fed_policy"] and macro_data["fed_policy"]["fed_funds"] is not None:
                    fed_df = macro_data["fed_policy"]["fed_funds"].copy()
                    fed_df["date"] = fed_df["date"].dt.strftime("%Y-%m-%d")
                    fed_df = fed_df[["date", "value"]]
                    fed_df.columns = ["date", "fed_funds_rate"]
                    fed_df.set_index("date", inplace=True)
                    
                    if combined_df is None:
                        combined_df = fed_df
                    else:
                        combined_df = combined_df.join(fed_df, how="outer")
            
            # 重置索引
            if combined_df is not None:
                combined_df.reset_index(inplace=True)
                
                # 排序并填充缺失值
                combined_df.sort_values("date", inplace=True)
                combined_df = combined_df.ffill().bfill()  # 前向和后向填充
                
                # 保存到CSV
                filename = os.path.join(self.data_dir, f"combined_macro_data_{self.get_timestamp()}.csv")
                combined_df.to_csv(filename, index=False)
                logger.info(f"综合宏观数据已保存到 {filename}")
        
        except Exception as e:
            logger.error(f"创建综合宏观数据时发生错误: {str(e)}")


# 示例使用方法
if __name__ == "__main__":
    # 初始化收集器
    collector = GoldMacroDataCollector()
    
    # 收集所有宏观经济数据
    macro_data = collector.get_all_macro_data()
    
    print("宏观经济数据收集完成，CSV文件已保存到 gold_macro_data 目录")