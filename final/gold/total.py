import os
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import json
import logging
from dotenv import load_dotenv
from scipy import stats
import matplotlib.pyplot as plt

# 加载环境变量
load_dotenv()

# 设置统一的时间格式和数据目录
DATA_DIR = "gold_data_unified"
TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"

# 确保数据目录存在
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, "gold_data.log")), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GoldDataCollector")

# 通用工具函数
def get_timestamp():
    """获取当前时间戳，格式为YYYY-MM-DD_HH-MM-SS"""
    return datetime.now().strftime(TIMESTAMP_FORMAT)

def get_filename(data_type, suffix=""):
    """生成统一格式的文件名"""
    timestamp = get_timestamp()
    return os.path.join(DATA_DIR, f"{data_type}_{suffix}_{timestamp}.csv")

class DataProcessor:
    """数据处理工具类：用于对收集的数据进行处理和优化"""
    
    @staticmethod
    def align_time_series(macro_df, price_df, freq='D'):
        """
        将宏观数据与价格数据在时间上对齐
        
        参数:
            macro_df (DataFrame): 宏观数据DataFrame，通常是低频数据(月度/季度)
            price_df (DataFrame): 价格数据DataFrame，通常是高频数据(日/分钟)
            freq (str): 重采样频率，默认为'D'(日)
            
        返回:
            DataFrame: 时间对齐后的DataFrame
        """
        try:
            logger.info("对时间序列数据进行对齐...")
            
            # 确保两个DataFrame都有日期列，并且是datetime格式
            if 'date' in macro_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(macro_df['date']):
                    macro_df['date'] = pd.to_datetime(macro_df['date'])
                macro_df = macro_df.set_index('date')
            
            if 'date' in price_df.columns:
                if not pd.api.types.is_datetime64_any_dtype(price_df['date']):
                    price_df['date'] = pd.to_datetime(price_df['date'])
                price_df = price_df.set_index('date')
            
            # 获取两个DataFrame的日期范围
            min_date = min(macro_df.index.min(), price_df.index.min())
            max_date = max(macro_df.index.max(), price_df.index.max())
            
            # 创建完整的日期范围
            date_range = pd.date_range(start=min_date, end=max_date, freq=freq)
            
            # 对宏观数据进行重采样和填充(前向填充)
            resampled_macro = macro_df.reindex(date_range).fillna(method='ffill')
            
            # 对价格数据进行重采样
            resampled_price = price_df.reindex(date_range)
            
            # 合并数据
            merged_df = pd.concat([resampled_macro, resampled_price], axis=1)
            
            # 处理可能的重复列名
            merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
            
            logger.info("时间序列对齐完成")
            return merged_df
            
        except Exception as e:
            logger.error(f"时间序列对齐时出错: {str(e)}")
            return None
    
    @staticmethod
    def detect_anomalies(df, col_name, window=20, threshold=3):
        """
        使用滚动Z得分检测异常值
        
        参数:
            df (DataFrame): 数据DataFrame
            col_name (str): 要检测的列名
            window (int): 滚动窗口大小
            threshold (float): Z得分阈值
            
        返回:
            DataFrame: 带有异常标记的DataFrame
        """
        try:
            logger.info(f"正在检测列 {col_name} 中的异常值...")
            
            # 计算滚动均值和标准差
            rolling_mean = df[col_name].rolling(window=window).mean()
            rolling_std = df[col_name].rolling(window=window).std()
            
            # 计算Z得分
            df[f'{col_name}_zscore'] = abs((df[col_name] - rolling_mean) / rolling_std)
            
            # 标记异常值
            df[f'{col_name}_is_anomaly'] = df[f'{col_name}_zscore'] > threshold
            
            # 计算异常值数量
            anomaly_count = df[f'{col_name}_is_anomaly'].sum()
            logger.info(f"检测到 {anomaly_count} 个异常值")
            
            return df
            
        except Exception as e:
            logger.error(f"异常值检测时出错: {str(e)}")
            return df
    
    @staticmethod
    def calculate_gold_etf_premium(futures_df, etf_df):
        """
        计算黄金ETF相对于期货的溢价率
        
        参数:
            futures_df (DataFrame): 黄金期货数据
            etf_df (DataFrame): 黄金ETF数据
            
        返回:
            DataFrame: 包含溢价率的DataFrame
        """
        try:
            logger.info("计算黄金ETF溢价率...")
            
            # 确保两个DataFrame都有date列
            if 'date' not in futures_df.columns or 'date' not in etf_df.columns:
                logger.error("数据中缺少date列")
                return None
            
            # 合并数据
            merged_df = pd.merge(
                futures_df[['date', 'close']],
                etf_df[['date', 'close']],
                on='date', 
                suffixes=('_futures', '_etf')
            )
            
            # 计算溢价率
            merged_df['premium_pct'] = (merged_df['close_etf'] - merged_df['close_futures']) / merged_df['close_futures'] * 100
            
            # 检测异常溢价
            merged_df = DataProcessor.detect_anomalies(merged_df, 'premium_pct', window=20, threshold=2)
            
            # 获取异常溢价数据
            anomalies = merged_df[merged_df['premium_pct_is_anomaly']]
            if not anomalies.empty:
                logger.warning(f"检测到 {len(anomalies)} 个异常溢价，可能表明市场不稳定或数据问题")
            
            return merged_df
            
        except Exception as e:
            logger.error(f"计算黄金ETF溢价率时出错: {str(e)}")
            return None
    
    @staticmethod
    def create_integrated_dataset(macro_data, price_data, fear_greed_data):
        """
        创建整合所有数据源的分析数据集
        
        参数:
            macro_data (dict): 宏观经济数据字典
            price_data (dict): 价格数据字典
            fear_greed_data (DataFrame): 恐惧与贪婪指数数据
            
        返回:
            DataFrame: 整合后的分析数据集
        """
        try:
            logger.info("创建整合数据集...")
            
            # 步骤1: 准备宏观数据
            macro_df = None
            
            # 处理美元指数数据
            if "dollar_index" in macro_data and macro_data["dollar_index"] is not None:
                dxy_df = macro_data["dollar_index"].copy()
                dxy_df['date'] = pd.to_datetime(dxy_df['date'])
                dxy_df = dxy_df[['date', 'dxy_value']]
                
                if macro_df is None:
                    macro_df = dxy_df
                else:
                    macro_df = pd.merge(macro_df, dxy_df, on='date', how='outer')
            
            # 处理实际利率数据
            if "real_interest_rate" in macro_data and macro_data["real_interest_rate"] is not None:
                rate_df = macro_data["real_interest_rate"].copy()
                rate_df = rate_df[['date', 'value']]
                rate_df.columns = ['date', 'real_interest_rate']
                
                if macro_df is None:
                    macro_df = rate_df
                else:
                    macro_df = pd.merge(macro_df, rate_df, on='date', how='outer')
            
            # 处理通胀数据
            if "inflation" in macro_data:
                # CPI
                if "cpi" in macro_data["inflation"] and macro_data["inflation"]["cpi"] is not None:
                    cpi_df = macro_data["inflation"]["cpi"].copy()
                    cpi_df = cpi_df[['date', 'value']]
                    cpi_df.columns = ['date', 'cpi']
                    
                    if macro_df is None:
                        macro_df = cpi_df
                    else:
                        macro_df = pd.merge(macro_df, cpi_df, on='date', how='outer')
                
                # 核心PCE
                if "core_pce" in macro_data["inflation"] and macro_data["inflation"]["core_pce"] is not None:
                    pce_df = macro_data["inflation"]["core_pce"].copy()
                    pce_df = pce_df[['date', 'value']]
                    pce_df.columns = ['date', 'core_pce']
                    
                    if macro_df is None:
                        macro_df = pce_df
                    else:
                        macro_df = pd.merge(macro_df, pce_df, on='date', how='outer')
            
            # 处理美联储政策数据
            if "fed_policy" in macro_data:
                # 联邦基金利率
                if "fed_funds" in macro_data["fed_policy"] and macro_data["fed_policy"]["fed_funds"] is not None:
                    fed_df = macro_data["fed_policy"]["fed_funds"].copy()
                    fed_df = fed_df[['date', 'value']]
                    fed_df.columns = ['date', 'fed_funds_rate']
                    
                    if macro_df is None:
                        macro_df = fed_df
                    else:
                        macro_df = pd.merge(macro_df, fed_df, on='date', how='outer')
            
            # 步骤2: 准备价格数据
            if "price_data" in price_data and price_data["price_data"] is not None:
                price_df = price_data["price_data"].copy()
                price_df['date'] = pd.to_datetime(price_df['date'])
            else:
                logger.error("无法找到价格数据")
                return None
            
            # 步骤3: 准备恐惧与贪婪指数数据
            if fear_greed_data is not None:
                # 如果是累积数据文件，直接使用
                if os.path.exists(os.path.join(DATA_DIR, "fear_greed_cumulative.csv")):
                    fear_greed_df = pd.read_csv(os.path.join(DATA_DIR, "fear_greed_cumulative.csv"))
                    fear_greed_df['date'] = pd.to_datetime(fear_greed_df['date'])
                    fear_greed_df = fear_greed_df[['date', 'value', 'classification']]
                    fear_greed_df.columns = ['date', 'fear_greed_value', 'fear_greed_class']
                else:
                    logger.warning("没有找到恐惧与贪婪指数累积数据文件")
                    fear_greed_df = None
            else:
                logger.warning("没有提供恐惧与贪婪指数数据")
                fear_greed_df = None
            
            # 步骤4: 合并所有数据
            # 首先合并宏观数据和价格数据
            if macro_df is not None and price_df is not None:
                # 将所有日期转换为日期格式
                macro_df['date'] = pd.to_datetime(macro_df['date'])
                price_df['date'] = pd.to_datetime(price_df['date'])
                
                # 合并数据
                combined_df = pd.merge(price_df, macro_df, on='date', how='outer')
            else:
                combined_df = price_df if price_df is not None else macro_df
            
            # 如果有恐惧与贪婪指数数据，也合并
            if fear_greed_df is not None and combined_df is not None:
                combined_df = pd.merge(combined_df, fear_greed_df, on='date', how='outer')
            
            # 步骤5: 对齐时间序列数据
            if combined_df is not None:
                # 确保日期列是日期类型
                if not pd.api.types.is_datetime64_any_dtype(combined_df['date']):
                    combined_df['date'] = pd.to_datetime(combined_df['date'])
                
                # 对日期进行排序
                combined_df = combined_df.sort_values('date')
                
                # 设置日期为索引
                combined_df = combined_df.set_index('date')
                
                # 对所有数据进行前向填充和后向填充
                combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
                
                # 重置索引，使日期再次成为列
                combined_df.reset_index(inplace=True)
                
                # 添加时间特征
                combined_df['year'] = combined_df['date'].dt.year
                combined_df['month'] = combined_df['date'].dt.month
                combined_df['day'] = combined_df['date'].dt.day
                combined_df['day_of_week'] = combined_df['date'].dt.dayofweek
                combined_df['is_month_end'] = combined_df['date'].dt.is_month_end
                combined_df['is_quarter_end'] = combined_df['date'].dt.is_quarter_end
                
                logger.info(f"整合数据集创建完成，包含 {len(combined_df)} 条记录")
                
                # 保存整合数据集
                filename = get_filename("analysis", "integrated_dataset")
                combined_df.to_csv(filename, index=False)
                logger.info(f"整合数据集已保存到 {filename}")
                
                return combined_df
            else:
                logger.error("无法创建整合数据集")
                return None
                
        except Exception as e:
            logger.error(f"创建整合数据集时出错: {str(e)}")
            return None

class GoldMacroDataCollector:
    """用于收集黄金市场相关的宏观经济数据的工具"""
    
    def __init__(self):
        """初始化数据收集器"""
        # 初始化API密钥
        self.api_keys = {
            'fred': os.getenv('FRED_API_KEY')
        }
        
        # 检查API密钥
        missing_keys = [key for key, value in self.api_keys.items() if not value]
        if missing_keys:
            logger.warning(f"以下API密钥未设置: {', '.join(missing_keys)}")
            
        logger.info("黄金市场宏观数据收集器已初始化")
    
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
            # 扩展默认时间范围，获取5年的数据而不是1年
            start_date = (datetime.now() - timedelta(days=5*365)).strftime("%Y-%m-%d")
            
        try:
            # 构建API URL
            url = f"https://api.stlouisfed.org/fred/series/observations"
            params = {
                "series_id": series_id,
                "api_key": self.api_keys['fred'],
                "file_type": "json",
                "observation_start": start_date,
                "observation_end": end_date,
                "limit": 10000  # 增加限制，获取更多数据点
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
                
                # 扩展：添加同比和环比变化率
                if len(df) > 1:
                    # 计算同比变化率(YoY)
                    if series_id in ["CPIAUCSL", "PCEPI", "PCEPILFE"]:
                        df["yoy_change"] = df["value"].pct_change(periods=12) * 100
                    
                    # 计算环比变化率(MoM)
                    df["mom_change"] = df["value"].pct_change() * 100
                
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
            # 扩展为获取2年的数据而不是30天
            start_time = end_time - (730 * 24 * 60 * 60)  # 730天前
            
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
                
                # 扩展：添加移动平均线
                if len(df) > 50:
                    df["dxy_ma_20"] = df["dxy_value"].rolling(window=20).mean()
                    df["dxy_ma_50"] = df["dxy_value"].rolling(window=50).mean()
                    df["dxy_ma_200"] = df["dxy_value"].rolling(window=200).mean()
                
                # 保存到CSV
                filename = get_filename("macro", "dollar_index")
                df.to_csv(filename, index=False)
                logger.info(f"美元指数数据已保存到 {filename}，包含 {len(df)} 条记录")
                
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
            # 计算实际利率的变化
            df["rate_change"] = df["value"].diff()
            
            # 保存到CSV
            filename = get_filename("macro", "real_interest_rate")
            df.to_csv(filename, index=False)
            logger.info(f"实际利率数据已保存到 {filename}，包含 {len(df)} 条记录")
            
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
            # 添加YoY和MoM变化率（如果尚未添加）
            if "yoy_change" not in cpi_df.columns:
                cpi_df["yoy_change"] = cpi_df["value"].pct_change(periods=12) * 100
                
            if "mom_change" not in cpi_df.columns:
                cpi_df["mom_change"] = cpi_df["value"].pct_change() * 100
            
            filename = get_filename("macro", "cpi")
            cpi_df.to_csv(filename, index=False)
            logger.info(f"CPI数据已保存到 {filename}，包含 {len(cpi_df)} 条记录")
            results["cpi"] = cpi_df
            
        # 获取PCE数据
        pce_df = self.get_fred_data("PCEPI")
        if pce_df is not None:
            # 添加YoY和MoM变化率（如果尚未添加）
            if "yoy_change" not in pce_df.columns:
                pce_df["yoy_change"] = pce_df["value"].pct_change(periods=12) * 100
                
            if "mom_change" not in pce_df.columns:
                pce_df["mom_change"] = pce_df["value"].pct_change() * 100
                
            filename = get_filename("macro", "pce")
            pce_df.to_csv(filename, index=False)
            logger.info(f"PCE数据已保存到 {filename}，包含 {len(pce_df)} 条记录")
            results["pce"] = pce_df
            
        # 获取核心PCE数据（美联储关注的指标）
        core_pce_df = self.get_fred_data("PCEPILFE")
        if core_pce_df is not None:
            # 添加YoY和MoM变化率（如果尚未添加）
            if "yoy_change" not in core_pce_df.columns:
                core_pce_df["yoy_change"] = core_pce_df["value"].pct_change(periods=12) * 100
                
            if "mom_change" not in core_pce_df.columns:
                core_pce_df["mom_change"] = core_pce_df["value"].pct_change() * 100
                
            filename = get_filename("macro", "core_pce")
            core_pce_df.to_csv(filename, index=False)
            logger.info(f"核心PCE数据已保存到 {filename}，包含 {len(core_pce_df)} 条记录")
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
            # 计算利率变化
            fed_funds_df["rate_change"] = fed_funds_df["value"].diff()
            
            filename = get_filename("macro", "fed_funds_rate")
            fed_funds_df.to_csv(filename, index=False)
            logger.info(f"联邦基金利率数据已保存到 {filename}，包含 {len(fed_funds_df)} 条记录")
            results["fed_funds"] = fed_funds_df
            
        # 获取美联储资产负债表
        fed_assets_df = self.get_fred_data("WALCL")
        if fed_assets_df is not None:
            # 计算资产负债表变化
            fed_assets_df["abs_change"] = fed_assets_df["value"].diff()
            fed_assets_df["pct_change"] = fed_assets_df["value"].pct_change() * 100
            
            filename = get_filename("macro", "fed_assets")
            fed_assets_df.to_csv(filename, index=False)
            logger.info(f"美联储资产负债表数据已保存到 {filename}，包含 {len(fed_assets_df)} 条记录")
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
        combined_df = self.create_combined_macro_csv(macro_data)
        
        # 保存合并后的完整数据集
        if combined_df is not None:
            # 为每个日期点增加更多特征
            # 日期特征
            combined_df["year"] = combined_df["date"].dt.year
            combined_df["month"] = combined_df["date"].dt.month
            combined_df["day_of_week"] = combined_df["date"].dt.dayofweek
            combined_df["is_month_end"] = combined_df["date"].dt.is_month_end
            combined_df["is_quarter_end"] = combined_df["date"].dt.is_quarter_end
            
            # 美元指数与实际利率的相对关系
            if "dxy_value" in combined_df.columns and "real_interest_rate" in combined_df.columns:
                combined_df["dxy_vs_real_rate"] = combined_df["dxy_value"] / combined_df["real_interest_rate"]
                
            # 通胀与利率的差值（实际实际利率）
            if "cpi" in combined_df.columns and "fed_funds_rate" in combined_df.columns:
                combined_df["real_fed_rate"] = combined_df["fed_funds_rate"] - combined_df["cpi_yoy_change"]
            
            # 保存完整数据集
            filename = os.path.join(DATA_DIR, "macro_full_dataset.csv")
            combined_df.to_csv(filename, index=False)
            logger.info(f"完整宏观数据集已保存到 {filename}")
            
            # 同时也保存一个带时间戳的版本
            timestamp_filename = get_filename("macro", "full_dataset")
            combined_df.to_csv(timestamp_filename, index=False)
            
            macro_data["full_dataset"] = combined_df
        
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
                rate_df["date"] = pd.to_datetime(rate_df["date"])
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
                    cpi_df["date"] = pd.to_datetime(cpi_df["date"])
                    # 包含YoY变化率
                    cpi_cols = ["date", "value", "yoy_change", "mom_change"]
                    cpi_cols = [col for col in cpi_cols if col in cpi_df.columns]
                    cpi_df = cpi_df[cpi_cols]
                    cpi_cols_new = ["date", "cpi", "cpi_yoy_change", "cpi_mom_change"]
                    cpi_cols_new = cpi_cols_new[:len(cpi_cols)]
                    cpi_df.columns = cpi_cols_new
                    cpi_df.set_index("date", inplace=True)
                    
                    if combined_df is None:
                        combined_df = cpi_df
                    else:
                        combined_df = combined_df.join(cpi_df, how="outer")
                
                # 核心PCE
                if "core_pce" in macro_data["inflation"] and macro_data["inflation"]["core_pce"] is not None:
                    pce_df = macro_data["inflation"]["core_pce"].copy()
                    pce_df["date"] = pd.to_datetime(pce_df["date"])
                    # 包含YoY变化率
                    pce_cols = ["date", "value", "yoy_change", "mom_change"]
                    pce_cols = [col for col in pce_cols if col in pce_df.columns]
                    pce_df = pce_df[pce_cols]
                    pce_cols_new = ["date", "core_pce", "core_pce_yoy_change", "core_pce_mom_change"]
                    pce_cols_new = pce_cols_new[:len(pce_cols)]
                    pce_df.columns = pce_cols_new
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
                    fed_df["date"] = pd.to_datetime(fed_df["date"])
                    fed_df = fed_df[["date", "value", "rate_change"]]
                    fed_df.columns = ["date", "fed_funds_rate", "fed_funds_change"]
                    fed_df.set_index("date", inplace=True)
                    
                    if combined_df is None:
                        combined_df = fed_df
                    else:
                        combined_df = combined_df.join(fed_df, how="outer")
                        
                # 美联储资产负债表
                if "fed_assets" in macro_data["fed_policy"] and macro_data["fed_policy"]["fed_assets"] is not None:
                    assets_df = macro_data["fed_policy"]["fed_assets"].copy()
                    assets_df["date"] = pd.to_datetime(assets_df["date"])
                    assets_df = assets_df[["date", "value", "pct_change"]]
                    assets_df.columns = ["date", "fed_assets", "fed_assets_pct_change"]
                    assets_df.set_index("date", inplace=True)
                    
                    if combined_df is None:
                        combined_df = assets_df
                    else:
                        combined_df = combined_df.join(assets_df, how="outer")
            
            # 重置索引
            if combined_df is not None:
                combined_df.reset_index(inplace=True)
                
                # 确保日期是日期类型
                combined_df["date"] = pd.to_datetime(combined_df["date"])
                
                # 排序并填充缺失值
                combined_df.sort_values("date", inplace=True)
                
                # 前向填充月度/季度数据
                combined_df = combined_df.fillna(method='ffill')
                
                # 保存到CSV
                filename = get_filename("macro", "combined_data")
                combined_df.to_csv(filename, index=False)
                logger.info(f"综合宏观数据已保存到 {filename}，包含 {len(combined_df)} 条记录")
                
                return combined_df
            else:
                logger.error("没有可用的宏观数据来创建综合数据集")
                return None
        
        except Exception as e:
            logger.error(f"创建综合宏观数据时发生错误: {str(e)}")
            return None

class GoldPriceDataCollector:
    """用于收集黄金价格数据的工具"""
    
    def __init__(self, history_file=os.path.join(DATA_DIR, "gold_history.json")):
        """
        初始化数据收集器
        
        参数:
            history_file (str): 历史数据文件路径
        """
        self.data_cache = {}
        self.history_file = history_file
        self.history_data = self.load_history()
        logger.info("黄金价格数据收集器已初始化")
    
    def load_history(self):
        """加载历史数据文件"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                logger.info(f"已加载历史数据，包含 {len(history)} 条记录")
                return history
            except Exception as e:
                logger.error(f"加载历史数据时出错: {str(e)}")
                return []
        else:
            logger.info("未找到历史数据文件，将创建新文件")
            return []
    
    def save_to_history(self):
        """将当前数据保存到历史记录"""
        # 添加时间戳
        self.data_cache["record_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 将当前数据添加到历史记录
        self.history_data.append(self.data_cache.copy())
        
        # 只保留最近30天的数据
        if len(self.history_data) > 30:
            self.history_data = self.history_data[-30:]
        
        # 保存到文件
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history_data, f, indent=4, ensure_ascii=False)
            logger.info(f"数据已保存到历史记录，当前有 {len(self.history_data)} 条记录")
            return True
        except Exception as e:
            logger.error(f"保存历史数据时出错: {str(e)}")
            return False
    
    def get_spot_gold_price(self):
        """获取当前黄金现货价格(XAU/USD)"""
        try:
            # 使用Yahoo Finance API获取黄金价格
            url = "https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers)
            data = response.json()
            
            if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
                result = data["chart"]["result"][0]
                
                # 获取最新价格
                latest_price = result["meta"]["regularMarketPrice"]
                timestamp = result["meta"]["regularMarketTime"]
                
                # 将时间戳转换为日期时间
                dt = datetime.fromtimestamp(timestamp)
                
                result = {
                    "price": latest_price,
                    "timestamp": dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": "Yahoo Finance"
                }
                
                self.data_cache["spot_gold"] = result
                logger.info(f"获取到黄金现货价格: ${latest_price}")
                return result
            else:
                logger.error(f"获取黄金现货价格出错: {data}")
                return None
                
        except Exception as e:
            logger.error(f"获取黄金现货价格时发生错误: {str(e)}")
            return None
    
    def get_historical_gold_data(self, symbol="GC=F", period1=None, period2=None, interval="1d"):
        """
        获取黄金历史价格数据
        
        参数:
            symbol (str): 获取数据的代码，默认为"GC=F"（黄金期货）
            period1 (str或datetime): 开始日期，格式为YYYY-MM-DD或datetime对象，默认为1年前
            period2 (str或datetime): 结束日期，格式为YYYY-MM-DD或datetime对象，默认为今天
            interval (str): 数据间隔，选项有："1d"、"1wk"、"1mo"（日、周、月）
            
        返回:
            DataFrame: 历史价格数据
        """
        try:
            # 设置默认日期（如果未提供）
            if period2 is None:
                period2 = datetime.now()
            elif isinstance(period2, str):
                period2 = datetime.strptime(period2, "%Y-%m-%d")
                
            if period1 is None:
                # 扩展时间范围，默认获取5年数据
                period1 = period2 - timedelta(days=5*365)
            elif isinstance(period1, str):
                period1 = datetime.strptime(period1, "%Y-%m-%d")
            
            # 将日期转换为Unix时间戳（自纪元以来的秒数）
            period1_unix = int(period1.timestamp())
            period2_unix = int(period2.timestamp())
            
            # 构建Yahoo Finance API的URL
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
            
            logger.info(f"获取{symbol}从{period1.strftime('%Y-%m-%d')}到{period2.strftime('%Y-%m-%d')}的历史数据")
            response = requests.get(url, params=params, headers=headers)
            data = response.json()
            
            if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
                result = data["chart"]["result"][0]
                
                # 提取时间戳和价格数据
                timestamps = result["timestamp"]
                
                # 提取OHLC数据
                quote = result["indicators"]["quote"][0]
                opens = quote.get("open", [])
                highs = quote.get("high", [])
                lows = quote.get("low", [])
                closes = quote.get("close", [])
                volumes = quote.get("volume", [])
                
                # 创建DataFrame
                df = pd.DataFrame({
                    "timestamp": timestamps,
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": closes,
                    "volume": volumes
                })
                
                # 处理可能的NaN值
                df = df.dropna()
                
                # 将时间戳转换为日期
                df["date"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
                
                # 转换日期为datetime类型以便后续处理
                df["date"] = pd.to_datetime(df["date"])
                
                # 添加技术指标
                # 1. 移动平均线
                for window in [20, 50, 200]:
                    df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
                
                # 2. RSI (相对强弱指标)
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # 3. MACD (移动平均收敛/发散)
                df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
                df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = df['ema_12'] - df['ema_26']
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
                df['macd_hist'] = df['macd'] - df['macd_signal']
                
                # 4. 布林带
                df['ma_20'] = df['close'].rolling(window=20).mean()
                df['upper_band'] = df['ma_20'] + (df['close'].rolling(window=20).std() * 2)
                df['lower_band'] = df['ma_20'] - (df['close'].rolling(window=20).std() * 2)
                
                # 5. 成交量变化
                df['volume_change'] = df['volume'].pct_change() * 100
                
                # 6. 波动率（用收盘价的标准差）
                df['volatility'] = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean() * 100
                
                # 7. 价格趋势（基于移动平均线）
                df['trend_ma_20_50'] = np.where(df['ma_20'] > df['ma_50'], 1, -1)
                
                # 8. 百分比价格震荡指标 (PPO)
                df['ppo'] = ((df['ema_12'] - df['ema_26']) / df['ema_26']) * 100
                
                # 9. 价格变化率
                df['price_change'] = df['close'].pct_change() * 100
                
                # 10. 累积收益
                df['cumulative_return'] = (1 + df['price_change'] / 100).cumprod() - 1
                
                # 重新排序列，将日期放在前面
                cols = ["date", "open", "high", "low", "close", "volume", "timestamp"]
                tech_cols = [col for col in df.columns if col not in cols]
                df = df[cols + tech_cols]
                
                # 保存到CSV
                filename = get_filename("price", f"{symbol}_{interval}")
                df.to_csv(filename, index=False)
                logger.info(f"历史数据已保存到 {filename} ({len(df)} 条记录)")
                
                # 同时保存一个固定文件名的版本，便于其他程序引用
                fixed_filename = os.path.join(DATA_DIR, f"{symbol}_{interval}_data.csv")
                df.to_csv(fixed_filename, index=False)
                logger.info(f"历史数据也保存到固定文件 {fixed_filename}")
                
                return df
            else:
                logger.error(f"获取历史数据出错: {data}")
                return None
                
        except Exception as e:
            logger.error(f"获取黄金历史数据时发生错误: {str(e)}")
            return None
    
    def get_minute_data(self, symbol="GLD"):
        """
        获取分钟级黄金价格数据
        
        参数:
            symbol (str): 股票代码，默认是"GLD"
            
        返回:
            DataFrame: 包含分钟级数据的DataFrame
        """
        try:
            # 计算时间戳（Yahoo Finance使用秒级Unix时间戳）
            end_time = int(time.time())
            # 从当天开始时间获取
            today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = int(today.timestamp())
            
            # Yahoo Finance API URL用于历史数据
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            
            # 设置分钟级数据的参数
            params = {
                "period1": start_time,
                "period2": end_time,
                "interval": "1m",        # 1分钟间隔
                "includePrePost": "true"  # 包括盘前和盘后数据
            }
            
            # 设置模拟浏览器的请求头
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, params=params, headers=headers)
            data = response.json()
            
            if "chart" not in data or "result" not in data["chart"] or not data["chart"]["result"]:
                logger.error("没有可用数据或从Yahoo Finance收到无效响应")
                return None
            
            # 提取时间戳和价格数据
            result = data["chart"]["result"][0]
            timestamps = result["timestamp"]
            quotes = result["indicators"]["quote"][0]
            
            # 创建DataFrame
            df = pd.DataFrame({
                "timestamp": timestamps,
                "open": quotes.get("open", []),
                "high": quotes.get("high", []),
                "low": quotes.get("low", []),
                "close": quotes.get("close", []),
                "volume": quotes.get("volume", [])
            })
            
            # 移除任何NaN值
            df = df.dropna()
            
            # 添加可读的时间列
            df["datetime"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d %H:%M:%S"))
            df["date"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x).strftime("%Y-%m-%d"))
            df["time"] = df["timestamp"].apply(lambda x: datetime.fromtimestamp(x).strftime("%H:%M:%S"))
            
            # 添加分钟级技术指标
            if len(df) > 20:
                # 短周期移动平均线
                df['ma_5'] = df['close'].rolling(window=5).mean()
                df['ma_10'] = df['close'].rolling(window=10).mean()
                df['ma_20'] = df['close'].rolling(window=20).mean()
                
                # 价格变化率
                df['price_change'] = df['close'].pct_change() * 100
                
                # 短周期RSI
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))
                
                # 检测异常波动
                df = DataProcessor.detect_anomalies(df, 'price_change', window=10, threshold=3)
            
            # 保存到CSV
            filename = get_filename("price", f"{symbol}_minute")
            df.to_csv(filename, index=False)
            logger.info(f"分钟级数据已保存到 {filename} ({len(df)} 条记录)")
            
            # 同时保存一个固定文件名的版本
            fixed_filename = os.path.join(DATA_DIR, f"{symbol}_minute_data.csv")
            df.to_csv(fixed_filename, index=False)
            logger.info(f"分钟级数据也保存到固定文件 {fixed_filename}")
            
            return df
            
        except Exception as e:
            logger.error(f"获取分钟级数据时发生错误: {str(e)}")
            return None
    
    def create_consolidated_dataset(self, symbols=None, start_date=None, end_date=None, interval="1d"):
        """
        为多个代码创建合并的历史数据集
        
        参数:
            symbols (list): 包含的代码列表。默认是GC=F和GLD
            start_date (str或datetime): 开始日期。默认是5年前
            end_date (str或datetime): 结束日期。默认是今天
            interval (str): 数据间隔（'1d', '1wk', '1mo'）
            
        返回:
            DataFrame: 合并的数据集
        """
        if symbols is None:
            symbols = ["GC=F", "GLD", "DX-Y.NYB"]  # 黄金期货、黄金ETF、美元指数
            
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        if start_date is None:
            start_date = end_date - timedelta(days=5*365)  # 5年的数据
        elif isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        # 存储每个代码的dataframes的字典
        dfs = {}
        
        # 获取每个代码的数据
        for symbol in symbols:
            logger.info(f"获取{symbol}的历史数据")
            
            df = self.get_historical_gold_data(
                symbol=symbol,
                period1=start_date,
                period2=end_date,
                interval=interval
            )
            
            if df is not None:
                # 以代码为键存储dataframe
                dfs[symbol] = df
                
                # 添加代码列
                df['symbol'] = symbol
                
                # 添加代码描述
                symbol_descriptions = {
                    "GC=F": "Gold Futures",
                    "GLD": "SPDR Gold Trust ETF",
                    "IAU": "iShares Gold Trust",
                    "DX-Y.NYB": "US Dollar Index",
                    "XAUUSD=X": "Gold Spot Price"
                }
                
                df['description'] = symbol_descriptions.get(symbol, symbol)
        
        if not dfs:
            logger.error("没有为任何代码检索到数据")
            return None
            
        # 创建以日期为索引的合并dataframe
        consolidated_df = None
        
        for symbol, df in dfs.items():
            # 将索引设置为日期列
            df.set_index('date', inplace=True)
            
            # 创建带有代码前缀的列名
            price_columns = {
                'open': f'{symbol}_open',
                'high': f'{symbol}_high',
                'low': f'{symbol}_low',
                'close': f'{symbol}_close',
                'volume': f'{symbol}_volume'
            }
            
            # 重命名列
            df = df.rename(columns=price_columns)
            
            # 选择要保留的列
            cols_to_keep = [f'{symbol}_open', f'{symbol}_high', f'{symbol}_low', f'{symbol}_close', f'{symbol}_volume']
            
            # 如果有技术指标，也加入
            tech_cols = [col for col in df.columns if col.startswith(('ma_', 'rsi', 'macd', 'volume_change', 'volatility'))]
            # 给技术指标加上前缀
            tech_cols_renamed = {}
            for col in tech_cols:
                tech_cols_renamed[col] = f'{symbol}_{col}'
            df = df.rename(columns=tech_cols_renamed)
            cols_to_keep.extend(list(tech_cols_renamed.values()))
            
            # 只保留选定的列
            df = df[cols_to_keep]
            
            if consolidated_df is None:
                consolidated_df = df
            else:
                # 与现有的合并dataframe连接
                consolidated_df = consolidated_df.join(df, how='outer')
        
        # 重置索引以使日期再次成为列
        consolidated_df.reset_index(inplace=True)
        
        # 添加计算列
        if 'GC=F_close' in consolidated_df.columns and 'GLD_close' in consolidated_df.columns:
            # 计算ETF相对于期货的溢价率
            consolidated_df['GLD_premium_pct'] = (consolidated_df['GLD_close'] - consolidated_df['GC=F_close']) / consolidated_df['GC=F_close'] * 100
            
            # 检测异常溢价
            consolidated_df = DataProcessor.detect_anomalies(consolidated_df, 'GLD_premium_pct', window=20, threshold=2)
            
        # 计算黄金和美元指数的相关性（滚动30天）
        if 'GC=F_close' in consolidated_df.columns and 'DX-Y.NYB_close' in consolidated_df.columns:
            # 确保没有NaN值
            tmp_df = consolidated_df[['GC=F_close', 'DX-Y.NYB_close']].copy()
            tmp_df = tmp_df.fillna(method='ffill').fillna(method='bfill')
            
            # 计算滚动相关性
            if len(tmp_df) > 30:
                tmp_df['corr_gold_usd'] = tmp_df['GC=F_close'].rolling(window=30).corr(tmp_df['DX-Y.NYB_close'])
                consolidated_df['corr_gold_usd'] = tmp_df['corr_gold_usd']
        
        # 先使用前向填充再使用后向填充来填充缺失值
        consolidated_df = consolidated_df.fillna(method='ffill').fillna(method='bfill')
        
        # 添加日期特征
        consolidated_df['date'] = pd.to_datetime(consolidated_df['date'])
        consolidated_df['year'] = consolidated_df['date'].dt.year
        consolidated_df['month'] = consolidated_df['date'].dt.month
        consolidated_df['day'] = consolidated_df['date'].dt.day
        consolidated_df['day_of_week'] = consolidated_df['date'].dt.dayofweek
        consolidated_df['is_month_end'] = consolidated_df['date'].dt.is_month_end
        consolidated_df['is_quarter_end'] = consolidated_df['date'].dt.is_quarter_end
        
        # 保存到CSV
        filename = get_filename("price", "consolidated_data")
        consolidated_df.to_csv(filename, index=False)
        logger.info(f"合并的历史数据已保存到 {filename} ({len(consolidated_df)} 条记录)")
        
        # 同时保存一个固定文件名的版本
        fixed_filename = os.path.join(DATA_DIR, "gold_consolidated_data.csv")
        consolidated_df.to_csv(fixed_filename, index=False)
        logger.info(f"合并的历史数据也保存到固定文件 {fixed_filename}")
        
        return consolidated_df
    
    def analyze_gold_etf_relationship(self):
        """
        分析黄金期货与ETF之间的关系
        
        返回:
            DataFrame: 包含分析结果的DataFrame
        """
        try:
            logger.info("分析黄金期货与ETF关系...")
            
            # 获取期货和ETF数据
            futures_df = self.get_historical_gold_data(symbol="GC=F")
            etf_df = self.get_historical_gold_data(symbol="GLD")
            
            if futures_df is None or etf_df is None:
                logger.error("无法获取期货或ETF数据")
                return None
            
            # 计算ETF溢价
            analysis_df = DataProcessor.calculate_gold_etf_premium(futures_df, etf_df)
            
            if analysis_df is not None:
                # 计算统计信息
                avg_premium = analysis_df['premium_pct'].mean()
                std_premium = analysis_df['premium_pct'].std()
                min_premium = analysis_df['premium_pct'].min()
                max_premium = analysis_df['premium_pct'].max()
                
                logger.info(f"ETF平均溢价率: {avg_premium:.2f}%, 标准差: {std_premium:.2f}%")
                logger.info(f"ETF溢价率范围: {min_premium:.2f}% to {max_premium:.2f}%")
                
                # 保存分析结果
                filename = get_filename("analysis", "gold_etf_premium")
                analysis_df.to_csv(filename, index=False)
                logger.info(f"黄金ETF溢价分析已保存到 {filename}")
                
                # 绘制溢价图表
                plt.figure(figsize=(12, 6))
                plt.plot(analysis_df['date'], analysis_df['premium_pct'])
                plt.axhline(y=avg_premium, color='r', linestyle='--', label=f'平均值: {avg_premium:.2f}%')
                plt.axhline(y=avg_premium + 2*std_premium, color='g', linestyle='--', label=f'上限 (+2σ): {avg_premium + 2*std_premium:.2f}%')
                plt.axhline(y=avg_premium - 2*std_premium, color='g', linestyle='--', label=f'下限 (-2σ): {avg_premium - 2*std_premium:.2f}%')
                
                # 标记异常值
                if 'premium_pct_is_anomaly' in analysis_df.columns:
                    anomalies = analysis_df[analysis_df['premium_pct_is_anomaly']]
                    if not anomalies.empty:
                        plt.scatter(anomalies['date'], anomalies['premium_pct'], color='red', s=50, label='异常值')
                
                plt.title('黄金ETF(GLD)相对于期货(GC=F)的溢价率')
                plt.xlabel('日期')
                plt.ylabel('溢价率 (%)')
                plt.legend()
                plt.grid(True)
                
                # 保存图表
                chart_filename = os.path.join(DATA_DIR, "gold_etf_premium_chart.png")
                plt.savefig(chart_filename)
                plt.close()
                logger.info(f"黄金ETF溢价图表已保存到 {chart_filename}")
                
                return analysis_df
            else:
                return None
                
        except Exception as e:
            logger.error(f"分析黄金期货与ETF关系时出错: {str(e)}")
            return None
    
    def get_all_price_data(self):
        """
        获取所有价格数据
        
        返回:
            dict: 包含所有价格数据的字典
        """
        logger.info("开始收集黄金价格数据...")
        
        price_data = {}
        
        # 获取实时现货价格
        price_data["spot_price"] = self.get_spot_gold_price()
        
        # 获取GLD的分钟级数据
        price_data["minute_data"] = self.get_minute_data()
        
        # 获取多个市场的历史数据
        symbols = ["GC=F", "GLD", "DX-Y.NYB", "IAU"]
        symbol_data = {}
        
        for symbol in symbols:
            symbol_data[symbol] = self.get_historical_gold_data(
                symbol=symbol,
                period1=datetime.now() - timedelta(days=5*365),  # 5年数据
                period2=datetime.now(),
                interval="1d"
            )
        
        price_data["historical_data"] = symbol_data
        
        # 创建合并数据集
        consolidated_data = self.create_consolidated_dataset()
        price_data["consolidated_data"] = consolidated_data
        
        # 分析黄金期货与ETF之间的关系
        etf_analysis = self.analyze_gold_etf_relationship()
        price_data["etf_analysis"] = etf_analysis
        
        # 将当前价格数据保存到历史记录
        if price_data["spot_price"]:
            self.save_to_history()
            
        logger.info("黄金价格数据收集完成")
        
        return price_data

class CNNFearGreedScraper:
    """精简版CNN恐惧与贪婪指数爬虫类"""
    
    def __init__(self):
        """初始化CNN恐惧与贪婪指数爬虫"""
        # 只保留有效的API端点
        self.api_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        logger.info("CNN恐惧与贪婪指数爬虫已初始化")
    
    def get_fear_greed_data(self):
        """获取CNN恐惧与贪婪指数数据"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36"
            }
            
            response = requests.get(self.api_url, headers=headers)
            
            if response.status_code == 200:
                response_text = response.text
                logger.info(f"API响应内容: {response_text[:100]}...")
                
                # 解析JSON
                data = json.loads(response_text)
                
                # 提取当前值和分类
                current_value = None
                current_classification = None
                
                if "fear_and_greed" in data:
                    fg_data = data["fear_and_greed"]
                    if isinstance(fg_data, dict):
                        current_value = fg_data.get("score")
                        current_classification = fg_data.get("rating")
                
                if current_value is not None:
                    current_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "value": current_value,
                        "classification": current_classification or self._get_classification(current_value),
                        "source": "CNN Fear & Greed API"
                    }
                    
                    # 获取历史数据
                    historical_data = []
                    if "fear_and_greed_historical" in data:
                        for item in data["fear_and_greed_historical"]:
                            if "x" in item and "y" in item:
                                # 转换时间戳为日期时间
                                date_str = item["x"]
                                try:
                                    # 尝试解析日期格式
                                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                                    formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                                except:
                                    # 如果解析失败，使用原始日期字符串
                                    formatted_date = date_str
                                
                                historical_data.append({
                                    "date": formatted_date,
                                    "value": item["y"],
                                    "classification": self._get_classification(item["y"])
                                })
                    
                    result = {
                        "current": current_data,
                        "historical": historical_data
                    }
                    
                    logger.info(f"从API获取到恐惧与贪婪指数: {current_data['value']} ({current_data['classification']})")
                    return result
                else:
                    logger.error("API响应中未找到恐惧与贪婪指数值")
            else:
                logger.error(f"API请求失败，状态码: {response.status_code}")
            
            # 如果API获取失败，使用默认值
            return self._get_default_data()
                
        except Exception as e:
            logger.error(f"获取恐惧与贪婪指数数据时出错: {str(e)}")
            # 如果出现异常，使用默认值
            return self._get_default_data()
    
    def _get_default_data(self):
        """返回默认的恐惧与贪婪指数数据"""
        logger.warning("使用默认数据")
        
        current_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "value": 50,  # 中性值
            "classification": "Neutral",
            "source": "Default Value"
        }
        
        return {
            "current": current_data,
            "historical": []
        }
    
    def _get_classification(self, value):
        """根据指数值获取对应的分类"""
        if value is None:
            return "Unknown"
            
        value = float(value)
        
        if value <= 25:
            return "Extreme Fear"
        elif value <= 45:
            return "Fear"
        elif value <= 55:
            return "Neutral"
        elif value <= 75:
            return "Greed"
        else:
            return "Extreme Greed"
    
    def save_data_to_csv(self, data):
        """将恐惧与贪婪指数数据保存到CSV文件"""
        try:
            if not data or "current" not in data:
                logger.error("没有可保存的数据")
                return False
                
            # 保存当前数据
            current_data = data["current"]
            current_df = pd.DataFrame([current_data])
            
            current_filename = get_filename("sentiment", "fear_greed_current")
            current_df.to_csv(current_filename, index=False)
            logger.info(f"当前恐惧与贪婪指数数据已保存到 {current_filename}")
            
            # 如果有历史数据，也保存
            if "historical" in data and data["historical"]:
                historical_data = data["historical"]
                historical_df = pd.DataFrame(historical_data)
                
                historical_filename = get_filename("sentiment", "fear_greed_historical")
                historical_df.to_csv(historical_filename, index=False)
                logger.info(f"历史恐惧与贪婪指数数据已保存到 {historical_filename}")
            
            # 将当前数据也保存到累积CSV文件中，包含时间信息
            cumulative_filename = os.path.join(DATA_DIR, "fear_greed_cumulative.csv")
            
            # 扩展当前数据，添加更多时间信息
            extended_data = current_data.copy()
            
            # 添加多种时间格式用于分析
            now = datetime.now()
            extended_data["date"] = now.strftime("%Y-%m-%d")
            extended_data["year"] = now.year
            extended_data["month"] = now.month
            extended_data["day"] = now.day
            extended_data["day_of_week"] = now.weekday()
            extended_data["hour"] = now.hour
            
            if os.path.exists(cumulative_filename):
                try:
                    cumulative_df = pd.read_csv(cumulative_filename)
                    # 添加新数据
                    cumulative_df = pd.concat([cumulative_df, pd.DataFrame([extended_data])], ignore_index=True)
                except Exception as e:
                    logger.error(f"读取累积数据文件时出错: {str(e)}")
                    cumulative_df = pd.DataFrame([extended_data])
            else:
                cumulative_df = pd.DataFrame([extended_data])
            
            # 保存累积数据
            cumulative_df.to_csv(cumulative_filename, index=False)
            logger.info(f"累积恐惧与贪婪指数数据已更新到 {cumulative_filename}")
            
            return True
                
        except Exception as e:
            logger.error(f"保存恐惧与贪婪指数数据时出错: {str(e)}")
            return False
    
    def run(self):
        """运行爬虫主程序"""
        logger.info("开始收集CNN恐惧与贪婪指数数据...")
        
        # 获取数据
        data = self.get_fear_greed_data()
        
        # 保存数据
        if data:
            self.save_data_to_csv(data)
            logger.info("CNN恐惧与贪婪指数数据收集完成")
            return data
        else:
            logger.error("未能获取CNN恐惧与贪婪指数数据")
            return None

# 主程序
def main():
    """主函数：收集所有数据"""
    # 创建收集器
    macro_collector = GoldMacroDataCollector()
    price_collector = GoldPriceDataCollector()
    fear_greed_scraper = CNNFearGreedScraper()
    
    # 收集宏观经济数据
    logger.info("开始收集宏观经济数据...")
    macro_data = macro_collector.get_all_macro_data()
    
    # 收集黄金价格数据
    logger.info("开始收集黄金价格数据...")
    price_data = price_collector.get_all_price_data()
    
    # 收集恐惧与贪婪指数数据
    logger.info("开始收集市场情绪数据...")
    fear_greed_data = fear_greed_scraper.run()
    
    # 创建整合数据集
    logger.info("创建整合数据集...")
    integrated_data = DataProcessor.create_integrated_dataset(
        macro_data=macro_data,
        price_data={"price_data": price_data.get("consolidated_data")},
        fear_greed_data=fear_greed_data
    )
    
    # 保存整合数据集
    if integrated_data is not None:
        integrated_filename = os.path.join(DATA_DIR, "gold_integrated_dataset.csv")
        integrated_data.to_csv(integrated_filename, index=False)
        logger.info(f"整合数据集已保存到 {integrated_filename}")
    
    logger.info("数据收集与分析完成，所有数据已保存到 %s 目录", DATA_DIR)
    
    return {
        "macro_data": macro_data,
        "price_data": price_data,
        "fear_greed_data": fear_greed_data,
        "integrated_data": integrated_data
    }

if __name__ == "__main__":
    main()