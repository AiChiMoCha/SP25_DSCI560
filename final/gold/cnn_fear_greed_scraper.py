import os
import requests
import pandas as pd
import json
import logging
from datetime import datetime, timedelta
import time

# 设置统一的时间格式和数据目录 - 与之前的代码保持一致
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
        logging.FileHandler(os.path.join(DATA_DIR, "fear_greed_data.log")), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("CNNFearGreedScraper")

def get_timestamp():
    """获取当前时间戳，格式为YYYY-MM-DD_HH-MM-SS"""
    return datetime.now().strftime(TIMESTAMP_FORMAT)

def get_filename(data_type, suffix=""):
    """生成统一格式的文件名"""
    timestamp = get_timestamp()
    return os.path.join(DATA_DIR, f"{data_type}_{suffix}_{timestamp}.csv")

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
            
            # 将当前数据也保存到累积CSV文件中，包含时间戳
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
if __name__ == "__main__":
    # 创建CNN恐惧与贪婪指数爬虫
    fear_greed_scraper = CNNFearGreedScraper()
    
    # 运行爬虫
    data = fear_greed_scraper.run()
    
    # 显示结果
    if data and "current" in data:
        current = data["current"]
        print(f"\n当前恐惧与贪婪指数: {current['value']} - {current['classification']}")
        print(f"数据来源: {current['source']}")
        print(f"数据已保存到 {DATA_DIR} 目录")
        
        # 显示历史数据数量
        if "historical" in data and data["historical"]:
            print(f"历史数据点数量: {len(data['historical'])}")
    else:
        print("\n未能获取恐惧与贪婪指数数据")