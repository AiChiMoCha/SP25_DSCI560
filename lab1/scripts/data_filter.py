import os
import csv
import logging
from bs4 import BeautifulSoup

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置路径
class Config:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "../data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
    WEB_DATA_FILE = os.path.join(RAW_DATA_DIR, "web_data.html")
    MARKET_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "market_data.csv")
    NEWS_DATA_CSV = os.path.join(PROCESSED_DATA_DIR, "news_data.csv")

# 确保文件夹存在
os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)

def parse_market_banner(soup):
    """
    提取市场横幅数据。
    """
    logging.info("Extracting market banner data...")
    market_banner = soup.find("div", class_="MarketsBanner-marketData")
    if not market_banner:
        logging.error("Market banner not found in the HTML file.")
        return []

    market_data = []
    for market_card in market_banner.find_all("a", class_="MarketCard-container"):
        symbol = market_card.find("span", class_="MarketCard-symbol")
        stock_position = market_card.find("span", class_="MarketCard-stockPosition")
        change_pct = market_card.find("span", class_="MarketCard-changePct")

        market_data.append({
            "symbol": symbol.text.strip() if symbol else "N/A",
            "stock_position": stock_position.text.strip() if stock_position else "N/A",
            "change_pct": change_pct.text.strip() if change_pct else "N/A",
        })

    return market_data

def parse_latest_news(soup):
    """
    提取最新新闻数据。
    """
    logging.info("Extracting latest news data...")
    latest_news_list = soup.find("ul", class_="LatestNews-list")
    if not latest_news_list:
        logging.error("Latest news list not found in the HTML file.")
        return []

    news_data = []
    for news_item in latest_news_list.find_all("li"):
        timestamp = news_item.find("time")
        title = news_item.find("a", class_="LatestNews-headline")
        link = news_item.find("a", class_="LatestNews-headline")

        news_data.append({
            "timestamp": timestamp.text.strip() if timestamp else "N/A",
            "title": title.text.strip() if title else "N/A",
            "link": link["href"].strip() if link and link.has_attr("href") else "N/A",
        })

    return news_data

def save_to_csv(data, file_path, fieldnames):
    """
    将数据保存为 CSV 文件。
    """
    logging.info(f"Saving data to {file_path}...")
    with open(file_path, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    logging.info(f"CSV file created at {file_path}")

if __name__ == "__main__":
    try:
        logging.info("Reading web_data.html file...")
        with open(Config.WEB_DATA_FILE, "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "html.parser")

        # 提取市场横幅数据
        market_data = parse_market_banner(soup)
        if market_data:
            save_to_csv(
                market_data,
                Config.MARKET_DATA_CSV,
                fieldnames=["symbol", "stock_position", "change_pct"]
            )

        # 提取最新新闻数据
        news_data = parse_latest_news(soup)
        if news_data:
            save_to_csv(
                news_data,
                Config.NEWS_DATA_CSV,
                fieldnames=["timestamp", "title", "link"]
            )

    except Exception as e:
        logging.error(f"An error occurred: {e}")
