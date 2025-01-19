import os
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置路径
class Config:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(ROOT_DIR, "../data")
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
    BASE_URL = "https://www.cnbc.com/world/?region=world"

# 确保文件夹存在
os.makedirs(Config.RAW_DATA_DIR, exist_ok=True)
os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)

# 初始化 Selenium WebDriver
def initialize_driver():
    """
    初始化 ChromeDriver，添加必要的选项。
    """
    options = Options()
    options.add_argument("--headless")  # 无头模式，适合服务器环境
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.binary_location = "/usr/bin/chromium-browser"  # 指定 Chromium 浏览器路径
    options.add_argument(f"--user-data-dir=/tmp/selenium_user_data_{os.getpid()}")  # 临时目录避免冲突

    logging.info("Initializing ChromeDriver...")
    driver = webdriver.Chrome(options=options)
    return driver

# 保存 HTML 内容到本地
def save_html_content(page_source, save_path):
    """
    从页面中提取 Market Banner 和 Latest News 并保存到 HTML 文件中。
    """
    soup = BeautifulSoup(page_source, "html.parser")

    # 提取 Market Banner 和 Latest News
    logging.info("Extracting Market Banner and Latest News...")
    market_banner = soup.find("div", class_="MarketsBanner-marketData")
    latest_news = soup.find("ul", class_="LatestNews-list")

    if not market_banner or not latest_news:
        logging.error("Could not find required elements on the page!")
        return False

    # 保存到文件
    with open(save_path, "w", encoding="utf-8") as file:
        file.write("=== Market Banner ===\n")
        file.write(market_banner.prettify())
        file.write("\n\n=== Latest News ===\n")
        file.write(latest_news.prettify())

    logging.info(f"HTML content saved to {save_path}")
    return True

# 主函数
if __name__ == "__main__":
    driver = initialize_driver()
    save_path = os.path.join(Config.RAW_DATA_DIR, "web_data.html")

    try:
        logging.info("Fetching page content...")
        driver.get(Config.BASE_URL)

        # 等待页面加载完成
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "MarketsBanner-marketData"))
        )

        # 保存 HTML 内容
        if save_html_content(driver.page_source, save_path):
            # 打印前十行
            logging.info("Printing the first ten lines of the saved HTML file:")
            with open(save_path, "r", encoding="utf-8") as file:
                for _ in range(10):
                    print(file.readline().strip())

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        driver.quit()
