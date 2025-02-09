import requests
from bs4 import BeautifulSoup
import csv

def scrape_stock_trades(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("请求失败，状态码：", response.status_code)
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    trades = []
    
    # 假设交易数据放在一个表格中，每行代表一条记录
    table = soup.find("table")
    if table:
        tbody = table.find("tbody")
        if tbody:
            rows = tbody.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                # 根据你提供的信息，每行至少应有6个单元格
                if len(cols) >= 6:
                    traded_issuer = cols[0].get_text(strip=True)
                    published      = cols[1].get_text(strip=True)
                    traded         = cols[2].get_text(strip=True)
                    filed_after    = cols[3].get_text(strip=True)
                    type_info      = cols[4].get_text(strip=True)
                    size           = cols[5].get_text(strip=True)
                    trades.append([traded_issuer, published, traded, filed_after, type_info, size])
    else:
        print("未找到表格结构，请检查页面的 HTML 结构。")
    
    return trades

def save_to_csv(data, filename):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Traded Issuer", "Published", "Traded", "Filed After", "Type", "Size"])
        writer.writerows(data)
    print("数据已保存到", filename)

if __name__ == "__main__":
    url = "https://www.capitoltrades.com/politicians/G000590?pageSize=96"
    trades_data = scrape_stock_trades(url)
    if trades_data:
        save_to_csv(trades_data, "Green.csv")
    else:
        print("未提取到任何数据")
