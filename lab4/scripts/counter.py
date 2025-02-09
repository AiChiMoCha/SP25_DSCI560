import pandas as pd
import glob
import re

def extract_stock_symbols(file_paths):
    stock_symbols = set()
    
    for file in file_paths:
        df = pd.read_csv(file)
        
        if "Traded Issuer" not in df.columns:
            print(f"Warning: 'Traded Issuer' column not found in {file}")
            continue
        
        # 提取股票代码（去掉公司名称，仅保留 :US 之前的股票代码）
        symbols = df["Traded Issuer"].dropna().apply(lambda x: x.split(":")[0].split()[-1] if ":US" in x else None)
        
        # 去除空值并添加到集合去重
        stock_symbols.update(symbols.dropna().tolist())

    return list(stock_symbols)

# 读取当前目录下所有 CSV 文件
csv_files = glob.glob("*.csv")

# 获取去重后的股票代码列表
stock_list = extract_stock_symbols(csv_files)
cleaned_symbols = [re.sub(r'^(Inc|Corp|LP|LLC|Fund|Trust|Co|Corporation)', '', symbol).strip('/') for symbol in stock_list]
print("去重后的股票代码列表：")
print(cleaned_symbols)
