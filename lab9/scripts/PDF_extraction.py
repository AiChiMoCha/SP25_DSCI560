import os
import sqlite3
from PyPDF2 import PdfReader

def extract_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            # 如果某些页面可能无法提取文本，返回值会是 None，需要加上 or ""
            text += page.extract_text() or ""
    return text

def store_text_in_db(filename, content):
    # 创建或连接本地数据库
    conn = sqlite3.connect("pdf_texts.db")
    cursor = conn.cursor()
    
    # 如果表不存在，就新建一个表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content TEXT
        )
    ''')
    
    # 插入数据
    cursor.execute(
        'INSERT INTO pdf_texts (filename, content) VALUES (?, ?)',
        (filename, content)
    )
    
    conn.commit()
    conn.close()

def main():
    # 指定 PDF 文件所在目录，这里假设和脚本在同级的 data 文件夹
    folder_path = os.path.join(os.path.dirname(__file__), "../data")
    
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, file_name)
            extracted_text = extract_pdf_text(pdf_path)
            store_text_in_db(file_name, extracted_text)
            print(f"Stored {file_name} in the database.")

if __name__ == "__main__":
    main()
