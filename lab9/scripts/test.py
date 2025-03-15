import sqlite3

def call_db():
    # 连接到 SQLite 数据库（如果不存在则会创建一个）
    conn = sqlite3.connect("pdf_texts.db")
    cursor = conn.cursor()
    
    # 执行查询，获取所有记录
    cursor.execute("SELECT * FROM pdf_texts")
    results = cursor.fetchall()
    
    # 遍历并打印每条记录（这里只显示前 200 个字符）
    for row in results:
        print("ID:", row[0])
        print("Filename:", row[1])
        print("Content:", row[2][:10000])  # 仅打印部分内容
        print("=" * 50)
    
    # 关闭数据库连接
    conn.close()

if __name__ == "__main__":
    call_db()
