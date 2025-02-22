from sqlalchemy import create_engine, text

# 使用提供的 MySQL 连接字符串
db_url = "mysql+pymysql://DSCI560:560560@172.16.161.128/"
engine = create_engine(db_url)

# 指定要创建的数据库名称
database_name = "oil_wells_db"

# 连接到 MySQL 服务器并执行创建数据库的命令
with engine.connect() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {database_name}"))
    conn.commit()

print(f"数据库 {database_name} 创建成功或已存在。")
