import pandas as pd
from sqlalchemy import create_engine, Column, String, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

# 数据库连接字符串，连接到已经建立好的 reddit 数据库
db_url = "mysql+pymysql://DSCI560:560560@172.16.161.128/reddit"
engine = create_engine(db_url, echo=True)

Base = declarative_base()

# 定义 posts 表，注意 id 为字符串，title 用 Text 类型
class Post(Base):
    __tablename__ = 'posts'
    id = Column(String(50), primary_key=True)
    title = Column(Text)
    comments = relationship("Comment", back_populates="post")

# 定义 comments 表，其中 post_id 为外键，关联 posts 表中的 id
class Comment(Base):
    __tablename__ = 'comments'
    id = Column(String(50), primary_key=True)
    post_id = Column(String(50), ForeignKey('posts.id'))
    body = Column(Text)
    post = relationship("Post", back_populates="comments")

# 删除旧表（会清除之前的数据），然后创建新表
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

# 读取 CSV 文件（请确保文件路径正确）
posts_df = pd.read_csv("posts.csv")
comments_df = pd.read_csv("comments.csv")

# 建立数据库会话
Session = sessionmaker(bind=engine)
session = Session()

# 插入 posts 数据
posts = []
for _, row in posts_df.iterrows():
    post = Post(
        id=str(row['id']).strip(),  # 强制转换为字符串，并去除可能的空格
        title=str(row['title']).strip() if pd.notnull(row['title']) else None
    )
    posts.append(post)
session.bulk_save_objects(posts)
session.commit()

# 插入 comments 数据
comments = []
for idx, row in comments_df.iterrows():
    comment = Comment(
        # 如果 comments.csv 中有 id 列则使用，否则用行号代替
        id=str(row['id']).strip() if 'id' in row and pd.notnull(row['id']) else str(idx),
        post_id=str(row['post_id']).strip(),
        body=str(row['body']).strip() if pd.notnull(row['body']) else None
    )
    comments.append(comment)
session.bulk_save_objects(comments)
session.commit()

session.close()
