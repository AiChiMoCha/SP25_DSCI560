from sqlalchemy import create_engine, Column, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# 数据库连接字符串，连接到 reddit 数据库
db_url = "mysql+pymysql://DSCI560:560560@172.16.161.128/reddit"
engine = create_engine(db_url, echo=True)

Base = declarative_base()

# 定义 posts 表
class Post(Base):
    __tablename__ = 'posts'
    id = Column(String(50), primary_key=True)
    title = Column(Text)
    comments = relationship("Comment", back_populates="post")

# 定义 comments 表
class Comment(Base):
    __tablename__ = 'comments'
    id = Column(String(50), primary_key=True)
    post_id = Column(String(50), ForeignKey('posts.id'))
    body = Column(Text)
    post = relationship("Post", back_populates="comments")

# 建立会话
Session = sessionmaker(bind=engine)
session = Session()

# 查询所有 posts 及其关联的 comments
posts = session.query(Post).all()

for post in posts:
    print("Post ID:", post.id)
    print("Title:", post.title)
    print("Comments:")
    for comment in post.comments:
        print("  Comment ID:", comment.id)
        print("  Body:", comment.body)
    print("-" * 40)

session.close()
