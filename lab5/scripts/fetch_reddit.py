import praw
import mysql.connector
import time
import re
import datetime

# set up Reddit API
reddit = praw.Reddit(
    user_agent="Comment Extraction (by /u/INSERTUSERNAME)",
    client_id="VMHdbuYaeJ5dVZiSAAqXmA",
    client_secret="9orK8kVOOW4HDOWPP82ID8Lr5wfvBQ"
)

# connect MySQL
db = mysql.connector.connect(
    host="localhost",
    user="phpmyadmin",
    password="Hyq010113!",
    database="reddit_db"
)
cursor = db.cursor()

# create tables
cursor.execute("""
    CREATE TABLE IF NOT EXISTS posts (
        id VARCHAR(20) PRIMARY KEY,
        title TEXT,
        selftext TEXT,
        subreddit VARCHAR(100),
        created_at DATETIME,
        image_url TEXT,
        ocr_text TEXT,
        cluster_id INT DEFAULT -1
    )
""")
db.commit()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        id VARCHAR(30) PRIMARY KEY,
        post_id VARCHAR(30),
        parent_id VARCHAR(30),
        body TEXT,
        author VARCHAR(50),
        created_at DATETIME,
        FOREIGN KEY (post_id) REFERENCES posts(id) ON DELETE CASCADE,
        cluster_id INT DEFAULT -1
    )
""")
db.commit()


# preprocess text
def preprocess_text(text):
    text = re.sub(r'<.*?>', '', text)  # remove html tags
    return text.lower().strip()


# get posts
def fetch_reddit_posts(subreddit_name, num_posts):
    subreddit = reddit.subreddit(subreddit_name)
    posts_fetched = 0
    after = None  # for paging
    batch_size = 1000

    while posts_fetched < num_posts:
        remaining = num_posts - posts_fetched
        limit = min(batch_size, remaining)

        try:
            posts = subreddit.new(limit=limit, params={"after": after})
            batch = list(posts)
            if not batch:
                break

            for post in batch:
                if "promoted" in post.title.lower() or "sponsored" in post.title.lower():
                    continue  # Filter ads

                cursor.execute("SELECT id FROM posts WHERE id = %s", (post.id,))
                if cursor.fetchone():
                    continue  # Skip existing posts

                timestamp = datetime.datetime.fromtimestamp(post.created_utc, datetime.UTC)
                image_url = post.url if post.url.endswith(('jpg', 'png', 'jpeg')) else None

                cursor.execute("""
                    INSERT IGNORE INTO posts (id, title, selftext, subreddit, created_at, image_url)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (post.id, preprocess_text(post.title), preprocess_text(post.selftext), post.subreddit.display_name,
                      timestamp, image_url))

                fetch_comments(post.id, post)

            db.commit()
            posts_fetched += len(batch)
            after = batch[-1].fullname
            print(f"Fetched {posts_fetched}/{num_posts} posts...")
            time.sleep(2)

        except Exception as e:
            print(f"Error: {e}")
            time.sleep(10)


def fetch_comments(post_id, post):
    """ Get all comments for a post and store them in the database """
    post.comments.replace_more(limit=None)  # Recursively expand all comments
    for comment in post.comments.list():
        cursor.execute("""
            INSERT IGNORE INTO comments (id, post_id, parent_id, body, author, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (comment.id, post_id, comment.parent_id, preprocess_text(comment.body),
              comment.author.name if comment.author else "Unknown",
              datetime.datetime.fromtimestamp(comment.created_utc, datetime.UTC)))

    db.commit()


fetch_reddit_posts("tech", 5000)
cursor.close()
db.close()