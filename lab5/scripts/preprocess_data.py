import mysql.connector
import pytesseract
from PIL import Image
import requests
from io import BytesIO
import re
from bs4 import BeautifulSoup

# connect MySQL
db = mysql.connector.connect(
    host="localhost",
    user="phpmyadmin",
    password="Hyq010113!",
    database="reddit_db"
)
cursor = db.cursor()


# clean text
def clean_text(text):
    if not isinstance(text, str):  # deal with None or non-string
        return ""

    text = text.strip()

    if text.startswith("http"):  # It may be a URL, skip it directly
        return text

    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except Exception as e:
        print(f"Error parsing text: {e}")  # detect failure
        return text  # If parsing fails, return the original text

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Only letters, numbers, and spaces are allowed
    return text.lower().strip()


# Processing OCR recognition
def extract_text_from_image(url):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        return pytesseract.image_to_string(img)
    except:
        return ""


# Process all posts
cursor.execute("SELECT id, selftext, image_url FROM posts WHERE ocr_text IS NULL")
posts = cursor.fetchall()

for post_id, text, image_url in posts:
    cleaned_text = clean_text(text)
    ocr_text = extract_text_from_image(image_url) if image_url else ""

    cursor.execute("UPDATE posts SET selftext=%s, ocr_text=%s WHERE id=%s", (cleaned_text, ocr_text, post_id))
db.commit()

cursor.execute("SELECT id, body FROM comments")
comments = cursor.fetchall()

for comment_id, text in comments:
    cleaned_text = clean_text(text)
    cursor.execute("UPDATE comments SET body=%s WHERE id=%s", (cleaned_text, comment_id))

db.commit()
cursor.close()
db.close()