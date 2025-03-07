import argparse
import time
import subprocess
import mysql.connector
import traceback
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
import threading
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# connect MySQL
def connect_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="phpmyadmin",
            password="Hyq010113!",
            database="reddit_db"
        )
    except mysql.connector.Error as e:
        print(f"Database connection failed: {e}")
        sys.exit(1)


# Get all clustered text (including posts and comments)
def fetch_clusters():
    db = connect_db()
    cursor = db.cursor(dictionary=True)

    # Get post data
    cursor.execute("SELECT id, selftext AS text, cluster_id FROM posts WHERE cluster_id IS NOT NULL")
    posts = cursor.fetchall()

    # Get comment data
    cursor.execute("SELECT post_id AS id, body AS text, cluster_id FROM comments WHERE cluster_id IS NOT NULL")
    comments = cursor.fetchall()

    db.close()
    return posts + comments


# Find the most relevant posts/comments and visualize them
def find_closest_cluster(user_input, posts_comments):
    if not posts_comments:
        print("There are no categorized posts or comments in the database")
        return

    # Extract text
    texts = [item["text"] for item in posts_comments] + [user_input]
    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts)

    # Calculating Similarity
    similarity = cosine_similarity(vectors[-1], vectors[:-1])  # Calculate the similarity between input and existing text
    closest_idx = np.argmax(similarity)  # Get the most similar index

    print("\nBest matching content:")
    print(f"{posts_comments[closest_idx]['text'][:200]}...\n（Cluster ID: {posts_comments[closest_idx]['cluster_id']}）")

    # Visualize matching posts/comments
    plot_cluster(posts_comments)


# Visualize matching posts/comments
def plot_cluster(posts_comments):
    # Extract text
    texts = [item["text"] for item in posts_comments]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
    vectors = vectorizer.fit_transform(texts)

    # Use PCA to reduce dimension to 2D
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors.toarray())

    # Get cluster ID
    cluster_ids = [item["cluster_id"] for item in posts_comments]

    # Draw a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=cluster_ids, cmap='viridis', alpha=0.6)
    plt.colorbar(label="Cluster ID")
    plt.title("Cluster Visualization (Posts & Comments)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


# Run subscript and catch errors
def run_script(script_name):
    try:
        print(f"\nRun {script_name} ...")
        subprocess.run(["python", script_name], check=True)
        print(f"{script_name} runs successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run {script_name}: {e}")
    except Exception as e:
        print(f"Unknown error: {traceback.format_exc()}")


def fetch_and_update(interval):
    """ The task of periodically fetching data """
    while True:
        print("\nStart crawling and processing data...")
        run_script("fetch_reddit.py")
        run_script("preprocess_data.py")
        run_script("cluster_analysis.py")

        print(f"Complete data update, wait {interval} minutes to next round...")
        time.sleep(interval * 60)


def search_posts():
    """ User query mode (allows query at any time) """
    print("\nNow you can enter keywords to search for the most relevant posts (type 'exit' to exit):")
    posts = fetch_clusters()

    while True:
        user_input = input("Please enter message: ").strip()
        if user_input.lower() == "exit":
            print("Exit query mode...")
            break
        find_closest_cluster(user_input, posts)


def main(interval):
    # Start a background thread to fetch data
    fetch_thread = threading.Thread(target=fetch_and_update, args=(interval,))
    fetch_thread.daemon = True  # Set as a daemon thread so that the main program can exit at any time
    fetch_thread.start()

    # The main thread enters query mode
    search_posts()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regularly crawl Reddit data and perform cluster analysis")
    parser.add_argument("interval", type=int, help="Data crawling interval (minutes)")
    args = parser.parse_args()

    print(f"Reddit scraping task started, updating every {args.interval} minutes")
    main(args.interval)