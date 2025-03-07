import mysql.connector
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# connect MySQL
db = mysql.connector.connect(
    host="localhost",
    user="phpmyadmin",
    password="Hyq010113!",
    database="reddit_db"
)
cursor = db.cursor()

# processing data
cursor.execute("SELECT id, selftext FROM posts")
post_data = cursor.fetchall()

cursor.execute("SELECT id, post_id, body FROM comments")
comment_data = cursor.fetchall()

posts = [{"id": p[0], "text": p[1]} for p in post_data]
comments = [{"id": c[0], "post_id": c[1], "text": c[2]} for c in comment_data]

all_texts = [p["text"] for p in posts] + [c["text"] for c in comments]

# Processing text using TF-IDF (removing stop words)
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)  # 限制特征数量，提高训练效果
X_tfidf = vectorizer.fit_transform(all_texts)  # 生成 TF-IDF 特征矩阵

# Generating Doc2Vec training data
documents = [TaggedDocument(words=text.split(), tags=[str(i)]) for i, text in enumerate(all_texts)]
model = Doc2Vec(documents, vector_size=100, min_count=2, epochs=20)

# Computing Embedding
doc2vec_vectors = np.array([model.dv[str(i)] for i in range(len(all_texts))])

# Combining TF-IDF & Doc2Vec vectors
X_combined = np.hstack((doc2vec_vectors, X_tfidf.toarray()))

# K-means clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X_combined)

# store in database
# update Posts
for i, post in enumerate(posts):
    cursor.execute("UPDATE posts SET cluster_id=%s WHERE id=%s", (int(clusters[i]), post["id"]))

# update Comments
post_count = len(posts)
for i, comment in enumerate(comments):
    cluster_index = post_count + i  # Because the index of Comments in all_texts starts after posts
    cursor.execute("UPDATE comments SET cluster_id=%s WHERE id=%s", (int(clusters[cluster_index]), comment["id"]))

db.commit()
cursor.close()
db.close()

# Use PCA to reduce the dimension to 2 dimensions
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(X_combined)  # X_combined 是 TF-IDF + Doc2Vec 组合向量

# Visualization
plt.figure(figsize=(8, 6))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=clusters, cmap="viridis", alpha=0.7)

# Add cluster centers
centroids = pca.transform(kmeans.cluster_centers_)  # K-means 聚类中心降维
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids")

# Show Legend
plt.colorbar(label="Cluster ID")
plt.legend()
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Cluster Visualization (PCA Reduced)")
plt.show()