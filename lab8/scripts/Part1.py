import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Connect to the MySQL reddit database
db_url = "mysql+pymysql://username:password@localhost/reddit_db"
engine = create_engine(db_url, echo=False)

# Load posts data
query = "SELECT id, title FROM posts"
posts_df = pd.read_sql(query, engine)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def preprocess(text):
    """Convert text to lowercase, remove special characters, split into tokens, and filter stopwords."""
    text = re.sub(r'[^\w\s]', '', text)
    return [word for word in text.lower().split() if word not in stop_words]


# Create tagged documents for Doc2Vec (using post id as tag)
tagged_docs = [
    TaggedDocument(words=preprocess(row['title']), tags=[str(row['id'])])
    for _, row in posts_df.iterrows() if pd.notnull(row['title'])
]

# Define three different Doc2Vec configurations
doc2vec_configs = [
    {"vector_size": 50, "min_count": 2, "epochs": 40},
    {"vector_size": 100, "min_count": 2, "epochs": 40},
    {"vector_size": 200, "min_count": 2, "epochs": 40}
]

# Number of clusters for KMeans clustering
n_clusters = 5

# Process each Doc2Vec configuration
for config in doc2vec_configs:
    print(f"Running Doc2Vec with vector_size = {config['vector_size']}")
    
    # Initialize and train the Doc2Vec model
    model = Doc2Vec(vector_size=config["vector_size"], min_count=config["min_count"], epochs=config["epochs"])
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    
    # Infer vectors for each document
    doc_vectors = []
    doc_ids = []
    for doc in tagged_docs:
        vec = model.infer_vector(doc.words)
        doc_vectors.append(vec)
        doc_ids.append(doc.tags[0])
    
    doc_vectors = np.array(doc_vectors)
    # Normalize vectors (L2 normalization) for cosine distance comparison
    norm_vectors = normalize(doc_vectors)
    
    # Cluster the normalized vectors using KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(norm_vectors)
    
    # Build a DataFrame to store the clustering results
    results_df = pd.DataFrame({
        "id": doc_ids,
        "title": posts_df.set_index('id').loc[doc_ids, 'title'].values,
        "cluster": clusters,
        "vector_size": config["vector_size"]
    })
    
    # Save the results to a CSV file
    output_filename = f"doc2vec_results_{config['vector_size']}.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"Doc2Vec clustering results saved to {output_filename}\n")

    # Silhouette Score - The closer it is to 1, the better the clustering effect is.
    sil_score = silhouette_score(norm_vectors, clusters, metric='cosine')
    print(f"Silhouette Score for vector_size {config['vector_size']}: {sil_score}")

    # Davies-Bouldin Index (DBI) - The lower the value, the higher the differentiation of clusters and the better the clustering effect.
    db_score = davies_bouldin_score(norm_vectors, clusters)
    print(f"Davies-Bouldin Index for vector_size {config['vector_size']}: {db_score}")

    # Within-Cluster SSE - The lower the value, the tighter the data in the cluster.
    print(f"Within-Cluster SSE for vector_size {config['vector_size']}: {kmeans.inertia_}")

    # PCA
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(norm_vectors)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=reduced_vectors[:, 0], y=reduced_vectors[:, 1], hue=clusters, palette='tab10')
    plt.title(f"PCA Visualization for vector_size {config['vector_size']}")
    plt.show()

    # Get the text of all clusters
    cluster_texts = {i: [] for i in range(n_clusters)}

    for _, row in results_df.iterrows():
        cluster_texts[row["cluster"]].append(row["title"])

    # calculate TF-IDF
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform([" ".join(texts) for texts in cluster_texts.values()])
    terms = tfidf.get_feature_names_out()

    # Display high TF-IDF words for each cluster
    for i in range(n_clusters):
        print(f"Cluster {i}:")
        top_indices = tfidf_matrix[i].toarray()[0].argsort()[-10:][::-1]
        print([terms[idx] for idx in top_indices])


