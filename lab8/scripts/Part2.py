import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Connect to the MySQL reddit database
db_url = "mysql+pymysql://DSCI560:560560@172.16.161.128/reddit"
engine = create_engine(db_url, echo=False)

# Load posts data
query = "SELECT id, title FROM posts"
posts_df = pd.read_sql(query, engine)

def preprocess(text):
    """Convert text to lowercase and split into tokens."""
    return text.lower().split()

# Build the corpus and collect document metadata
corpus = []
doc_ids = []
doc_titles = []
for _, row in posts_df.iterrows():
    if pd.notnull(row['title']):
        tokens = preprocess(row['title'])
        corpus.append(tokens)
        doc_ids.append(str(row['id']).strip())
        doc_titles.append(row['title'])

# Train the Word2Vec model on the entire corpus
word2vec_model = Word2Vec(sentences=corpus, vector_size=100, window=5, min_count=1, workers=4)

# Define three different bin configurations (dimensions)
bin_configs = [50, 100, 200]
n_clusters = 5  # Number of clusters for document clustering

for bins in bin_configs:
    print(f"\nRunning Word2Vec-Bag-of-Words with {bins} bins")
    
    # Extract all words and their vectors from the model
    vocab_words = list(word2vec_model.wv.key_to_index.keys())
    word_vectors = np.array([word2vec_model.wv[word] for word in vocab_words])
    
    # Cluster the word vectors into 'bins' clusters using KMeans
    kmeans_words = KMeans(n_clusters=bins, random_state=42)
    word_cluster_labels = kmeans_words.fit_predict(word_vectors)
    
    # Create a dictionary mapping each word to its bin/cluster label
    word_to_cluster = {word: label for word, label in zip(vocab_words, word_cluster_labels)}
    
    # Generate a normalized histogram vector for each document
    doc_vectors = []
    for tokens in corpus:
        hist = np.zeros(bins)
        word_count = 0
        for token in tokens:
            if token in word_to_cluster:
                hist[word_to_cluster[token]] += 1
                word_count += 1
        if word_count > 0:
            hist = hist / word_count  # Normalize the histogram
        doc_vectors.append(hist)
    
    doc_vectors = np.array(doc_vectors)
    
    # Cluster the document vectors using KMeans (normalize for cosine similarity)
    kmeans_docs = KMeans(n_clusters=n_clusters, random_state=42)
    doc_cluster_labels = kmeans_docs.fit_predict(normalize(doc_vectors))
    
    # Build a DataFrame to store the clustering results
    results_df = pd.DataFrame({
        "id": doc_ids,
        "title": doc_titles,
        "cluster": doc_cluster_labels,
        "bins": bins
    })
    
    # Save the results to a CSV file
    output_filename = f"word2vec_results_{bins}_bins.csv"
    results_df.to_csv(output_filename, index=False)
    print(f"Word2Vec-Bag-of-Words clustering results saved to {output_filename}\n")
