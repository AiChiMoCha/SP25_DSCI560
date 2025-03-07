import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
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
