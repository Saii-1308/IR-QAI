# IR Practical 7 - Text Similarity, LSI and Document Retrieval System
# Topic: Latent Semantic Indexing (LSI/SVD), cosine similarity matrix, document-to-document similarity, query expansion, and building a mini IR system

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Step 1: Load document corpus
data = pd.read_csv("spam.csv", encoding='latin-1')
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Use first 200 messages as corpus
corpus = data['message'].head(200).tolist()
labels = data['label'].head(200).tolist()

print("Corpus Size:", len(corpus))

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english',
                             max_features=500,
                             min_df=2)
tfidf_matrix = vectorizer.fit_transform(corpus)

print("TF-IDF Matrix:", tfidf_matrix.shape)

# Step 3: LSI using Truncated SVD (reduces to 50 latent topics)
svd = TruncatedSVD(n_components=50, random_state=42)
normalizer = Normalizer(copy=False)
lsi_pipeline = make_pipeline(svd, normalizer)

lsi_matrix = lsi_pipeline.fit_transform(tfidf_matrix)

print("LSI Matrix Shape:", lsi_matrix.shape)
print("Variance Explained by 50 components:",
      f"{svd.explained_variance_ratio_.sum():.2%}")

# Step 4: Document-to-Document Similarity Matrix (first 20)
sim_matrix = cosine_similarity(lsi_matrix[:20])

plt.figure(figsize=(10, 8))
sns.heatmap(sim_matrix, cmap='YlOrRd', annot=False)
plt.title("Document-to-Document Cosine Similarity (First 20 Docs)")
plt.xlabel("Document Index")
plt.ylabel("Document Index")
plt.show()

# Step 5: Query-based Document Retrieval
def retrieve_documents(query, top_n=5):
    query_tfidf = vectorizer.transform([query])
    query_lsi = lsi_pipeline.transform(query_tfidf)
    scores = cosine_similarity(query_lsi, lsi_matrix).flatten()
    ranked = scores.argsort()[::-1][:top_n]
    print(f"\nQuery: '{query}'")
    print("=" * 60)
    for rank, idx in enumerate(ranked, 1):
        print(f"Rank {rank} | Score: {scores[idx]:.4f} | [{labels[idx].upper()}]")
        print(f"  {corpus[idx][:80]}...")
        print()

# Step 6: Run retrieval for different queries
retrieve_documents("win free cash prize lottery")
retrieve_documents("meeting tomorrow call me")

# Step 7: Plot Singular Values (Topic Strength)
plt.figure(figsize=(10, 4))
plt.plot(range(1, 21), svd.singular_values_[:20],
         marker='o', color='teal', linewidth=2)
plt.title("SVD Singular Values (Top 20 Latent Topics)")
plt.xlabel("Topic Index")
plt.ylabel("Singular Value (Topic Strength)")
plt.grid(True)
plt.show()

# Step 8: Find most similar document pairs
print("\nTop 5 Most Similar Document Pairs:")
full_sim = cosine_similarity(lsi_matrix[:50])
np.fill_diagonal(full_sim, 0)

for _ in range(5):
    idx = np.unravel_index(full_sim.argmax(), full_sim.shape)
    i, j = idx
    score = full_sim[i, j]
    full_sim[i, j] = 0
    full_sim[j, i] = 0
    print(f"Doc {i} <-> Doc {j} | Similarity: {score:.4f}")
    print(f"  Doc {i}: {corpus[i][:60]}...")
    print(f"  Doc {j}: {corpus[j][:60]}...")
    print()

print("\nLSI Document Retrieval System Completed!")
