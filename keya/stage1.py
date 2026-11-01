import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import joblib

# Resolve directories dynamically so script works from root or inside keya/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "goodreads-10k")

# 1. Load the CSV files from your goodreads-10k folder
books = pd.read_csv(os.path.join(DATA_DIR, "books.csv"))
book_tags = pd.read_csv(os.path.join(DATA_DIR, "book_tags.csv"))
tags = pd.read_csv(os.path.join(DATA_DIR, "tags.csv"))

print(f"Loaded {len(books)} books, {len(book_tags)} book-tag links, and {len(tags)} unique tags.")
print("Columns in books.csv:", list(books.columns))

# Standardize ID columns in books.csv so it never throws a KeyError
if "goodreads_book_id" not in books.columns:
    if "best_book_id" in books.columns and "id" in books.columns:
        # 'id' is 1..10000 (for ratings.csv), 'book_id' is the Goodreads ID (for book_tags.csv)
        books["goodreads_book_id"] = books["book_id"]
        books["book_id"] = books["id"]
    else:
        books["goodreads_book_id"] = books["book_id"]

# Also check book_tags ID column name just in case
tag_book_col = "goodreads_book_id" if "goodreads_book_id" in book_tags.columns else "book_id"

# 2. Merge book_tags with tags to get actual tag names
merged_tags = pd.merge(book_tags, tags, on="tag_id", how="inner")

# Filter out non-genre Goodreads shelf tags so K-Means forms real thematic clusters
noise_tags = {
    "to-read", "currently-reading", "favorites", "owned", "books-i-own",
    "owned-books", "library", "default", "kindle", "audiobook", "audiobooks",
    "ebook", "ebooks", "wish-list", "my-books", "my-library", "read-in-2016",
    "read-in-2015", "read-in-2017", "i-own", "series", "favourites", "re-read"
}
merged_tags["tag_name_clean"] = merged_tags["tag_name"].astype(str).str.lower().str.strip()
merged_tags = merged_tags[~merged_tags["tag_name_clean"].isin(noise_tags)]

# 3. Sort by tag popularity ('count') and take the Top 15 tags per book
merged_tags = merged_tags.sort_values([tag_book_col, "count"], ascending=[True, False])
top_15_tags = merged_tags.groupby(tag_book_col).head(15)

# Group the 15 tags into both a space-separated string (for TF-IDF) and a comma-separated string (for UI)
book_tag_strings = (
    top_15_tags.groupby(tag_book_col)["tag_name_clean"]
    .agg(
        tags_soup=lambda x: " ".join(x.str.replace("-", " ")),
        top_tags=lambda x: ", ".join(x.head(5))
    )
    .reset_index()
    .rename(columns={tag_book_col: "goodreads_book_id"})
)

# 4. Merge the aggregated tags back into books.csv
books = pd.merge(books, book_tag_strings, on="goodreads_book_id", how="left")
books["tags_soup"] = books["tags_soup"].fillna("")
books["top_tags"] = books["top_tags"].fillna("general fiction")
books["authors"] = books["authors"].fillna("")
books["title"] = books["title"].fillna("")

# 5. Build the final text "soup" (Weighting tags 2x + authors + title)
books["soup"] = (
    (books["tags_soup"] + " ") * 2 +
    books["authors"] + " " +
    books["title"]
)

# 6. TF-IDF Vectorization (Top 500 features)
tfidf = TfidfVectorizer(max_features=500, stop_words="english")
tfidf_matrix = tfidf.fit_transform(books["soup"])

# 7. Train K-Means (K = 20) to assign every book a cluster_id
kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
books["cluster_id"] = kmeans.fit_predict(tfidf_matrix)

# 8. Save books_clustered.csv for Satya and Sneha
cols_to_save = [
    "book_id", "goodreads_book_id", "title", "authors",
    "average_rating", "ratings_count", "image_url", "top_tags", "cluster_id"
]
# Only keep columns that exist in books
cols_to_save = [c for c in cols_to_save if c in books.columns]
output_clustered_csv = os.path.join(DATA_DIR, "books_clustered.csv")
books[cols_to_save].to_csv(output_clustered_csv, index=False)
print(f"Saved {output_clustered_csv}! Ready for Satya and Sneha.")

# 9. Train KNN (Cosine Distance) for Stage 1 Candidate Retrieval
knn_model = NearestNeighbors(n_neighbors=30, metric="cosine", algorithm="brute")
knn_model.fit(tfidf_matrix)

# Save artifacts inside keya/ folder with compress=3
joblib.dump(tfidf, os.path.join(BASE_DIR, "keya_tfidf.joblib"), compress=3)
joblib.dump(tfidf_matrix, os.path.join(BASE_DIR, "keya_tfidf_matrix.joblib"), compress=3)
joblib.dump(knn_model, os.path.join(BASE_DIR, "keya_knn.joblib"), compress=3)
print("Saved keya_tfidf.joblib, keya_tfidf_matrix.joblib, and keya_knn.joblib inside keya/!")
