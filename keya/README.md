# Sift Book Recommendation — Clustering & Retrieval Module (`keya`)

This module powers **Stage 1 (Candidate Retrieval & Thematic Clustering)** for the Sift Book Recommendation System.

---

## 1. Responsibilities
1. **Thematic Content Soup**: Clean Goodreads tags (filtering out noise tags like `to-read`, `owned`, `books-i-own`), weight top 15 tags $2\times$, and combine with author and title.
2. **TF-IDF Vectorization**: Extract top 500 thematic and lexical features across 10,000 books.
3. **K-Means Clustering ($K=20$)**: Partition 10,000 books into 20 coherent genre/thematic clusters to generate the `cluster_id` feature used by Stage 2 and Stage 3.
4. **KNN Cosine Similarity**: Fast cosine distance nearest neighbor search to retrieve Top 30 candidate books for any input seed book.

---

## 2. Deliverables
- `stage1.py`: Complete pipeline script.
- `keya_tfidf.joblib`: Pre-trained TF-IDF vectorizer.
- `keya_tfidf_matrix.joblib`: TF-IDF matrix for all books.
- `keya_knn.joblib`: NearestNeighbors model fitted on TF-IDF vectors.
- `goodreads-10k/books_clustered.csv`: Enriched book dataset with `cluster_id` and `top_tags`.

## How to Run

```bash
cd keya
python stage1.py
```
