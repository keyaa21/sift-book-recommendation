import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "goodreads-10k")


class SiftRecommender:
    """
    Unified 3-Stage Book Recommendation System:
      - Stage 1: Thematic & Tag-based Candidate Retrieval (Keya)
      - Stage 2: Supervised Preference / Likelihood Classification (Satya)
      - Stage 3: Fine-grained Rating Prediction & Final Ranking (Sneha)
    """

    def __init__(self, data_dir: str = DATA_DIR):
        self.data_dir = data_dir
        self.books_df: Optional[pd.DataFrame] = None
        self.user_stats_df: Optional[pd.DataFrame] = None
        self.user_cluster_affinity: Optional[pd.DataFrame] = None
        self.global_mean_rating: float = 3.90
        
        # Models
        self.tfidf = None
        self.tfidf_matrix = None
        self.knn = None
        self.clf = None
        self.clf_scaler = None
        self.clf_features = None
        self.regressor = None
        self.reg_scaler = None
        self.reg_features = None

        self._load_all()

    def _load_all(self):
        print("Initializing Sift Recommendation Engine...")
        
        # 1. Load Clustered Books
        books_path = os.path.join(self.data_dir, "books_clustered.csv")
        if not os.path.exists(books_path):
            raise FileNotFoundError(f"Missing {books_path}. Run keya/stage1.py first.")
        self.books_df = pd.read_csv(books_path)
        # Ensure fast index lookup by title and book_id
        self.books_df["title_clean"] = self.books_df["title"].astype(str).str.lower().str.strip()

        # 2. Load Stage 1 Models (Keya)
        keya_dir = os.path.join(ROOT_DIR, "keya")
        self.tfidf = joblib.load(os.path.join(keya_dir, "keya_tfidf.joblib"))
        self.tfidf_matrix = joblib.load(os.path.join(keya_dir, "keya_tfidf_matrix.joblib"))
        self.knn = joblib.load(os.path.join(keya_dir, "keya_knn.joblib"))

        # 3. Load Stage 2 Models (Satya)
        satya_dir = os.path.join(ROOT_DIR, "satya")
        clf_bundle = joblib.load(os.path.join(satya_dir, "best_classifier.joblib"))
        self.clf = clf_bundle["classifier"]
        self.clf_scaler = clf_bundle["scaler"]
        self.clf_features = clf_bundle["feature_cols"]

        # 4. Load Stage 3 Models (Sneha)
        reg_path = os.path.join(ROOT_DIR, "best_regressor.joblib")
        if not os.path.exists(reg_path):
            reg_path = os.path.join(ROOT_DIR, "sneha", "best_regressor.joblib")
        self.regressor = joblib.load(reg_path)
        self.reg_scaler = getattr(self.regressor, "scaler_", None)
        self.reg_features = getattr(
            self.regressor,
            "feature_cols_",
            ["average_rating", "ratings_count", "cluster_id", "user_mean_rating", "user_rating_count", "user_cluster_affinity"]
        )

        # 5. Pre-aggregate User Stats for instant (<10ms) lookup
        self._build_user_profiles()
        print("Sift Recommendation Engine ready!")

    def _build_user_profiles(self):
        ratings_path = os.path.join(self.data_dir, "ratings.csv")
        if not os.path.exists(ratings_path):
            return

        # Load ratings with memory-efficient dtypes
        ratings = pd.read_csv(
            ratings_path,
            usecols=["user_id", "book_id", "rating"],
            dtype={"user_id": "int32", "book_id": "int32", "rating": "float32"}
        )
        self.global_mean_rating = float(ratings["rating"].mean())

        # Precompute per-user stats
        self.user_stats_map = (
            ratings.groupby("user_id")
            .agg(user_mean_rating=("rating", "mean"), user_rating_count=("rating", "count"))
            .to_dict(orient="index")
        )

        # Precompute user cluster affinity
        merged = ratings.merge(self.books_df[["book_id", "cluster_id"]], on="book_id", how="inner")
        affinity = merged.groupby(["user_id", "cluster_id"])["rating"].mean().to_dict()
        self.user_cluster_affinity_map = affinity

    def get_user_profile(self, user_id: int) -> Dict[str, Any]:
        """Fetch user profile with graceful cold-start fallback."""
        if hasattr(self, "user_stats_map") and user_id in self.user_stats_map:
            stats = self.user_stats_map[user_id]
            return {
                "user_mean_rating": float(stats["user_mean_rating"]),
                "user_rating_count": int(stats["user_rating_count"]),
                "is_cold_start": False
            }
        # Cold start fallback for new users
        return {
            "user_mean_rating": self.global_mean_rating,
            "user_rating_count": 0,
            "is_cold_start": True
        }

    def get_user_cluster_affinity(self, user_id: int, cluster_id: int, default_rating: float) -> float:
        """Fetch user affinity for a specific cluster with fallback."""
        if hasattr(self, "user_cluster_affinity_map"):
            val = self.user_cluster_affinity_map.get((user_id, cluster_id))
            if val is not None:
                return float(val)
        return default_rating

    def find_book(self, query: str) -> Optional[pd.Series]:
        """Fuzzy/substring book search by title."""
        query_clean = query.lower().strip()
        # 1. Exact match
        exact = self.books_df[self.books_df["title_clean"] == query_clean]
        if not exact.empty:
            return exact.iloc[0]
        # 2. Substring match
        contains = self.books_df[self.books_df["title_clean"].str.contains(query_clean, regex=False)]
        if not contains.empty:
            return contains.iloc[0]
        # 3. Author search
        author_match = self.books_df[self.books_df["authors"].str.lower().str.contains(query_clean, regex=False, na=False)]
        if not author_match.empty:
            return author_match.iloc[0]
        return None

    def recommend(
        self,
        user_id: int,
        seed_title: Optional[str] = None,
        seed_book_id: Optional[int] = None,
        top_n: int = 5,
        min_like_prob: float = 0.40,
        candidate_pool_size: int = 30
    ) -> Dict[str, Any]:
        """
        Executes the full 3-Stage recommendation funnel:
          Stage 1: Retrieve candidate_pool_size books using KNN on TF-IDF soup.
          Stage 2: Filter candidates using Random Forest Classifier (P(rating >= 4)).
          Stage 3: Score surviving candidates with Regressor and rank Top-N.
        """
        # Resolve seed book
        seed_row = None
        if seed_book_id is not None:
            matches = self.books_df[self.books_df["book_id"] == seed_book_id]
            if not matches.empty:
                seed_row = matches.iloc[0]
        elif seed_title:
            seed_row = self.find_book(seed_title)

        if seed_row is None:
            raise ValueError(f"Could not find seed book matching query: {seed_title or seed_book_id}")

        seed_idx = self.books_df.index[self.books_df["book_id"] == seed_row["book_id"]][0]

        # -------------------------------------------------------------
        # STAGE 1: CANDIDATE GENERATION (Keya's TF-IDF + KNN)
        # -------------------------------------------------------------
        seed_vector = self.tfidf_matrix[seed_idx]
        distances, indices = self.knn.kneighbors(seed_vector, n_neighbors=candidate_pool_size + 1)
        
        # Exclude seed book itself
        cand_indices = [idx for idx in indices[0] if idx != seed_idx][:candidate_pool_size]
        candidates = self.books_df.iloc[cand_indices].copy().reset_index(drop=True)

        # -------------------------------------------------------------
        # STAGE 2: CLASSIFICATION FILTER (Satya's Classifier)
        # -------------------------------------------------------------
        user_profile = self.get_user_profile(user_id)
        u_mean = user_profile["user_mean_rating"]
        u_count = user_profile["user_rating_count"]

        # Build feature matrix for Stage 2
        candidates["log_ratings_count"] = np.log1p(candidates["ratings_count"]).astype("float32")
        candidates["user_mean_rating"] = u_mean
        candidates["user_rating_count"] = u_count
        candidates["user_cluster_affinity"] = [
            self.get_user_cluster_affinity(user_id, cid, u_mean) for cid in candidates["cluster_id"]
        ]

        X_clf = candidates[self.clf_features].fillna(0)
        X_clf_scaled = self.clf_scaler.transform(X_clf)
        
        # Predict probability of user liking (rating >= 4)
        like_probs = self.clf.predict_proba(X_clf_scaled)[:, 1]
        candidates["like_probability"] = like_probs

        # Filter by threshold (fallback to top candidates if strict threshold drops all)
        survivors = candidates[candidates["like_probability"] >= min_like_prob].copy()
        if len(survivors) < min(top_n, len(candidates)):
            # Fallback to top-scoring candidates to avoid returning empty results
            survivors = candidates.sort_values("like_probability", ascending=False).head(max(top_n, 10)).copy()

        # -------------------------------------------------------------
        # STAGE 3: FINE-GRAINED RATING PREDICTION & RANKING (Sneha)
        # -------------------------------------------------------------
        X_reg = survivors[self.reg_features].fillna(0)
        if self.reg_scaler is not None:
            X_reg = self.reg_scaler.transform(X_reg)
        
        predicted_ratings = self.regressor.predict(X_reg)
        predicted_ratings = np.clip(predicted_ratings, 1.0, 5.0)
        survivors["predicted_rating"] = np.round(predicted_ratings, 2)

        # Final Ranking by predicted rating descending
        ranked = survivors.sort_values(
            by=["predicted_rating", "like_probability", "average_rating"],
            ascending=[False, False, False]
        ).head(top_n)

        # Build clean output
        recommendations = []
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            recommendations.append({
                "rank": rank,
                "book_id": int(row["book_id"]),
                "title": str(row["title"]),
                "authors": str(row["authors"]),
                "cluster_id": int(row["cluster_id"]),
                "top_tags": str(row.get("top_tags", "")),
                "average_rating": float(row["average_rating"]),
                "like_probability": round(float(row["like_probability"]) * 100, 1),
                "predicted_rating": float(row["predicted_rating"]),
                "image_url": str(row.get("image_url", ""))
            })

        return {
            "user_id": user_id,
            "user_profile": user_profile,
            "seed_book": {
                "book_id": int(seed_row["book_id"]),
                "title": str(seed_row["title"]),
                "authors": str(seed_row["authors"]),
                "cluster_id": int(seed_row["cluster_id"]),
                "average_rating": float(seed_row["average_rating"]),
                "image_url": str(seed_row.get("image_url", ""))
            },
            "funnel_stats": {
                "stage1_candidates": len(candidates),
                "stage2_survivors": len(survivors),
                "stage3_final_ranked": len(recommendations)
            },
            "recommendations": recommendations
        }
