import os
import sys
import unittest

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, ROOT_DIR)

from sift.engine import SiftRecommender


class TestSiftRecommendationPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("Setting up SiftRecommender for testing...")
        cls.recommender = SiftRecommender()

    def test_initialization(self):
        """Verify all models and datasets are loaded properly."""
        self.assertIsNotNone(self.recommender.books_df)
        self.assertIsNotNone(self.recommender.knn)
        self.assertIsNotNone(self.recommender.clf)
        self.assertIsNotNone(self.recommender.regressor)
        self.assertEqual(len(self.recommender.books_df), 10000)

    def test_book_search(self):
        """Verify search finds books by exact title or substring."""
        hobbit = self.recommender.find_book("The Hobbit")
        self.assertIsNotNone(hobbit)
        self.assertIn("Hobbit", hobbit["title"])

        potter = self.recommender.find_book("Harry Potter")
        self.assertIsNotNone(potter)
        self.assertIn("Harry Potter", potter["title"])

    def test_full_pipeline_recommendation(self):
        """Verify end-to-end 3-stage recommendation output."""
        res = self.recommender.recommend(
            user_id=1,
            seed_title="The Hobbit",
            top_n=5,
            min_like_prob=0.30
        )
        self.assertEqual(res["user_id"], 1)
        self.assertIn("seed_book", res)
        self.assertIn("funnel_stats", res)
        self.assertEqual(len(res["recommendations"]), 5)

        first_rec = res["recommendations"][0]
        self.assertEqual(first_rec["rank"], 1)
        self.assertIn("title", first_rec)
        self.assertIn("like_probability", first_rec)
        self.assertIn("predicted_rating", first_rec)
        self.assertTrue(1.0 <= first_rec["predicted_rating"] <= 5.0)

    def test_cold_start_user(self):
        """Verify recommendations work seamlessly for unknown/new users."""
        res = self.recommender.recommend(
            user_id=9999999,  # Unseen user
            seed_title="The Hunger Games",
            top_n=3
        )
        self.assertTrue(res["user_profile"]["is_cold_start"])
        self.assertEqual(len(res["recommendations"]), 3)

    def test_file_sizes_under_github_limit(self):
        """Verify all tracked joblib artifacts are strictly under GitHub's 100 MB limit."""
        joblib_files = [
            os.path.join(ROOT_DIR, "best_regressor.joblib"),
            os.path.join(ROOT_DIR, "keya", "keya_knn.joblib"),
            os.path.join(ROOT_DIR, "keya", "keya_tfidf_matrix.joblib"),
            os.path.join(ROOT_DIR, "satya", "best_classifier.joblib"),
            os.path.join(ROOT_DIR, "sneha", "best_regressor.joblib"),
            os.path.join(ROOT_DIR, "sneha", "best_regressor_bundle.joblib"),
        ]
        for path in joblib_files:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                self.assertLess(size_mb, 100.0, f"File {path} ({size_mb:.2f} MB) exceeds 100MB!")
                print(f"Verified {os.path.basename(path)}: {size_mb:.4f} MB (< 100 MB limit)")


if __name__ == "__main__":
    unittest.main()
