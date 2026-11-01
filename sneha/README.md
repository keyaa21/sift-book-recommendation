# Sift Book Recommendation — Regression Module (`sneha`)

This module formulates book recommendation scoring and fine-grained ranking on **Goodbooks-10k** (`../goodreads-10k`) as a **Supervised Continuous Regression** problem:
> *"Given a user's reading history and a candidate book's metadata/cluster, what exact rating (1.0 to 5.0 stars) will the user assign?"*

---

## 1. Problem Formulation & Feature Engineering

### Target ($y$)
- **Continuous User Rating**: Target values ranging from $1.0$ to $5.0$ stars.

### Engineered Features ($X$)
1. `average_rating`: Global Goodreads average rating for the book.
2. `ratings_count`: Book popularity (total number of ratings received).
3. `cluster_id`: Thematic tag cluster ($0$ to $19$) derived from Keya's Stage 1 TF-IDF clustering.
4. `user_mean_rating`: Historical average rating given by the user across all past reads.
5. `user_rating_count`: Total books read and rated by the user.
6. `user_cluster_affinity`: User's average rating within that specific `cluster_id` (imputed with `user_mean_rating` for new clusters).

---

## 2. Benchmark Results (80/20 Split, 150k Ratings)

| Model | RMSE | MAE | MSE | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | **0.5074** | 0.3067 | **0.2575** | **0.7348** |
| **Ridge Regression** | **0.5074** | 0.3067 | **0.2575** | **0.7348** |
| **Random Forest Regressor** | 0.5107 | **0.2953** | 0.2609 | 0.7313 |
| **LinearSVR** | 0.5115 | 0.2869 | 0.2616 | 0.7305 |

> **Key Finding**: Feature engineering with `user_mean_rating` and `user_cluster_affinity` creates a strong linear correlation with user satisfaction. Linear and Ridge Regression achieve an outstanding $R^2 = 0.7348$ with an RMSE of $0.5074$, while Random Forest provides a competitive MAE of $0.2953$.

---

## 3. Storage & Deployment Optimization

Previous uncompressed tree models resulted in multi-gigabyte files that exceeded GitHub's 100 MB push limit. Using:
1. **Depth regularized trees** (`max_depth=12`, `min_samples_leaf=5`), and
2. **Lossless Zlib compression** (`joblib.dump(..., compress=3)`),
the production models now have negligible footprint with **100% precision preservation**:
- `best_regressor.joblib`: **~1.0 KB** (Instant loading & 100% GitHub-Safe)
- `best_regressor_bundle.joblib`: **~2.5 MB** (Includes all models, scaler, and metadata)

---

## 4. Generated Artifacts

- `stage3.py` — Dynamic training, benchmarking, chart generation, and model export script.
- `regression_metrics.csv` — Comprehensive benchmark evaluation table.
- `regression_comparison.png` — Bar chart comparing models across RMSE, MAE, and $R^2$.
- `predicted_vs_actual.png` — Scatter plot illustrating predicted vs actual rating concordance.
- `best_regressor.joblib` — Optimized, compressed production regressor ready for live ranking.
- `best_regressor_bundle.joblib` — Self-contained deployment bundle.

## How to Run

```bash
cd sneha
python stage3.py
```
