# Sift: Multi-Stage Hybrid Book Recommendation System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Pipeline-Verified%20%26%20Optimized-brightgreen.svg)]()

**Sift** is an industrial-grade, multi-stage hybrid book recommendation engine built on the [Goodbooks-10k](goodreads-10k/) dataset. It mirrors modern recommendation funnel architectures used by Netflix, Spotify, and YouTube:

```
                  All 10,000 Goodreads Books
                              │
                              ▼
        ┌───────────────────────────────────────────┐
        │   Stage 1: Retrieval (Keya)               │
        │   • TF-IDF Text Soup (Tags 2x + Author)   │
        │   • K-Means (20 Thematic Clusters)        │
        │   • KNN Cosine Similarity Retrieval       │
        └─────────────────────┬─────────────────────┘
                              │  30 Candidate Books
                              ▼
        ┌───────────────────────────────────────────┐
        │   Stage 2: Filtering / Preference (Satya) │
        │   • Calibrated Random Forest Classifier   │
        │   • Predicts Likelihood: P(Liked >= 4★)   │
        │   • Filters Out Low-Probability Matches   │
        └─────────────────────┬─────────────────────┘
                              │  Qualified Candidates
                              ▼
        ┌───────────────────────────────────────────┐
        │   Stage 3: Scoring & Ranking (Sneha)      │
        │   • Continuous Rating Regressor           │
        │   • Predicts Exact Rating (1.0 to 5.0★)   │
        │   • Final Ranked Top-N Recommendations    │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
                 Personalized Book Recommendations
```

---

## 👥 Module Breakdown & Team Contributions

| Module | Member | Role | Core Techniques & Deliverables |
| :--- | :--- | :--- | :--- |
| **`keya/`** | **Keya** | Stage 1: Retrieval & Clustering | Noise tag filtering, 2x-weighted tag soup, TF-IDF (500 features), K-Means ($K=20$), KNN Cosine distance retrieval, `books_clustered.csv`. |
| **`satya/`** | **Satya** | Stage 2: Preference Classification | Feature engineering (`user_cluster_affinity`, `log_ratings_count`), StandardScaler, Calibrated Random Forest ($F_1 = 0.8534$, $\text{AUC} = 0.8782$), threshold tuning. |
| **`sneha/`** | **Sneha** | Stage 3: Rating Regression & Ranking | Collaborative user/item feature fusion, Linear & Ridge Regression ($R^2 = 0.7348$, $\text{RMSE} = 0.5074$), regularized Random Forest, lossless model compression. |

---

## 📊 Benchmark Results

### Stage 2: Supervised Binary Classification (Satya)
> Predicts whether a user will like a candidate book ($\text{rating} \ge 4$).

| Model | Accuracy | Precision | Recall | $F_1$-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes (`GaussianNB`)** | 0.7931 | **0.8301** | 0.8542 | 0.8420 | 0.8599 |
| **Logistic Regression** | 0.8036 | 0.8261 | 0.8812 | 0.8528 | 0.8724 |
| **SVM (`Calibrated LinearSVC`)** | 0.8036 | 0.8258 | 0.8817 | 0.8528 | 0.8724 |
| **Random Forest Classifier** | **0.8044** | 0.8266 | **0.8820** | **0.8534** | **0.8782** |

### Stage 3: Supervised Continuous Regression (Sneha)
> Predicts exact star rating ($1.0$ to $5.0$) for personalized ranking.

| Model | RMSE | MAE | MSE | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** | **0.5074** | 0.3067 | **0.2575** | **0.7348** |
| **Ridge Regression** | **0.5074** | 0.3067 | **0.2575** | **0.7348** |
| **Random Forest Regressor** | 0.5107 | **0.2953** | 0.2609 | 0.7313 |
| **LinearSVR** | 0.5115 | 0.2869 | 0.2616 | 0.7305 |

---

## 🚀 How to Run the Application

### 1. Interactive Web Dashboard
Launch the zero-dependency browser UI with cover images, interactive sliders, and live stage funnels:
```bash
python app.py --web
# Or directly:
python web_app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in any web browser.

### 2. Command-Line Interface (CLI)
Get instant recommendations directly in the terminal:
```bash
# Recommend books for User 1 based on The Hobbit
python app.py --user 1 --book "The Hobbit" --top 5

# Search by author or title query
python app.py --user 42 --book "Pride and Prejudice" --top 5

# Cold-start recommendation for a new user
python app.py --user 999999 --book "The Hunger Games" --top 5
```

### 3. Interactive Terminal Mode
```bash
python app.py --interactive
```

### 4. Automated Multi-User Demo
```bash
python app.py --demo
```

---

## 🧪 Running Automated Tests

Run the full end-to-end integration and unit test suite:
```bash
python -m unittest tests/test_sift.py
```
Tests verify:
- Stage 1 candidate retrieval
- Stage 2 preference classification
- Stage 3 continuous rating scoring
- Cold-start handling for unseen users
- Model file size verification (all artifacts strictly $< 100\text{ MB}$)

---

## 📁 Repository Structure

```
sift-book-recommendation/
├── app.py                      # Unified CLI & Interactive Recommender Entry Point
├── web_app.py                  # Standalone Web Dashboard & REST API
├── best_regressor.joblib       # Optimized Stage 3 Model (~1.0 KB, GitHub-safe)
├── regression_metrics.csv      # Stage 3 Benchmark Results
├── sift/                       # Recommender Core Package
│   ├── __init__.py
│   └── engine.py               # 3-Stage Pipeline & User Profile Caching
├── keya/                       # Stage 1: Retrieval & Clustering (Keya)
│   ├── stage1.py
│   ├── keya_tfidf.joblib
│   ├── keya_tfidf_matrix.joblib
│   ├── keya_knn.joblib
│   └── README.md
├── satya/                      # Stage 2: Binary Classification (Satya)
│   ├── my_classification.py
│   ├── best_classifier.joblib
│   ├── classification_metrics.csv
│   ├── classification_comparison.png
│   ├── confusion_matrix.png
│   ├── threshold_tuning.csv
│   └── README.md
├── sneha/                      # Stage 3: Regression & Ranking (Sneha)
│   ├── stage3.py
│   ├── best_regressor.joblib
│   ├── best_regressor_bundle.joblib
│   ├── regression_metrics.csv
│   ├── regression_comparison.png
│   ├── predicted_vs_actual.png
│   └── README.md
├── tests/
│   └── test_sift.py            # Automated Unit & Funnel Tests
└── goodreads-10k/              # Dataset & Cluster Maps
    ├── books.csv
    ├── books_clustered.csv
    ├── book_tags.csv
    ├── ratings.csv
    └── tags.csv
```
