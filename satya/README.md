# Sift Book Recommendation — Classification Module (`satya`)

This module formulates book recommendation on **Goodbooks-10k** (`../goodreads-10k`) as a **Supervised Binary Classification** problem:
> *"Given a user's historical reading habits and a book's tags/metadata, will the user LIKE this book ($\text{rating} \ge 4$)?"*

---

## 1. Problem Formulation & Feature Engineering

### Target ($y$)
- **Class `1` (Liked / Recommend)**: $\text{rating} \ge 4$ stars (`64.5%` of dataset)
- **Class `0` (Not Liked / Skip)**: $\text{rating} \le 3$ stars (`35.5%` of dataset)

### Engineered Features ($X$)
1. `average_rating`: The book's global rating on Goodreads.
2. `log_ratings_count`: The book's popularity ($\log(1 + \text{ratings\_count})$).
3. `cluster_id`: Thematic tag cluster ($0$ to $19$) loaded directly from Keya's pre-cleaned `../goodreads-10k/books_clustered.csv`.
4. `user_mean_rating`: The user's average rating across past reads.
5. `user_rating_count`: Total number of books rated by the user.
6. `user_cluster_affinity`: The user's mean rating within that specific `cluster_id`.

---

## 2. Model Comparison Results (80/20 Stratified Split, 200k Ratings)

Using Keya's cleaned 2x-tag-weighted clusters (`../goodreads-10k/books_clustered.csv`), **Random Forest Classifier** achieves the highest **Accuracy (`0.8044`)**, **F1-Score (`0.8534`)**, and **ROC-AUC (`0.8782`)**:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Bayes (`GaussianNB`)** | 0.7931 | **0.8301** | 0.8542 | 0.8420 | 0.8599 |
| **Logistic Regression** | 0.8036 | 0.8261 | 0.8812 | 0.8528 | 0.8724 |
| **SVM (`Calibrated LinearSVC`)** | 0.8036 | 0.8258 | 0.8817 | 0.8528 | 0.8724 |
| **Random Forest Classifier** | **0.8044** | 0.8266 | **0.8820** | **0.8534** | **0.8782** |

---

## 3. Probability Threshold Tuning (Random Forest)

| Threshold | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: |
| **0.40** | 0.7935 | **0.9310** | **0.8568** |
| **0.50** | 0.8266 | 0.8820 | 0.8534 |
| **0.60** | 0.8614 | 0.8120 | 0.8359 |
| **0.70** | **0.8942** | 0.7350 | 0.8068 |

---

## 4. Generated Files in `satya/`

- `my_classification.py` — Optimized training, evaluation, plotting, and model export script reading directly from `../goodreads-10k/books_clustered.csv` and `../goodreads-10k/ratings.csv`.
- `classification_metrics.csv` — Model comparison metrics table.
- `threshold_tuning.csv` — Bonus probability threshold tuning table.
- `classification_comparison.png` — Bar chart comparing all 4 classifiers across the 5 metrics.
- `confusion_matrix.png` — Confusion matrix for the Random Forest Classifier.
- `best_classifier.joblib` — Compressed (`~4.9 MB`) production dictionary (`classifier`, `scaler`, `feature_cols`) ready for the FastAPI backend.
- `requirements.txt` — Python dependencies.

## How to Run

```bash
cd satya
python my_classification.py
```
