import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

# Resolve paths relative to satya/ and ../goodreads-10k/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "goodreads-10k"))

# ==========================================
# 1. LOAD DATA (Directly from Keya's books_clustered.csv + ratings.csv)
# ==========================================
print("Step 1: Loading Goodreads ratings and Keya's books_clustered.csv...")
ratings_path = os.path.join(DATA_DIR, "ratings.csv")
clustered_path = os.path.join(DATA_DIR, "books_clustered.csv")

ratings = pd.read_csv(
    ratings_path,
    usecols=["user_id", "book_id", "rating"],
    dtype={"user_id": "int32", "book_id": "int32", "rating": "int8"},
    nrows=200000
)  # 200k ratings for fast training

books = pd.read_csv(
    clustered_path,
    usecols=["book_id", "average_rating", "ratings_count", "cluster_id"],
    dtype={
        "book_id": "int32",
        "average_rating": "float32",
        "ratings_count": "int32",
        "cluster_id": "int16"
    }
)
print(f"Loaded {len(books)} pre-clustered books from {clustered_path} across {books['cluster_id'].nunique()} tag clusters.")

# ==========================================
# 2. FEATURE ENGINEERING FOR CLASSIFICATION
# ==========================================
print("Step 2: Engineering User + Book + Tag Cluster features...")
books["log_ratings_count"] = np.log1p(books["ratings_count"]).astype("float32")

# Merge book metadata with ratings first
df = ratings.merge(
    books[["book_id", "average_rating", "log_ratings_count", "cluster_id"]],
    on="book_id",
    how="inner"
)

# Vectorized user historical behavior stats & user-cluster affinity via groupby transform
user_grp = df.groupby("user_id")["rating"]
df["user_mean_rating"] = user_grp.transform("mean").astype("float32")
df["user_rating_count"] = user_grp.transform("count").astype("int32")
df["user_cluster_affinity"] = (
    df.groupby(["user_id", "cluster_id"])["rating"]
    .transform("mean")
    .astype("float32")
)

feature_cols = [
    "average_rating",
    "log_ratings_count",
    "cluster_id",
    "user_mean_rating",
    "user_rating_count",
    "user_cluster_affinity"
]

X = df[feature_cols].fillna(0)
# Binary Target: 1 if rating >= 4 (Liked), 0 if rating <= 3 (Not Liked)
y = (df["rating"] >= 4).astype(int)

print(f"Dataset ready: {len(df)} samples | Positive Class (Liked >= 4★): {y.mean()*100:.1f}%")

# 80% Train, 20% Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (Important for Logistic Regression & SVM!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. TRAIN & COMPARE 4 CLASSIFICATION MODELS
# ==========================================
print("\nStep 3: Training Naive Bayes, Logistic Regression, SVM, and Random Forest...")

models = {
    "Naive Bayes (GaussianNB)": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "SVM (Calibrated LinearSVC)": CalibratedClassifierCV(LinearSVC(random_state=42, max_iter=2000), cv=3),
    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=120, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
    )
}

results = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    trained_models[name] = model

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1-Score": round(f1_score(y_test, y_pred), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4)
    })

results_df = pd.DataFrame(results)
print("\n=== CLASSIFICATION MODEL COMPARISON TABLE ===")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(BASE_DIR, "classification_metrics.csv"), index=False)

# ==========================================
# 4. BONUS EXPERIMENT: THRESHOLD TUNING
# ==========================================
print("\n=== BONUS EXPERIMENT: PROBABILITY THRESHOLD TUNING (Random Forest) ===")
rf_probs = trained_models["Random Forest Classifier"].predict_proba(X_test_scaled)[:, 1]
thresh_rows = []
for thresh in [0.40, 0.50, 0.60, 0.70]:
    t_pred = (rf_probs >= thresh).astype(int)
    thresh_rows.append({
        "Threshold": thresh,
        "Precision": round(precision_score(y_test, t_pred), 4),
        "Recall": round(recall_score(y_test, t_pred), 4),
        "F1-Score": round(f1_score(y_test, t_pred), 4)
    })
thresh_df = pd.DataFrame(thresh_rows)
print(thresh_df.to_string(index=False))
thresh_df.to_csv(os.path.join(BASE_DIR, "threshold_tuning.csv"), index=False)

# ==========================================
# 5. SAVE CHARTS FOR YOUR PRESENTATION SLIDES
# ==========================================
# Chart 1: Bar Chart Comparing Models
ax = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]].plot(
    kind="bar", figsize=(10, 5), ylim=(0.5, 1.0), rot=15
)
plt.title("Sift Classification Ablation: Predicting User Book Preference (Rating >= 4)")
plt.ylabel("Score")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "classification_comparison.png"), dpi=300)
plt.close()

# Chart 2: Confusion Matrix of Best Model (Random Forest)
best_rf = trained_models["Random Forest Classifier"]
cm = confusion_matrix(y_test, best_rf.predict(X_test_scaled))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Liked (<=3★)", "Liked (>=4★)"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Random Forest Classifier - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"), dpi=300)
plt.close()

# ==========================================
# 6. SAVE ARTIFACT FOR LIVE SIFT DEPLOYMENT
# ==========================================
joblib.dump({
    "classifier": best_rf,
    "scaler": scaler,
    "feature_cols": feature_cols
}, os.path.join(BASE_DIR, "best_classifier.joblib"), compress=3)

print("\nDone! Saved in satya/:")
print("1. classification_metrics.csv (Put this table in your report)")
print("2. threshold_tuning.csv (Bonus experiment table)")
print("3. classification_comparison.png & confusion_matrix.png (Put these in your slides)")
print("4. best_classifier.joblib (Ready for FastAPI backend)")
