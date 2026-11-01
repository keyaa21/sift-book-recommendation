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
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 0. RESOLVE DIRECTORIES DYNAMICALLY
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_DIR = os.path.join(ROOT_DIR, "goodreads-10k")

# ==========================================
# 1. LOAD DATASETS
# ==========================================
print("Step 1: Loading datasets...")
books_path = os.path.join(DATA_DIR, "books_clustered.csv")
ratings_path = os.path.join(DATA_DIR, "ratings.csv")

books_df = pd.read_csv(books_path)
ratings_df = pd.read_csv(
    ratings_path,
    usecols=["user_id", "book_id", "rating"],
    dtype={"user_id": "int32", "book_id": "int32", "rating": "float32"}
)

# Sample 150,000 ratings for fast and robust training
ratings_df = ratings_df.sample(n=min(150000, len(ratings_df)), random_state=42)
print(f"Loaded {len(books_df)} books and {len(ratings_df)} sampled ratings.")

# ==========================================
# 2. FEATURE ENGINEERING (X and y)
# ==========================================
print("Step 2: Engineering user and cluster features...")

# Calculate each user's overall mean rating and total rating count
user_stats = ratings_df.groupby("user_id").agg(
    user_mean_rating=("rating", "mean"),
    user_rating_count=("rating", "count")
).reset_index()

# Merge user statistics into ratings dataframe
df = ratings_df.merge(user_stats, on="user_id", how="left")

# Merge book statistics and cluster_id from Keya's clustered books
df = df.merge(
    books_df[["book_id", "average_rating", "ratings_count", "cluster_id"]],
    on="book_id",
    how="inner"
)

# Calculate user affinity score per cluster
cluster_affinity = (
    df.groupby(["user_id", "cluster_id"])["rating"]
    .mean()
    .reset_index()
    .rename(columns={"rating": "user_cluster_affinity"})
)

# Merge cluster affinity back into main dataframe
df = df.merge(cluster_affinity, on=["user_id", "cluster_id"], how="left")

# Clean missing values without deprecated inplace
df["user_cluster_affinity"] = df["user_cluster_affinity"].fillna(df["user_mean_rating"])
df = df.dropna(subset=["average_rating", "cluster_id"])

# Define feature matrix X and target vector y
feature_cols = [
    "average_rating",
    "ratings_count",
    "cluster_id",
    "user_mean_rating",
    "user_rating_count",
    "user_cluster_affinity"
]

X = df[feature_cols].copy()
y = df["rating"].values

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================
print("Step 3: Splitting data into 80% Train / 20% Test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features for models sensitive to scale (Linear Regression, Ridge, LinearSVR)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. MODEL TRAINING & COMPARISON
# ==========================================
print("Step 4: Training regression models...")

# Define models
models = {
    "Linear Regression": (LinearRegression(), True),
    "Ridge Regression": (Ridge(alpha=1.0, random_state=42), True),
    "LinearSVR": (LinearSVR(random_state=42, max_iter=3000, dual="auto"), True),
    "Random Forest Regressor": (
        RandomForestRegressor(
            n_estimators=50,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ),
        False
    )
}

results = []
trained_models = {}
best_model = None
best_name = None
best_rmse = float("inf")
best_scaler = None

for name, (model, use_scaled) in models.items():
    print(f" -> Training {name}...")
    X_tr = X_train_scaled if use_scaled else X_train
    X_te = X_test_scaled if use_scaled else X_test
    
    model.fit(X_tr, y_train)
    trained_models[name] = model
    
    y_pred = model.predict(X_te)
    y_pred = np.clip(y_pred, 1.0, 5.0)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "RMSE": round(float(rmse), 4),
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "R2 Score": round(float(r2), 4)
    })
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_name = name
        best_scaler = scaler if use_scaled else None

# ==========================================
# 5. EXPORT DELIVERABLES
# ==========================================
metrics_df = pd.DataFrame(results)

# Save metrics CSV in both sneha/ and root
metrics_csv_sneha = os.path.join(BASE_DIR, "regression_metrics.csv")
metrics_csv_root = os.path.join(ROOT_DIR, "regression_metrics.csv")
metrics_df.to_csv(metrics_csv_sneha, index=False)
metrics_df.to_csv(metrics_csv_root, index=False)

print("\n================ BENCHMARK RESULTS ================")
print(metrics_df.to_string(index=False))
print(f"================ Best Model: {best_name} (RMSE: {best_rmse:.4f}) ================\n")

# Attach metadata to best_model for full production self-containment
best_model.feature_cols_ = feature_cols
best_model.scaler_ = best_scaler
best_model.rmse_ = best_rmse

# 1. Save optimized best_regressor.joblib with lossless compression (compress=3)
model_path_sneha = os.path.join(BASE_DIR, "best_regressor.joblib")
model_path_root = os.path.join(ROOT_DIR, "best_regressor.joblib")

joblib.dump(best_model, model_path_sneha, compress=3)
joblib.dump(best_model, model_path_root, compress=3)

# 2. Also save full production bundle (dictionary with all models & metadata)
bundle = {
    "best_model": best_model,
    "best_model_name": best_name,
    "scaler": best_scaler,
    "feature_cols": feature_cols,
    "metrics": results,
    "models": {
        "linear_regression": trained_models["Linear Regression"],
        "ridge_regression": trained_models["Ridge Regression"],
        "random_forest": trained_models["Random Forest Regressor"]
    }
}
joblib.dump(bundle, os.path.join(BASE_DIR, "best_regressor_bundle.joblib"), compress=3)

# ==========================================
# 6. GENERATE PRESENTATION CHARTS
# ==========================================
print("Step 6: Generating presentation charts...")

# Chart 1: Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(9, 5))
plot_df = metrics_df.set_index("Model")[["RMSE", "MAE", "R2 Score"]]
plot_df.plot(kind="bar", ax=ax, rot=15, colormap="viridis")
plt.title("Sift Stage 3: Regression Model Benchmark Comparison", fontsize=13, fontweight="bold")
plt.ylabel("Score", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
chart1_path = os.path.join(BASE_DIR, "regression_comparison.png")
plt.savefig(chart1_path, dpi=300)
plt.close()

# Chart 2: Predicted vs Actual Scatter Plot (Sample 1000 points)
best_X_te = X_test_scaled if (best_scaler is not None) else X_test
sample_idx = np.random.choice(len(y_test), size=min(1000, len(y_test)), replace=False)
y_actual_sample = y_test[sample_idx]
y_pred_sample = np.clip(best_model.predict(best_X_te[sample_idx]), 1.0, 5.0)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_actual_sample, y_pred_sample, alpha=0.25, color="#2b5c8f", edgecolors="none")
ax.plot([1, 5], [1, 5], "--", color="#d9534f", linewidth=2, label="Ideal Fit (y = x)")
plt.title(f"Predicted vs. Actual Ratings ({best_name})", fontsize=13, fontweight="bold")
plt.xlabel("Actual Rating", fontsize=11)
plt.ylabel("Predicted Rating", fontsize=11)
plt.xlim(0.8, 5.2)
plt.ylim(0.8, 5.2)
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(loc="upper left")
plt.tight_layout()
chart2_path = os.path.join(BASE_DIR, "predicted_vs_actual.png")
plt.savefig(chart2_path, dpi=300)
plt.close()

size_kb = os.path.getsize(model_path_root) / 1024
size_mb = size_kb / 1024

print("\nAll deliverables generated successfully:")
print(f"1. {model_path_sneha} & {model_path_root}")
print(f"   -> Optimized Model File Size: {size_kb:.2f} KB ({size_mb:.4f} MB) [100% GitHub-Safe!]")
print(f"2. {metrics_csv_sneha} & {metrics_csv_root}")
print(f"3. {chart1_path}")
print(f"4. {chart2_path}")
print(f"5. {os.path.join(BASE_DIR, 'best_regressor_bundle.joblib')}")
