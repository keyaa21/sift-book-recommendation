import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==========================================
# 1. LOAD DATASETS
# ==========================================
print("Loading datasets...")

# Adjust path if running from inside sneha/ folder
books_df = pd.read_csv("goodreads-10k/books_clustered.csv")
ratings_df = pd.read_csv("goodreads-10k/ratings.csv")

# Sample 150,000 ratings so it runs instantly and safely on a laptop
ratings_df = ratings_df.sample(n=min(150000, len(ratings_df)), random_state=42)

# ==========================================
# 2. FEATURE ENGINEERING (X and y)
# ==========================================
print("Engineering user and cluster features...")

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
cluster_affinity = df.groupby(["user_id", "cluster_id"])["rating"].mean().reset_index()
cluster_affinity.rename(columns={"rating": "user_cluster_affinity"}, inplace=True)

# Merge cluster affinity back into main dataframe
df = df.merge(cluster_affinity, on=["user_id", "cluster_id"], how="left")

# Clean missing values
df["user_cluster_affinity"] = df["user_cluster_affinity"].fillna(df["user_mean_rating"])
df.dropna(subset=["average_rating", "cluster_id"], inplace=True)

# Define feature matrix X and target vector y
feature_cols = [
    "average_rating",
    "ratings_count",
    "cluster_id",
    "user_mean_rating",
    "user_rating_count",
    "user_cluster_affinity"
]

X = df[feature_cols]
y = df["rating"]

# ==========================================
# 3. TRAIN-TEST SPLIT
# ==========================================
print("Splitting data into 80% Train / 20% Test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 4. MODEL TRAINING & COMPARISON
# ==========================================
print("Training the 4 regression models...")
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0, random_state=42),
    "LinearSVR": LinearSVR(random_state=42, max_iter=2000),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1)
}

results = []
best_model = None
best_rmse = float("inf")

for name, model in models.items():
    print(f" -> Training {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 1.0, 5.0)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "MSE": round(mse, 4),
        "R2 Score": round(r2, 4)
    })
    
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model

# ==========================================
# 5. EXPORT DELIVERABLES
# ==========================================
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("regression_metrics.csv", index=False)

print("\n================ BENCHMARK RESULTS ================")
print(metrics_df.to_string(index=False))
print("===================================================\n")

joblib.dump(best_model, "best_regressor.joblib")

print("Files generated successfully:")
print("1. regression_metrics.csv")
print("2. best_regressor.joblib")