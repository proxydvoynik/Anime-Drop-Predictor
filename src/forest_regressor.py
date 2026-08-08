import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Don't truncate the bucket-stats table when printing
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/processed/phase1_training.csv")

TARGET = "target_drop_episode"
GROUP_COL = "user_id"

NUMERICAL_FEATURES = [
    "score", "episode_count", "member_count", "airing_year", "duration_minutes",
    "is_source_original", "favorites_to_members_ratio", "ptw_ratio",
    "score_std_dev", "show_baseline_drop_rate", "popularity_ratio",
    "drops_slow_start", "length_fit",
]

CATEGORICAL_FEATURES = [
    "genres", "user_history_size", "user_completion_rate",
    "user_avr_drop_ep", "length_tolerance", "status_preference", "name",
]
# Note: the original guide listed "rating" and "airing_status" as categorical
# features, but the actual training CSV doesn't contain those columns, so
# they're excluded here. It also doesn't use "anime_id" as a feature (it's
# an identifier, like user_id, not a predictive signal).

X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]
groups = df[GROUP_COL]

# ---------------------------------------------------------------------------
# 2. Anti-leakage split: GroupKFold by user_id
# ---------------------------------------------------------------------------
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(X, y, groups=groups))

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
# IMPORTANT: y_val stays untouched, in RAW episode-number scale, the whole
# script. It is your ground truth — you only ever transform PREDICTIONS
# back to match it, never the other way around.

assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[val_idx])), \
    "Leakage! A user_id appears in both train and validation."

# ---------------------------------------------------------------------------
# 3. Preprocessing — fit ONLY on training fold
# ---------------------------------------------------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERICAL_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# ---------------------------------------------------------------------------
# 4. Baseline: dummy mean predictor (raw scale, for reference)
# ---------------------------------------------------------------------------
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
dummy_preds = dummy.predict(X_val)

dummy_mae = mean_absolute_error(y_val, dummy_preds)
dummy_rmse = np.sqrt(mean_squared_error(y_val, dummy_preds))

print("=== Baseline (Dummy Mean Predictor) ===")
print(f"MAE:  {dummy_mae:.4f}")
print(f"RMSE: {dummy_rmse:.4f}\n")

# ---------------------------------------------------------------------------
# 5. Random Forest pipeline + hyperparameter tuning — TRAINED ON LOG TARGET
# ---------------------------------------------------------------------------
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(random_state=42, n_jobs=-1)),
])

param_grid = {
    "regressor__n_estimators": [300],
    "regressor__max_depth": [10],
    "regressor__min_samples_split": [10],
    "regressor__min_samples_leaf": [12],
}

# --- Train on RAW target, no log-transform, no sample weights ---
search = GridSearchCV(
    rf_pipeline,
    param_grid,
    scoring="neg_mean_absolute_error",
    cv=3,
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

best_pipeline = search.best_estimator_
joblib.dump(best_pipeline, "data/processed/random_forest_model.pkl")
print("=== Best Random Forest Hyperparameters ===")
print(search.best_params_, "\n")

# --- predict directly, already in RAW scale, no conversion needed ---
rf_preds = best_pipeline.predict(X_val)

# --- STEP E: now compare REAL predictions to REAL y_val ---
rf_mae = mean_absolute_error(y_val, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
rf_r2 = r2_score(y_val, rf_preds)

print("=== Random Forest (raw target, no tricks) — Validation Performance ===")
print(f"MAE:  {rf_mae:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R²:   {rf_r2:.4f}")
print(f"Improvement over baseline MAE: {dummy_mae - rf_mae:.4f}\n")

# ---------------------------------------------------------------------------
# 6. Error breakdown by episode-drop bucket — uses REAL-scale rf_preds
# ---------------------------------------------------------------------------
results = X_val.copy()
results["actual"] = y_val.values
results["predicted"] = rf_preds        # already converted back in step D
results["abs_error"] = np.abs(results["actual"] - results["predicted"])
results["squared_error"] = (results["actual"] - results["predicted"]) ** 2


def bucket(ep):
    if ep <= 3:
        return "early (1-3)"
    elif ep <= 12:
        return "midway (4-12)"
    else:
        return "late (13+)"


def bucket_matching_friend(ep):
    """Same cutoffs as Person B's ensemble, for direct apples-to-apples comparison."""
    if ep == 1:
        return "ep1 (1)"
    elif ep <= 3:
        return "early (1-3)"
    elif ep <= 12:
        return "mid (4-12)"
    elif ep <= 50:
        return "late (13-50)"
    else:
        return "ultra-late (51+)"


results["bucket"] = results["actual"].apply(bucket)
results["bucket_matching_friend"] = results["actual"].apply(bucket_matching_friend)

# --- Per-bucket metrics: MAE, median AE, max error, RMSE, count ---
# Why each one matters, beyond MAE alone:
#   median AE - the "typical" error, immune to a few huge outliers dragging
#               the mean up. If median is much lower than MAE, a handful of
#               bad predictions are inflating the mean.
#   max error - your single worst prediction in that bucket. Useful for
#               spotting "how bad can it possibly get" for a report.
#   RMSE      - punishes big misses harder than MAE (squares errors first).
#               A big RMSE-vs-MAE gap in a bucket = a few large misses there,
#               not consistently-moderate error across the whole bucket.
bucket_stats = results.groupby("bucket").agg(
    count=("abs_error", "count"),
    mae=("abs_error", "mean"),
    median_ae=("abs_error", "median"),
    max_error=("abs_error", "max"),
    rmse=("squared_error", lambda s: np.sqrt(s.mean())),
)

print("=== Error Breakdown by Drop-Episode Bucket (your original 3-bucket scheme) ===")
print(bucket_stats.round(3))
print()

# --- Same cutoffs as your friend's ensemble, for a true apples-to-apples check ---
bucket_stats_matching = results.groupby("bucket_matching_friend").agg(
    count=("abs_error", "count"),
    mae=("abs_error", "mean"),
    median_ae=("abs_error", "median"),
    max_error=("abs_error", "max"),
    rmse=("squared_error", lambda s: np.sqrt(s.mean())),
)
# reorder rows to match the natural episode progression, not alphabetical
bucket_order = ["ep1 (1)", "early (1-3)", "mid (4-12)", "late (13-50)", "ultra-late (51+)"]
bucket_stats_matching = bucket_stats_matching.reindex(bucket_order)

print("=== Error Breakdown Using Your Friend's Bucket Cutoffs (for direct comparison) ===")
print(bucket_stats_matching.round(3))
print()

print("=== Top 10 Worst Predictions (for the project report) ===")
worst = results.sort_values("abs_error", ascending=False).head(10)
print(worst[["actual", "predicted", "abs_error", "bucket"]])

# ---------------------------------------------------------------------------
# 7. Hand off to Person B / joint selection step
#    best_pipeline.predict() outputs RAW episode-number scale directly —
#    no log/expm1 conversion needed for this run.
# ---------------------------------------------------------------------------