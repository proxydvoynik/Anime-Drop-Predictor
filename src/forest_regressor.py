"""
Anime Drop Episode Predictor — Improved Random Forest Model
============================================================
Improvements over Person A's Original Baseline:
  1. Drops `name` column (no title string memorization / overfitting)
  2. Expands `genres` into 20 binary flags (same as Person B)
  3. Adds 23 domain-specific synthetic interaction features
  4. Uses GroupShuffleSplit (random_state=42) for reproducible 80/20 splits
  5. Log1p target transformation to handle right-skewed episode distributions
  6. Tuned hyperparameters for generalization
  7. Prediction clamping to valid [1, episode_count] range
"""

import os
import sys
import warnings
import multiprocessing
import joblib
import numpy as np
import pandas as pd

# Windows multiprocessing safeguards
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())
warnings.filterwarnings("ignore")

try:
    import joblib.externals.loky.backend.context as loky_context
    loky_context._count_physical_cores = lambda: (multiprocessing.cpu_count(), None)
except Exception:
    pass

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Import shared utilities
sys.path.insert(0, os.path.dirname(__file__))
from train_split import get_train_test_splits
from train_person_b import add_synthetic_features

# Don't truncate the bucket-stats table when printing
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Constructs the preprocessor pipeline.
    - Numeric features: Impute median + StandardScaler
    - Categorical features: Impute + OrdinalEncoder (NOT OneHotEncoder)
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    transformers = [("num", numeric_transformer, numeric_cols)]
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    return ColumnTransformer(transformers=transformers)


def main():
    # =========================================================================
    # 1. Load data & apply synthetic feature engineering
    # =========================================================================
    print("[+] Loading data and engineering features...")
    X_train_raw, X_test_raw, y_train, y_test = get_train_test_splits(
        test_size=0.20, random_state=42
    )

    # Apply the same 43 synthetic features used by Person B
    # (drops `name` and `genres`, adds 20 genre binary flags + 23 interaction features)
    X_train = add_synthetic_features(X_train_raw)
    X_test = add_synthetic_features(X_test_raw)

    print(f"  * Train set: {X_train.shape[0]} rows × {X_train.shape[1]} features")
    print(f"  * Test set:  {X_test.shape[0]} rows × {X_test.shape[1]} features")
    print(f"  * Name column included: {'name' in X_train.columns}")

    # =========================================================================
    # 2. Baseline: dummy mean predictor
    # =========================================================================
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    dummy_preds = dummy.predict(X_test)

    dummy_mae = mean_absolute_error(y_test, dummy_preds)
    dummy_rmse = np.sqrt(mean_squared_error(y_test, dummy_preds))

    print(f"\n=== Baseline (Dummy Mean Predictor) ===")
    print(f"MAE:  {dummy_mae:.4f}")
    print(f"RMSE: {dummy_rmse:.4f}\n")

    # =========================================================================
    # 3. Build improved Random Forest pipeline
    # =========================================================================
    preprocessor = build_preprocessor(X_train)

    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=500,          # More trees for stability
            max_depth=18,              # Deeper trees for complex interactions
            min_samples_split=8,       # Fine-tuned regularization
            min_samples_leaf=5,        # Allow capturing smaller patterns
            max_features="sqrt",       # Standard RF decorrelation
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # =========================================================================
    # 4. Train on log1p(target) to handle right-skewed episode distributions
    # =========================================================================
    print("[+] Training Improved Random Forest (log1p target)...")
    y_train_log = np.log1p(y_train)
    rf_pipeline.fit(X_train, y_train_log)

    # =========================================================================
    # 5. Predict and invert log transform
    # =========================================================================
    rf_preds_log = rf_pipeline.predict(X_test)
    rf_preds_raw = np.expm1(rf_preds_log)

    # Clamp predictions to valid episode range [1, episode_count]
    episode_counts = X_test_raw["episode_count"].values
    rf_preds = np.clip(np.round(rf_preds_raw), 1, episode_counts)

    # =========================================================================
    # 6. Compute metrics
    # =========================================================================
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_medae = median_absolute_error(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)
    rf_r2_log = r2_score(np.log1p(y_test), np.log1p(rf_preds))

    print(f"\n{'=' * 65}")
    print(f"     IMPROVED RANDOM FOREST MODEL — EVALUATION METRICS")
    print(f"{'=' * 65}")
    print(f"  * Overall MAE            : {rf_mae:.4f} episodes")
    print(f"  * Median Absolute Error  : {rf_medae:.4f} episodes")
    print(f"  * Overall RMSE           : {rf_rmse:.4f} episodes")
    print(f"  * R² Score (Raw Scale)   : {rf_r2:.4f}")
    print(f"  * R² Score (Log Scale)   : {rf_r2_log:.4f}")
    print(f"  * Improvement over MAE   : {dummy_mae - rf_mae:.4f} episodes")

    # =========================================================================
    # 7. Cumulative tolerance accuracy
    # =========================================================================
    diffs = np.abs(y_test.values - rf_preds)
    n_total = len(y_test)

    print(f"\n{'=' * 65}")
    print(f"             CUMULATIVE TOLERANCE ACCURACY")
    print(f"{'=' * 65}")
    for tol in [0, 1, 2, 3, 5, 10]:
        acc = (diffs <= tol).mean() * 100
        count = (diffs <= tol).sum()
        print(f"  * Within ± {tol:2d} eps : {acc:6.2f}%  ({count:5d} / {n_total} samples)")

    # =========================================================================
    # 8. Per-bucket error breakdown
    # =========================================================================
    results = X_test_raw.copy()
    results["actual"] = y_test.values
    results["predicted"] = rf_preds
    results["abs_error"] = diffs
    results["squared_error"] = (results["actual"] - results["predicted"]) ** 2

    def bucket_label(ep):
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

    results["bucket"] = results["actual"].apply(bucket_label)

    bucket_stats = results.groupby("bucket").agg(
        count=("abs_error", "count"),
        mae=("abs_error", "mean"),
        median_ae=("abs_error", "median"),
        max_error=("abs_error", "max"),
        rmse=("squared_error", lambda s: np.sqrt(s.mean())),
    )

    bucket_order = ["ep1 (1)", "early (1-3)", "mid (4-12)", "late (13-50)", "ultra-late (51+)"]
    bucket_stats = bucket_stats.reindex(bucket_order)

    print(f"\n{'=' * 65}")
    print(f"             PER-BUCKET ERROR BREAKDOWN")
    print(f"{'=' * 65}")
    print(bucket_stats.round(3))

    # =========================================================================
    # 9. Top 10 worst predictions
    # =========================================================================
    print(f"\n{'=' * 65}")
    print(f"             TOP 10 WORST PREDICTIONS")
    print(f"{'=' * 65}")
    worst = results.sort_values("abs_error", ascending=False).head(10)
    print(worst[["actual", "predicted", "abs_error", "bucket"]].to_string())

    # =========================================================================
    # 10. Save model artifacts
    # =========================================================================
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_pipeline, "models/random_forest_model.joblib")
    joblib.dump(rf_pipeline, "data/processed/random_forest_model.pkl")
    print(f"\n[+] Model saved to models/random_forest_model.joblib")
    print(f"[+] Model saved to data/processed/random_forest_model.pkl")


if __name__ == "__main__":
    main()