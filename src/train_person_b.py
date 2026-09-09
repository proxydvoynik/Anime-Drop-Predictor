# """
# Anime Drop Episode Predictor — 5-Stage Specialist Ensemble Model
# =======================================================================
# Architecture:
#   - 5-Stage Probabilistic Specialist Routing (Ep1, Early, Mid, Late, Ultra-Late)
#   - 20 Binary Genre Flags + Domain-Specific Synthetic Interaction Features
#   - Log1p Target Transformation for Right-Skewed Specialist Regressors
#   - Weighted Sample Importance for Ultra-Late (51+ ep) Drops
#   - Hyperparameter-Tuned HistGradientBoosting Regressors + Global Fallback Anchor
#   - GroupShuffleSplit (80/20 User Split) Anti-Leakage Holdout Evaluation
# """

# import os
# import sys
# import warnings
# import multiprocessing
# from typing import Tuple

# import numpy as np
# import pandas as pd
# import joblib

# from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
# from sklearn.preprocessing import StandardScaler, OrdinalEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import (
#     HistGradientBoostingRegressor,
#     HistGradientBoostingClassifier,
#     VotingRegressor,
# )
# from sklearn.metrics import (
#     mean_absolute_error,
#     root_mean_squared_error,
#     mean_squared_error,
#     r2_score,
#     median_absolute_error,
#     explained_variance_score,
# )

# from train_split import get_train_test_splits

# # Environment configuration & CPU optimization for parallel training
# os.environ["PYTHONUTF8"] = "1"
# os.environ["PYTHONIOENCODING"] = "utf-8"
# os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())
# warnings.filterwarnings("ignore")

# try:
#     import joblib.externals.loky.backend.context as loky_context
#     loky_context._count_physical_cores = lambda: (multiprocessing.cpu_count(), None)
# except Exception:
#     pass

# # Top 20 genres to extract as binary indicator features
# TOP_GENRES = [
#     "Comedy", "Action", "Romance", "Fantasy", "School",
#     "Shounen", "Drama", "Supernatural", "Adventure", "Sci-Fi",
#     "Slice of Life", "Ecchi", "Mystery", "Seinen", "Magic",
#     "Harem", "Mecha", "Historical", "Sports", "Psychological",
# ]


# def add_synthetic_features(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Expands genres to 20 binary flags, drops raw high-cardinality string columns,
#     and adds domain-specific synthetic interaction features.
#     """
#     df = df.copy()

#     genres_str = df["genres"].fillna("").astype(str)
#     genre_count = genres_str.apply(lambda s: len(s.split(",")) if s.strip() else 0)

#     for genre in TOP_GENRES:
#         col_name = f"genre_{genre.lower().replace(' ', '_').replace('-', '_')}"
#         df[col_name] = genres_str.str.contains(genre, case=False, regex=False).astype(int)

#     df = df.drop(columns=["name", "genres"], errors="ignore")

#     drop_ep_map = {"early": 2.0, "midway": 6.5, "late": 18.0}
#     user_drop_num = df["user_avr_drop_ep"].map(drop_ep_map).fillna(6.5)
#     eps = df["episode_count"].clip(lower=1)

#     df["expected_drop_ratio"]    = user_drop_num / eps
#     df["drop_risk_factor"]       = df["show_baseline_drop_rate"] * (10.0 - df["score"])
#     df["log_episode_count"]      = np.log1p(eps)
#     df["log_member_count"]       = np.log1p(df["member_count"])
#     df["ptw_popularity_product"] = df["ptw_ratio"] * df["popularity_ratio"]
#     df["score_deviation"]        = df["score"] - 7.5
#     df["user_drop_ep_num"]       = user_drop_num

#     hist_map = {"<10": 0, "10-50": 1, "50+": 2}
#     comp_map = {"low": 0, "medium": 1, "high": 2}
#     df["user_conservatism"] = (
#         df["user_history_size"].map(hist_map).fillna(1)
#         + df["user_completion_rate"].map(comp_map).fillna(1)
#     )
#     df["drop_signal"]            = user_drop_num * df["show_baseline_drop_rate"]
#     df["score_x_log_members"]   = df["score"] * np.log1p(df["member_count"])
#     df["long_show_user_pref"]   = (
#         (eps > 50).astype(int) * (df["length_tolerance"] == "long").astype(int)
#     )
#     df["relative_drop_pos"]     = user_drop_num / eps.clip(lower=1)
#     df["ep_short"]              = (eps <= 13).astype(int)
#     df["ep_medium"]             = ((eps > 13) & (eps <= 52)).astype(int)
#     df["ep_long"]               = (eps > 52).astype(int)

#     df["score_squared"]          = df["score"] ** 2
#     df["fav_per_member_x_score"] = df["favorites_to_members_ratio"] * df["score"]
#     df["year_recency"]           = 2020.0 - df["airing_year"]
#     df["genre_count"]            = genre_count
#     df["is_long_running"]        = (eps > 100).astype(int)
#     df["user_engagement_score"]  = df["user_conservatism"] * df["score"]
#     df["baseline_risk_x_eps"]    = df["show_baseline_drop_rate"] * np.log1p(eps)

#     return df


# def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
#     """Constructs the preprocessor pipeline for numeric scaling and categorical encoding."""
#     num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
#     cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

#     return ColumnTransformer(transformers=[
#         ("num", Pipeline([
#             ("imp", SimpleImputer(strategy="median")),
#             ("scl", StandardScaler()),
#         ]), num_cols),
#         ("cat", Pipeline([
#             ("imp", SimpleImputer(strategy="most_frequent")),
#             ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
#         ]), cat_cols),
#     ], remainder="drop")


# # Module alias for cross-module unpickling compatibility
# import sys as _sys
# if __name__ == "__main__":
#     _sys.modules["train_person_b"] = _sys.modules["__main__"]


# class SpecialistEnsemble:
#     """
#     5-Stage Specialist Ensemble Classifier-Regressor Pipeline.
#     Routes input samples to segment-specific regressors and blends
#     results with a global fallback anchor.
#     """
#     __module__ = "train_person_b"

#     def _stage_labels(self, y_arr: np.ndarray) -> np.ndarray:
#         return np.select(
#             [y_arr <= 1, y_arr <= 3, y_arr <= 12, y_arr <= 50],
#             [0, 1, 2, 3],
#             default=4
#         )

#     def fit(self, X_ext: pd.DataFrame, y: pd.Series, ep_series: pd.Series):
#         """Fits preprocessor, routing classifier, 5 specialist regressors, and global fallback."""
#         y_arr = y.values
#         stage = self._stage_labels(y_arr)

#         print("  [1/7] Fitting shared preprocessor...")
#         self.preprocessor_ = build_preprocessor(X_ext)
#         X_np = self.preprocessor_.fit_transform(X_ext)

#         print("  [2/7] Training 5-stage routing classifier...")
#         self.classifier_ = HistGradientBoostingClassifier(
#             max_iter=500, learning_rate=0.015,
#             max_leaf_nodes=47, l2_regularization=0.3,
#             random_state=42,
#         )
#         self.classifier_.fit(X_np, stage)

#         print("  [3/7] Training Ep1 specialist (y=1)...")
#         m0 = stage == 0
#         self.ep1_ = TransformedTargetRegressor(
#             regressor=HistGradientBoostingRegressor(
#                 loss="absolute_error", max_iter=400,
#                 learning_rate=0.015, max_leaf_nodes=25,
#                 l2_regularization=0.4, min_samples_leaf=25, random_state=42,
#             ),
#             func=np.log1p, inverse_func=np.expm1,
#         )
#         self.ep1_.fit(X_np[m0], y_arr[m0])

#         print("  [4/7] Training Early specialist (eps 2-3)...")
#         m1 = stage == 1
#         self.early_ = TransformedTargetRegressor(
#             regressor=HistGradientBoostingRegressor(
#                 loss="absolute_error", max_iter=400,
#                 learning_rate=0.015, max_leaf_nodes=25,
#                 l2_regularization=0.4, min_samples_leaf=25, random_state=42,
#             ),
#             func=np.log1p, inverse_func=np.expm1,
#         )
#         self.early_.fit(X_np[m1], y_arr[m1])

#         print("  [5/7] Training Mid specialist (eps 4-12)...")
#         m2 = stage == 2
#         self.mid_ = HistGradientBoostingRegressor(
#             loss="absolute_error", max_iter=500,
#             learning_rate=0.015, max_leaf_nodes=63,
#             l2_regularization=0.2, min_samples_leaf=15, random_state=42,
#         )
#         self.mid_.fit(X_np[m2], y_arr[m2])

#         print("  [6/7] Training Late specialist (eps 13-50)...")
#         m3 = stage == 3
#         self.late_ = TransformedTargetRegressor(
#             regressor=HistGradientBoostingRegressor(
#                 loss="absolute_error", max_iter=600,
#                 learning_rate=0.01, max_leaf_nodes=79,
#                 l2_regularization=0.10, min_samples_leaf=6,
#                 random_state=42,
#             ),
#             func=np.log1p, inverse_func=np.expm1,
#         )
#         self.late_.fit(X_np[m3], y_arr[m3])

#         print("  [7/7] Training Ultra-Late specialist (eps 51+, weighted)...")
#         m4 = stage == 4
#         self.ultra_late_ = TransformedTargetRegressor(
#             regressor=HistGradientBoostingRegressor(
#                 loss="absolute_error", max_iter=700,
#                 learning_rate=0.01, max_leaf_nodes=95,
#                 l2_regularization=0.08, min_samples_leaf=3,
#                 random_state=42,
#             ),
#             func=np.log1p, inverse_func=np.expm1,
#         )
#         sample_weights_u = np.log1p(y_arr[m4])
#         self.ultra_late_.fit(X_np[m4], y_arr[m4], sample_weight=sample_weights_u)

#         gm1 = HistGradientBoostingRegressor(
#             loss="absolute_error", max_iter=400, learning_rate=0.02,
#             max_leaf_nodes=63, l2_regularization=0.15, min_samples_leaf=10, random_state=42,
#         )
#         gm2 = HistGradientBoostingRegressor(
#             loss="squared_error", max_iter=300, learning_rate=0.02,
#             max_leaf_nodes=47, l2_regularization=0.20, random_state=0,
#         )
#         self.global_ = TransformedTargetRegressor(
#             regressor=VotingRegressor(
#                 estimators=[("a", gm1), ("b", gm2)],
#                 weights=[0.7, 0.3],
#             ),
#             func=np.log1p, inverse_func=np.expm1,
#         )
#         self.global_.fit(X_np, y)
#         print("  All 5 specialists + global fallback anchor trained successfully.")
#         return self

#     def predict(self, X_ext: pd.DataFrame, ep_series: pd.Series) -> np.ndarray:
#         """Predicts drop episode clamped between 1 and episode_count."""
#         ep   = np.array(ep_series).ravel()
#         X_np = self.preprocessor_.transform(X_ext)

#         proba = self.classifier_.predict_proba(X_np)
#         p_0   = self.ep1_.predict(X_np)
#         p_1   = self.early_.predict(X_np)
#         p_2   = self.mid_.predict(X_np)
#         p_3   = self.late_.predict(X_np)
#         p_4   = self.ultra_late_.predict(X_np)
#         p_g   = self.global_.predict(X_np)

#         final = (
#             proba[:, 0] * (0.50 * p_0 + 0.30 * p_1 + 0.20 * p_g)
#             + proba[:, 1] * (0.25 * p_0 + 0.55 * p_1 + 0.20 * p_g)
#             + proba[:, 2] * (0.80 * p_2 + 0.20 * p_g)
#             + proba[:, 3] * (0.80 * p_3 + 0.20 * p_g)
#             + proba[:, 4] * (0.85 * p_4 + 0.15 * p_g)
#         )
#         return np.clip(np.round(final).astype(float), 1, ep)


# def train_and_evaluate() -> Tuple[float, float]:
#     """Runs model training on 80% train split and evaluates metrics on 20% test split."""
#     print("=" * 65)
#     print("   ANIME DROP PREDICTOR - 5-STAGE SPECIALIST ENSEMBLE MODEL")
#     print("=" * 65)

#     X_train, X_test, y_train, y_test = get_train_test_splits(test_size=0.20)
#     X_train_ext = add_synthetic_features(X_train)
#     X_test_ext  = add_synthetic_features(X_test)

#     print(f"Train Shape: {X_train_ext.shape} (80%) | Test Shape: {X_test_ext.shape} (20%)")

#     model = SpecialistEnsemble()
#     model.fit(X_train_ext, y_train, X_train["episode_count"])

#     preds = model.predict(X_test_ext, X_test["episode_count"])
#     y_true = y_test.values
#     abs_err = np.abs(y_true - preds)

#     mae   = mean_absolute_error(y_true, preds)
#     rmse  = root_mean_squared_error(y_true, preds)
#     mse   = mean_squared_error(y_true, preds)
#     r2    = r2_score(y_true, preds)
#     medae = median_absolute_error(y_true, preds)
#     evs   = explained_variance_score(y_true, preds)

#     print("\n" + "=" * 65)
#     print("                 OVERALL EVALUATION METRICS")
#     print("=" * 65)
#     print(f"  * Overall MAE            : {mae:.4f} episodes")
#     print(f"  * Median Absolute Error  : {medae:.4f} episodes")
#     print(f"  * Overall RMSE           : {rmse:.4f} episodes")
#     print(f"  * Overall MSE            : {mse:.4f}")
#     print(f"  * Overall R² Score       : {r2:.4f}")
#     print(f"  * Explained Variance     : {evs:.4f}")
#     print(f"  * Max Error              : {abs_err.max():.4f} episodes")

#     print("\n" + "=" * 65)
#     print("                 PER-BUCKET PERFORMANCE")
#     print("=" * 65)
#     ep1_m  = y_true == 1
#     early_m = (y_true >= 1) & (y_true <= 3)
#     mid_m   = (y_true > 3) & (y_true <= 12)
#     late_m  = (y_true > 12) & (y_true <= 50)
#     ultra_m = y_true > 50
#     overall_late_m = y_true > 12

#     print(f"  * Ep 1 Drops (y=1)    [n={ep1_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[ep1_m], preds[ep1_m]):.4f} | RMSE: {root_mean_squared_error(y_true[ep1_m], preds[ep1_m]):.4f}")
#     print(f"  * Early Drops (1-3)   [n={early_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[early_m], preds[early_m]):.4f} | RMSE: {root_mean_squared_error(y_true[early_m], preds[early_m]):.4f}")
#     print(f"  * Mid Drops (4-12)    [n={mid_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[mid_m], preds[mid_m]):.4f} | RMSE: {root_mean_squared_error(y_true[mid_m], preds[mid_m]):.4f}")
#     print(f"  * Late Drops (13-50)  [n={late_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[late_m], preds[late_m]):.4f} | RMSE: {root_mean_squared_error(y_true[late_m], preds[late_m]):.4f}")
#     print(f"  * Ultra-Late (51+)    [n={ultra_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[ultra_m], preds[ultra_m]):.4f} | RMSE: {root_mean_squared_error(y_true[ultra_m], preds[ultra_m]):.4f}")
#     print(f"  * Overall Late (13+)  [n={overall_late_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[overall_late_m], preds[overall_late_m]):.4f} | RMSE: {root_mean_squared_error(y_true[overall_late_m], preds[overall_late_m]):.4f}")

#     print("\n" + "=" * 65)
#     print("             CUMULATIVE TOLERANCE ACCURACY")
#     print("=" * 65)
#     for tol in [0, 1, 2, 3, 5, 10, 15, 20]:
#         pct = (abs_err <= tol).mean() * 100
#         cnt = (abs_err <= tol).sum()
#         print(f"  * Within ±{tol:2d} eps : {pct:5.2f}%  ({cnt:5d} / {len(y_true)} test samples)")

#     os.makedirs("models", exist_ok=True)
#     joblib.dump(model, "models/model_pipeline.joblib")
#     print("\n[+] Model successfully saved to: models/model_pipeline.joblib")
#     print("=" * 65 + "\n")
#     return mae, rmse


# if __name__ == "__main__":
#     train_and_evaluate()
"""
Anime Drop Episode Predictor — 5-Stage Specialist Ensemble Model
=======================================================================
Architecture:
  - 5-Stage Probabilistic Specialist Routing (Ep1, Early, Mid, Late, Ultra-Late)
  - 20 Binary Genre Flags + Domain-Specific Synthetic Interaction Features
  - Log1p Target Transformation for Right-Skewed Specialist Regressors
  - Weighted Sample Importance for Ultra-Late (51+ ep) Drops
  - Hyperparameter-Tuned HistGradientBoosting Regressors + Global Fallback Anchor
  - GroupShuffleSplit (80/20 User Split) Anti-Leakage Holdout Evaluation
"""

import os
import sys
import warnings
import multiprocessing
from typing import Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
    VotingRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score,
)

from train_split import get_train_test_splits

# Environment configuration & CPU optimization for parallel training
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["LOKY_MAX_CPU_COUNT"] = str(multiprocessing.cpu_count())
warnings.filterwarnings("ignore")

try:
    import joblib.externals.loky.backend.context as loky_context
    loky_context._count_physical_cores = lambda: (multiprocessing.cpu_count(), None)
except Exception:
    pass

# Top 20 genres to extract as binary indicator features
TOP_GENRES = [
    "Comedy", "Action", "Romance", "Fantasy", "School",
    "Shounen", "Drama", "Supernatural", "Adventure", "Sci-Fi",
    "Slice of Life", "Ecchi", "Mystery", "Seinen", "Magic",
    "Harem", "Mecha", "Historical", "Sports", "Psychological",
]


def add_synthetic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expands genres to 20 binary flags, drops raw high-cardinality string columns,
    and adds domain-specific synthetic interaction features.
    """
    df = df.copy()

    genres_str = df["genres"].fillna("").astype(str)
    genre_count = genres_str.apply(lambda s: len(s.split(",")) if s.strip() else 0)

    for genre in TOP_GENRES:
        col_name = f"genre_{genre.lower().replace(' ', '_').replace('-', '_')}"
        df[col_name] = genres_str.str.contains(genre, case=False, regex=False).astype(int)

    df = df.drop(columns=["name", "genres"], errors="ignore")

    drop_ep_map = {"early": 2.0, "midway": 6.5, "late": 18.0}
    user_drop_num = df["user_avr_drop_ep"].map(drop_ep_map).fillna(6.5)
    eps = df["episode_count"].clip(lower=1)

    df["expected_drop_ratio"]    = user_drop_num / eps
    df["drop_risk_factor"]       = df["show_baseline_drop_rate"] * (10.0 - df["score"])
    df["log_episode_count"]      = np.log1p(eps)
    df["log_member_count"]       = np.log1p(df["member_count"])
    df["ptw_popularity_product"] = df["ptw_ratio"] * df["popularity_ratio"]
    df["score_deviation"]        = df["score"] - 7.5
    df["user_drop_ep_num"]       = user_drop_num

    hist_map = {"<10": 0, "10-50": 1, "50+": 2}
    comp_map = {"low": 0, "medium": 1, "high": 2}
    df["user_conservatism"] = (
        df["user_history_size"].map(hist_map).fillna(1)
        + df["user_completion_rate"].map(comp_map).fillna(1)
    )
    df["drop_signal"]            = user_drop_num * df["show_baseline_drop_rate"]
    df["score_x_log_members"]   = df["score"] * np.log1p(df["member_count"])
    df["long_show_user_pref"]   = (
        (eps > 50).astype(int) * (df["length_tolerance"] == "long").astype(int)
    )
    df["relative_drop_pos"]     = user_drop_num / eps.clip(lower=1)
    df["ep_short"]              = (eps <= 13).astype(int)
    df["ep_medium"]             = ((eps > 13) & (eps <= 52)).astype(int)
    df["ep_long"]               = (eps > 52).astype(int)

    df["score_squared"]          = df["score"] ** 2
    df["fav_per_member_x_score"] = df["favorites_to_members_ratio"] * df["score"]
    df["year_recency"]           = 2020.0 - df["airing_year"]
    df["genre_count"]            = genre_count
    df["is_long_running"]        = (eps > 100).astype(int)
    df["user_engagement_score"]  = df["user_conservatism"] * df["score"]
    df["baseline_risk_x_eps"]    = df["show_baseline_drop_rate"] * np.log1p(eps)

    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Constructs the preprocessor pipeline for numeric scaling and categorical encoding."""
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    return ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ], remainder="drop")


# Module alias for cross-module unpickling compatibility
import sys as _sys
if __name__ == "__main__":
    _sys.modules["train_person_b"] = _sys.modules["__main__"]


class SpecialistEnsemble:
    """
    5-Stage Specialist Ensemble Classifier-Regressor Pipeline.
    Routes input samples to segment-specific regressors and blends
    results with a global fallback anchor.
    """
    __module__ = "train_person_b"

    def _stage_labels(self, y_arr: np.ndarray) -> np.ndarray:
        return np.select(
            [y_arr <= 1, y_arr <= 3, y_arr <= 12, y_arr <= 50],
            [0, 1, 2, 3],
            default=4
        )

    def fit(self, X_ext: pd.DataFrame, y: pd.Series, ep_series: pd.Series):
        """Fits preprocessor, routing classifier, 5 specialist regressors, and global fallback."""
        y_arr = y.values
        stage = self._stage_labels(y_arr)

        print("  [1/7] Fitting shared preprocessor...")
        self.preprocessor_ = build_preprocessor(X_ext)
        X_np = self.preprocessor_.fit_transform(X_ext)

        print("  [2/7] Training 5-stage routing classifier...")
        self.classifier_ = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.015,
            max_leaf_nodes=47, l2_regularization=0.3,
            random_state=42,
        )
        self.classifier_.fit(X_np, stage)

        print("  [3/7] Training Ep1 specialist (y=1)...")
        m0 = stage == 0
        self.ep1_ = TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=400,
                learning_rate=0.015, max_leaf_nodes=25,
                l2_regularization=0.4, min_samples_leaf=25, random_state=42,
            ),
            func=np.log1p, inverse_func=np.expm1,
        )
        self.ep1_.fit(X_np[m0], y_arr[m0])

        print("  [4/7] Training Early specialist (eps 2-3)...")
        m1 = stage == 1
        self.early_ = TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=400,
                learning_rate=0.015, max_leaf_nodes=25,
                l2_regularization=0.4, min_samples_leaf=25, random_state=42,
            ),
            func=np.log1p, inverse_func=np.expm1,
        )
        self.early_.fit(X_np[m1], y_arr[m1])

        print("  [5/7] Training Mid specialist (eps 4-12)...")
        m2 = stage == 2
        self.mid_ = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=500,
            learning_rate=0.015, max_leaf_nodes=63,
            l2_regularization=0.2, min_samples_leaf=15, random_state=42,
        )
        self.mid_.fit(X_np[m2], y_arr[m2])

        print("  [6/7] Training Late specialist (eps 13-50)...")
        m3 = stage == 3
        self.late_ = TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=600,
                learning_rate=0.01, max_leaf_nodes=79,
                l2_regularization=0.10, min_samples_leaf=6,
                random_state=42,
            ),
            func=np.log1p, inverse_func=np.expm1,
        )
        self.late_.fit(X_np[m3], y_arr[m3])

        print("  [7/7] Training Ultra-Late specialist (eps 51+, weighted)...")
        m4 = stage == 4
        self.ultra_late_ = TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                loss="absolute_error", max_iter=700,
                learning_rate=0.01, max_leaf_nodes=95,
                l2_regularization=0.08, min_samples_leaf=3,
                random_state=42,
            ),
            func=np.log1p, inverse_func=np.expm1,
        )
        sample_weights_u = np.log1p(y_arr[m4])
        self.ultra_late_.fit(X_np[m4], y_arr[m4], sample_weight=sample_weights_u)

        gm1 = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=400, learning_rate=0.02,
            max_leaf_nodes=63, l2_regularization=0.15, min_samples_leaf=10, random_state=42,
        )
        gm2 = HistGradientBoostingRegressor(
            loss="squared_error", max_iter=300, learning_rate=0.02,
            max_leaf_nodes=47, l2_regularization=0.20, random_state=0,
        )
        self.global_ = TransformedTargetRegressor(
            regressor=VotingRegressor(
                estimators=[("a", gm1), ("b", gm2)],
                weights=[0.7, 0.3],
            ),
            func=np.log1p, inverse_func=np.expm1,
        )
        self.global_.fit(X_np, y)
        print("  All 5 specialists + global fallback anchor trained successfully.")
        return self

    def predict(self, X_ext: pd.DataFrame, ep_series: pd.Series) -> np.ndarray:
        """Predicts drop episode clamped between 1 and episode_count."""
        ep   = np.array(ep_series).ravel()
        X_np = self.preprocessor_.transform(X_ext)

        proba = self.classifier_.predict_proba(X_np)
        p_0   = self.ep1_.predict(X_np)
        p_1   = self.early_.predict(X_np)
        p_2   = self.mid_.predict(X_np)
        p_3   = self.late_.predict(X_np)
        p_4   = self.ultra_late_.predict(X_np)
        p_g   = self.global_.predict(X_np)

        final = (
            proba[:, 0] * (0.50 * p_0 + 0.30 * p_1 + 0.20 * p_g)
            + proba[:, 1] * (0.25 * p_0 + 0.55 * p_1 + 0.20 * p_g)
            + proba[:, 2] * (0.80 * p_2 + 0.20 * p_g)
            + proba[:, 3] * (0.80 * p_3 + 0.20 * p_g)
            + proba[:, 4] * (0.85 * p_4 + 0.15 * p_g)
        )
        return np.clip(np.round(final).astype(float), 1, ep)


def train_and_evaluate() -> Tuple[float, float]:
    """Runs model training on 80% train split and evaluates metrics on 20% test split."""
    print("=" * 65)
    print("   ANIME DROP PREDICTOR - 5-STAGE SPECIALIST ENSEMBLE MODEL")
    print("=" * 65)

    # FIX: previously called without random_state, so this model was scored
    # on a DIFFERENT train/test split than forrestregressor.py's RF model
    # (which pins random_state=42). Pinned here to match, so MAE/RMSE
    # comparisons between the two models are actually apples-to-apples.
    X_train, X_test, y_train, y_test = get_train_test_splits(test_size=0.20, random_state=42)
    X_train_ext = add_synthetic_features(X_train)
    X_test_ext  = add_synthetic_features(X_test)

    print(f"Train Shape: {X_train_ext.shape} (80%) | Test Shape: {X_test_ext.shape} (20%)")

    model = SpecialistEnsemble()
    model.fit(X_train_ext, y_train, X_train["episode_count"])

    preds = model.predict(X_test_ext, X_test["episode_count"])
    y_true = y_test.values
    abs_err = np.abs(y_true - preds)

    mae   = mean_absolute_error(y_true, preds)
    rmse  = root_mean_squared_error(y_true, preds)
    mse   = mean_squared_error(y_true, preds)
    r2    = r2_score(y_true, preds)
    medae = median_absolute_error(y_true, preds)
    evs   = explained_variance_score(y_true, preds)

    print("\n" + "=" * 65)
    print("                 OVERALL EVALUATION METRICS")
    print("=" * 65)
    print(f"  * Overall MAE            : {mae:.4f} episodes")
    print(f"  * Median Absolute Error  : {medae:.4f} episodes")
    print(f"  * Overall RMSE           : {rmse:.4f} episodes")
    print(f"  * Overall MSE            : {mse:.4f}")
    print(f"  * Overall R² Score       : {r2:.4f}")
    print(f"  * Explained Variance     : {evs:.4f}")
    print(f"  * Max Error              : {abs_err.max():.4f} episodes")

    print("\n" + "=" * 65)
    print("                 PER-BUCKET PERFORMANCE")
    print("=" * 65)
    ep1_m  = y_true == 1
    early_m = (y_true >= 1) & (y_true <= 3)
    mid_m   = (y_true > 3) & (y_true <= 12)
    late_m  = (y_true > 12) & (y_true <= 50)
    ultra_m = y_true > 50
    overall_late_m = y_true > 12

    print(f"  * Ep 1 Drops (y=1)    [n={ep1_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[ep1_m], preds[ep1_m]):.4f} | RMSE: {root_mean_squared_error(y_true[ep1_m], preds[ep1_m]):.4f}")
    print(f"  * Early Drops (1-3)   [n={early_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[early_m], preds[early_m]):.4f} | RMSE: {root_mean_squared_error(y_true[early_m], preds[early_m]):.4f}")
    print(f"  * Mid Drops (4-12)    [n={mid_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[mid_m], preds[mid_m]):.4f} | RMSE: {root_mean_squared_error(y_true[mid_m], preds[mid_m]):.4f}")
    print(f"  * Late Drops (13-50)  [n={late_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[late_m], preds[late_m]):.4f} | RMSE: {root_mean_squared_error(y_true[late_m], preds[late_m]):.4f}")
    print(f"  * Ultra-Late (51+)    [n={ultra_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[ultra_m], preds[ultra_m]):.4f} | RMSE: {root_mean_squared_error(y_true[ultra_m], preds[ultra_m]):.4f}")
    print(f"  * Overall Late (13+)  [n={overall_late_m.sum():5d}] -> MAE: {mean_absolute_error(y_true[overall_late_m], preds[overall_late_m]):.4f} | RMSE: {root_mean_squared_error(y_true[overall_late_m], preds[overall_late_m]):.4f}")

    print("\n" + "=" * 65)
    print("             CUMULATIVE TOLERANCE ACCURACY")
    print("=" * 65)
    for tol in [0, 1, 2, 3, 5, 10, 15, 20]:
        pct = (abs_err <= tol).mean() * 100
        cnt = (abs_err <= tol).sum()
        print(f"  * Within ±{tol:2d} eps : {pct:5.2f}%  ({cnt:5d} / {len(y_true)} test samples)")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model_pipeline.joblib")
    print("\n[+] Model successfully saved to: models/model_pipeline.joblib")
    print("=" * 65 + "\n")
    return mae, rmse


if __name__ == "__main__":
    train_and_evaluate()