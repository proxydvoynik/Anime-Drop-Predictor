# import os
# import sys
# import joblib
# import numpy as np
# import pandas as pd

# sys.path.insert(0, os.path.dirname(__file__))
# from train_person_b import add_synthetic_features  # required for both models' preprocessing
# from jikkan import fetch_from_user                  # your live MAL fetch pipeline
# # ^ import of train_person_b also makes SpecialistEnsemble's class definition
# #   available, which joblib needs when unpickling model_pipeline.joblib

# # NOE: 50/50 is a guess, not a measured choice. Per the weight-sweep
# # discussed earlier, you should pick these by testing several splits on
# # a shared validation set and taking whichever minimizes MAE.
# RF_WEIGHT = 0.5
# ENSEMBLE_WEIGHT = 0.5

# # The exact raw columns both models expect (post add_synthetic_features
# # input, i.e. BEFORE it's applied). fetch_from_user() returns extra columns
# # too (watching_status, user_score, episodes, etc.) — harmless, sklearn's
# # ColumnTransformer only pulls the columns it was fit on by name and
# # ignores the rest, but we select explicitly here to keep this obvious.
# REQUIRED_RAW_COLUMNS = [
#     "anime_id", "score", "genres", "episode_count", "member_count",
#     "airing_year", "duration_minutes", "is_source_original",
#     "favorites_to_members_ratio", "ptw_ratio", "score_std_dev",
#     "show_baseline_drop_rate", "popularity_ratio", "user_history_size",
#     "user_completion_rate", "user_avr_drop_ep", "drops_slow_start",
#     "length_tolerance", "status_preference", "length_fit",
# ]


# def predict_drop_episode(anime_data: dict) -> float:
#     """
#     anime_data: the RAW 20-key input dict (before add_synthetic_features
#     has been applied) — score, episode_count, genres, etc.
#     """
#     row_raw = pd.DataFrame([anime_data])
#     row_ext = add_synthetic_features(row_raw)  # both models need this, not raw
#     episode_count = row_raw["episode_count"]

#     # --- Model 1: friend's 5-stage SpecialistEnsemble ---
#     ensemble_model = joblib.load("models/model_pipeline.joblib")
#     ensemble_pred = ensemble_model.predict(row_ext, episode_count)[0]

#     # --- Model 2: your Random Forest, trained on log1p(target) ---
#     rf_model = joblib.load("models/random_forest_model.joblib")
#     rf_pred_log = rf_model.predict(row_ext)[0]
#     rf_pred_raw = np.expm1(rf_pred_log)
#     rf_pred = np.clip(round(rf_pred_raw), 1, anime_data["episode_count"])

#     blended = RF_WEIGHT * rf_pred + ENSEMBLE_WEIGHT * ensemble_pred
#     return float(blended)


# def predict_for_mal_user(username: str, only_currently_watching: bool = True) -> pd.DataFrame:
#     """
#     Fetches a live MAL user's list via jikan.py, then predicts a drop
#     episode for each anime.

#     only_currently_watching: if True (default), only predicts for shows
#     with watching_status == 1 (still watching) — predicting for shows
#     already completed/dropped is moot, the real outcome's already known.
#     Set False to predict across the whole fetched list anyway.
#     """
#     print(f"[+] Fetching MAL data for '{username}'...")
#     user_df = fetch_from_user(username)

#     if only_currently_watching:
#         user_df = user_df[user_df["watching_status"] == 1]

#     if user_df.empty:
#         print("[!] No matching anime found for this user/filter.")
#         return pd.DataFrame(columns=["name", "predicted_drop_episode"])

#     results = []
#     for _, row in user_df.iterrows():
#         missing = [c for c in REQUIRED_RAW_COLUMNS if c not in row.index or pd.isna(row[c])]
#         if missing:
#             print(f"[!] Skipping '{row.get('name', '?')}' — missing/NaN fields: {missing}")
#             continue

#         anime_data = {col: row[col] for col in REQUIRED_RAW_COLUMNS}
#         try:
#             prediction = predict_drop_episode(anime_data)
#         except Exception as e:
#             print(f"[!] Skipping '{row.get('name', '?')}' — prediction failed: {e}")
#             continue

#         results.append({
#             "name": row.get("name"),
#             "episode_count": row.get("episode_count"),
#             "watched_episodes": row.get("watched_episodes"),
#             "predicted_drop_episode": round(prediction, 1),
#         })

#     results_df = pd.DataFrame(results)
#     return results_df


# if __name__ == "__main__":
#     MAL_USERNAME = "iitjeeair001"  # replace with a real MAL username

#     predictions = predict_for_mal_user(MAL_USERNAME, only_currently_watching=True)

#     print(f"\n{'=' * 60}")
#     print(f"  Drop-episode predictions for '{MAL_USERNAME}'")
#     print(f"{'=' * 60}")
#     print(predictions.to_string(index=False))
import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train_person_b import add_synthetic_features  # required for both models' preprocessing
from jikkan import fetch_from_user                  # your live MAL fetch pipeline
# ^ import of train_person_b also makes SpecialistEnsemble's class definition
#   available, which joblib needs when unpickling model_pipeline.joblib

# NOTE: 50/50 is a guess, not a measured choice. Per the weight-sweep
# discussed earlier, you should pick these by testing several splits on
# a shared validation set and taking whichever minimizes MAE.
RF_WEIGHT = 0.5
ENSEMBLE_WEIGHT = 0.5

# The exact raw columns both models expect (post add_synthetic_features
# input, i.e. BEFORE it's applied). fetch_from_user() returns extra columns
# too (watching_status, user_score, episodes, etc.) — harmless, sklearn's
# ColumnTransformer only pulls the columns it was fit on by name and
# ignores the rest, but we select explicitly here to keep this obvious.
REQUIRED_RAW_COLUMNS = [
    "anime_id", "score", "genres", "episode_count", "member_count",
    "airing_year", "duration_minutes", "is_source_original",
    "favorites_to_members_ratio", "ptw_ratio", "score_std_dev",
    "show_baseline_drop_rate", "popularity_ratio", "user_history_size",
    "user_completion_rate", "user_avr_drop_ep", "drops_slow_start",
    "length_tolerance", "status_preference", "length_fit",
]


# FIX: previously both models were joblib.load()-ed inside
# predict_drop_episode(), which was called once per anime in the loop below
# — i.e. both model files were deserialized from disk repeatedly for every
# single title in a user's list. Load them once here and reuse.
_ENSEMBLE_MODEL = None
_RF_MODEL = None


def _get_models():
    global _ENSEMBLE_MODEL, _RF_MODEL
    if _ENSEMBLE_MODEL is None:
        _ENSEMBLE_MODEL = joblib.load("models/model_pipeline.joblib")
    if _RF_MODEL is None:
        _RF_MODEL = joblib.load("models/random_forest_model.joblib")
    return _ENSEMBLE_MODEL, _RF_MODEL


def predict_drop_episode(anime_data: dict) -> float:
    """
    anime_data: the RAW 20-key input dict (before add_synthetic_features
    has been applied) — score, episode_count, genres, etc.
    """
    row_raw = pd.DataFrame([anime_data])
    row_ext = add_synthetic_features(row_raw)  # both models need this, not raw
    episode_count = row_raw["episode_count"]

    ensemble_model, rf_model = _get_models()

    # --- Model 1: friend's 5-stage SpecialistEnsemble ---
    ensemble_pred = ensemble_model.predict(row_ext, episode_count)[0]

    # --- Model 2: your Random Forest, trained on log1p(target) ---
    rf_pred_log = rf_model.predict(row_ext)[0]
    rf_pred_raw = np.expm1(rf_pred_log)
    rf_pred = np.clip(round(rf_pred_raw), 1, anime_data["episode_count"])

    blended = RF_WEIGHT * rf_pred + ENSEMBLE_WEIGHT * ensemble_pred
    return float(blended)


def predict_for_mal_user(username: str, only_currently_watching: bool = True) -> pd.DataFrame:
    """
    Fetches a live MAL user's list via jikkan.py, then predicts a drop
    episode for each anime.

    only_currently_watching: if True (default), only predicts for shows
    with watching_status == 1 (still watching) — predicting for shows
    already completed/dropped is moot, the real outcome's already known.
    Set False to predict across the whole fetched list anyway.
    """
    print(f"[+] Fetching MAL data for '{username}'...")
    user_df = fetch_from_user(username)

    # FIX: check for an empty result BEFORE filtering by watching_status.
    # If every anime failed to fetch (e.g. Jikan/MAL rate-limited or down),
    # fetch_from_user now returns an empty DataFrame with no columns at all.
    # Filtering `user_df["watching_status"] == 1` on that used to raise
    # KeyError before we ever got a chance to report "no data" cleanly.
    if user_df.empty:
        print(f"[!] No data returned for '{username}' — Jikan/MAL may be "
              f"rate-limiting or unavailable. Try again shortly.")
        return pd.DataFrame(columns=["name", "predicted_drop_episode"])

    if only_currently_watching:
        user_df = user_df[user_df["watching_status"] == 1]

    if user_df.empty:
        print("[!] No matching anime found for this user/filter.")
        return pd.DataFrame(columns=["name", "predicted_drop_episode"])

    results = []
    for _, row in user_df.iterrows():
        missing = [c for c in REQUIRED_RAW_COLUMNS if c not in row.index or pd.isna(row[c])]
        if missing:
            print(f"[!] Skipping '{row.get('name', '?')}' — missing/NaN fields: {missing}")
            continue

        anime_data = {col: row[col] for col in REQUIRED_RAW_COLUMNS}
        try:
            prediction = predict_drop_episode(anime_data)
        except Exception as e:
            print(f"[!] Skipping '{row.get('name', '?')}' — prediction failed: {e}")
            continue

        results.append({
            "name": row.get("name"),
            "episode_count": row.get("episode_count"),
            "watched_episodes": row.get("watched_episodes"),
            "predicted_drop_episode": round(prediction, 1),
        })

    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":
    MAL_USERNAME = "iitjeeair001"  # replace with a real MAL username

    predictions = predict_for_mal_user(MAL_USERNAME, only_currently_watching=True)

    print(f"\n{'=' * 60}")
    print(f"  Drop-episode predictions for '{MAL_USERNAME}'")
    print(f"{'=' * 60}")
    print(predictions.to_string(index=False))