import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from train_person_b import add_synthetic_features  
from jikkan import fetch_from_user                 
from anime_lookup import get_anime_features          

RF_WEIGHT = 0.4
ENSEMBLE_WEIGHT = 0.6

# The exact raw columns both models expect (post add_synthetic_features
# input, i.e. BEFORE it's applied).
REQUIRED_RAW_COLUMNS = [
    "anime_id", "score", "genres", "episode_count", "member_count",
    "airing_year", "duration_minutes", "is_source_original",
    "favorites_to_members_ratio", "ptw_ratio", "score_std_dev",
    "show_baseline_drop_rate", "popularity_ratio", "user_history_size",
    "user_completion_rate", "user_avr_drop_ep", "drops_slow_start",
    "length_tolerance", "status_preference", "length_fit",
]

_ENSEMBLE_MODEL = None
_RF_MODEL = None

# These 6 features come from a user's WATCH HISTORY, not the anime itself
# — there's no way to derive them for someone typing in a title with no
# MAL account attached. This is a generic "typical user" stand-in used
# when no username is given to predict_by_title().
DEFAULT_USER_FEATURES = {
    "user_history_size": "10-50",
    "user_completion_rate": "medium",
    "user_avr_drop_ep": "midway",
    "drops_slow_start": 0,
    "length_tolerance": "medium",
    "status_preference": "completed",
}


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


def _get_user_features(username: str = None) -> dict:
    """
    Returns the 6 per-user engineered features either from a real MAL
    account (if given) or a generic 'typical user' default profile
    (if not).
    """
    if not username:
        print("[i] No MAL username given — using a generic 'typical user' "
              "profile for the user-level features.")
        return dict(DEFAULT_USER_FEATURES)

    user_df = fetch_from_user(username)
    if user_df.empty:
        print(f"[!] Could not fetch '{username}' — falling back to generic user profile.")
        return dict(DEFAULT_USER_FEATURES)

    row = user_df.iloc[0]
    return {col: row[col] for col in DEFAULT_USER_FEATURES}


def predict_by_title(title: str, username: str = None, watched_episodes: int = 0):
    """
    Predicts a drop episode for ANY anime by typing its title directly —
    it doesn't need to be on anyone's MAL "Currently Watching" list.

    username: optional. If given, uses that account's real watch-history
        features (user_history_size, user_completion_rate, etc.). If
        omitted, uses a generic 'typical user' default profile instead —
        useful for a quick "where might people generally drop this"
        check without needing a MAL account at all.
    watched_episodes: how many episodes in you already are, if any
        (defaults to 0, i.e. "haven't started").

    Returns the predicted drop episode (float), or None if the title
    isn't found in the catalog.
    """
    anime_features = get_anime_features(anime_id=-1, name=title)
    if anime_features is None:
        print(f"[!] '{title}' not found in the anime catalog — can't predict.")
        return None

    user_features = _get_user_features(username)

    anime_data = {
        **anime_features,
        **user_features,
        "anime_id": -1,
        "watched_episodes": watched_episodes,
        "length_fit": 1,
    }

    missing = [
        c for c in REQUIRED_RAW_COLUMNS
        if c not in anime_data or pd.isna(anime_data[c])
    ]
    if missing:
        print(f"[!] Missing/NaN fields, cannot predict: {missing}")
        return None

    return predict_drop_episode(anime_data)


def predict_for_mal_user(username: str, only_currently_watching: bool = True) -> pd.DataFrame:
    """
    Fetches a live MAL user's list (user-level features only — see
    jikkan.fetch_from_user), then for each anime that needs a prediction,
    looks up ITS static features from the training catalog on demand
    (see anime_lookup.get_anime_features) and predicts a drop episode.

    only_currently_watching: if True (default), only predicts for shows
    with watching_status == 1 (still watching) — predicting for shows
    already completed/dropped is moot, the real outcome's already known.
    Set False to predict across the whole fetched list anyway.
    """
    print(f"[+] Fetching MAL data for '{username}'...")
    user_df = fetch_from_user(username)

    if user_df.empty:
        print(f"[!] No data returned for '{username}' — Jikan/MAL may be "
              f"rate-limiting or unavailable, or the list is empty. Try again shortly.")
        return pd.DataFrame(columns=["name", "predicted_drop_episode"])

    if only_currently_watching:
        user_df = user_df[user_df["watching_status"] == 1]

    if user_df.empty:
        print("[!] No matching anime found for this user/filter.")
        return pd.DataFrame(columns=["name", "predicted_drop_episode"])

    print(f"[i] Predicting for {len(user_df)} anime...")

    results = []
    for _, row in user_df.iterrows():
        # This is the only place a catalog lookup happens now — one call
        # per anime actually being predicted, not the user's whole history.
        anime_features = get_anime_features(row["anime_id"], row.get("name", ""))
        if anime_features is None:
            print(f"[!] Skipping '{row.get('name', '?')}' — no catalog match, cannot predict")
            continue

        # Catalog features first, then the live/user-level fields from
        # jikkan on top — live data should win over anything the catalog
        # might have (e.g. airing_status can be stale in a static CSV).
        anime_data = {**anime_features, **row.to_dict()}

        # PLACEHOLDER — "length_fit" formula was never defined anywhere in
        # the original code. Still a stand-in.
        anime_data["length_fit"] = 1

        missing = [
            c for c in REQUIRED_RAW_COLUMNS
            if c not in anime_data or pd.isna(anime_data[c])
        ]
        if missing:
            print(f"[!] Skipping '{row.get('name', '?')}' — missing/NaN fields: {missing}")
            continue

        try:
            prediction = predict_drop_episode(anime_data)
        except Exception as e:
            print(f"[!] Skipping '{row.get('name', '?')}' — prediction failed: {e}")
            continue

        results.append({
            "name": row.get("name"),
            "episode_count": anime_data.get("episode_count"),
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

    # Predict for ANY anime by title — doesn't need to be on a MAL list.
    # Pass username=None to use a generic "typical user" profile instead
    # of a real account's watch-history features.
    title = "Attack on Titan"
    pred = predict_by_title(title, username=MAL_USERNAME, watched_episodes=5)
    if pred is not None:
        print(f"\n[+] Predicted drop episode for '{title}' "
              f"(watched {5} eps so far): {pred}")