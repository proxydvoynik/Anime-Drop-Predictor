"""
Thin Flask API wrapper around predict.py's predict_drop_episode().
No changes to any existing Python files — just imports and calls.

Usage:
    python api.py

Serves on http://127.0.0.1:5000
"""

import os
import sys
import requests
import pandas as pd

# Make sure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow local web page to call us

# Load the anime dataset globally for fast search
try:
    anime_df_global = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'raw', 'anime.csv'))
    anime_df_global = anime_df_global.fillna("")
except Exception as e:
    print(f"Warning: Could not load data/raw/anime.csv. Search will not work. Error: {e}")
    anime_df_global = None

@app.route('/search_anime', methods=['GET'])
def search_anime():
    q = request.args.get("q", "").lower()
    if not q or anime_df_global is None:
        return jsonify({"data": []})
    
    # Simple search in Name or English name
    mask = anime_df_global["Name"].str.lower().str.contains(q, na=False, regex=False) | \
           anime_df_global["English name"].str.lower().str.contains(q, na=False, regex=False)
           
    matches = anime_df_global[mask].head(5)
    
    results = []
    import re
    for _, row in matches.iterrows():
        year = None
        if row["Aired"]:
            m = re.search(r'\b(19\d\d|20\d\d)\b', str(row["Aired"]))
            if m: year = int(m.group(1))
                
        genres_str = str(row["Genres"]) if row["Genres"] else ""
        genres = [{"name": g.strip()} for g in genres_str.split(",") if g.strip()]
        
        score_val = row["Score"]
        if score_val == "Unknown" or not score_val: score_val = 0.0
        
        ep_val = row["Episodes"]
        if ep_val == "Unknown" or not ep_val: ep_val = 0
        
        results.append({
            "mal_id": row["MAL_ID"],
            "title": row["Name"] if row["Name"] else row["English name"],
            "score": float(score_val),
            "episodes": int(ep_val),
            "year": year,
            "images": {
                "jpg": {
                    "image_url": "",
                    "small_image_url": "",
                    "large_image_url": ""
                }
            },
            "members": int(row["Members"]) if row["Members"] else 0,
            "favorites": int(row["Favorites"]) if row["Favorites"] else 0,
            "genres": genres,
            "source": row["Source"] if row["Source"] else "Unknown",
            "duration": str(row["Duration"]),
            "stats": {
                "plan_to_watch": int(row["Plan to Watch"]) if "Plan to Watch" in row and row["Plan to Watch"] else 0,
                "dropped": int(row["Dropped"]) if "Dropped" in row and row["Dropped"] else 0
            }
        })
        
    return jsonify({"data": results})

# Helper to compute user features exactly like useful_extract.py does, 
# but extremely fast by hitting MAL directly just once.
def compute_user_features(username: str) -> dict:
    url = f"https://myanimelist.net/animelist/{username}/load.json?status=7"
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise ValueError(f"Could not fetch MAL list for user '{username}'.")
    
    data = resp.json()
    if not data:
        raise ValueError(f"User '{username}' has an empty list or is private.")
        
    df = pd.DataFrame(data)
    
    # MAL load.json fields: 
    # status: 1=watching, 2=completed, 3=onhold, 4=dropped, 6=ptw
    # anime_num_episodes, num_watched_episodes, anime_airing_status (1=ongoing, 2=finished)
    
    df = df.rename(columns={
        "status": "watching_status",
        "num_watched_episodes": "watched_episodes"
    })
    
    df["airing_status"] = df["anime_airing_status"].apply(lambda x: "ongoing" if x == 1 else "completed")
    
    tot_anime = len(df)
    
    # 1. user_history_size
    if tot_anime > 50:
        user_history_size = "50+"
    elif tot_anime >= 10:
        user_history_size = "10-50"
    elif tot_anime > 0:
        user_history_size = "1-9"
    else:
        user_history_size = "0"
        
    # 2. user_completion_rate
    completed_count = len(df[df["watching_status"] == 2])
    completion_rate = completed_count / tot_anime if tot_anime > 0 else 0
    if completion_rate > 0.7:
        user_completion_rate = "high"
    elif completion_rate > 0.3:
        user_completion_rate = "medium"
    else:
        user_completion_rate = "low"
        
    # 3. user_avr_drop_ep
    dropped_df = df[df["watching_status"] == 4]
    avr_drop_ep = dropped_df["watched_episodes"].mean() if len(dropped_df) > 0 else 0
    if avr_drop_ep > 13:
        user_avr_drop_ep = "late"
    elif avr_drop_ep > 3:
        user_avr_drop_ep = "midway"
    else:
        user_avr_drop_ep = "early"
        
    # 4. drops_slow_start
    dropped_early_count = len(dropped_df[dropped_df["watched_episodes"] < 4])
    if len(dropped_df) > 0 and (dropped_early_count / len(dropped_df)) > 0.3:
        drops_slow_start = 1
    else:
        drops_slow_start = 0
        
    # 5. length_tolerance
    sample_space = df[(df["watching_status"] == 2) | (df["watching_status"] == 1)]
    if len(sample_space) > 0:
        length_tol = sample_space["watched_episodes"].max()
    else:
        length_tol = 0
        
    if length_tol > 50:
        length_tolerance = "long"
    elif length_tol > 16:
        length_tolerance = "medium"
    else:
        length_tolerance = "short"
        
    # 6. status_preference
    tot_ongoing = len(df[df["airing_status"] == "ongoing"])
    status_preference = "ongoing" if (tot_anime > 0 and tot_ongoing/tot_anime > 0.2) else "completed"
    
    return {
        "user_history_size": user_history_size,
        "user_completion_rate": user_completion_rate,
        "user_avr_drop_ep": user_avr_drop_ep,
        "drops_slow_start": drops_slow_start,
        "length_tolerance": length_tolerance,
        "status_preference": status_preference
    }

@app.route('/predict_user_anime', methods=['POST'])
def predict_user_anime():
    """
    Expects:
    {
      "mal_username": "iitjeeair001",
      "anime_data": { ... from Jikan ... },
      "stats_data": { ... from Jikan stats ... }
    }
    """
    req = request.get_json(force=True)
    if not req or "mal_username" not in req or "anime_data" not in req:
        return jsonify({"error": "Requires mal_username and anime_data"}), 400
        
    username = req["mal_username"]
    anime_data = req["anime_data"]
    stats_data = req.get("stats_data", {})
    anime_id = anime_data.get("mal_id")
    
    try:
        # 1. Fetch User Features (FAST - Hits MAL directly, usually bypasses Cloudflare)
        user_features = compute_user_features(username)
        
        # Calculate derived anime features
        score = anime_data.get("score")
        if score is None: score = 0.0
        
        episodes = anime_data.get("episodes")
        if not episodes or episodes <= 0: episodes = 12
        
        members = anime_data.get("members", 1)
        if members <= 0: members = 1
        
        favorites = anime_data.get("favorites", 0)
        ptw = stats_data.get("plan_to_watch", 0)
        dropped = stats_data.get("dropped", 0)
        
        favorites_to_members_ratio = favorites / members
        ptw_ratio = ptw / members
        show_baseline_drop_rate = dropped / members
        
        # We don't have perfect metrics for these 2 without heavy calculation, so use reasonable defaults for the model
        score_std_dev = 50000 
        popularity_ratio = 0.6
        
        # Source calculation
        source_str = anime_data.get("source", "").lower()
        is_source_original = 1 if source_str == "original" else 0
        
        # Length fit calculation
        eps = episodes
        tol = user_features["length_tolerance"]
        if tol == "long":
            length_fit = 1
        elif tol == "medium":
            length_fit = 1 if eps <= 50 else 0
        else: # short
            length_fit = 1 if eps <= 16 else 0

        # Create the full payload for the model
        combined_features = {
            "anime_id": anime_id,
            "score": float(score),
            "genres": ",".join([g["name"] for g in anime_data.get("genres", [])]),
            "episode_count": int(episodes),
            "member_count": int(members),
            "airing_year": int(anime_data.get("year") or 2020),
            "duration_minutes": 24, # Jikan's duration is a string like "24 min per ep"
            "is_source_original": is_source_original,
            "favorites_to_members_ratio": float(favorites_to_members_ratio),
            "ptw_ratio": float(ptw_ratio),
            "score_std_dev": score_std_dev,
            "show_baseline_drop_rate": float(show_baseline_drop_rate),
            "popularity_ratio": popularity_ratio,
            
            # User features
            "user_history_size": user_features["user_history_size"],
            "user_completion_rate": user_features["user_completion_rate"],
            "user_avr_drop_ep": user_features["user_avr_drop_ep"],
            "drops_slow_start": user_features["drops_slow_start"],
            "length_tolerance": user_features["length_tolerance"],
            "status_preference": user_features["status_preference"],
            "length_fit": length_fit
        }
        
        from predict import predict_drop_episode
        result = predict_drop_episode(combined_features)
        
        return jsonify({
            "predicted_drop_episode": round(result, 1),
            "user_features_calculated": True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print("[*] Starting Drop Predictor API on http://127.0.0.1:5000")
    print("[*] Press Ctrl+C to stop")
    app.run(host='127.0.0.1', port=5000, debug=False)
