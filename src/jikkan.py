import requests
import pandas as pd
from useful_extract import extract

def fetch_from_user(username: str) -> pd.DataFrame:
    status = 7
    base_url = f"https://myanimelist.net/animelist/{username}/load.json?status={status}"

    # MAL's default response to requests' default User-Agent can be to
    # silently stall rather than return an error.
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        )
    }

    try:
        user_response = requests.get(base_url, headers=headers, timeout=(10, 20))
        user_response.raise_for_status()
        data = user_response.json()
        print(f"[i] MAL returned {len(data)} entries for '{username}'")
    except requests.exceptions.Timeout:
        print(f"[!] Timed out fetching MAL list for '{username}' — "
              f"MAL may be slow, blocking this request, or the username may be wrong/private.")
        return pd.DataFrame()
    except requests.RequestException as e:
        print(f"[!] Failed to fetch MAL list for '{username}': {e}")
        return pd.DataFrame()

    rows = []
    for item in data:
        rows.append({
            "anime_id": int(item["anime_id"]),
            "name": item.get("anime_title", ""),
            "user_score": item.get("score"),
            "watching_status": item["status"],
            "watched_episodes": item["num_watched_episodes"],
            "episodes": item.get("anime_num_episodes"),
            # Same field this used to be computed as before — still just
            # comes straight off the MAL list item, no extra API call.
            "airing_status": "ongoing" if item["anime_end_date_string"] is None else "completed",
        })

    full_list_df = pd.DataFrame(rows)
    if full_list_df.empty:
        print(f"[!] '{username}' has no anime on their list.")
        return full_list_df

    user_features = extract(chunk=full_list_df, user_id=0)
    for col in user_features.columns:
        full_list_df[col] = user_features[col].iloc[0]

    return full_list_df
