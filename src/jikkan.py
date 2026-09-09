# # import requests
# # from useful_extract import extract
# # import numpy as np
# # import pandas as pd
# # import time


# # def fetch_from_user(username: str) -> pd.DataFrame:
# #     final_obj = pd.DataFrame()
# #     each_row = dict()
# #     status=7
# #     base_url = f"https://myanimelist.net/animelist/{username}/load.json?status={status}"
# #     user_response = requests.get(base_url)
# #     user_response.raise_for_status()
# #     data = user_response.json()

# #     for item in data:
# #         stats_response = f"https://api.jikan.moe/v4/anime/{item['anime_id']}/stats"
# #         stats = requests.get(stats_response).json()["data"]
# #         time.sleep(0.4)
# #         anime_response = requests.get(f"https://api.jikan.moe/v4/anime/{item['anime_id']}")
# #         anime_details = anime_response.json()["data"]
# #         time.sleep(0.4)

# #         each_row["anime_id"]=item["anime_id"]
# #         each_row["name"]=item["anime_title"]
# #         each_row["watching_status"]=item["score"]
# #         each_row["genres"]=f'{item["genres"][0]["name"]},{item["genres"][1]["name"]},{item["genres"][2]["name"]}'
# #         each_row["episode_count"]=item["anime_num_episodes"]
# #         each_row["member_count"]=item["anime_total_members"]
# #         each_row["airing year"]=f'{20 if item["anime_start_date_string"][:2]>26 else 19}+{item["anime_start_date_string"][:2]}'
# #         each_row["duration_minutes"]=24
# #         each_row["is_source_original"]=1 if anime_details["source"]=="Original" else 0
# #         each_row["favourites_to_members_ratio"] = anime_details["favorites"]/anime_details["members"]
# #         each_row["ptw_ratio"] = stats["plan_to_watch"] / anime_details["members"] if anime_details["members"] else 0
# #         each_row["score_std_dev"] = np.std([stats[f"score_{i}"] for i in range(1, 11)])
# #         each_row["show_baseline_drop_rate"] = stats["dropped"]/(stats["completed"]+stats["dropped"]) if (stats["completed"]+stats["dropped"]) else 0
# #         each_row["popularity_ratio"] = stats["completed"]/anime_details["members"] if anime_details["members"] else 0
# #         each_row["watching_status"] = item["status"]
# #         each_row["watched_episodes"] = item["num_watched_episodes"]
# #         each_row["episodes"] = item["anime_num_episodes"] 
# #         each_row["airing_status"] = "ongoing" if item["anime_end_date_string"]=="null" else "completed"
# #         final_obj = pd.concat(
# #             [final_obj, pd.DataFrame([each_row])],
# #             ignore_index=True
# #         )
# #     extract(chunk=final_obj,user_id=0)
# #     return final_obj
# import requests
# from useful_extract import extract
# import numpy as np
# import pandas as pd
# import time


# def fetch_from_user(username: str) -> pd.DataFrame:
#     final_obj = pd.DataFrame()
#     status=7
#     base_url = f"https://myanimelist.net/animelist/{username}/load.json?status={status}"
#     user_response = requests.get(base_url)
#     user_response.raise_for_status()
#     data = user_response.json()

#     for item in data:
#         each_row = dict()  # FIX: moved inside loop — was outside, so every
#                             # row was overwriting the same dict object instead
#                             # of each row being independent (all rows in the
#                             # final DataFrame would've ended up identical)
#         stats_response = f"https://api.jikan.moe/v4/anime/{item['anime_id']}/stats"
#         stats_raw = requests.get(stats_response)
#         # FIX: no error-checking before — assumed every response has a
#         # "data" key. Jikan returns a different shape on rate-limits (429)
#         # or errors, with no "data" key at all, which crashed here.
#         if stats_raw.status_code != 200:
#             print(f"[!] Skipping anime_id={item['anime_id']} — stats fetch failed "
#                   f"(status {stats_raw.status_code}): {stats_raw.text[:200]}")
#             continue
#         stats = stats_raw.json().get("data")
#         if stats is None:
#             print(f"[!] Skipping anime_id={item['anime_id']} — no stats data returned")
#             continue
#         time.sleep(0.4)

#         anime_response = requests.get(f"https://api.jikan.moe/v4/anime/{item['anime_id']}")
#         if anime_response.status_code != 200:
#             print(f"[!] Skipping anime_id={item['anime_id']} — anime fetch failed "
#                   f"(status {anime_response.status_code}): {anime_response.text[:200]}")
#             continue
#         anime_details = anime_response.json().get("data")
#         if anime_details is None:
#             print(f"[!] Skipping anime_id={item['anime_id']} — no anime data returned")
#             continue
#         time.sleep(0.4)

#         each_row["anime_id"]=item["anime_id"]
#         each_row["name"]=item["anime_title"]
#         each_row["user_score"]=item["score"]
#         # FIX: was "watching_status", immediately overwritten a few lines
#         # down by item["status"] — this value was silently discarded before

#         # FIX: was hardcoded to exactly 3 genres, crashed (IndexError) on
#         # any anime with fewer than 3 genres
#         genre_names = [g["name"] for g in item["genres"]]
#         each_row["genres"] = ",".join(genre_names)

#         each_row["episode_count"]=item["anime_num_episodes"]
#         each_row["member_count"]=item["anime_total_members"]

#         # FIX: original compared a string slice to an int (item[...][:2] > 26)
#         # — TypeError every time. Also had a literal "+" character baked into
#         # the resulting string ("20+03") instead of concatenating "20"+"03".
#         yy = int(item["anime_start_date_string"][:2])
#         century = "19" if yy > 26 else "20"
#         each_row["airing_year"] = int(f"{century}{item['anime_start_date_string'][:2]}")
#         # FIX: key was "airing year" (space) — add_synthetic_features expects
#         # "airing_year" (underscore); the space would've caused a silent
#         # missing-column failure downstream

#         # FIX: was hardcoded to 24 for every anime regardless of actual data
#         duration_str = anime_details.get("duration") or "24 min"
#         digits = "".join(c for c in duration_str.split("min")[0] if c.isdigit())
#         each_row["duration_minutes"] = int(digits) if digits else 24

#         each_row["is_source_original"]=1 if anime_details["source"]=="Original" else 0

#         # FIX: was never set at all — the model needs the show's overall
#         # MAL rating, not the user's personal score (that's user_score above)
#         each_row["score"] = anime_details.get("score") or 0.0

#         # FIX: was spelled "favourites_..." (British) — training data column
#         # is "favorites_to_members_ratio" (American spelling); mismatched key
#         # would've caused a silent missing-column failure downstream
#         each_row["favorites_to_members_ratio"] = anime_details["favorites"]/anime_details["members"] if anime_details["members"] else 0

#         each_row["ptw_ratio"] = stats["plan_to_watch"] / anime_details["members"] if anime_details["members"] else 0
#         each_row["score_std_dev"] = np.std([stats[f"score_{i}"] for i in range(1, 11)])
#         each_row["show_baseline_drop_rate"] = stats["dropped"]/(stats["completed"]+stats["dropped"]) if (stats["completed"]+stats["dropped"]) else 0
#         each_row["popularity_ratio"] = stats["completed"]/anime_details["members"] if anime_details["members"] else 0
#         each_row["watching_status"] = item["status"]
#         each_row["watched_episodes"] = item["num_watched_episodes"]
#         each_row["episodes"] = item["anime_num_episodes"]
#         # FIX: was "each-row" (hyphen) — NameError, `each` isn't a variable

#         # FIX: anime_end_date_string is a real None from JSON when an anime
#         # is still airing, not the literal string "null" — the old
#         # comparison was always False, so airing_status was always
#         # "completed" no matter what
#         each_row["airing_status"] = "ongoing" if item["anime_end_date_string"] is None else "completed"

#         # PLACEHOLDER — "length_fit" formula was never defined anywhere in
#         # your original code (jikan.py or useful_extract.py). I'm not
#         # guessing at real logic here — this is a stand-in so the pipeline
#         # runs end-to-end. Replace with your team's actual intended formula.
#         each_row["length_fit"] = 1

#         final_obj = pd.concat(
#             [final_obj, pd.DataFrame([each_row])],
#             ignore_index=True
#         )

#     # FIX: extract() previously computed the user-level engineered features
#     # (user_history_size, user_completion_rate, etc.) but never returned or
#     # attached them anywhere for the live-fetch path — they were silently
#     # discarded. Now merged onto every row, since these are per-USER stats
#     # (same value for every anime this user has watched).
#     user_features = extract(chunk=final_obj, user_id=0)
#     for col in user_features.columns:
#         final_obj[col] = user_features[col].iloc[0]

#     return final_obj


import requests
from useful_extract import extract
import numpy as np
import pandas as pd
import time


# ---------------------------------------------------------------------------
# Retry / backoff helper
# ---------------------------------------------------------------------------
# Jikan's public API enforces a hard rate limit (roughly 1 request/sec
# sustained, with an even tighter burst cap). Firing 2 requests per anime
# back-to-back with only a flat 0.4s sleep blows through that limit almost
# immediately, which is why every request was coming back as 429
# (rate-limited) or 504 (Jikan/MAL refusing to connect under load).
#
# This wrapper:
#   - Retries on 429 with exponential backoff, honoring a `Retry-After`
#     header if Jikan sends one.
#   - Retries on 504 a couple of times (transient), but gives up if MAL/Jikan
#     is genuinely down rather than retrying forever.
#   - Returns None (rather than raising) if all retries are exhausted, so
#     the caller can skip that anime and move on.
def _get_with_backoff(url: str, max_retries: int = 5, base_delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=15)
        except requests.RequestException as e:
            print(f"[!] Network error on {url} (attempt {attempt + 1}/{max_retries}): {e}")
            time.sleep(base_delay * (2 ** attempt))
            continue

        if response.status_code == 200:
            return response

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else base_delay * (2 ** attempt)
            print(f"[!] Rate-limited on {url} — waiting {wait:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue

        if response.status_code == 504:
            wait = base_delay * (2 ** attempt)
            print(f"[!] 504 from {url} — MAL/Jikan may be down, retrying in {wait:.1f}s "
                  f"(attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
            continue

        # Any other non-200 status — don't keep retrying, just report and bail.
        print(f"[!] Request to {url} failed (status {response.status_code}): "
              f"{response.text[:200]}")
        return None

    print(f"[!] Giving up on {url} after {max_retries} attempts")
    return None


def fetch_from_user(username: str) -> pd.DataFrame:
    final_obj = pd.DataFrame()
    status = 7
    base_url = f"https://myanimelist.net/animelist/{username}/load.json?status={status}"
    user_response = requests.get(base_url)
    user_response.raise_for_status()
    data = user_response.json()

    for item in data:
        each_row = dict()

        stats_response = _get_with_backoff(
            f"https://api.jikan.moe/v4/anime/{item['anime_id']}/stats"
        )
        if stats_response is None:
            print(f"[!] Skipping anime_id={item['anime_id']} — stats fetch failed after retries")
            continue
        stats = stats_response.json().get("data")
        if stats is None:
            print(f"[!] Skipping anime_id={item['anime_id']} — no stats data returned")
            continue
        time.sleep(0.5)

        anime_response = _get_with_backoff(
            f"https://api.jikan.moe/v4/anime/{item['anime_id']}"
        )
        if anime_response is None:
            print(f"[!] Skipping anime_id={item['anime_id']} — anime fetch failed after retries")
            continue
        anime_details = anime_response.json().get("data")
        if anime_details is None:
            print(f"[!] Skipping anime_id={item['anime_id']} — no anime data returned")
            continue
        time.sleep(0.5)

        each_row["anime_id"] = item["anime_id"]
        each_row["name"] = item["anime_title"]
        each_row["user_score"] = item["score"]

        genre_names = [g["name"] for g in item["genres"]]
        each_row["genres"] = ",".join(genre_names)

        each_row["episode_count"] = item["anime_num_episodes"]
        each_row["member_count"] = item["anime_total_members"]

        yy = int(item["anime_start_date_string"][:2])
        century = "19" if yy > 26 else "20"
        each_row["airing_year"] = int(f"{century}{item['anime_start_date_string'][:2]}")

        duration_str = anime_details.get("duration") or "24 min"
        digits = "".join(c for c in duration_str.split("min")[0] if c.isdigit())
        each_row["duration_minutes"] = int(digits) if digits else 24

        each_row["is_source_original"] = 1 if anime_details["source"] == "Original" else 0

        each_row["score"] = anime_details.get("score") or 0.0

        each_row["favorites_to_members_ratio"] = (
            anime_details["favorites"] / anime_details["members"]
            if anime_details["members"] else 0
        )

        each_row["ptw_ratio"] = (
            stats["plan_to_watch"] / anime_details["members"]
            if anime_details["members"] else 0
        )

        # FIX: Jikan's /stats endpoint does NOT return flat keys like
        # "score_1".."score_10" — score breakdowns come back as a `scores`
        # array of objects, e.g. [{"score": 10, "votes": 1234, "percentage": ...}, ...].
        # The old code (`stats[f"score_{i}"] for i in range(1, 11)`) would
        # KeyError on essentially every anime. We pull vote counts from the
        # array instead, matched up by the "score" field rather than assumed
        # ordering (defensive against Jikan returning entries out of order
        # or omitting a score bucket with zero votes).
        scores_list = stats.get("scores", [])
        votes_by_score = {s.get("score"): s.get("votes", 0) for s in scores_list}
        vote_counts = [votes_by_score.get(i, 0) for i in range(1, 11)]
        each_row["score_std_dev"] = float(np.std(vote_counts)) if vote_counts else 0.0

        each_row["show_baseline_drop_rate"] = (
            stats["dropped"] / (stats["completed"] + stats["dropped"])
            if (stats["completed"] + stats["dropped"]) else 0
        )
        each_row["popularity_ratio"] = (
            stats["completed"] / anime_details["members"]
            if anime_details["members"] else 0
        )
        each_row["watching_status"] = item["status"]
        each_row["watched_episodes"] = item["num_watched_episodes"]
        each_row["episodes"] = item["anime_num_episodes"]

        each_row["airing_status"] = "ongoing" if item["anime_end_date_string"] is None else "completed"

        # PLACEHOLDER — "length_fit" formula was never defined anywhere in
        # the original code. Still a stand-in; replace with the real formula
        # once it's decided, otherwise every live prediction feeds the model
        # a constant here instead of a real signal.
        each_row["length_fit"] = 1

        final_obj = pd.concat(
            [final_obj, pd.DataFrame([each_row])],
            ignore_index=True
        )

    # FIX: if every anime failed to fetch (e.g. Jikan/MAL was down, or the
    # rate limit couldn't be worked around within max_retries), final_obj is
    # an empty DataFrame with NO columns at all. Calling extract() on that
    # used to crash with `KeyError: 'watching_status'` deep inside
    # useful_extract.py. Guard against it here with a clear message instead.
    if final_obj.empty:
        print(f"[!] No anime were successfully fetched for '{username}' — "
              f"Jikan/MAL may be rate-limiting or unavailable. Returning empty result.")
        return final_obj

    # These are per-USER stats (same value for every anime this user has
    # watched), computed once from the whole assembled list and broadcast
    # onto every row.
    user_features = extract(chunk=final_obj, user_id=0)
    for col in user_features.columns:
        final_obj[col] = user_features[col].iloc[0]

    return final_obj
