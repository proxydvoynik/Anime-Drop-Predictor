"""
Anime-level feature lookup for the live prediction pipeline.

Looks up ONLY the anime being predicted from cleaned_anime.csv.

Title matching uses both:
    - name
    - English name

so a user can enter either the Japanese/romanized title or the
English title used by MAL.
"""

import difflib
import re
import pandas as pd


ANIME_CSV_PATH = "data/raw/anime.csv"

_ANIME_DF = None
_NAME_TO_ID = None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _to_number(val) -> float:
    """
    Safely converts a CSV value to float.
    """

    if val is None:
        return 0.0

    if isinstance(val, float) and pd.isna(val):
        return 0.0

    if isinstance(val, (int, float)):
        return float(val)

    try:
        result = float(str(val).replace(",", "").strip())
        return 0.0 if pd.isna(result) else result

    except (ValueError, TypeError):
        return 0.0


def _parse_duration_minutes(duration_str) -> int:

    if not isinstance(duration_str, str) or not duration_str.strip():
        return 24

    digits = "".join(
        c for c in duration_str.split("min")[0]
        if c.isdigit()
    )

    return int(digits) if digits else 24


def _normalize_title(name) -> str:
    """
    Normalizes titles so capitalization, punctuation, etc.
    don't affect matching.
    """

    if not isinstance(name, str):
        return ""

    s = name.lower().strip()

    # Replace punctuation with spaces
    s = re.sub(r"[^a-z0-9]+", " ", s)

    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()

    return s


# ---------------------------------------------------------------------------
# Whole-token matching
# ---------------------------------------------------------------------------

def _token_subsequence_match(short_tokens: list,
                              long_tokens: list) -> bool:

    n = len(short_tokens)
    m = len(long_tokens)

    if n == 0 or n > m:
        return False

    for i in range(m - n + 1):

        if long_tokens[i:i + n] == short_tokens:
            return True

    return False


# ---------------------------------------------------------------------------
# Title lookup
# ---------------------------------------------------------------------------

def _find_by_name(title: str, name_to_id: dict):

    """
    Searches for an anime using:

        1. Exact normalized title
        2. Whole-token substring match
        3. Fuzzy matching

    name_to_id contains BOTH:
        - normal/name titles
        - English names
    """

    norm = _normalize_title(title)

    if not norm:
        return None, None

    # ---------------------------------------------------------------
    # 1. Exact match
    # ---------------------------------------------------------------

    if norm in name_to_id:
        return name_to_id[norm], "name-exact"

    # ---------------------------------------------------------------
    # 2. Whole-token substring match
    # ---------------------------------------------------------------

    norm_tokens = norm.split()

    substring_candidates = []

    for cat_norm, cat_id in name_to_id.items():

        cat_tokens = cat_norm.split()

        if len(norm_tokens) <= len(cat_tokens):
            shorter = norm_tokens
            longer = cat_tokens
        else:
            shorter = cat_tokens
            longer = norm_tokens

        # Avoid tiny titles causing huge numbers of false matches.
        if len(" ".join(shorter)) < 4:
            continue

        if _token_subsequence_match(shorter, longer):

            substring_candidates.append(
                (cat_id, cat_norm)
            )

    if substring_candidates:

        substring_candidates.sort(
            key=lambda c: abs(len(c[1]) - len(norm))
        )

        return substring_candidates[0][0], "name-substring"

    # ---------------------------------------------------------------
    # 3. Fuzzy match
    # ---------------------------------------------------------------

    close = difflib.get_close_matches(
        norm,
        name_to_id.keys(),
        n=1,
        cutoff=0.9
    )

    if close:

        return name_to_id[close[0]], "name-fuzzy"

    return None, None


# ---------------------------------------------------------------------------
# Load anime catalog
# ---------------------------------------------------------------------------

def _load_anime_catalog(path: str = ANIME_CSV_PATH) -> pd.DataFrame:

    global _ANIME_DF
    global _NAME_TO_ID

    if _ANIME_DF is None:

        df = pd.read_csv(path)

        # -----------------------------------------------------------
        # anime_id
        # -----------------------------------------------------------

        first_col = df.columns[0]

        if first_col != "anime_id":

            df = df.rename(
                columns={
                    first_col: "anime_id"
                }
            )

        df["anime_id"] = pd.to_numeric(
            df["anime_id"],
            errors="coerce"
        )

        df = df.dropna(
            subset=["anime_id"]
        )

        df["anime_id"] = df["anime_id"].astype(int)

        df = df.set_index(
            "anime_id",
            drop=False
        )

        _ANIME_DF = df

        # -----------------------------------------------------------
        # Build title lookup
        #
        # IMPORTANT:
        # Both "name" AND "English name" are mapped to the same
        # anime_id.
        # -----------------------------------------------------------

        name_map = {}

        for _, row in df.iterrows():

            anime_id = int(row["anime_id"])

            # -------------------------------------------------------
            # Original/name column
            # -------------------------------------------------------

            original_name = row.get("name")

            normalized_name = _normalize_title(
                original_name
            )

            if normalized_name:

                if normalized_name not in name_map:

                    name_map[normalized_name] = anime_id

            # -------------------------------------------------------
            # English name column
            # -------------------------------------------------------

            english_name = row.get("English name")

            normalized_english = _normalize_title(
                english_name
            )

            if normalized_english:

                if normalized_english not in name_map:

                    name_map[normalized_english] = anime_id

        _NAME_TO_ID = name_map

        print(
            f"[i] Loaded anime catalog: "
            f"{len(df)} rows, "
            f"{len(name_map)} normalized titles "
            f"(including English names), "
            f"from {path}"
        )

    return _ANIME_DF


# ---------------------------------------------------------------------------
# Convert catalog row to model features
# ---------------------------------------------------------------------------

def _anime_row_to_features(anime_row: pd.Series) -> dict:

    members = _to_number(
        anime_row.get("Members")
    )

    completed = _to_number(
        anime_row.get("Completed")
    )

    dropped = _to_number(
        anime_row.get("Dropped")
    )

    favorites = _to_number(
        anime_row.get("Favorites")
    )

    ptw = _to_number(
        anime_row.get("Plan to Watch")
    )

    score_votes = [
        _to_number(
            anime_row.get(f"Score-{i}")
        )
        for i in range(1, 11)
    ]

    # Import only when needed.
    import numpy as np

    return {

        "name":
            anime_row.get("name", ""),

        "score":
            _to_number(
                anime_row.get("Score")
            ),

        "genres":
            anime_row.get("Genres", "") or "",

        "episode_count":
            int(
                _to_number(
                    anime_row.get("Episodes")
                )
            ),

        "member_count":
            members,

        "airing_year":
            int(
                _to_number(
                    anime_row.get("Release_year")
                )
            ),

        "duration_minutes":
            _parse_duration_minutes(
                anime_row.get("Duration")
            ),

        "is_source_original":
            (
                1
                if str(
                    anime_row.get("Source", "")
                ).strip().lower() == "original"
                else 0
            ),

        "favorites_to_members_ratio":
            (
                favorites / members
                if members
                else 0
            ),

        "ptw_ratio":
            (
                ptw / members
                if members
                else 0
            ),

        "score_std_dev":
            float(
                np.std(score_votes)
            )
            if score_votes
            else 0.0,

        "show_baseline_drop_rate":
            (
                dropped / (completed + dropped)
                if (completed + dropped)
                else 0
            ),

        "popularity_ratio":
            (
                completed / members
                if members
                else 0
            ),

        "airing_status":
            str(
                anime_row.get(
                    "airing_status",
                    "completed"
                )
                or "completed"
            ).strip().lower(),
    }


# ---------------------------------------------------------------------------
# Public lookup function
# ---------------------------------------------------------------------------

def get_anime_features(
    anime_id: int,
    name: str = ""
) -> dict | None:

    """
    Looks up ONE anime's static features.

    Lookup order:

        1. Catalog ID
        2. Original/name title
        3. English name
        4. Whole-token matching
        5. Fuzzy matching

    The returned features always come from the SAME catalog row.
    """

    catalog = _load_anime_catalog()

    anime_id = int(anime_id)

    # ---------------------------------------------------------------
    # 1. Try ID
    # ---------------------------------------------------------------

    if anime_id in catalog.index:

        row = catalog.loc[anime_id]

        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        return _anime_row_to_features(row)

    # ---------------------------------------------------------------
    # 2. Try name / English name
    # ---------------------------------------------------------------

    fallback_id, match_type = _find_by_name(
        name,
        _NAME_TO_ID
    )

    if fallback_id is None:

        print(
            f"[!] No catalog match for "
            f"anime_id={anime_id} ('{name}')"
        )

        return None

    row = catalog.loc[fallback_id]

    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    print(
        f"[i] anime_id={anime_id} ('{name}') "
        f"matched via {match_type} -> "
        f"catalog '{row['name']}' "
        f"(catalog id {int(row['anime_id'])})"
    )

    return _anime_row_to_features(row)