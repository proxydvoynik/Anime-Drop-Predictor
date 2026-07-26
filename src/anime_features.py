import re
import pandas as pd
import numpy as np

def parse_duration(d_str):
    """Converts duration string to minutes float."""
    if pd.isna(d_str) or d_str == 'Unknown':
        return 24.0
        
    d_str = str(d_str).lower()
    numbers = [int(s) for s in re.findall(r'\d+', d_str)]
    if not numbers:
        return 24.0
        
    if 'hr' in d_str or 'hour' in d_str:
        hours = numbers[0]
        minutes = numbers[1] if len(numbers) > 1 else 0
        return float(hours * 60 + minutes)
    elif 'min' in d_str:
        return float(numbers[0])
    elif 'sec' in d_str:
        return 1.0
        
    return 24.0

def extract_airing_year(row):
    """Extracts 4-digit airing year from Premiered or Aired columns."""
    prem = str(row.get('Premiered', '')).strip()
    aired = str(row.get('Aired', '')).strip()
    
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', prem)
    if year_match:
        return int(year_match.group(1))
        
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', aired)
    if year_match:
        return int(year_match.group(1))
        
    return 2000

def calculate_score_std(row):
    """Calculates standard deviation of rating votes (Score-10 down to Score-1)."""
    counts = np.array([pd.to_numeric(row.get(f'Score-{i}', 0), errors='coerce') for i in range(1, 11)], dtype=float)
    counts = np.nan_to_num(counts)
    values = np.array(range(1, 11))
    total_votes = counts.sum()
    
    if total_votes <= 1.0:
        return 0.0
        
    mean = np.average(values, weights=counts)
    variance = np.average((values - mean)**2, weights=counts)
    return float(np.sqrt(variance))

def build_anime_features(df):
    """
    Engineers and returns show-level features.
    """
    df = df.copy()
    
    # 1. Episode, member counts and raw score
    df['episode_count'] = df['episodes']
    df['member_count'] = df['members']
    
    # 2. Parse episode duration to minutes
    df['duration_minutes'] = df['Duration'].apply(parse_duration)
    
    # 3. Extract Airing Year
    df['airing_year'] = df.apply(extract_airing_year, axis=1)
    
    # 4. Is original source flag
    df['is_source_original'] = (df['Source'] == 'Original').astype(int)
    
    # 5. Engagement ratios
    df['favorites_to_members_ratio'] = df['Favorites'].fillna(0) / (df['member_count'] + 1e-5)
    df['ptw_ratio'] = df['Plan to Watch'].fillna(0) / (df['member_count'] + 1e-5)
    
    # 6. Score polarization (std dev)
    df['score_std_dev'] = df.apply(calculate_score_std, axis=1)
    
    # 7. Baseline rates
    df['show_baseline_drop_rate'] = df['dropped'] / (df['completed'] + df['dropped'] + 1e-5)
    df['popularity_ratio'] = df['completed'] / (df['member_count'] + 1e-5)
    
    # Clean NaN values
    numeric_cols = ['score', 'episode_count', 'member_count', 'airing_year', 
                    'duration_minutes', 'is_source_original', 'favorites_to_members_ratio',
                    'ptw_ratio', 'score_std_dev', 'show_baseline_drop_rate', 'popularity_ratio']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
    cols_to_keep = [
        'anime_id', 'name', 'score', 'genres', 'rating',
        'episode_count', 'member_count', 'airing_year',
        'duration_minutes', 'is_source_original', 'favorites_to_members_ratio',
        'ptw_ratio', 'score_std_dev', 'show_baseline_drop_rate', 'popularity_ratio',
        'airing_status'
    ]
        
    return df[cols_to_keep]
