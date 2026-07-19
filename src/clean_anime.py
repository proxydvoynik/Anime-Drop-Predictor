import pandas as pd
import numpy as np
import re

def extract_year(aired_str):
    aired = str(aired_str).strip()
    if aired and aired != 'Unknown' and aired != 'nan':
        # Extract the year by searching for first 4-digit year starting with 19 or 20
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', aired)
        if year_match:
            return int(year_match.group(1))
    return np.nan

def clean_anime_data(filePath):
    df = pd.read_csv(filePath)

    # 1. CRITICAL FIX: Reassign df here to save the renamed columns!
    df = df.rename(columns={
        'MAL_ID': 'anime_id',
        'Name': 'name',
        'Score': 'score',
        'Genres': 'genres', 
        'Type': 'type', 
        'Episodes': 'episodes',
        'Members': 'members',
        'Completed': 'completed',
        'Dropped': 'dropped',
        'Rating': 'rating'
    })

    # 2. Drop duplicates
    df = df.drop_duplicates(subset='anime_id')

    # 3. Clean numeric columns (forcing numeric, filling missing values, and casting to integer)
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce').fillna(1).astype(int)
    
    df['members'] = pd.to_numeric(df['members'], errors='coerce').fillna(0).astype(int)
    df['completed'] = pd.to_numeric(df['completed'], errors='coerce').fillna(0).astype(int)
    df['dropped'] = pd.to_numeric(df['dropped'], errors='coerce').fillna(0).astype(int) 

    # Filter out Movies, OVAs, and single-episode shows
    df = df[~df['type'].isin(['Movie', 'OVA'])]
    df = df[df['episodes'] > 1]

    # 4. Clean text/categoricals
    df['genres'] = df['genres'].fillna('Unknown')
    df['type'] = df['type'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Unknown').apply(lambda x: str(x).split(' - ')[0].strip())

    # 5. Extract year from aired column
    df['release_year'] = df['Aired'].apply(extract_year)

    # 6. Clean Studios (Keep top 20 + Other)
    primary_studios = df['Studios'].fillna('Unknown').apply(lambda x: str(x).split(',')[0].strip())
    top_studios = primary_studios.value_counts().index[:20]
    df['studio'] = primary_studios.apply(lambda x: x if x in top_studios or x == 'Unknown' else 'Other')

    # 7. Keep only required columns
    columns_to_keep = [
        'anime_id', 'name', 'score', 'genres', 'type', 'episodes', 
        'members', 'completed', 'dropped', 'release_year', 'studio', 'rating'
    ]
    
    return df[columns_to_keep]