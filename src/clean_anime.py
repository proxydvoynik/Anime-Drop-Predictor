import pandas as pd
import numpy as np
import re

def extract_year(aired_str):
    aired = str(aired_str).strip()
    if aired and aired != 'Unknown' and aired != 'nan':
        # Grab first 4-digit year (19xx or 20xx)
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', aired)
        if year_match:
            return int(year_match.group(1))
    return np.nan

def clean_anime_data(filePath):
    df = pd.read_csv(filePath)

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

    # Drop duplicate show entries
    df = df.drop_duplicates(subset='anime_id')

    # Convert numeric fields
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce').fillna(1).astype(int)
    
    df['members'] = pd.to_numeric(df['members'], errors='coerce').fillna(0).astype(int)
    df['completed'] = pd.to_numeric(df['completed'], errors='coerce').fillna(0).astype(int)
    df['dropped'] = pd.to_numeric(df['dropped'], errors='coerce').fillna(0).astype(int) 

    # Only keep TV series with >1 episode
    df = df[~df['type'].isin(['Movie', 'OVA'])]
    df = df[df['episodes'] > 1]

    # Clean text columns
    df['genres'] = df['genres'].fillna('Unknown')
    df['type'] = df['type'].fillna('Unknown')
    df['rating'] = df['rating'].fillna('Unknown').apply(lambda x: str(x).split(' - ')[0].strip())

    # Airing status and release year
    df['airing_status'] = df['Aired'].apply(lambda x: 'ongoing' if '?' in str(x) else 'completed')
    df['release_year'] = df['Aired'].apply(extract_year)

    # Clean studio names (keep top 20, bucket rest into Other)
    primary_studios = df['Studios'].fillna('Unknown').apply(lambda x: str(x).split(',')[0].strip())
    top_studios = primary_studios.value_counts().index[:20]
    df['studio'] = primary_studios.apply(lambda x: x if x in top_studios or x == 'Unknown' else 'Other')

    columns_to_keep = [
        'anime_id', 'name', 'score', 'genres', 'type', 'episodes', 
        'members', 'completed', 'dropped', 'release_year', 'studio', 'rating',
        'Duration', 'Source', 'Favorites', 'Plan to Watch', 'Premiered', 'Aired',
        'airing_status',
        'Score-10', 'Score-9', 'Score-8', 'Score-7', 'Score-6',
        'Score-5', 'Score-4', 'Score-3', 'Score-2', 'Score-1'
    ]
    
    return df[columns_to_keep]