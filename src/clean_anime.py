import pandas as pd
import numpy as np
import re

def extract_year(aired_str):

    aired = str(aired_str).strip()
    if aired and aired != 'Unknown' and aired != 'nan':
        # Extract the year from the aired string by searching for first 4 digit year starting with 19 or 20
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', aired)
        if year_match:
            return int(year_match.group(1))
    
    return np.nan

def clean_anime_data(filePath):

    df = pd.read_csv(filePath)

    #Renaming columns which are required for analysis (MAL_ID, Name, Score, Genre, Type, Episodes, Members, Completed, Dropped, Rating)
    df.rename(columns={
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

    #Dropping duplicate anime entries
    df = df.drop_duplicates(subset='anime_id')

    #Cleaning numeric columns (score, episodes)
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    df['episodes'] = pd.to_numeric(df['episodes'], errors='coerce')

    #Cleaning count columns (members, completed, dropped)
    df['members'] = pd.to_numeric(df['members'], errors='coerce').fillna(0).astype(np.int32)
    df['completed'] = pd.to_numeric(df['completed'], errors='coerce').fillna(0).astype(np.int32)
    df['dropped'] = pd.to_numeric(df['dropped'], errors='coerce').fillna(0).astype(np.int32) 

    #Extracting year from aired column and creating a new column 'release_year' 
    df['release_year'] = df['Aired'].apply(extract_year)

    #Cleaning Studios (Taking only top 20 + Other)
    df['studios'] = df['Studios'].fillna('Unknown').apply(lambda x: str(x).split(',')[0].strip())
    #We consider only the first studio listed for each anime

    #Top 20 studios based on the number of anime they have produced
    top_studios = df['studios'].value_counts().index[:20]

    #Grouping studios into 'Others' category for those not in the top 20
    df['studio'] = df['studios'].apply(lambda x: x if x in top_studios or x=='Unknown' else 'Other')

    #Cleaning rating column (rating)
    df['rating'] = df['rating'].fillna('Unknown').apply(lambda x: str(x).split(' - ')[0].strip())

    #Cleaning text columns (genres, type)
    df['genres'] = df['genres'].fillna('Unknown')
    df['type'] = df['type'].fillna('Unknown')


    #Assigning columns considered for analysis
    columns_to_keep = [
        'anime_id', 'name', 'score', 'genres', 'type', 'episodes', 'members', 'completed', 'dropped', 'release_year', 'studio', 'rating'
    ]
    
    return df[columns_to_keep]

