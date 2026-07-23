import pandas as pd

def build_merged_training_table():
    """
    Merges cleaned user interaction logs with engineered anime features,
    filters exclusively for dropped shows (watching_status == 4)
    """
    print("Merging user interactions with show features")
    
    cleaned_interactions_path = "data/processed/animelist_cleaned.csv"
    anime_features_path = "data/processed/cleaned_anime_features.csv"
    output_path = "data/processed/phase1_training.csv"
    
    df_anime = pd.read_csv(anime_features_path)
    chunk_size = 500000
    first_chunk = True
    
    for chunk in pd.read_csv(cleaned_interactions_path, chunksize=chunk_size):
        # Filter for dropped shows only (watching_status == 4)
        chunk = chunk[chunk['watching_status'] == 4].copy()
        if chunk.empty:
            continue
            
        chunk['target_drop_episode'] = chunk['watched_episodes']
        merged_chunk = pd.merge(chunk, df_anime, on='anime_id', how='inner')
        
        cols_order = [
            'user_id', 'anime_id', 'name', 'score', 'genres', 'rating',
            'episode_count', 'member_count', 'airing_year', 'duration_minutes',
            'is_source_original', 'favorites_to_members_ratio', 'ptw_ratio',
            'score_std_dev', 'show_baseline_drop_rate', 'popularity_ratio',
            'airing_status', 'target_drop_episode'
        ]
        
        final_cols = [c for c in cols_order if c in merged_chunk.columns]
        merged_chunk = merged_chunk[final_cols]
        
        merged_chunk.to_csv(
            output_path,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )
        first_chunk = False
        
    print(f"Saved merged phase1_training dataset to: {output_path}")

if __name__ == '__main__':
    build_merged_training_table()
