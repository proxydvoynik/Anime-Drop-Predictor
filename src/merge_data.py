import pandas as pd

def calculate_length_fit(row):
    # Match show episode count against user's length preference
    tol = str(row.get('length_tolerance', '')).lower()
    eps = row.get('episode_count', 0)
    
    if tol == 'long':
        return 1
    elif tol == 'medium':
        return 1 if eps <= 50 else 0
    elif tol == 'short':
        return 1 if eps <= 16 else 0
    return 0

def build_merged_training_table():
    print("Merging interaction logs, anime features, and user features...")
    
    cleaned_interactions_path = "data/processed/animelist_updated.csv"
    anime_features_path = "data/processed/cleaned_anime_features.csv"
    user_features_path = "data/processed/feature_list.csv"
    output_path = "data/processed/phase1_training.csv"
    
    df_anime = pd.read_csv(anime_features_path)
    df_user = pd.read_csv(user_features_path)
    
    chunk_size = 500000
    first_chunk = True
    
    for chunk in pd.read_csv(cleaned_interactions_path, chunksize=chunk_size):
        # Keep dropped interactions only
        chunk = chunk[chunk['watching_status'] == 4].copy()
        if chunk.empty:
            continue
            
        chunk['target_drop_episode'] = chunk['watched_episodes']
        
        merged_chunk = pd.merge(chunk, df_anime, on='anime_id', how='inner')
        merged_chunk = pd.merge(merged_chunk, df_user, on='user_id', how='inner')
        
        merged_chunk['length_fit'] = merged_chunk.apply(calculate_length_fit, axis=1)
        
        cols_order = [
            'user_id', 'anime_id', 'name', 'score', 'genres', 'rating',
            'episode_count', 'member_count', 'airing_year', 'duration_minutes',
            'is_source_original', 'favorites_to_members_ratio', 'ptw_ratio',
            'score_std_dev', 'show_baseline_drop_rate', 'popularity_ratio',
            'airing_status', 'user_history_size', 'user_completion_rate',
            'user_avr_drop_ep', 'drops_slow_start', 'length_tolerance',
            'status_preference', 'length_fit', 'target_drop_episode'
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
        
    print(f"Saved merged dataset to: {output_path}")

if __name__ == '__main__':
    build_merged_training_table()
