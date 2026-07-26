from clean_anime import clean_anime_data
from anime_features import build_anime_features
from data_clenser import process_huge_file
from useful_extract import loader as extract_user_features
from merge_data import build_merged_training_table

def main():
    print("=== STARTING DATA PREPARATION ===")
    
    # Clean anime metadata
    df_anime = clean_anime_data("data/raw/anime_2020_corrected.csv")
    df_anime.to_csv("data/processed/cleaned_anime.csv", index=False)
    
    # Feature engineering for anime
    df_engineered = build_anime_features(df_anime)
    df_engineered.to_csv("data/processed/cleaned_anime_features.csv", index=False)
    
    # Process interaction logs in chunks
    process_huge_file()
    
    # Extract user watch-style features
    extract_user_features()
    
    # Merge dropped interactions for training
    build_merged_training_table()
    
    print("=== DATA PREPARATION COMPLETED ===")

if __name__ == '__main__':
    main()