from clean_anime import clean_anime_data
from anime_features import build_anime_features
from data_loader import process_huge_file
from merge_data import build_merged_training_table

def main():
    print("===> STARTING COMPLETE DATA PREPARATION <===\n")
    
    # 1. Clean anime metadata
    print("1: Cleaning raw anime metadata")
    df_anime = clean_anime_data("data/raw/anime_2020_corrected.csv")
    df_anime.to_csv("data/processed/cleaned_anime.csv", index=False)
    print("== Saved cleaned anime data to: data/processed/cleaned_anime.csv ==\n")
    
    # 2. Build show-level features
    print("2: Precomputing show-level features")
    df_engineered = build_anime_features(df_anime)
    df_engineered.to_csv("data/processed/cleaned_anime_features.csv", index=False)
    print("== Saved anime features to: data/processed/cleaned_anime_features.csv ==\n")
    
    # 3. Clean user interaction logs in chunks
    print("3: Processing user interaction data (filtering TV shows)")
    process_huge_file()
    print("== Saved cleaned interactions to: data/processed/animelist_cleaned.csv ==\n")
    
    # 4. Merge dropped interactions with show features for training
    print("4: Merging dropped interactions into phase1_training.csv")
    build_merged_training_table()
    
    print("\n===> DATA PREPARATION COMPLETED SUCCESSFULLY <===")

if __name__ == '__main__':
    main()