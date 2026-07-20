import pandas as pd
from clean_anime import clean_anime_data
from data_loader import process_huge_file

def main():
    # Clean anime metadata and save to CSV
    print("Cleaning anime metadata")
    df_anime = clean_anime_data("data/raw/anime_2020_corrected.csv")
    df_anime.to_csv("data/processed/anime_cleaned.csv", index=False)
    
    # Clean user data in chunks and save to CSV
    print("Processing user interaction data in chunks")
    process_huge_file()

    print("Cleaning done.")
    
    #Merge both processed files into a single master CSV
    print("Merging cleaned files into anime-user interaction CSV")
    
    # Load the clean anime features
    df_anime_clean = pd.read_csv("data/processed/anime_cleaned.csv")
    
    # Setup chunked reading for the filtered interactions to prevent memory errors
    input_user_path = "data/processed/animelist_cleaned.csv"
    output_merged_path = "data/processed/anime_user_merged.csv"
    chunk_size = 1000000
    first_chunk = True
    
    for chunk in pd.read_csv(input_user_path, chunksize=chunk_size):
        # Drop episodes from user chunk if present to avoid duplicate columns
        if 'episodes' in chunk.columns:
            chunk = chunk.drop(columns=['episodes'])
            
        merged_chunk = pd.merge(chunk, df_anime_clean, on='anime_id', how='inner')
        
        merged_chunk.to_csv(
            output_merged_path,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )
        first_chunk = False
        
    print(f"Saved merged dataset to: {output_merged_path}")
    print("Merging done.")

if __name__ == '__main__':
    main()