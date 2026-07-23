import pandas as pd

def process_huge_file():
    anime_df = pd.read_csv("data/processed/cleaned_anime.csv", usecols=['anime_id', 'type', 'episodes', 'airing_status'])
    
    # Setup chunking for the huge file
    input_file = "data/raw/animelist.csv"
    output_file = "data/processed/animelist_updated.csv"
    chunk_size = 500000
    first_chunk = True
    
    #Process in chunks
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Merge the small slice with our reference dataframe
        merged_chunk = pd.merge(chunk, anime_df, on='anime_id', how='left')
        merged_chunk = merged_chunk[merged_chunk["type"]=="TV"]
                                
        del merged_chunk["type"]
    # Write to file
        merged_chunk.to_csv(
            output_file,
            mode='w' if first_chunk else 'a',
            header=first_chunk,
            index=False
        )
        first_chunk = False

if __name__ == "__main__":
    process_huge_file()