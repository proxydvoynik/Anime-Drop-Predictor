import pandas as pd

def process_huge_file():
    anime_df = pd.read_csv('cleaned_anime_data.csv', usecols=['anime_id', 'type'])
    
    #Setup chunking for the huge file
    input_file = 'animelist.csv'
    output_file = 'animelist_updated.csv'
    chunk_size = 500000 
    first_chunk = True
    
    #Process in chunks
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Merge the small slice with our reference dataframe
        merged_chunk = pd.merge(chunk, anime_df, on='anime_id', how='left')
        
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