import pandas as pd

first_chunk = True
input_file = "data/processed/animelist_updated.csv"
output_file = "data/processed/feature_list.csv"

def extract(chunk,user_id):
    global first_chunk

    #user_history_size
    tot_anime = len(chunk)
    if tot_anime > 50:
        tot_anime_status = "50+"
    elif tot_anime >= 10:
        tot_anime_status = "10-50"
    elif tot_anime > 0:
        tot_anime_status = "1-9"
    else:
        tot_anime_status = "0"

    #user_completion_rate
    completion_rate = len(chunk[chunk["watching_status"]==2])/tot_anime if tot_anime > 0 else 0
    if completion_rate > 0.7:
        completion_rate_status = "high"
    elif completion_rate > 0.3:
        completion_rate_status = "medium"
    else:
        completion_rate_status = "low"

    #user_avr_drop_ep
    avr_drop_ep = chunk[chunk["watching_status"]==4]["watched_episodes"].mean() if len(chunk[chunk["watching_status"]==4]) > 0 else 0
    if avr_drop_ep > 13:
        avr_drop_ep_status = "late"
    elif avr_drop_ep > 3:
        avr_drop_ep_status = "midway"
    else:
        avr_drop_ep_status = "early"

    #drops_slow_start
    if (chunk[(chunk["watching_status"]==4) & (chunk["watched_episodes"]<4)].shape[0]/chunk[chunk["watching_status"]==4].shape[0] if chunk[chunk["watching_status"]==4].shape[0] > 0 else 0) > 0.3:
        drop_slow = 1
    else:
        drop_slow = 0

    #length_tolenrance
    sample_space = chunk[(chunk["watching_status"]==2) | chunk["watching_status"]==1]
    length_tol = sample_space["watched_episodes"].max()
    if length_tol > 50:
        length_tol_status = "long"
    elif length_tol > 16:
        length_tol_status = "medium"
    else:
        length_tol_status = "short"

    #status preference
    tot_ongoing = len(chunk[chunk["airing_status"]=="ongoing"])
    if tot_ongoing/tot_anime > 0.2:
        status_pref = "ongoing"
    else:
        status_pref = "completed"

    # Write to file
    feature_list = pd.DataFrame({
                "user_id": [user_id],
                "user_history_size": [tot_anime_status],
                "user_completion_rate": [completion_rate_status],
                "user_avr_drop_ep": [avr_drop_ep_status],
                "drops_slow_start": [drop_slow],
                "length_tolerance": [length_tol_status],
                "status_preference": [status_pref]
            })
    feature_list.to_csv(
                output_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
            )
    
    first_chunk = False

#function for chunked loading and managing chunk splits
def loader():
    leftover = pd.DataFrame()
    for big_chunk in pd.read_csv(input_file,chunksize=500000,usecols=["user_id", "watching_status", "watched_episodes","airing_status"]):
        last_user = big_chunk["user_id"].iloc[-1]
        complete_chunk = big_chunk[big_chunk["user_id"]!=last_user]
        complete_chunk=pd.concat([leftover,complete_chunk])
        leftover = big_chunk[big_chunk["user_id"]==last_user]
        for user_id,smaller_chunk in complete_chunk.groupby("user_id"):
            extract(smaller_chunk,user_id)
    for user_id, smaller_chunk in leftover.groupby("user_id"):
        extract(smaller_chunk, user_id)

if __name__ == "__main__":
    loader()
