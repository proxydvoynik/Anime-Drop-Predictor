# import pandas as pd
# import inspect

# first_chunk = True
# input_file = "data/processed/animelist_updated.csv"
# output_file = "data/processed/feature_list.csv"

# def extract(**kwargs):
#     global first_chunk

#     #user_history_size
#     chunk = kwargs["chunk"]
#     user_id = kwargs["user_id"]
#     tot_anime = len(chunk)
#     if tot_anime > 50:
#         tot_anime_status = "50+"
#     elif tot_anime >= 10:
#         tot_anime_status = "10-50"
#     elif tot_anime > 0:
#         tot_anime_status = "1-9"
#     else:
#         tot_anime_status = "0"

#     #user_completion_rate
#     completion_rate = len(chunk[chunk["watching_status"]==2])/tot_anime if tot_anime > 0 else 0
#     if completion_rate > 0.7:
#         completion_rate_status = "high"
#     elif completion_rate > 0.3:
#         completion_rate_status = "medium"
#     else:
#         completion_rate_status = "low"

#     #user_avr_drop_ep
#     avr_drop_ep = chunk[chunk["watching_status"]==4]["watched_episodes"].mean() if len(chunk[chunk["watching_status"]==4]) > 0 else 0
#     if avr_drop_ep > 13:
#         avr_drop_ep_status = "late"
#     elif avr_drop_ep > 3:
#         avr_drop_ep_status = "midway"
#     else:
#         avr_drop_ep_status = "early"

#     #drops_slow_start
#     if (chunk[(chunk["watching_status"]==4) & (chunk["watched_episodes"]<4)].shape[0]/chunk[chunk["watching_status"]==4].shape[0] if chunk[chunk["watching_status"]==4].shape[0] > 0 else 0) > 0.3:
#         drop_slow = 1
#     else:
#         drop_slow = 0

#     #length_tolenrance
#     sample_space = chunk[(chunk["watching_status"]==2) | chunk["watching_status"]==1]
#     length_tol = sample_space["watched_episodes"].max()
#     if length_tol > 50:
#         length_tol_status = "long"
#     elif length_tol > 16:
#         length_tol_status = "medium"
#     else:
#         length_tol_status = "short"

#     #status preference
#     tot_ongoing = len(chunk[chunk["airing_status"]=="ongoing"])
#     if tot_ongoing/tot_anime > 0.2:
#         status_pref = "ongoing"
#     else:
#         status_pref = "completed"

#     # Write to file
#     temp_dict={
#                 "user_history_size": [tot_anime_status],
#                 "user_completion_rate": [completion_rate_status],
#                 "user_avr_drop_ep": [avr_drop_ep_status],
#                 "drops_slow_start": [drop_slow],
#                 "length_tolerance": [length_tol_status],
#                 "status_preference": [status_pref]
#             }
#     caller = inspect.currentframe().f_back.f_code.co_name
#     if caller==loader :
#         temp_dict["user_id"]=user_id 
#     feature_list = pd.DataFrame(temp_dict)

#     if caller==loader:
#         feature_list.to_csv(
#                 output_file,
#                 mode="w" if first_chunk else "a",
#                 header=first_chunk,
#                 index=False
#                 )
#     else:
#         chunk = pd.concat(
#                     [chunk, pd.DataFrame([feature_list])],
#                     ignore_index=True
#                 )
#     first_chunk = False

# #function for chunked loading and managing chunk splits
# def loader():
#     leftover = pd.DataFrame()
#     for big_chunk in pd.read_csv(input_file,chunksize=500000,usecols=["user_id", "watching_status", "watched_episodes","airing_status"]):
#         last_user = big_chunk["user_id"].iloc[-1]
#         complete_chunk = big_chunk[big_chunk["user_id"]!=last_user]
#         complete_chunk=pd.concat([leftover,complete_chunk])
#         leftover = big_chunk[big_chunk["user_id"]==last_user]
#         for user_id,smaller_chunk in complete_chunk.groupby("user_id"):
#             extract(chunk = smaller_chunk,user_id = user_id)
#     for user_id, smaller_chunk in leftover.groupby("user_id"):
#         extract(chunk=smaller_chunk,user_id=user_id)

# if __name__ == "__main__":
#     loader()
import pandas as pd
import inspect

first_chunk = True
input_file = "data/processed/animelist_updated.csv"
output_file = "data/processed/feature_list.csv"

def extract(**kwargs):
    global first_chunk

    #user_history_size
    chunk = kwargs["chunk"]
    user_id = kwargs["user_id"]
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
    # FIX: missing parentheses around the second condition. Because `|`
    # binds tighter than `==` in Python, the original
    # `(chunk["watching_status"]==2) | chunk["watching_status"]==1`
    # did NOT evaluate as "status is 2 OR status is 1" like intended.
    sample_space = chunk[(chunk["watching_status"]==2) | (chunk["watching_status"]==1)]
    length_tol = sample_space["watched_episodes"].max()
    if length_tol > 50:
        length_tol_status = "long"
    elif length_tol > 16:
        length_tol_status = "medium"
    else:
        length_tol_status = "short"

    #status preference
    tot_ongoing = len(chunk[chunk["airing_status"]=="ongoing"])
    status_pref = "ongoing" if (tot_anime > 0 and tot_ongoing/tot_anime > 0.2) else "completed"
    # FIX: original divided by tot_anime with no zero-check — ZeroDivisionError
    # possible if tot_anime is 0

    # Write to file
    temp_dict={
                "user_history_size": [tot_anime_status],
                "user_completion_rate": [completion_rate_status],
                "user_avr_drop_ep": [avr_drop_ep_status],
                "drops_slow_start": [drop_slow],
                "length_tolerance": [length_tol_status],
                "status_preference": [status_pref]
            }
    caller = inspect.currentframe().f_back.f_code.co_name
    # FIX: `caller` is a STRING (the function name, e.g. "loader"), but the
    # original compared it to the actual function object `loader` — always
    # False, so the loader-only branch below never ran when it should have.
    if caller == "loader":
        temp_dict["user_id"]=user_id
        feature_list = pd.DataFrame(temp_dict)
        # FIX: was `pd.Dataframe` (lowercase "frame") — AttributeError,
        # pandas has no such name, every single call would've crashed
        feature_list.to_csv(
                output_file,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False
                )
        first_chunk = False
    else:
        # FIX: the original branch built `feature_list` from concatenating
        # it onto `chunk` (anime-level rows) which doesn't make sense
        # (mismatched shapes/columns), AND the function had no return
        # statement at all — the computed features vanished into nothing
        # for any live/non-loader caller (like jikan.py's fetch_from_user).
        feature_list = pd.DataFrame(temp_dict)
        first_chunk = False
        return feature_list

#function for chunked loading and managing chunk splits
def loader():
    leftover = pd.DataFrame()
    for big_chunk in pd.read_csv(input_file,chunksize=500000,usecols=["user_id", "watching_status", "watched_episodes","airing_status"]):
        last_user = big_chunk["user_id"].iloc[-1]
        complete_chunk = big_chunk[big_chunk["user_id"]!=last_user]
        complete_chunk=pd.concat([leftover,complete_chunk])
        leftover = big_chunk[big_chunk["user_id"]==last_user]
        for user_id,smaller_chunk in complete_chunk.groupby("user_id"):
            extract(chunk = smaller_chunk,user_id = user_id)
    for user_id, smaller_chunk in leftover.groupby("user_id"):
        extract(chunk=smaller_chunk,user_id=user_id)

if __name__ == "__main__":
    loader()