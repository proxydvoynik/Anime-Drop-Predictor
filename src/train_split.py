# """
# Data Splitting Module for Anime Drop Episode Predictor
# ======================================================
# Provides user-grouped anti-leakage train/test splitting using GroupShuffleSplit.
# """

# from typing import Tuple
# import pandas as pd
# from sklearn.model_selection import GroupShuffleSplit


# def get_train_test_splits(
#     csv_path: str = "data/processed/phase1_training.csv",
#     test_size: float = 0.20,
#     random_state: int = 42,
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
#     """
#     Splits dataset into train and test sets using GroupShuffleSplit on 'user_id'
#     to prevent data leakage across user records.

#     Parameters
#     ----------
#     csv_path : str
#         Path to processed training CSV file.
#     test_size : float
#         Proportion of data reserved for testing (default 0.20 = 80/20 split).
#     random_state : int
#         Random seed for reproducibility.

#     Returns
#     -------
#     Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
#         (X_train, X_test, y_train, y_test)
#     """
#     df = pd.read_csv(csv_path)

#     groups = df["user_id"]
#     X = df.drop(columns=["user_id", "target_drop_episode"])
#     y = df["target_drop_episode"]

#     gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
#     train_idx, test_idx = next(gss.split(X, y, groups=groups))

#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
#     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#     return X_train, X_test, y_train, y_test


# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test = get_train_test_splits()
#     print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
"""
Data Splitting Module for Anime Drop Episode Predictor
======================================================
Provides user-grouped anti-leakage train/test splitting using GroupShuffleSplit.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def get_train_test_splits(
    csv_path: str = "data/processed/phase1_training.csv",
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits dataset into train and test sets using GroupShuffleSplit on 'user_id'
    to prevent data leakage across user records.

    Parameters
    ----------
    csv_path : str
        Path to processed training CSV file.
    test_size : float
        Proportion of data reserved for testing (default 0.20 = 80/20 split).
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        (X_train, X_test, y_train, y_test)
    """
    df = pd.read_csv(csv_path)

    groups = df["user_id"]
    X = df.drop(columns=["user_id", "target_drop_episode"])
    y = df["target_drop_episode"]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_train_test_splits()
    print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")