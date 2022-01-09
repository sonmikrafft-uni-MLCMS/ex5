# module for generic utility functions

import pandas as pd
import numpy as np


def augment_with_prediction(df: pd.DataFrame, y_pred: np.ndarray) -> pd.DataFrame:
    """Add a column with the prediction to the dataframe and set the index to the "x" column.

    Args:
        df (pd.DataFrame): Dataframe to augment. Must have a column named "x".
        y_pred (np.ndarray): Predictions for the data to add to the dataframe.

    Returns:
        pd.DataFrame: Augmented dataframe.
    """
    df_augmented = df.copy()
    df_augmented["y_pred"] = y_pred
    df_augmented.index = df_augmented["x"]
    df_augmented.drop(columns=["x"], inplace=True)
    return df_augmented
