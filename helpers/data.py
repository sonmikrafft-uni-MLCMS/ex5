# module to load data files

import pandas as pd


def load_dataset(filename: str = "data/linear_vectorfield_data_x0.txt") -> pd.DataFrame:
    """Load linear dataset.
    Args:
        filename (str, optional): Path to file. Defaults to "data/linear_function_data.txt".
    Returns:
        pd.DataFrame: Dataframe with x and y columns.
    """

    data = pd.read_csv(filename, sep=" ", header=None)
    data.columns = ["x", "y"]
    return data