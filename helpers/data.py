# module to load data files

import pandas as pd


def load_linear_dataset(filename: str = "data/linear_function_data.txt") -> pd.DataFrame:
    """Load linear dataset.

    Args:
        filename (str, optional): Path to file. Defaults to "data/linear_function_data.txt".

    Returns:
        pd.DataFrame: Dataframe with x and y columns.
    """
    data = pd.read_csv(filename, sep=" ", header=None)
    data.columns = ["x", "y"]
    return data


def load_nonlinear_dataset(filename: str = "data/nonlinear_function_data.txt") -> pd.DataFrame:
    """Load nonlinear dataset.

    Args:
        filename (str, optional): Path to file. Defaults to "data/nonlinear_function_data.txt".

    Returns:
        pd.DataFrame: Dataframe with x and y columns.
    """
    data = pd.read_csv(filename, sep=" ", header=None)
    data.columns = ["x", "y"]
    return data
