# module to load data files

import pandas as pd


def load_mi_data(filename: str = "data/MI_timesteps.txt", skip_first: int = 1000) -> pd.DataFrame:
    """Load the MI data from the given file and skip the first n timesteps.

    Without skipping any rows, the MI data is of shape (15001, 9), where the rows correspond to the timesteps and the
    columns correspond to the different measurement values at different areas on the campus.

    Args:
        filename (str, optional): The filename of the MI data. Defaults to "data/MI_timesteps.txt".
        skip_first (int, optional): N timesteps to skip at the beginning of the data. Defaults to 1000.

    Returns:
        pd.DataFrame: Pandas dataframe with the MI data.
    """
    df = pd.read_csv(filename, delimiter=' ', index_col=0)
    df = df.iloc[skip_first:]
    return df
