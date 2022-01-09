# module to plot data

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def plot_pandas_dataset(
    df: pd.DataFrame, x_label: str, y_label: str, title: str, ax: plt.Axes = None
) -> plt.Axes:
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        x_label (str): [description]
        y_label (str): [description]
        title (str): [description]
        filename (str, optional): [description]. Defaults to None.
        ax (plt.Axes, optional): [description]. Defaults to None.

    Returns:
        plt.Axes: [description]
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    ax = sns.lineplot(data=df, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return ax
