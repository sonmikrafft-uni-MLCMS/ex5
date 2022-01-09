# module to plot data

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional


def plot_pandas_dataset(
    df: pd.DataFrame,
    x_label: str,
    y_label: str,
    title: str,
    x: Optional[str] = None,
    y: Optional[str] = None,
    ax: plt.Axes = None,
    **kwargs,
) -> plt.Axes:
    """Plot a dataset using seaborn.

    Args:
        df (pd.DataFrame): Pandas DataFrame to plot
        x_label (str): Label for the x-axis
        y_label (str): Label for the y-axis
        title (str): Title of the plot
        x (Optional[str], optional): Column to plot on the x-axis. Defaults to None.
        y (Optional[str], optional): Column to plot on the y-axis. Defaults to None.
        ax (plt.Axes, optional): Axes to plot on. Defaults to None.
        kwargs (dict, optional): Arguments passed to sns.lineplot. Defaults to {}.

    Returns:
        plt.Axes: [description]
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 8))
    ax = sns.lineplot(data=df, x=x, y=y, ax=ax, **kwargs)
    ax.grid(True)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return ax
