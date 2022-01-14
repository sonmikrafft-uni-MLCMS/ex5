# module to plot data

import seaborn as sns
import pandas as pd
import numpy as np
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
    legend_loc: str = "best",
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
        legend_loc (str, optional): Where to place the legend. Defaults to "best".
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
    ax.legend(loc=legend_loc)
    return ax


def plot_3d_plot(
    data: np.ndarray,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    title: str,
    title_colorbar: str = None,
    ax: plt.Axes = None,
    fig: plt.Figure = None,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a 3D plot.

    Args:
        data (np.ndarray): Data to plot of shape (num_points, 3)
        xlabel (str): Label for the x-axis
        ylabel (str): Label for the y-axis
        zlabel (str): Label for the z-axis
        title (str): Title of the plot
        title_colorbar (str, optional): Title of the colorbar. Defaults to None.
        ax (plt.Axes, optional): Axes to plot on. Defaults to None.
        fig (plt.Figure, optional): Figure to plot on. Defaults to None.
        kwargs (dict, optional): Arguments passed to plt.scatter. Defaults to {}.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and axes of the plot.
    """    
    if ax is None or fig is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(*data.T, **kwargs)
    if title_colorbar is not None:
        cbar = fig.colorbar(scatter, location="left", anchor=(0, 0.3), pad=0.02, shrink=0.7)
        cbar.set_label(title_colorbar, rotation=90)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return fig, ax
