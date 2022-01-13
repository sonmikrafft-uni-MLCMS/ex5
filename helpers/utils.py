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


def create_delay_embedding(
    df: pd.DataFrame, num_windows: int, num_delays: int, column_index_to_look_at: list[int]
) -> np.ndarray:
    """Create a delay embedding from the dataframe.

    Args:
        df (pd.DataFrame): Dataframe to create the embedding from.
        num_windows (int): Number of windows to create
        num_delays (int): Number of delays to consider in one window.
        column_index_to_look_at (list[int]): List of column indices to look at within one window.

    Returns:
        np.ndarray: Delay embedding of shape (num_windows, (num_delays + 1) * len(column_index_to_look_at))).
    """
    window_height = num_delays + 1
    window_width = len(column_index_to_look_at)

    out = np.zeros((num_windows, window_height * window_width))
    for i in range(0, num_windows):
        out[i, :] = df.iloc[i : i + window_height, column_index_to_look_at].to_numpy().reshape(-1)

    return out


def compute_arc_length(curve: np.ndarray) -> np.ndarray:
    """Compute the arc length of a curve.

    Args:
        curve (np.ndarray): Curve to compute the length of, of shape (num_points, D).

    Returns:
        np.ndarray: List of length num_points, where each element is the length of the argument of the curve at the
            corresponding point starting from the first point. Of shape (num_points,).
    """
    lengths = np.zeros(curve.shape[0])
    curr_length = 0

    for i in range(lengths.shape[0]):
        if i > 0:
            curr_length += np.linalg.norm(curve[i] - curve[i - 1])
        lengths[i] = curr_length

    return lengths


def integrate_over_periodic_velocity(velocities: np.ndarray, timesteps_to_integrate: int) -> np.ndarray:
    """Integrate over a one-dimensional periodic velocity field over a given number of timesteps.

    Args:
        velocities (np.ndarray): One-dimensional velocity field.
        timesteps_to_integrate (int): Number of timesteps to integrate over.

    Returns:
        np.ndarray: Integrated velocity field, of shape (timesteps_to_integrate,).
    """
    integrated_velocities = np.zeros(timesteps_to_integrate)

    current_integral = 0
    for i in range(timesteps_to_integrate):
        if i > 0:
            current_integral += velocities[i % velocities.shape[0]]
        integrated_velocities[i] = current_integral

    return integrated_velocities


def reset_to_0_every_n(X: np.ndarray, n: int) -> np.ndarray:
    """After every n-th element, set the value to 0 and continue from there

    Example:
    >>> reset_to_0_every_n(np.arange(10), 3)
    array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])

    Args:
        X (np.ndarray): Array to reset
        n (int): Reset after every n-th element

    Returns:
        np.ndarray: Array with values reset to 0 after every n-th element
    """
    i = 0
    X_return = np.zeros_like(X)
    while (i * n) < X.shape[0]:
        X_return[n * i :] = X[n * i :] - X[n * i]
        i += 1
    return X_return
