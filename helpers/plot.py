# module to plot data
import matplotlib.pyplot as plt
import numpy as np


def plot_phase_portrait(
    mesh_tuple: np.ndarray,
    flow_tuple: np.ndarray,
    **kwargs,
):
    """
    Plot phase portrait.
    Args:
        mesh_tuple (np.ndarray): 2D mesh with two elements of shape (N,)
        flow_tuple (np.ndarray): 2D flow tuple of two elements of shape (N,)
        **kwargs (): additional arguments
    """
    (X1, X2) = mesh_tuple
    (U, V) = flow_tuple

    _, ax = plt.subplots(1, 1)

    ax.streamplot(X1, X2, U, V, density=1.0, **kwargs)
    ax.set_aspect("equal")
    ax.set_xlim([X1[0, 0], X1[-1, -1]])
    ax.set_ylim([X2[0, 0], X2[-1, -1]])
