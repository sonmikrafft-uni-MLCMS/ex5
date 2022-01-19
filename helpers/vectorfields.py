import pandas as pd
import numpy as np
from scipy.integrate import odeint
from Exercise5.helpers.models import RBFRegression


def compute_finite_difference(x_0: pd.DataFrame, x_1: pd.DataFrame, delta_t: float = 0.1) -> pd.DataFrame:
    """
    computes finite difference between x_0 and x_1 for a given delta_t

    Args:
        x_0 (pd.DataFrame): points of shape (N,D) at time t=0 of
        x_1 (pd.DataFrame): points of shape (N,D) at time delta_t
        delta_t (float): delta of time

    Returns: v_k, approximation of the vector field
    """
    v_k = x_1 - x_0
    v_k /= delta_t
    return v_k


def compute_closed_form_linear(x: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    """
    computes closed form solution A.T for linear functions that minimizes the least-squares error
    Args:
        x (pd.DataFrame): input of shape (N,D)
        f (pd.DataFrame): f \approx Ax of shape (N,D)

    Returns: A.T of shape (D,D), closed form solution
    """

    x_t = x.T
    cov = x_t @ x
    cov_inv = pd.DataFrame(np.linalg.pinv(cov.values), cov.columns, cov.index)

    result = cov_inv @ x_t
    result = result @ f

    return result

def deriv_linear(x: np.ndarray, t: np.ndarray, A_T: np.ndarray, dim1: int, dim2: int) -> np.ndarray:
    """
    Args:
        x (np.ndarray): input of shape (N,D)
        t (np.ndarray): time interval
        A (np.ndarray): matrix of shape (D,D)
        dim1 (int): N
        dim2 (int): D

    Returns: x_dot of shape (N,D), x_dot = Ax

    """
    x = x.reshape(dim1, dim2)
    x_dot = x @ A_T
    return x_dot.reshape(-1)

def solve_ode_linear(t: np.array, A_T: np.ndarray, x: pd.DataFrame) -> pd.DataFrame:
    """
    Args:
        A (pd.DataFrame): matrix of shape (D,D)
        x (pd.DataFrame): input of shape (N,D)

    Returns: solution to differential equation x_dot = Ax for a given input x and matrix A

    """
    dim1 = x.shape[0]
    dim2 = x.shape[1]
    initial = x.to_numpy().reshape(-1)
    sol = odeint(deriv_linear, initial, t, args=(A_T, dim1, dim2))
    return sol

def deriv_nonlinear(x: np.ndarray, t: np.ndarray, model: RBFRegression) -> np.ndarray:
    """
    Args:
        x (np.ndarray): input of shape (N,1)
        t (np.ndarray): time interval
        model (RBFRegression): instance of RBFRegression

    Returns: x_dot of shape (N,D), x_dot = Ax

    """
    return model.predict(x)

def solve_ode_nonlinear(t: np.array, model: RBFRegression, x: pd.DataFrame) -> np.ndarray:
    """
    Args:
        model (RBFRegression): instance of RBFRegression
        x (pd.DataFrame): input of shape (N,1)

    Returns: solution to differential equation x_dot = Ax for a given input x and matrix A

    """
    sol = odeint(deriv_nonlinear, x.to_numpy(), t, args=(model,))
    return sol


def mean_squared_error(x_head: np.ndarray, x: np.ndarray) -> float:
    """
    Args:
        x_head (np.ndarray): approximated solution of shape (N,D)
        x (np.ndarray): observed input of shape (N,D)

    Returns: Mean Squared Error

    """
    result = x_head - x
    result = result ** 2
    return result.mean()

