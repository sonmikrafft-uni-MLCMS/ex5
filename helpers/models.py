# module for model implemnetations

import numpy as np

from typing import Optional


class RidgeRegression:
    def __init__(self, lamb: float = 0):
        """Init method for Ridge Regression.

        See https://stats.stackexchange.com/questions/69205/how-to-derive-the-ridge-regression-solution for a
        mathematical derivation of the Ridge Regression normal form.

        Args:
            lamb (float, optional): Regularization strength; must be a positive float. Defaults to 0.
        """
        self.lamb = lamb
        self.w = None  # type: Optional[np.ndarray]
        self.d = None  # type: Optional[int]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the data

        Fits the RidgeRegression model to the data and stores the weights in self.w.

        Args:
            X (np.ndarray): Training data, of shape (n_samples, n_features)
            y (np.ndarray): Training targets, of shape (n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # add feature dimension
        self.d = X.shape[1]

        # see normal form for ridge regression
        self.w = np.linalg.lstsq(X.T.dot(X) + self.lamb * np.identity(self.d), X.T.dot(y), rcond=None)[0]

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Calculate new prediction based on the trained model

        Args:
            X (np.ndarray): Data to predict on, of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predictions, of shape (n_samples,)
        """
        assert self.w is not None, "Model not trained yet"
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # add feature dimension
        return X @ self.w

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the model and calculate predictions

        Args:
            X (np.ndarray): Training data, of shape (n_samples, n_features)
            y (np.ndarray): Training targets, of shape (n_samples,)

        Returns:
            np.ndarray: Predictions, of shape (n_samples,)
        """
        self.fit(X, y)
        return self.transform(X)
