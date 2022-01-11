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


class RBFRegression:
    def __init__(self, eps, L, lamb: float = 0):
        """Init method for Radial Basis Function

        Args:
            lamb (float, optional): Regularization strength; must be a positive float. Defaults to 0.
            eps (float): bandwidth parameter (controls radius/flatness)
            L (int): number of basis functions
        """

        self.eps = eps
        self.L = L
        self.lamb = lamb
        self.w = None  # type: Optional[np.ndarray]
        self.d = None  # type: Optional[int]

    def gaussian_rbf(self, x_l: np.array, x: np.array) -> float:
        """Calculate the gaussian radial function for given vector pair

        Args:
            x_l (np.array): center vector Dx1
            x (np.array): data vector Dx1
            eps (float): bandwidth parameter (controls radius/flatness) 1x1

        Returns:
            [float]: radial basis function value
        """
        return np.exp(-np.linalg.norm(x_l - x) ** 2 / (self.eps ** 2))

    def sample_interpolation_center(self, X: np.ndarray):
        """randomly sample L number of dataset points

        Args:
            X (np.ndarray): dataset

        Returns:
            [np.ndarray]: matrix with L Number of dataset point pairs
        """
        random_interpolation_centers = np.random.choice(X[:, 0], size=self.L)
        return random_interpolation_centers

    def calculate_design_matrix(self, X: np.ndarray, interpolation_centers: np.ndarray):
        """Fill the designmatrix by applying rbf function on X at centers

        Args:
            X (np.ndarray): dataset add shapes !!!!
            interpolation_centers (np.ndarray): sampled approximation points (num=L)

        Returns:
            Phi (np.ndarray): Design matrix Phi
        """
        shape = (X.shape[0], self.L)
        Phi = np.zeros(shape)  # NxL

        for idx_centers, center in enumerate(interpolation_centers):
            Phi[:, idx_centers] = self.gaussian_rbf(center, X)

        print(Phi.shape)
        return Phi

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the data

        Fits the RBF Regression model to the data and stores the weights in self.w.

        Args:
            X (np.ndarray): Training data, of shape (n_samples, n_features)
            y (np.ndarray): Training targets, of shape (n_samples,)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # add feature dimension
        self.d = X.shape[1]

        # sample center points to fit gaussian
        interpolation_centers = self.sample_interpolation_center(X)

        # fill design matrix
        Phi = self.calculate_design_matrix(X, interpolation_centers)

        # find optimal w* for design matrix Phi
        self.w = np.linalg.lstsq(Phi.T.dot(Phi) + self.lamb * np.identity(self.d), Phi.T.dot(y), rcond=None)[0]

    def transform(self, Phi: np.ndarray) -> np.ndarray:
        """Calculate new prediction based on the trained model

        Args:
            Phi (np.ndarray): Data to predict on, of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predictions, of shape (n_samples,)
        """
        assert self.w is not None, "Model not trained yet"
        if Phi.ndim == 1:
            Phi = Phi.reshape(-1, 1)  # add feature dimension
        return Phi @ self.w

    def fit_transform(self, Phi: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the model and calculate predictions

        Args:
            X (np.ndarray): Training data, of shape (n_samples, n_features)
            y (np.ndarray): Training targets, of shape (n_samples,)

        Returns:
            np.ndarray: Predictions, of shape (n_samples,)
        """
        self.fit(Phi, y)
        return self.transform(Phi)
