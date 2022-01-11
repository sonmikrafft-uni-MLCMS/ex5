# module for model implemnetations

import numpy as np
from typing import Optional

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import RandomizedSearchCV


def get_best_model(
    param_grid: dict, model: BaseEstimator, X: np.ndarray, Y: np.ndarray, n_iter: int = 40, cv: int = 5
) -> BaseEstimator:
    """Tries to find the best model for the given parameters by using a randomized search with cross-validation.

    Args:
        param_grid (dict): dictionary with the parameters to try
        model (BaseEstimator): model to use
        X (np.ndarray): input data
        Y (np.ndarray): output data
        n_iter (int, optional): number of iterations for the randomized search. Defaults to 40.
        cv (int, optional): number of cross-validation folds. Defaults to 5.

    Returns:
        BaseEstimator: best model found
    """
    clf = RandomizedSearchCV(model, param_grid, n_iter=n_iter, cv=cv, n_jobs=-1, refit=True, verbose=1)
    clf.fit(X, Y)

    print(f"Best R2 score:\n{clf.best_score_}")
    print(f"Best params:\n{clf.best_params_}")

    return clf.best_estimator_


class RidgeRegression(BaseEstimator, RegressorMixin):
    def __init__(self, lamb: float = 0):
        """Init method for Ridge Regression.

        Finds a weight matrix W of shape (n_features, n_targets) that linearly maps from the feature space of
        dimension n_features to the output space of dimension n_targets such that the L2 - norm between the
        prediction and the target is minimized.

        Ridge regression finds weights W such that:
        ||y - XW||^2 = ||y - XW||^2 + lamb * ||W||^2
        is minimized.

        See https://stats.stackexchange.com/questions/69205/how-to-derive-the-ridge-regression-solution for a
        mathematical derivation of the Ridge Regression normal form.

        For an official implementation see:
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

        Attributes:
            lamb (float): Regularization parameter
            W (np.ndarray): Weights of the model, of shape (n_features, n_targets)

        Args:
            lamb (float, optional): Regularization strength; must be a positive float. Defaults to 0.
        """
        self.lamb = lamb
        self.W = None  # type: Optional[np.ndarray]

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fit the model to the data

        Fits the RidgeRegression model to the data and stores the weights in self.w.

        Args:
            X (np.ndarray): Training data, of shape (n_samples, n_features)
            Y (np.ndarray): Training targets, of shape (n_samples, n_targets)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # add feature dimension
        D = X.shape[1]

        # see normal form for ridge regression
        self.W = np.linalg.lstsq(X.T.dot(X) + self.lamb * np.identity(D), X.T.dot(Y), rcond=None)[0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Calculate new prediction based on the trained model

        Args:
            X (np.ndarray): Data to predict on, of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predictions, of shape (n_samples, n_targets)
        """
        assert self.W is not None, "Model not trained yet."
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # add feature dimension
        return X @ self.W


class RBFRegression(BaseEstimator, RegressorMixin):
    def __init__(self, eps: float = 0.1, L: int = 100, lamb: float = 0) -> None:
        """Init method for RBF Regression.

        First transforms the data (n_samples, n_features) into a design matrix Phi of dimension (n_samples, L),
        then finds a weight matrix W of shape (L, n_targets) that linearly maps from the intermediate feature space
        if dimension L to the output space of dimension n_targets such that the L2 - norm between the prediction and
        the target is minimized.

        RBF regression finds weights W such that:
        ||y - PhiW||^2 = ||y - PhiW||^2 + lamb * ||W||^2
        is minimized, where Phi is the design matrix and the intermediate feature space.

        Note that the intermediate feature space is not learnt from the data, but is instead determined by the
        bandwidth parameter eps and the number of RBFs L.

        For an official implementation see:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html

        Args:
            eps (float, optional): Bandwidth parameter. Defaults to 0.1.
            L (int, optional): Number of RBFs. Defaults to 100.
            lamb (float, optional): Regularization parameter. Defaults to 0.
        """
        self.lamb = lamb
        self.eps = eps
        self.L = L
        self.interpolation_centers = None  # type: Optional[np.ndarray]
        self.linear_regression = RidgeRegression(lamb)

    def gaussian_rbf(self, x_l: np.ndarray, x: np.ndarray, eps: float) -> float:
        """Calculate the gaussian radial function for given vector pair

        The gaussian radial function is defined as:
        gaussian_rbf(x_l, x) = exp(-||x_l - x||^2 / (eps^2))

        where:
        - x_l: center point
        - x: point to calculate the gaussian for
        - eps: bandwidth parameter (controls radius/flatness)

        Args:
            x_l (np.array): center of shape (n_features,)
            x (np.array): incoming data of shape (n_features,)
            eps (float): bandwidth parameter (controls radius/flatness) shape (1,)

        Returns:
            float: value for gaussian radial function, shape (1,)
        """
        return np.exp(-np.linalg.norm(x_l - x, axis=-1) ** 2 / (eps ** 2))

    def sample_interpolation_center(self, X: np.ndarray) -> np.ndarray:
        """randomly sample L number of dataset points.

        Args:
            X (np.ndarray): dataset of shape (n_samples, n_features)

        Returns:
            np.ndarray: matrix with L dataset point pairs of shape (L, n_features)
        """
        random_interpolation_centers = np.random.choice(X[:, 0], size=self.L)
        return random_interpolation_centers

    def calculate_design_matrix(self, X: np.ndarray, interpolation_centers: np.ndarray, eps: float) -> np.ndarray:
        """Fill the designmatrix by applying rbf function on X at centers.

        Args:
            X (np.ndarray): dataset of shape (n_samples, n_features)
            interpolation_centers (np.ndarray): interpolation centers for RBFs of shape (L, n_features)
            eps (float): bandwidth parameter (controls radius/flatness) shape (1,)

        Returns:
            np.ndarray: Design matrix Phi of shape (n_samples, L) where L is the number of RBFs to use
        """
        shape = (X.shape[0], self.L)
        Phi = np.zeros(shape)  # shape (n_samples, L)

        for idx_centers, center in enumerate(interpolation_centers):
            Phi[:, idx_centers] = self.gaussian_rbf(center, X, eps)

        return Phi

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """Fit the model to the data

        Fits the RBF Regression model to the data.

        Args:
            X (np.ndarray): Training data, of shape (n_samples, n_features)
            y (np.ndarray): Training targets, of shape (n_samples, n_targets)
        """
        if X.ndim == 1:
            X = X.reshape(-1, 1)  # add feature dimension
        self.d = X.shape[1]

        # sample center points to fit gaussian
        self.interpolation_centers = self.sample_interpolation_center(X)

        # fill design matrix
        Phi = self.calculate_design_matrix(X, self.interpolation_centers, self.eps)

        # find optimal w* for design matrix Phi
        self.linear_regression.fit(Phi, Y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Calculate new prediction based on the trained model.

        Args:
            X (np.ndarray): Data to predict on, of shape (n_samples, n_features)

        Returns:
            np.ndarray: Predictions, of shape (n_samples, n_targets)
        """
        assert self.interpolation_centers is not None, "Model not trained yet"
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        Phi = self.calculate_design_matrix(X, self.interpolation_centers, self.eps)
        return self.linear_regression.predict(Phi)
