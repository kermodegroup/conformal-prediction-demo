"""
Regression model classes for uncertainty quantification.

Includes Bayesian Ridge, Conformal Prediction, Neural Network Ensemble,
and Quantile Regression wrappers.
"""

import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import QuantileRegressor


class MyBayesianRidge(BayesianRidge):
    """
    Extended BayesianRidge with additional uncertainty methods.

    Adds:
    - Option to return epistemic-only or total (epistemic + aleatoric) uncertainty
    - Log marginal likelihood accessor
    - Leave-one-out cross-validation log likelihood
    """

    def predict(self, X, return_std=False, aleatoric=False):
        """
        Predict with optional uncertainty.

        Parameters
        ----------
        X : array (n_samples, n_features)
            Input features
        return_std : bool
            If True, return (prediction, std)
        aleatoric : bool
            If True, include aleatoric noise in uncertainty

        Returns
        -------
        y_pred : array
            Predictions
        y_std : array (if return_std=True)
            Standard deviations
        """
        y_pred = super().predict(X)
        if not return_std:
            return y_pred
        y_var = np.sum((X @ self.sigma_) * X, axis=1)
        if aleatoric:
            y_var += 1.0 / self.alpha_
        return y_pred, np.sqrt(y_var)

    def sample_posterior(self, X, n_samples=10):
        """
        Draw samples from the posterior predictive distribution.

        Parameters
        ----------
        X : array (n_points, n_features)
            Input features
        n_samples : int
            Number of posterior samples to draw

        Returns
        -------
        samples : array (n_points, n_samples)
            Posterior predictive samples
        """
        if not hasattr(self, "coef_"):
            raise ValueError("Model must be fitted first")

        # Sample coefficients from posterior N(coef_, sigma_)
        coef_samples = np.random.multivariate_normal(
            self.coef_, self.sigma_, size=n_samples
        )
        # Compute predictions for each sample: X @ coef^T for each coef sample
        # coef_samples shape: (n_samples, n_features)
        # X shape: (n_points, n_features)
        # Result shape: (n_points, n_samples)
        return X @ coef_samples.T

    def log_marginal_likelihood(self):
        """
        Get log marginal likelihood for the fitted model.

        Returns
        -------
        float
            Log marginal likelihood (from final EM iteration)
        """
        if not hasattr(self, "coef_"):
            raise ValueError("Model must be fitted first")

        if hasattr(self, "scores_") and len(self.scores_) > 0:
            return self.scores_[-1]
        else:
            return 0.0

    def loo_log_likelihood(self, X, y, aleatoric=False):
        """
        Compute Leave-One-Out cross-validation log likelihood (per point average).

        Uses efficient closed-form formula without refitting.

        Parameters
        ----------
        X : array (n, p)
            Design matrix (same as used for fitting)
        y : array (n,)
            Target values
        aleatoric : bool
            Include aleatoric noise in variance

        Returns
        -------
        float
            Average LOO log likelihood per point
        """
        if not hasattr(self, "coef_"):
            raise ValueError("Model must be fitted first")

        n = len(y)
        y_pred = X @ self.coef_

        # Predictive variance at each training point (epistemic only)
        pred_var = np.sum((X @ self.sigma_) * X, axis=1)

        # Add aleatoric noise variance
        noise_var = 1.0 / self.alpha_
        if aleatoric:
            pred_var = pred_var + noise_var

        # Leverage (hat matrix diagonal)
        h = np.sum((X @ self.sigma_) * X, axis=1) * self.alpha_
        h = np.clip(h, 0, 0.999)

        # LOO residuals and variances
        residuals = y - y_pred
        loo_residuals = residuals / (1 - h)
        loo_var = (pred_var + noise_var) / (1 - h)

        # Gaussian log likelihood
        loo_var = np.maximum(loo_var, 1e-10)
        log_lik = -0.5 * (np.log(2 * np.pi * loo_var) + loo_residuals**2 / loo_var)

        return np.mean(log_lik)


class ConformalPrediction(MyBayesianRidge):
    """
    Conformal prediction wrapper for Bayesian Ridge.

    Calibrates prediction intervals using a held-out calibration set.
    """

    def get_scores(self, X, y, aleatoric=False):
        """Compute conformity scores (normalized residuals)."""
        y_pred, y_std = self.predict(X, return_std=True, rescale=False, aleatoric=aleatoric)
        residuals = (y_pred - y) / y_std
        scores = np.abs(residuals)
        return scores

    def calibrate(self, X_calib, y_calib, zeta=0.05, aleatoric=False):
        """
        Calibrate prediction intervals using calibration set.

        Parameters
        ----------
        X_calib : array
            Calibration features
        y_calib : array
            Calibration targets
        zeta : float
            Miscoverage rate (e.g., 0.05 for 95% coverage)
        aleatoric : bool
            Include aleatoric noise

        Returns
        -------
        float
            Calibrated quantile (qhat)
        """
        scores = self.get_scores(X_calib, y_calib, aleatoric=aleatoric)
        n = float(len(y_calib))
        self.qhat = np.quantile(
            scores, np.ceil((n + 1) * (1.0 - zeta)) / n, method="higher"
        )
        return self.qhat

    def predict(self, X, return_std=False, aleatoric=False, rescale=True):
        """
        Predict with optional conformal rescaling.

        Parameters
        ----------
        X : array
            Input features
        return_std : bool
            Return uncertainty
        aleatoric : bool
            Include aleatoric noise
        rescale : bool
            Apply conformal calibration factor

        Returns
        -------
        y_pred : array
            Predictions
        y_std : array (if return_std=True)
            Calibrated standard deviations
        """
        res = super().predict(X, return_std=return_std, aleatoric=aleatoric)
        if not return_std:
            return res
        y_pred, y_std = res
        if rescale:
            y_std = y_std * self.qhat
        return y_pred, y_std


class NeuralNetworkRegression:
    """
    Ensemble of MLPRegressors for uncertainty quantification.

    Uses sklearn's MLPRegressor with ensemble for uncertainty estimation.
    """

    def __init__(
        self,
        hidden_layer_sizes=(20,),
        alpha=0.001,
        n_ensemble=5,
        max_iter=1000,
        tol=1e-3,
        activation="tanh",
        ensemble_method="seed",
    ):
        """
        Parameters
        ----------
        hidden_layer_sizes : tuple
            Number of neurons in each hidden layer
        alpha : float
            L2 regularization strength
        n_ensemble : int
            Number of networks in ensemble
        max_iter : int
            Maximum iterations for LBFGS solver
        tol : float
            Tolerance for optimization
        activation : str
            Activation function ('tanh' or 'relu')
        ensemble_method : str
            'seed' for different initializations, 'bootstrap' for data resampling
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.n_ensemble = n_ensemble
        self.max_iter = max_iter
        self.tol = tol
        self.activation = activation
        self.ensemble_method = ensemble_method
        self.ensemble_ = []
        self.x_mean_ = None
        self.x_std_ = None

    def fit(self, X, y):
        """Fit ensemble of neural networks."""
        # Standardize inputs
        self.x_mean_ = np.mean(X, axis=0, keepdims=True)
        self.x_std_ = np.std(X, axis=0, keepdims=True) + 1e-8
        X_scaled = (X - self.x_mean_) / self.x_std_

        # Train ensemble
        self.ensemble_ = []
        for i in range(self.n_ensemble):
            if self.ensemble_method == "bootstrap":
                # Bootstrap: resample data with replacement
                n_samples = len(X_scaled)
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_boot = X_scaled[indices]
                y_boot = y[indices]
                random_state = 42
            else:
                # Random seed: different initialization
                X_boot = X_scaled
                y_boot = y
                random_state = i

            model = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation=self.activation,
                solver="lbfgs",
                alpha=self.alpha,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=random_state,
                warm_start=False,
            )
            model.fit(X_boot, y_boot)
            self.ensemble_.append(model)

        return self

    def predict(self, X, return_std=False, aleatoric=False):
        """
        Predict using ensemble.

        Parameters
        ----------
        X : array
            Input data
        return_std : bool
            If True, return (predictions, std_dev)
        aleatoric : bool
            Not implemented for NN (ensemble variance includes both)

        Returns
        -------
        y_pred : array
            Mean predictions
        y_std : array (if return_std=True)
            Ensemble standard deviation
        """
        X_scaled = (X - self.x_mean_) / self.x_std_

        predictions = np.array([model.predict(X_scaled) for model in self.ensemble_])
        y_pred = np.mean(predictions, axis=0)

        if return_std:
            y_std = np.std(predictions, axis=0)
            return y_pred, y_std
        else:
            return y_pred

    def predict_ensemble(self, X):
        """
        Return predictions from all ensemble members for CRPS computation.

        Returns
        -------
        array (n_samples, n_ensemble)
            Predictions from each ensemble member
        """
        X_scaled = (X - self.x_mean_) / self.x_std_
        predictions = np.array([model.predict(X_scaled) for model in self.ensemble_])
        return predictions.T


class QuantileRegressionUQ:
    """
    Quantile regression for uncertainty quantification.

    Directly predicts prediction intervals by fitting models at different quantiles.
    """

    def __init__(self, confidence=0.9, fit_intercept=False, alpha=1.0):
        """
        Parameters
        ----------
        confidence : float
            Confidence level for prediction intervals (e.g., 0.9 for 90%)
        fit_intercept : bool
            Whether to fit intercept
        alpha : float
            Regularization strength
        """
        self.confidence = confidence
        self.fit_intercept = fit_intercept
        self.significance = 1 - confidence
        self.alpha = alpha

        self.model_lower = QuantileRegressor(
            quantile=self.significance / 2,
            alpha=alpha,
            solver="highs",
            fit_intercept=fit_intercept,
        )
        self.model_median = QuantileRegressor(
            quantile=0.5,
            alpha=alpha,
            solver="highs",
            fit_intercept=fit_intercept,
        )
        self.model_upper = QuantileRegressor(
            quantile=1 - self.significance / 2,
            alpha=alpha,
            solver="highs",
            fit_intercept=fit_intercept,
        )

    def fit(self, X, y):
        """Fit three quantile regression models."""
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            try:
                self.model_lower.fit(X, y)
                self.model_median.fit(X, y)
                self.model_upper.fit(X, y)
                self.fit_succeeded_ = True
            except Exception:
                self.fit_succeeded_ = False
                self.fallback_mean_ = np.mean(y)
                self.fallback_std_ = np.std(y)

        return self

    def predict(self, X, return_std=False, aleatoric=False):
        """
        Predict using quantile regression.

        Parameters
        ----------
        X : array
            Input features
        return_std : bool
            If True, return (predictions, std_deviations)
        aleatoric : bool
            Not used (intervals include all uncertainty)

        Returns
        -------
        y_pred : array
            Median predictions
        y_std : array (if return_std=True)
            Approximate std from interval width
        """
        if not self.fit_succeeded_:
            n_samples = X.shape[0]
            y_pred = np.full(n_samples, self.fallback_mean_)
            if return_std:
                y_std = np.full(n_samples, self.fallback_std_)
                return y_pred, y_std
            return y_pred

        y_pred = self.model_median.predict(X)

        if not return_std:
            return y_pred

        y_lower = self.model_lower.predict(X)
        y_upper = self.model_upper.predict(X)

        # Approximate std from interval width
        from scipy.stats import norm

        z_score = norm.ppf(1 - self.significance / 2)
        y_std = (y_upper - y_lower) / (2 * z_score)

        return y_pred, y_std

    def predict_quantiles(self, X):
        """
        Return raw quantile predictions for CRPS computation.

        Returns
        -------
        y_lower : array
            Lower quantile predictions
        y_median : array
            Median predictions
        y_upper : array
            Upper quantile predictions
        alpha : float
            Significance level
        """
        if not self.fit_succeeded_:
            n_samples = X.shape[0]
            return (
                np.full(n_samples, self.fallback_mean_),
                np.full(n_samples, self.fallback_mean_),
                np.full(n_samples, self.fallback_mean_),
                self.significance,
            )

        y_lower = self.model_lower.predict(X)
        y_median = self.model_median.predict(X)
        y_upper = self.model_upper.predict(X)

        return y_lower, y_median, y_upper, self.significance
