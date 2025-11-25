# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "watchdog",
#     "matplotlib==3.10.1",
#     "numpy==2.2.5",
#     "popsregression @ git+https://github.com/tomswinburne/POPS-Regression.git",
#     "scikit-learn==1.6.1",
#     "seaborn==0.13.2",
#     "qrcode==8.2",
#     "scipy",
# ]
# ///

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.Html('''
    <style>
        body, .marimo-container {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }

        /* Main layout container - use full viewport height */
        body {
            height: 100vh;
            overflow: hidden;
        }

        .marimo-container {
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }

        /* Header section - fixed height */
        .app-header {
            flex-shrink: 0;
            padding: 10px 0;
        }

        /* Dashboard section - fixed height with internal scrolling if needed */
        .app-dashboard {
            flex-shrink: 0;
            max-height: 35vh;
            overflow-y: auto;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 8px;
            margin: 0 auto 10px auto;
            max-width: 90%;
        }

        /* Plot section - takes remaining space */
        .app-plot {
            flex-grow: 1;
            flex-shrink: 1;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            min-height: 0;
        }

        /* Ensure matplotlib figures scale properly */
        .app-plot img,
        .app-plot svg {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
        }
    </style>
    ''')
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import PolynomialFeatures
    from POPSRegression import POPSRegression
    from scipy.linalg import cho_factor, cho_solve

    # Customize default plotting style
    import seaborn as sns
    sns.set_context('talk')
    return (
        BayesianRidge,
        POPSRegression,
        PolynomialFeatures,
        cho_factor,
        cho_solve,
        mo,
        np,
        plt,
        train_test_split,
    )


@app.cell
def _(np):
    def g(X, noise_variance, function_type='sin'):
        """Generate data from different ground truth functions"""
        if function_type == 'sin':
            y = np.sin(X)
        elif function_type == 'witch':
            # Witch of Agnesi: 1/(1+x²)
            y = 1.0 / (1.0 + X**2)
        elif function_type == 'sinc':
            # sinc(x) = sin(x)/x, handle x=0 carefully
            y = np.where(X == 0, 1.0, np.sin(X) / X)
        elif function_type == 'step':
            # Step function
            y = np.where(X < 0, -0.5, 0.5)
        elif function_type == 'runge':
            # Runge function: 1/(1+25x²) - pathological for polynomial interpolation
            y = 1.0 / (1.0 + 25.0 * X**2)
        else:
            y = np.sin(X)  # Default to sin

        return y + np.random.normal(scale=np.sqrt(noise_variance), size=X.shape)
    return (g,)


@app.cell
def _(BayesianRidge, np):
    class MyBayesianRidge(BayesianRidge):
        def predict(self, X, return_std=False, aleatoric=False):
            y_pred = super().predict(X)
            if not return_std:
                return y_pred
            y_var = np.sum((X @ self.sigma_) * X, axis=1)
            if aleatoric:
                y_var += 1.0 / self.alpha_
            return y_pred, np.sqrt(y_var)

        def log_marginal_likelihood(self):
            """
            Compute log marginal likelihood for the fitted Bayesian Ridge model

            Uses the evidence lower bound from the fitted model
            """
            if not hasattr(self, 'coef_'):
                raise ValueError("Model must be fitted first")

            # BayesianRidge stores the negative log marginal likelihood in scores_
            # Return the final score (which is already log marginal likelihood)
            if hasattr(self, 'scores_') and len(self.scores_) > 0:
                return self.scores_[-1]
            else:
                return 0.0

    class ConformalPrediction(MyBayesianRidge):
        def get_scores(self, X, y, aleatoric=False):
            y_pred, y_std = self.predict(X, return_std=True, rescale=False, aleatoric=aleatoric)
            residuals = (y_pred - y) / y_std
            scores = np.abs(residuals)
            return scores

        def calibrate(self, X_calib, y_calib, zeta=0.05, aleatoric=False):
            scores = self.get_scores(X_calib, y_calib, aleatoric=aleatoric)
            n = float(len(y_calib))
            self.qhat = np.quantile(scores, np.ceil((n+1)*(1.0-zeta))/n, method='higher')
            return self.qhat

        def predict(self, X, return_std=False, aleatoric=False, rescale=True):
            res = super().predict(X, return_std=return_std, aleatoric=aleatoric)
            if not return_std:
                return res
            y_pred, y_std = res
            if rescale:
                y_std = y_std * self.qhat
            return y_pred, y_std
    return ConformalPrediction, MyBayesianRidge


@app.cell
def _(np):
    from sklearn.neural_network import MLPRegressor

    class NeuralNetworkRegression:
        """
        Ensemble of MLPRegressors for uncertainty quantification

        Uses sklearn's MLPRegressor with ensemble for uncertainty estimation
        """
        def __init__(self, hidden_layer_sizes=(20,), alpha=0.001, n_ensemble=5, max_iter=1000, tol=1e-3, activation='tanh', ensemble_method='seed'):
            """
            Parameters:
            -----------
            hidden_layer_sizes : tuple, default=(20,)
                Number of neurons in each hidden layer, e.g., (20,) or (20, 10)
            alpha : float, default=0.001
                L2 regularization strength
            n_ensemble : int, default=5
                Number of networks in ensemble for uncertainty estimation
            max_iter : int, default=1000
                Maximum number of iterations for LBFGS solver
            tol : float, default=1e-3
                Tolerance for optimization (relaxed for faster convergence)
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
            """Fit ensemble of neural networks using either different seeds or bootstrap"""
            # Standardize inputs for better NN training
            self.x_mean_ = np.mean(X, axis=0, keepdims=True)
            self.x_std_ = np.std(X, axis=0, keepdims=True) + 1e-8  # Avoid division by zero
            X_scaled = (X - self.x_mean_) / self.x_std_

            # Train ensemble
            self.ensemble_ = []
            for i in range(self.n_ensemble):
                if self.ensemble_method == 'bootstrap':
                    # Bootstrap: resample data with replacement
                    n_samples = len(X_scaled)
                    indices = np.random.choice(n_samples, size=n_samples, replace=True)
                    X_boot = X_scaled[indices]
                    y_boot = y[indices]
                    # Use same random seed for all bootstrap models
                    random_state = 42
                else:
                    # Random seed: use different random initialization for each model
                    X_boot = X_scaled
                    y_boot = y
                    random_state = i

                model = MLPRegressor(
                    hidden_layer_sizes=self.hidden_layer_sizes,
                    activation=self.activation,
                    solver='lbfgs',
                    alpha=self.alpha,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    random_state=random_state,
                    warm_start=False
                )
                model.fit(X_boot, y_boot)
                self.ensemble_.append(model)

            return self

        def predict(self, X, return_std=False, aleatoric=False):
            """
            Predict using ensemble

            Parameters:
            -----------
            X : array-like
                Input data
            return_std : bool, default=False
                If True, return (predictions, std_dev)
            aleatoric : bool, default=False
                If True, add noise variance to uncertainty (not implemented for NN)

            Returns:
            --------
            y_pred : array
                Mean predictions across ensemble
            y_std : array (if return_std=True)
                Standard deviation across ensemble
            """
            # Standardize inputs using training statistics
            X_scaled = (X - self.x_mean_) / self.x_std_

            # Get predictions from all ensemble members
            predictions = np.array([model.predict(X_scaled) for model in self.ensemble_])

            # Compute mean and std across ensemble
            y_pred = np.mean(predictions, axis=0)

            if return_std:
                y_std = np.std(predictions, axis=0)
                # Note: aleatoric uncertainty not implemented for NN ensemble
                # The ensemble variance already captures both epistemic and some aleatoric uncertainty
                return y_pred, y_std
            else:
                return y_pred
    return (NeuralNetworkRegression,)


@app.cell
def _(np):
    from sklearn.linear_model import QuantileRegressor

    class QuantileRegressionUQ:
        """
        Quantile regression for uncertainty quantification

        Directly predicts prediction intervals by fitting models at different quantiles.
        Non-parametric approach that doesn't assume a specific distribution.
        """
        def __init__(self, confidence=0.9, fit_intercept=False, alpha=1.0):
            """
            Parameters:
            -----------
            confidence : float, default=0.9
                Confidence level for prediction intervals (e.g., 0.9 for 90% intervals)
            fit_intercept : bool, default=False
                Whether to fit intercept (usually False when using polynomial basis)
            alpha : float, default=1.0
                Regularization strength (higher = more regularization)
            """
            self.confidence = confidence
            self.fit_intercept = fit_intercept
            self.significance = 1 - confidence  # Significance level
            self.alpha = alpha  # Regularization parameter

            # Three quantile models: lower, median, upper
            self.model_lower = QuantileRegressor(
                quantile=self.significance/2,
                alpha=alpha,
                solver='highs',
                fit_intercept=fit_intercept
            )
            self.model_median = QuantileRegressor(
                quantile=0.5,
                alpha=alpha,
                solver='highs',
                fit_intercept=fit_intercept
            )
            self.model_upper = QuantileRegressor(
                quantile=1 - self.significance/2,
                alpha=alpha,
                solver='highs',
                fit_intercept=fit_intercept
            )

        def fit(self, X, y):
            """Fit three quantile regression models"""
            import warnings

            # Suppress convergence warnings and catch fit failures
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')  # Suppress all warnings during fit

                try:
                    self.model_lower.fit(X, y)
                    self.model_median.fit(X, y)
                    self.model_upper.fit(X, y)
                    self.fit_succeeded_ = True
                except Exception as e:
                    # If fitting fails, mark as failed
                    # Predictions will fall back to simple mean
                    self.fit_succeeded_ = False
                    self.fallback_mean_ = np.mean(y)
                    self.fallback_std_ = np.std(y)

            return self

        def predict(self, X, return_std=False, aleatoric=False):
            """
            Predict using quantile regression

            Parameters:
            -----------
            X : array-like
                Input features
            return_std : bool, default=False
                If True, return (predictions, std_deviations)
            aleatoric : bool, default=False
                Not used for quantile regression (intervals already include all uncertainty)

            Returns:
            --------
            y_pred : array
                Median predictions
            y_std : array (if return_std=True)
                Approximate standard deviation from prediction interval width
            """
            # If fitting failed, return fallback predictions
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
            # For normal distribution, 90% interval is roughly ±1.645σ
            # So interval width ≈ 2*1.645σ → σ ≈ width / 3.29
            # For 95%: width / 3.92, for 80%: width / 2.56
            from scipy.stats import norm
            z_score = norm.ppf(1 - self.significance/2)
            y_std = (y_upper - y_lower) / (2 * z_score)

            return y_pred, y_std

    return (QuantileRegressionUQ,)


@app.cell
def _(cho_factor, cho_solve, np):
    # Pure numpy/scipy GP implementation for WebAssembly compatibility

    def bump_kernel(X1, X2, lengthscale, support_radius):
        """Wendland C2 compactly supported kernel"""
        dists = np.abs(X1[:, None] - X2[None, :])
        r = dists / (lengthscale * support_radius)
        r_clipped = np.clip(r, 0.0, 1.0)
        K = np.where(r < 1.0, (1.0 - r_clipped)**4 * (4.0 * r_clipped + 1.0), 0.0)
        return K

    def polynomial_kernel(X1, X2, degree, sigma):
        """Polynomial kernel"""
        # Normalize inputs
        x1_norm = X1 * 0.1
        x2_norm = X2 * 0.1
        K = (sigma + x1_norm[:, None] * x2_norm[None, :])**degree
        return K

    def rbf_kernel(X1, X2, lengthscale):
        """RBF/squared exponential kernel"""
        dists_sq = (X1[:, None] - X2[None, :])**2
        K = np.exp(-0.5 * dists_sq / lengthscale**2)
        return K

    def matern_kernel(X1, X2, lengthscale, nu=1.5):
        """Matérn kernel with nu=1.5 (Matérn-3/2)"""
        dists = np.abs(X1[:, None] - X2[None, :])
        r = dists / lengthscale
        sqrt3_r = np.sqrt(3.0) * r
        K = (1.0 + sqrt3_r) * np.exp(-sqrt3_r)
        return K

    def compute_kernel_matrix(X1, X2, kernel_type, params):
        """Helper to compute kernel matrix given type and params"""
        if kernel_type == 'bump':
            return bump_kernel(X1, X2, params['lengthscale'], params['support_radius'])
        elif kernel_type == 'polynomial':
            return polynomial_kernel(X1, X2, params['degree'], params['sigma'])
        elif kernel_type == 'rbf':
            return rbf_kernel(X1, X2, params['lengthscale'])
        elif kernel_type == 'matern':
            return matern_kernel(X1, X2, params['lengthscale'])
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def gp_marginal_likelihood(X_train, y_train, K_train, noise):
        """
        Compute log marginal likelihood

        log p(y|X) = -0.5 * y^T K^{-1} y - 0.5 * log|K| - n/2 * log(2π)
        """
        n = len(y_train)
        K_noisy = K_train + noise * np.eye(n)

        try:
            L, lower = cho_factor(K_noisy, lower=True)
        except np.linalg.LinAlgError:
            return -np.inf

        # y^T K^{-1} y
        alpha = cho_solve((L, lower), y_train)
        data_fit = -0.5 * np.dot(y_train, alpha)

        # log|K|
        complexity = -np.sum(np.log(np.diag(L)))

        # constant
        constant = -0.5 * n * np.log(2 * np.pi)

        return data_fit + complexity + constant

    def gp_joint_marginal_likelihood(X_train, y_train, K_train, noise, Phi_train, mean_regularization_strength, poly_degree):
        """
        Compute log marginal likelihood with joint Bayesian inference over polynomial mean coefficients

        Model: y = Φβ + f + ε
        where β ~ N(0, λ⁻¹I), f ~ GP(0, K), ε ~ N(0, σ²I)

        Marginal: y ~ N(0, K_total) where K_total = ΦΛ⁻¹Φᵀ + K + σ²I

        log p(y|X) = -0.5 * y^T K_total^{-1} y - 0.5 * log|K_total| - n/2 * log(2π)

        Parameters:
        -----------
        X_train : array (n_train,)
        y_train : array (n_train,)
        K_train : array (n_train, n_train) - GP kernel on training data
        noise : float - observation noise variance
        Phi_train : array (n_train, n_features) - polynomial feature matrix
        mean_regularization_strength : float - regularization strength for polynomial coefficients
        poly_degree : int - polynomial degree (for degree-adaptive regularization)

        Returns:
        --------
        log_marginal_likelihood : float
        """
        from sklearn.preprocessing import PolynomialFeatures

        n = len(y_train)

        # Create polynomial features if not provided
        if Phi_train is None:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
            Phi_train = poly.fit_transform(X_train.reshape(-1, 1))

        # Normalize polynomial features for numerical stability
        Phi_train_std = np.std(Phi_train, axis=0, keepdims=True)
        Phi_train_std[Phi_train_std < 1e-10] = 1.0
        Phi_train_normalized = Phi_train / Phi_train_std

        # Prior precision for polynomial coefficients (degree-adaptive)
        lambda_prior = mean_regularization_strength * (poly_degree / 3.0)
        Lambda_inv = (1.0 / lambda_prior) * np.eye(Phi_train_normalized.shape[1])

        # Adjust for normalization
        Lambda_inv_normalized = Lambda_inv / (Phi_train_std.T @ Phi_train_std + 1e-10)

        # Build total covariance: K_total = ΦΛ⁻¹Φᵀ + K_GP + σ²I
        K_beta = Phi_train_normalized @ Lambda_inv_normalized @ Phi_train_normalized.T
        K_total = K_beta + K_train + noise * np.eye(n)

        # Add small jitter for numerical stability
        jitter = 1e-8
        K_total += jitter * np.eye(n)

        try:
            L, lower = cho_factor(K_total, lower=True)
        except np.linalg.LinAlgError:
            return -np.inf

        # y^T K_total^{-1} y
        alpha = cho_solve((L, lower), y_train)
        data_fit = -0.5 * np.dot(y_train, alpha)

        # log|K_total|
        complexity = -np.sum(np.log(np.diag(L)))

        # constant
        constant = -0.5 * n * np.log(2 * np.pi)

        return data_fit + complexity + constant

    def gp_predict(X_train, y_train, X_test, K_train, K_test_train, K_test, noise):
        """
        GP prediction using Cholesky decomposition

        Parameters:
        -----------
        X_train : array (n_train,)
        y_train : array (n_train,)
        X_test : array (n_test,)
        K_train : array (n_train, n_train) - kernel on training data
        K_test_train : array (n_test, n_train) - cross kernel
        K_test : array (n_test, n_test) - kernel on test data
        noise : float - observation noise variance

        Returns:
        --------
        y_mean : array (n_test,)
        y_std : array (n_test,)
        """
        n_train = len(X_train)

        # Add noise to diagonal
        K_noisy = K_train + noise * np.eye(n_train)

        # Cholesky decomposition
        try:
            L, lower = cho_factor(K_noisy, lower=True)
        except np.linalg.LinAlgError:
            # Fallback: add more noise
            K_noisy += 1e-3 * np.eye(n_train)
            L, lower = cho_factor(K_noisy, lower=True)

        # Solve for alpha = K^{-1} y
        alpha = cho_solve((L, lower), y_train)

        # Predictive mean
        y_mean = K_test_train @ alpha

        # Predictive variance
        v = cho_solve((L, lower), K_test_train.T)
        y_var = np.diag(K_test) - np.sum(K_test_train * v.T, axis=1)
        y_var = np.maximum(y_var, 0.0)  # Ensure non-negative
        y_std = np.sqrt(y_var)

        return y_mean, y_std

    def optimize_gp_hyperparameters(X_train, y_train, kernel_type, initial_params, max_iter=50,
                                     use_polynomial_mean=False, poly_degree=None, joint_inference=False,
                                     mean_regularization_strength=0.1):
        """
        Optimize GP hyperparameters by maximizing marginal likelihood

        Parameters:
        -----------
        X_train : array (n_train,)
        y_train : array (n_train,)
        kernel_type : str
        initial_params : dict with initial hyperparameter values
        max_iter : int, maximum optimization iterations
        use_polynomial_mean : bool, whether to use polynomial mean function
        poly_degree : int, degree of polynomial mean (for joint inference)
        joint_inference : bool, if True optimize mean regularization jointly
        mean_regularization_strength : float, initial mean regularization strength

        Returns:
        --------
        optimized_params : dict with optimized hyperparameters
        log_marginal_likelihood : float
        """
        from scipy.optimize import minimize
        from sklearn.preprocessing import PolynomialFeatures

        X_train = np.atleast_1d(X_train).reshape(-1)
        y_train = np.atleast_1d(y_train).reshape(-1)

        # Precompute polynomial features if using joint inference
        Phi_train = None
        if use_polynomial_mean and joint_inference and poly_degree is not None:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
            Phi_train = poly.fit_transform(X_train.reshape(-1, 1))

        # Define bounds and parameterization (work in log space for positive params)
        if kernel_type == 'rbf':
            if use_polynomial_mean and joint_inference:
                # Optimize: log_lengthscale, log_noise, log_mean_regularization
                def pack_params(lengthscale, noise, mean_reg):
                    return np.array([np.log(lengthscale), np.log(noise), np.log(mean_reg)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'noise': np.exp(x[1]),
                           'mean_regularization': np.exp(x[2])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('noise', 0.1),
                               mean_regularization_strength)
                bounds = [(-2, 2), (-6, 0), (-6, 1)]  # mean_reg: [0.000001, 10]
            else:
                # Optimize: log_lengthscale, log_noise
                def pack_params(lengthscale, noise):
                    return np.array([np.log(lengthscale), np.log(noise)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'noise': np.exp(x[1])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('noise', 0.1))
                bounds = [(-2, 2), (-6, 0)]

        elif kernel_type == 'matern':
            # Same hyperparameters as RBF: lengthscale and noise
            if use_polynomial_mean and joint_inference:
                # Optimize: log_lengthscale, log_noise, log_mean_regularization
                def pack_params(lengthscale, noise, mean_reg):
                    return np.array([np.log(lengthscale), np.log(noise), np.log(mean_reg)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'noise': np.exp(x[1]),
                           'mean_regularization': np.exp(x[2])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('noise', 0.1),
                               mean_regularization_strength)
                bounds = [(-2, 2), (-6, 0), (-6, 1)]  # mean_reg: [0.000001, 10]
            else:
                # Optimize: log_lengthscale, log_noise
                def pack_params(lengthscale, noise):
                    return np.array([np.log(lengthscale), np.log(noise)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'noise': np.exp(x[1])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('noise', 0.1))
                bounds = [(-2, 2), (-6, 0)]

        elif kernel_type == 'bump':
            if use_polynomial_mean and joint_inference:
                # Optimize: log_lengthscale, log_support_radius, log_noise, log_mean_regularization
                def pack_params(lengthscale, support_radius, noise, mean_reg):
                    return np.array([np.log(lengthscale), np.log(support_radius),
                                   np.log(noise), np.log(mean_reg)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'support_radius': np.exp(x[1]),
                           'noise': np.exp(x[2]), 'mean_regularization': np.exp(x[3])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('support_radius', 2.0),
                               initial_params.get('noise', 0.1),
                               mean_regularization_strength)
                bounds = [(-2, 2), (-1, 2), (-6, 0), (-6, 1)]
            else:
                # Optimize: log_lengthscale, log_support_radius, log_noise
                def pack_params(lengthscale, support_radius, noise):
                    return np.array([np.log(lengthscale), np.log(support_radius), np.log(noise)])

                def unpack_params(x):
                    return {'lengthscale': np.exp(x[0]), 'support_radius': np.exp(x[1]),
                           'noise': np.exp(x[2])}

                x0 = pack_params(initial_params.get('lengthscale', 1.0),
                               initial_params.get('support_radius', 2.0),
                               initial_params.get('noise', 0.1))
                bounds = [(-2, 2), (-1, 2), (-6, 0)]

        elif kernel_type == 'polynomial':
            # Optimize: log_sigma, log_noise (keep degree fixed)
            degree = initial_params.get('degree', 5)

            if use_polynomial_mean and joint_inference:
                def pack_params(sigma, noise, mean_reg):
                    return np.array([np.log(sigma), np.log(noise), np.log(mean_reg)])

                def unpack_params(x):
                    return {'degree': degree, 'sigma': np.exp(x[0]), 'noise': np.exp(x[1]),
                           'mean_regularization': np.exp(x[2])}

                x0 = pack_params(initial_params.get('sigma', 0.1),
                               initial_params.get('noise', 0.1),
                               mean_regularization_strength)
                bounds = [(-3, 1), (-6, 0), (-6, 1)]
            else:
                def pack_params(sigma, noise):
                    return np.array([np.log(sigma), np.log(noise)])

                def unpack_params(x):
                    return {'degree': degree, 'sigma': np.exp(x[0]), 'noise': np.exp(x[1])}

                x0 = pack_params(initial_params.get('sigma', 0.1),
                               initial_params.get('noise', 0.1))
                bounds = [(-3, 1), (-6, 0)]
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        # Negative log marginal likelihood (to minimize)
        def neg_log_marginal_likelihood(x):
            params = unpack_params(x)
            noise = params.pop('noise')
            mean_reg = params.pop('mean_regularization', None)

            K_train = compute_kernel_matrix(X_train, X_train, kernel_type, params)

            # Use joint marginal likelihood if optimizing mean regularization
            if mean_reg is not None and use_polynomial_mean and joint_inference:
                log_ml = gp_joint_marginal_likelihood(X_train, y_train, K_train, noise,
                                                     Phi_train, mean_reg, poly_degree)
            else:
                log_ml = gp_marginal_likelihood(X_train, y_train, K_train, noise)

            # Return negative for minimization
            return -log_ml

        # Optimize
        result = minimize(neg_log_marginal_likelihood, x0, bounds=bounds,
                         method='L-BFGS-B', options={'maxiter': max_iter})

        optimized_params = unpack_params(result.x)
        log_ml = -result.fun

        return optimized_params, log_ml

    def fit_gp_numpy(X_train, y_train, X_test, kernel_type='rbf', use_polynomial_mean=False, poly_degree=None, joint_inference=False, mean_regularization_strength=0.1, **kernel_params):
        """
        Fit GP using pure numpy/scipy

        Parameters:
        -----------
        X_train : array (n_train,)
        y_train : array (n_train,)
        X_test : array (n_test,)
        kernel_type : str, one of 'rbf', 'bump', 'polynomial'
        use_polynomial_mean : bool, whether to use polynomial mean function
        poly_degree : int, degree of polynomial mean function
        joint_inference : bool, if True do joint Bayesian inference over mean params and GP
        **kernel_params : kernel hyperparameters

        Returns:
        --------
        y_mean : array (n_test,) - full GP predictions (with polynomial mean added back)
        y_std : array (n_test,) - total uncertainty (GP + mean if joint inference)
        poly_mean : array (n_test,) or None - polynomial mean function (for plotting)
        y_std_gp : array (n_test,) or None - GP uncertainty component only (for joint inference)
        y_std_mean : array (n_test,) or None - mean uncertainty component only (for joint inference)
        """
        X_train = np.atleast_1d(X_train).reshape(-1)
        y_train = np.atleast_1d(y_train).reshape(-1)
        X_test = np.atleast_1d(X_test).reshape(-1)

        # Polynomial mean function (if requested)
        if use_polynomial_mean and poly_degree is not None:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import BayesianRidge

            # Create polynomial features
            poly = PolynomialFeatures(degree=poly_degree, include_bias=True)
            Phi_train = poly.fit_transform(X_train.reshape(-1, 1))
            Phi_test = poly.transform(X_test.reshape(-1, 1))

            if joint_inference:
                # Joint Bayesian inference: marginalize over polynomial coefficients
                # Model: y = Φβ + f + ε
                # where β ~ N(0, λ⁻¹I), f ~ GP(0, K), ε ~ N(0, σ²I)
                #
                # Marginal: y ~ N(0, ΦΛ⁻¹Φᵀ + K + σ²I)
                # This integrates out uncertainty in β!

                # Prior precision for polynomial coefficients
                # Use degree-adaptive regularization to prevent numerical issues
                # Higher degree → stronger regularization (smaller prior variance)
                # This prevents the polynomial contribution from dominating
                # User-controlled base strength * degree-adaptive scaling
                lambda_prior = mean_regularization_strength * (poly_degree / 3.0)  # Increases with degree
                Lambda_inv = (1.0 / lambda_prior) * np.eye(Phi_train.shape[1])

                # Normalize polynomial features to improve conditioning
                # This is crucial for numerical stability with high-degree polynomials
                Phi_train_std = np.std(Phi_train, axis=0, keepdims=True)
                Phi_train_std[Phi_train_std < 1e-10] = 1.0  # Avoid division by zero
                Phi_train_normalized = Phi_train / Phi_train_std
                Phi_test_normalized = Phi_test / Phi_train_std

                # Update Lambda_inv to account for normalization
                # After normalization, we need to adjust the prior accordingly
                Lambda_inv_normalized = Lambda_inv / (Phi_train_std.T @ Phi_train_std + 1e-10)

                # Store normalized features for joint inference
                Phi_train = Phi_train_normalized
                Phi_test = Phi_test_normalized
                Lambda_inv = Lambda_inv_normalized

                # Posterior mean will be computed after we have K_total
                # We need to defer this until after kernel matrices are computed
                use_polynomial_mean = True
                y_train_residual = None  # Not used in joint path
                mean_test = None  # Will be computed in joint inference path after seeing data
            else:
                # Sequential inference: fit polynomial first (point estimate)
                poly_model = BayesianRidge(fit_intercept=False)
                poly_model.fit(Phi_train, y_train)

                # Get polynomial predictions
                mean_train = poly_model.predict(Phi_train)
                mean_test = poly_model.predict(Phi_test)

                # Subtract mean from training data (fit GP on residuals)
                y_train_residual = y_train - mean_train
        else:
            y_train_residual = y_train
            mean_test = np.zeros_like(X_test)
            use_polynomial_mean = False  # Track if we're using it for return value
            joint_inference = False  # Not using joint inference

        # Get noise level
        noise = kernel_params.get('noise', 0.1)

        # Compute kernel matrices
        if kernel_type == 'bump':
            lengthscale = kernel_params.get('lengthscale', 1.0)
            support_radius = kernel_params.get('support_radius', 2.0)
            noise = max(noise, 1e-4)  # Higher noise for stability

            K_train = bump_kernel(X_train, X_train, lengthscale, support_radius)
            K_test_train = bump_kernel(X_test, X_train, lengthscale, support_radius)
            K_test = bump_kernel(X_test, X_test, lengthscale, support_radius)

        elif kernel_type == 'polynomial':
            degree = min(kernel_params.get('degree', 10), 8)  # Cap at 8
            sigma = kernel_params.get('sigma', 0.1)
            noise = max(noise, 1e-3)  # Higher noise for stability

            K_train = polynomial_kernel(X_train, X_train, degree, sigma)
            K_test_train = polynomial_kernel(X_test, X_train, degree, sigma)
            K_test = polynomial_kernel(X_test, X_test, degree, sigma)

        elif kernel_type == 'rbf':
            lengthscale = kernel_params.get('lengthscale', 1.0)
            noise = max(noise, 1e-6)

            K_train = rbf_kernel(X_train, X_train, lengthscale)
            K_test_train = rbf_kernel(X_test, X_train, lengthscale)
            K_test = rbf_kernel(X_test, X_test, lengthscale)

        elif kernel_type == 'matern':
            lengthscale = kernel_params.get('lengthscale', 1.0)
            noise = max(noise, 1e-6)

            K_train = matern_kernel(X_train, X_train, lengthscale)
            K_test_train = matern_kernel(X_test, X_train, lengthscale)
            K_test = matern_kernel(X_test, X_test, lengthscale)

        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        try:
            if joint_inference and use_polynomial_mean:
                # Joint Bayesian inference path
                # Build total covariance: K_total = ΦΛ⁻¹Φᵀ + K_GP + σ²I
                K_beta = Phi_train @ Lambda_inv @ Phi_train.T  # Uncertainty from β
                K_total = K_beta + K_train + noise * np.eye(len(X_train))

                try:
                    # Adaptive jitter for numerical stability
                    # Start with small jitter and increase if needed
                    jitter = 1e-8
                    max_attempts = 5

                    for attempt in range(max_attempts):
                        try:
                            # Solve for α = K_total⁻¹ y
                            L = np.linalg.cholesky(K_total + jitter * np.eye(len(X_train)))
                            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                            break  # Success!
                        except np.linalg.LinAlgError:
                            if attempt < max_attempts - 1:
                                # Increase jitter and try again
                                jitter *= 10
                                continue
                            else:
                                # Final attempt failed
                                raise

                    # Prediction mean: [Φ(x*) k(x*,X)] α
                    # This automatically includes contribution from both β and GP
                    K_test_beta = Phi_test @ Lambda_inv @ Phi_train.T
                    K_combined = K_test_beta + K_test_train
                    y_mean = K_combined @ alpha

                    # Prediction variance: includes both β and GP uncertainty
                    # Σ(x*) = Φ(x*)Λ⁻¹Φ(x*)ᵀ + k(x*,x*) - K_combined K_total⁻¹ K_combinedᵀ
                    v = np.linalg.solve(L, K_combined.T)
                    K_test_beta_diag = np.sum((Phi_test @ Lambda_inv) * Phi_test, axis=1)

                    # Separate the uncertainty components:
                    # 1. Prior uncertainty from polynomial mean (before seeing data)
                    y_var_mean_prior = K_test_beta_diag

                    # 2. Prior uncertainty from GP (before seeing data)
                    y_var_gp_prior = np.diag(K_test)

                    # 3. Reduction in uncertainty from observing data
                    y_var_reduction = np.sum(v**2, axis=0)

                    # Total posterior variance
                    y_var_total = y_var_mean_prior + y_var_gp_prior - y_var_reduction
                    y_var_total = np.maximum(y_var_total, 1e-10)  # Numerical stability
                    y_std_total = np.sqrt(y_var_total)

                    # For visualization: approximate the split between mean and GP contributions
                    # This is approximate because they're coupled through the data
                    fraction_mean = y_var_mean_prior / (y_var_mean_prior + y_var_gp_prior + 1e-10)
                    fraction_gp = y_var_gp_prior / (y_var_mean_prior + y_var_gp_prior + 1e-10)

                    y_std_mean_component = y_std_total * np.sqrt(fraction_mean)
                    y_std_gp_component = y_std_total * np.sqrt(fraction_gp)

                    # Compute posterior mean of polynomial coefficients for visualization
                    # E[β|y] = Λ⁻¹Φᵀ K_total⁻¹ y
                    # This gives us the posterior mean estimate of the polynomial component
                    Phi_beta_coef = Lambda_inv @ Phi_train.T @ alpha
                    mean_test_posterior = Phi_test @ Phi_beta_coef

                    # For plotting: return the polynomial posterior mean for visualization
                    # Note: In joint inference, this represents E[Φβ|y], the posterior mean
                    # of the polynomial component after seeing the data
                    return y_mean, y_std_total, mean_test_posterior, y_std_gp_component, y_std_mean_component

                except np.linalg.LinAlgError:
                    print("Warning: Cholesky failed for joint inference")
                    return np.zeros_like(X_test), np.ones_like(X_test), None, None, None
            else:
                # Sequential inference path: fit GP on residuals
                y_mean_residual, y_std = gp_predict(X_train, y_train_residual, X_test,
                                                    K_train, K_test_train, K_test, noise)

                # Add polynomial mean back to predictions
                y_mean = y_mean_residual + mean_test

                # Check for NaNs
                if np.any(np.isnan(y_mean)) or np.any(np.isnan(y_std)):
                    print(f"Warning: NaNs in GP prediction with {kernel_type} kernel")
                    return np.zeros_like(X_test), np.ones_like(X_test), None, None, None

                # Return polynomial mean for plotting (None if not using it)
                # Sequential inference doesn't separate uncertainty components
                return y_mean, y_std, (mean_test if use_polynomial_mean and not joint_inference else None), None, None

        except Exception as e:
            print(f"Error in GP fitting with {kernel_type} kernel: {e}")
            return np.zeros_like(X_test), np.ones_like(X_test), None, None, None
    return (
        bump_kernel,
        compute_kernel_matrix,
        fit_gp_numpy,
        gp_marginal_likelihood,
        optimize_gp_hyperparameters,
        polynomial_kernel,
        rbf_kernel,
    )


@app.cell
def _():
    import qrcode
    import io
    import base64

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://kermodegroup.github.io/demos')
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    qr_base64 = base64.b64encode(buffer.read()).decode()
    return (qr_base64,)


@app.cell
def _(mo, qr_base64):
    mo.Html(f'''
    <div class="app-header">
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 0; padding: 0;">
            <div>
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>Bayesian Regression and UQ Demo</b>
                <br><span style="font-size: 16px;"><i>Live demo:</i>
                <a href="https://kermodegroup.github.io/demos" target="_blank" style="color: #0066cc; text-decoration: none;">kermodegroup.github.io/demos</a>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <i>Code:</i>
                <a href="https://github.com/kermodegroup/demos" target="_blank" style="color: #0066cc; text-decoration: none;">github.com/kermodegroup/demos</a>
                </span></p>
            </div>
            <img src="data:image/png;base64,{qr_base64}" alt="QR Code" style="width: 100px; height: 100px; flex-shrink: 0;" />
        </div>
    </div>
    ''')
    return


@app.cell(hide_code=True)
def _(
    ConformalPrediction,
    MyBayesianRidge,
    N_samples,
    NeuralNetworkRegression,
    POPSRegression,
    PolynomialFeatures,
    aleatoric,
    aleatoric_gp,
    bayesian,
    bump_kernel,
    compute_kernel_matrix,
    conformal,
    fit_gp_numpy,
    function_dropdown,
    g,
    get_N_samples,
    get_P,
    get_calib_frac,
    get_filter_max,
    get_filter_min,
    get_gp_joint_inference,
    get_gp_kernel_type,
    get_gp_lengthscale,
    get_gp_mean_regularization,
    get_gp_poly_mean_degree,
    get_gp_support_radius,
    get_last_enabled_method,
    get_nn_activation,
    get_nn_ensemble_method,
    get_nn_ensemble_size,
    get_nn_hidden_units,
    get_nn_num_layers,
    get_nn_regularization,
    get_percentile_clipping,
    get_quantile_confidence,
    get_quantile_regularization,
    get_seed,
    get_sigma,
    get_zeta,
    gp_marginal_likelihood,
    gp_regression,
    gp_use_poly_mean,
    mo,
    neural_network,
    np,
    plt,
    polynomial_kernel,
    pops,
    quantile,
    QuantileRegressionUQ,
    rbf_kernel,
    seed,
    sigma,
    train_test_split,
):
    def get_data(N_samples=500, sigma=0.1, function_type='sin'):
        x_train = np.append(np.random.uniform(-10, 10, size=N_samples), np.linspace(-10, 10, 2))
        x_train = x_train[(x_train < get_filter_min()) | (x_train > get_filter_max())]
        x_train = np.sort(x_train)
        y_train = g(x_train, noise_variance=sigma**2, function_type=function_type)
        X_train = x_train[:, None]

        x_test = np.linspace(-10, 10, 1000)
        y_test = g(x_test, 0, function_type=function_type)
        X_test = x_test[:, None]

        return X_train, y_train, X_test, y_test

    fig, ax = plt.subplots(figsize=(14, 5))
    np.random.seed(seed.value)
    _func_type = function_dropdown.value
    X_data, y_data, X_test, y_test = get_data(N_samples.value, sigma=sigma.value, function_type=_func_type)

    # Check if filter removed all data
    mo.stop(len(X_data) < 5, mo.md(f"""
    ⚠️ **Error: Filter range removed all data!**

    The current filter range ({get_filter_min():.1f} to {get_filter_max():.1f}) excludes all data points.

    Please adjust the filter range to include some data points.
    """))

    X_train, X_calib, y_train, y_calib = train_test_split(X_data, y_data, test_size=get_calib_frac(), random_state=get_seed())
    n = len(y_calib)

    poly = PolynomialFeatures(degree=get_P()-1, include_bias=True)
    Phi_train = poly.fit_transform(X_train)
    Phi_test = poly.transform(X_test)
    Phi_calib = poly.transform(X_calib)

    b = MyBayesianRidge(fit_intercept=False)
    # Note: POPS mode selection UI is available but not yet implemented in the library
    # The library currently only supports hypercube sampling mode
    p = POPSRegression(fit_intercept=False, percentile_clipping=get_percentile_clipping(), leverage_percentile=0)
    c = ConformalPrediction(fit_intercept=False)
    q = QuantileRegressionUQ(confidence=get_quantile_confidence(), fit_intercept=False, alpha=get_quantile_regularization())

    ax.plot(X_test[:, 0], y_test, 'k-', label='Truth')
    ax.plot(X_train[:, 0], y_train, 'b.', label='Train')
    ax.plot(X_calib[:, 0], y_calib, 'c.', label='Calibration')
    ax.axvline(get_filter_min(), ls='--', color='k')
    ax.axvline(get_filter_max(), ls='--', color='k')

    gp_log_ml = 0.0  # Initialize
    bayes_log_ml = 0.0  # Initialize
    gp_sparsity = 0.0  # Initialize GP covariance sparsity

    # Dictionary to store predictions and metrics for each method
    import time
    method_predictions = {}  # {label: (y_pred, y_std, fit_time)}

    models_to_plot = []
    if bayesian.value:
        models_to_plot.append((b, Phi_train, Phi_test, 'C2', 'Bayesian uncertainty', True))
    if conformal.value:
        models_to_plot.append((c, Phi_train, Phi_test, 'C1', 'Conformal prediction', True))
    if pops.value:
        models_to_plot.append((p, Phi_train, Phi_test, 'C0', 'POPS regression', True))
    if quantile.value:
        models_to_plot.append((q, Phi_train, Phi_test, 'C4', 'Quantile regression', True))
    if gp_regression.value:
        models_to_plot.append((None, X_train, X_test, 'C3', 'GP regression', False))
    if neural_network.value:
        models_to_plot.append((None, X_train, X_test, 'C5', 'Neural network', False))

    for model_info in models_to_plot:
        if len(model_info) == 6:
            model, X_train_model, X_test_model, color, label, use_poly = model_info
        else:
            continue

        # Start timing for this method
        start_time = time.time()

        if label == 'GP regression':
            # Fit GP with numpy
            y_pred, y_std, poly_mean, y_std_gp, y_std_mean = fit_gp_numpy(
                X_train_model[:, 0], y_train, X_test_model[:, 0],
                kernel_type=get_gp_kernel_type(),
                lengthscale=get_gp_lengthscale(),
                support_radius=get_gp_support_radius(),
                degree=get_P(),
                sigma=1.0,
                noise=sigma.value**2,
                use_polynomial_mean=gp_use_poly_mean.value,
                poly_degree=get_gp_poly_mean_degree(),
                joint_inference=get_gp_joint_inference(),
                mean_regularization_strength=get_gp_mean_regularization()
            )

            # Compute log marginal likelihood for display
            params = {
                'lengthscale': get_gp_lengthscale(),
                'support_radius': get_gp_support_radius(),
                'degree': get_P(),
                'sigma': 1.0
            }
            K_train = compute_kernel_matrix(X_train_model[:, 0], X_train_model[:, 0],
                                           get_gp_kernel_type(), params)
            gp_log_ml = gp_marginal_likelihood(X_train_model[:, 0], y_train, K_train, sigma.value**2)

            # Calculate sparsity of covariance matrix (% of elements < 1e-6)
            gp_sparsity = 100.0 * np.sum(np.abs(K_train) < 1e-6) / K_train.size

            # Plot polynomial mean function if enabled (to show partitioning)
            if poly_mean is not None:
                ax.plot(X_test[:, 0], poly_mean, 'C3--', lw=2, label='GP prior mean', alpha=0.7)

            # Plot separate uncertainty components if joint inference is enabled
            if get_gp_joint_inference() and y_std_mean is not None:
                # Plot total uncertainty (darker shade)
                ax.fill_between(X_test[:, 0], y_pred - y_std, y_pred + y_std,
                               alpha=0.35, color='purple', label='Total uncertainty')
                # Plot polynomial mean uncertainty component (lighter shade)
                ax.fill_between(X_test[:, 0], y_pred - y_std_mean, y_pred + y_std_mean,
                               alpha=0.25, color='orange', label='Mean uncertainty')

            if aleatoric_gp.value:
                # Add aleatoric noise to GP predictions
                y_std = np.sqrt(y_std**2 + sigma.value**2)
        elif label == 'Neural network':
            # Fit neural network ensemble
            # Build hidden layer architecture
            num_layers = get_nn_num_layers()
            hidden_units = get_nn_hidden_units()
            hidden_layer_sizes = tuple([hidden_units] * num_layers)

            # Create and fit neural network
            nn = NeuralNetworkRegression(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=10**get_nn_regularization(),  # Convert from log10 scale
                n_ensemble=get_nn_ensemble_size(),
                activation=get_nn_activation(),
                ensemble_method=get_nn_ensemble_method()
                # Uses default max_iter=1000, tol=1e-3 for reliable convergence
            )
            nn.fit(X_train_model, y_train)

            # Get predictions with uncertainty
            y_pred, y_std = nn.predict(X_test_model, return_std=True)
        else:
            # Existing polynomial-basis models
            model.fit(X_train_model, y_train)

            # Store log ML for Bayesian models
            if label in ['Bayesian uncertainty', 'Conformal prediction']:
                bayes_log_ml = model.log_marginal_likelihood()

            kwargs = {
                'return_std': True,
                'aleatoric': aleatoric.value,
            }        
            if label == 'Conformal prediction':
                qhat = model.calibrate(Phi_calib, y_calib, zeta=get_zeta(), aleatoric=aleatoric.value)
                kwargs['rescale'] = True

            if label == 'POPS regression':
                y_pred, y_std, y_min, y_max = model.predict(X_test_model, return_std=True, return_bounds=True)
                if aleatoric.value:
                    y_std = np.sqrt(y_std**2 + 1.0 / model.alpha_)
            else:
                y_pred, y_std = model.predict(X_test_model, **kwargs)

        # Store predictions and timing for metrics computation
        fit_time = time.time() - start_time
        method_predictions[label] = (y_pred, y_std, fit_time)

        ax.plot(X_test[:, 0], y_pred, color=color, lw=3)
        ax.fill_between(X_test[:, 0], y_pred - y_std, y_pred + y_std, alpha=0.5, color=color, label=label)

        if label == 'POPS regression':
            ax.plot(X_test[:, 0], y_min, 'k--', lw=1, label='POPS min/max')
            ax.plot(X_test[:, 0], y_max, 'k--', lw=1)

    # No title needed - all info is in the dashboard and outputs bar
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(loc='lower left')

    # Apply tight_layout before adding inset (to avoid warning)
    plt.tight_layout()

    # Add inset plot based on most recently enabled method
    linear_methods_active = bayesian.value or conformal.value or pops.value or quantile.value
    last_method = get_last_enabled_method()

    # Determine which inset to show based on last enabled method
    show_gp_inset = last_method == 'gp' and gp_regression.value
    show_nn_inset = last_method == 'nn' and neural_network.value
    show_linear_inset = last_method == 'linear' and linear_methods_active

    if show_gp_inset or show_nn_inset or show_linear_inset:
        # Create inset axes in the lower right
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        # Position: [x, y, width, height] in axes coordinates
        axins = inset_axes(ax, width="12%", height="25%", loc='lower right',
                          bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax.transAxes,
                          borderpad=1.5)

        if show_gp_inset:
            # Plot kernel function
            x_kernel = np.linspace(0, 2.0, 100)
            kernel_type = get_gp_kernel_type()

            if kernel_type == 'rbf':
                K_values = rbf_kernel(np.array([0.0]), x_kernel, get_gp_lengthscale())[0, :]
            elif kernel_type == 'bump':
                K_values = bump_kernel(np.array([0.0]), x_kernel, get_gp_lengthscale(), get_gp_support_radius())[0, :]
                # Add vertical line for support radius
                axins.axvline(get_gp_support_radius(), color='k', ls='dashed', lw=0.8, alpha=0.5)
            elif kernel_type == 'matern':
                K_values = matern_kernel(np.array([0.0]), x_kernel, get_gp_lengthscale())[0, :]
            elif kernel_type == 'polynomial':
                K_values = polynomial_kernel(np.array([0.0]), x_kernel, get_P(), 1.0)[0, :]

            axins.plot(x_kernel, K_values, 'C3', lw=1.5)
            axins.set_xlabel('$r$', fontsize=8)
            axins.set_ylabel('$K(0, r)$', fontsize=8)
            axins.set_title(f'Kernel: {kernel_type}', fontsize=9, pad=3)
            axins.tick_params(labelsize=7)
            axins.grid(True, alpha=0.3, linewidth=0.5)

        elif show_nn_inset:
            # Plot activation function
            x_act = np.linspace(-3, 3, 100)
            activation_type = get_nn_activation()

            if activation_type == 'tanh':
                y_act = np.tanh(x_act)
            elif activation_type == 'relu':
                y_act = np.maximum(0, x_act)

            axins.plot(x_act, y_act, 'C5', lw=1.5)
            axins.set_xlabel('$x$', fontsize=8)
            axins.set_ylabel(f'${activation_type}(x)$', fontsize=8)
            axins.set_title(f'Activation: {activation_type}', fontsize=9, pad=3)
            axins.tick_params(labelsize=7)
            axins.grid(True, alpha=0.3, linewidth=0.5)
            axins.axhline(0, color='k', lw=0.5, alpha=0.3)
            axins.axvline(0, color='k', lw=0.5, alpha=0.3)

        elif show_linear_inset:
            # Plot all polynomial basis functions
            x_basis = np.linspace(-1, 1, 100)
            P_degree = get_P()

            # Use colormap for varying colors
            import matplotlib.cm as cm
            cmap = cm.get_cmap('viridis')
            colors = [cmap(i / P_degree) for i in range(P_degree + 1)]

            for i in range(P_degree + 1):
                y_basis = x_basis ** i
                # Use thinner lines for higher degrees to reduce clutter
                lw = 1.2 if i < 4 else 0.8
                alpha = 0.8 if i < 4 else 0.5
                axins.plot(x_basis, y_basis, color=colors[i], lw=lw, alpha=alpha)

            axins.set_xlabel('$x$', fontsize=8)
            axins.set_ylabel('Basis', fontsize=8)
            axins.set_title(f'Polynomial basis (P={P_degree})', fontsize=9, pad=3)
            axins.tick_params(labelsize=7)
            axins.grid(True, alpha=0.3, linewidth=0.5)
            # Add text labels for first and last basis
            axins.text(0.95, 0.05, '$x^0$', transform=axins.transAxes, fontsize=6, ha='right', va='bottom')
            axins.text(0.95, 0.95, f'$x^{{{P_degree}}}$', transform=axins.transAxes, fontsize=6, ha='right', va='top')

    # Compute metrics for all active methods
    metrics_dict = {}
    for method_label, (method_y_pred, method_y_std, method_fit_time) in method_predictions.items():
        # Coverage: % of test points within predicted intervals
        method_in_interval = np.abs(y_test - method_y_pred) <= method_y_std
        method_coverage = 100.0 * np.mean(method_in_interval)

        # Mean interval width (sharpness)
        method_mean_width = np.mean(2 * method_y_std)

        # MSE (accuracy)
        method_mse = np.mean((y_test - method_y_pred) ** 2)

        metrics_dict[method_label] = {
            'coverage': method_coverage,
            'mean_width': method_mean_width,
            'mse': method_mse,
            'fit_time': method_fit_time
        }

    mo.Html(f'''
    <div class="app-plot">
        {mo.center(fig)}
    </div>
    ''')
    return (bayes_log_ml, n, qhat, gp_log_ml, gp_sparsity, metrics_dict)


@app.cell(hide_code=True)
def _(
    bayesian,
    conformal,
    get_N_samples,
    get_calib_frac,
    get_filter_max,
    get_filter_min,
    get_function_type,
    get_gp_joint_inference,
    get_gp_kernel_type,
    get_gp_mean_regularization,
    get_gp_poly_mean_degree,
    get_nn_activation,
    get_nn_ensemble_method,
    get_nn_ensemble_size,
    get_nn_hidden_units,
    get_nn_num_layers,
    get_nn_regularization,
    get_percentile_clipping,
    get_quantile_confidence,
    get_quantile_regularization,
    get_seed,
    get_sigma,
    get_zeta,
    gp_optimize_button,
    gp_regression,
    gp_use_poly_mean,
    mo,
    neural_network,
    np,
    pops,
    quantile,
    set_N_samples,
    set_P,
    set_calib_frac,
    set_filter_max,
    set_filter_min,
    set_function_type,
    set_gp_joint_inference,
    set_gp_kernel_type,
    set_gp_lengthscale,
    set_gp_mean_regularization,
    set_gp_poly_mean_degree,
    set_gp_support_radius,
    set_nn_activation,
    set_nn_ensemble_method,
    set_nn_ensemble_size,
    set_nn_hidden_units,
    set_nn_num_layers,
    set_nn_regularization,
    set_percentile_clipping,
    set_quantile_confidence,
    set_quantile_regularization,
    set_seed,
    set_sigma,
    set_zeta,
):
    # NOTE: This cell does NOT depend on get_P, get_gp_lengthscale, or get_gp_support_radius
    # to avoid circular dependency with optimization cells that modify those values
    aleatoric = mo.ui.checkbox(False, label="Include aleatoric uncertainty")

    data_label = mo.md("**Dataset parameters**")

    # Valid function types
    function_dropdown = mo.ui.dropdown(
        options=[
            'sin',
            'witch',
            'sinc',
            'step',
            'runge',
        ],
        label='Ground truth function',
        value=get_function_type(),
        on_change=set_function_type,
    )
    N_samples = mo.ui.slider(50, 1000, 50, get_N_samples(), label='Samples $N$', on_change=set_N_samples)
    sigma = mo.ui.slider(0.001, 0.3, 0.005, get_sigma(), label=r'$\sigma$ noise', on_change=set_sigma)
    seed = mo.ui.slider(0, 10, 1, get_seed(), label="Seed", on_change=set_seed)
    filter_range = mo.ui.range_slider(
        start=-10, stop=10, step=0.5,
        value=[get_filter_min(), get_filter_max()],
        label='Filter range',
        on_change=lambda v: (set_filter_min(v[0]), set_filter_max(v[1]))
    )

    # Regression parameters with conditional styling
    reg_enabled = bayesian.value or conformal.value or pops.value or quantile.value

    if reg_enabled:
        # Use fixed default value (not state) to avoid circular dependency
        # Manual slider changes still update state via on_change, but slider doesn't react to state changes
        P_slider = mo.ui.slider(5, 15, 1, 10, label="Degree $P$", on_change=set_P)
        P_elem = P_slider  # For display
        aleatoric = mo.ui.checkbox(False, label="Include aleatoric uncertainty")
        reg_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd;'>")
    else:
        P_slider = None  # No slider when disabled
        P_elem = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(5, 15, 1, 10, label='Degree $P$', disabled=True, on_change=set_P)}</div>")
        aleatoric = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(False, label='Include aleatoric uncertainty', disabled=True)}</div>")
        reg_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd; opacity: 0.4;'>")

    # Conformal prediction section with conditional styling
    if conformal.value:
        calib_frac = mo.ui.slider(0.05, 0.5, 0.05, get_calib_frac(), label="Calibration fraction", on_change=set_calib_frac)
        zeta = mo.ui.slider(0.05, 0.3, 0.05, get_zeta(), label=r"Coverage $\zeta$", on_change=set_zeta)
        cp_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd;'>")
    else:
        calib_frac = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.05, 0.5, 0.05, get_calib_frac(), label='Calibration fraction', disabled=True, on_change=set_calib_frac)}</div>")
        zeta = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.05, 0.3, 0.05, get_zeta(), label=r'Coverage $\zeta$', disabled=True, on_change=set_zeta)}</div>")
        cp_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd; opacity: 0.4;'>")

    # POPS regression section with conditional styling
    if pops.value:
        percentile_clipping = mo.ui.slider(0, 10, 1, get_percentile_clipping(), label="Percentile clipping", on_change=set_percentile_clipping)
    else:
        percentile_clipping = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0, 10, 1, get_percentile_clipping(), label='Percentile clipping', disabled=True, on_change=set_percentile_clipping)}</div>")

    # Quantile regression section with conditional styling
    if quantile.value:
        quantile_confidence = mo.ui.slider(0.80, 0.95, 0.05, get_quantile_confidence(), label="Confidence level", on_change=set_quantile_confidence)
        quantile_regularization = mo.ui.slider(0.0, 0.1, 0.001, get_quantile_regularization(), label="Regularization", on_change=set_quantile_regularization)
    else:
        quantile_confidence = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.80, 0.95, 0.05, get_quantile_confidence(), label='Confidence level', disabled=True, on_change=set_quantile_confidence)}</div>")
        quantile_regularization = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.0, 0.1, 0.001, get_quantile_regularization(), label='Regularization', disabled=True, on_change=set_quantile_regularization)}</div>")

    # GP regression section with conditional styling
    if gp_regression.value:
        aleatoric_gp = mo.ui.checkbox(False, label="Include aleatoric uncertainty")
        gp_kernel_dropdown = mo.ui.dropdown(
            options=['rbf', 'matern', 'bump'],
            value=get_gp_kernel_type(),
            label='Kernel type',
            on_change=set_gp_kernel_type
        )
        # Use fixed default values (not state) to avoid circular dependency
        # Manual slider changes still update state via on_change, but sliders don't react to state changes
        gp_lengthscale_slider = mo.ui.slider(0.1, 5.0, 0.1, 0.5, label='Lengthscale', on_change=set_gp_lengthscale)
        gp_support_radius_slider = mo.ui.slider(0.5, 5.0, 0.1, 1.5, label='Support radius (bump only)', on_change=set_gp_support_radius)

        # gp_use_poly_mean checkbox is now created in a separate cell and passed in as dependency
        # We can check its .value here to conditionally show controls
        if gp_use_poly_mean.value:
            gp_poly_mean_degree = mo.ui.slider(1, 15, 1, get_gp_poly_mean_degree(), label='Mean polynomial degree', on_change=set_gp_poly_mean_degree)
            gp_joint_inference = mo.ui.checkbox(get_gp_joint_inference(), label="Joint Bayesian inference", on_change=set_gp_joint_inference)
            # Log-scale slider for mean regularization (better range control)
            # Maps -6 to 1 in log10 space → 10^-6 to 10^1 = 0.000001 to 10
            _log_reg = np.log10(get_gp_mean_regularization())
            gp_mean_regularization_log = mo.ui.slider(-6, 1, 0.1, _log_reg, label='Mean regularization (log₁₀)', on_change=lambda v: set_gp_mean_regularization(10**v))
            gp_mean_regularization = gp_mean_regularization_log  # For layout
        else:
            gp_poly_mean_degree = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(1, 15, 1, get_gp_poly_mean_degree(), label='Mean polynomial degree', disabled=True)}</div>")
            gp_joint_inference = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(get_gp_joint_inference(), label='Joint Bayesian inference', disabled=True)}</div>")
            _log_reg = np.log10(get_gp_mean_regularization())
            gp_mean_regularization = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(-6, 1, 0.1, _log_reg, label='Mean regularization (log₁₀)', disabled=True)}</div>")

        gp_lengthscale = gp_lengthscale_slider  # For display
        gp_support_radius = gp_support_radius_slider  # For display
        gp_opt_button_elem = gp_optimize_button
        # Add horizontal separator between kernel hyperparameters and mean function controls
        gp_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd;'>")
        # Show gp_use_poly_mean checkbox normally when GP regression is enabled
        gp_use_poly_mean_elem = gp_use_poly_mean
    else:
        aleatoric_gp = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(False, label='Include aleatoric uncertainty', disabled=True)}</div>")
        # Dropdown doesn't have disabled attribute, just show greyed out
        gp_kernel_dropdown = mo.Html(f"<div style='opacity: 0.4; pointer-events: none;'>{mo.ui.dropdown(['rbf', 'matern', 'bump'], value='rbf', label='Kernel type')}</div>")
        gp_lengthscale_slider = None  # No slider when disabled
        gp_support_radius_slider = None  # No slider when disabled
        gp_lengthscale = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.1, 5.0, 0.1, 0.5, label='Lengthscale', disabled=True)}</div>")
        gp_support_radius = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.5, 5.0, 0.1, 1.5, label='Support radius (bump only)', disabled=True)}</div>")
        gp_separator = mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd; opacity: 0.4;'>")
        # Wrap gp_use_poly_mean checkbox with disabled styling when GP regression is off
        gp_use_poly_mean_elem = mo.Html(f"<div style='opacity: 0.4; pointer-events: none;'>{gp_use_poly_mean}</div>")
        gp_poly_mean_degree = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(1, 10, 1, 3, label='Mean polynomial degree', disabled=True)}</div>")
        gp_joint_inference = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.checkbox(False, label='Joint Bayesian inference', disabled=True)}</div>")
        gp_mean_regularization = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(-6, 1, 0.1, -1, label='Mean regularization (log₁₀)', disabled=True)}</div>")
        gp_opt_button_elem = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.button(label='Optimize hyperparameters', disabled=True)}</div>")

    # Neural network section with conditional styling
    if neural_network.value:
        nn_activation = mo.ui.dropdown(
            options=['tanh', 'relu'],
            value=get_nn_activation(),
            label='Activation function',
            on_change=set_nn_activation
        )
        nn_ensemble_method = mo.ui.dropdown(
            options=['seed', 'bootstrap'],
            value=get_nn_ensemble_method(),
            label='Ensemble method',
            on_change=set_nn_ensemble_method
        )
        nn_hidden_units = mo.ui.slider(5, 50, 5, get_nn_hidden_units(), label='Hidden units', on_change=set_nn_hidden_units)
        nn_num_layers = mo.ui.slider(1, 3, 1, get_nn_num_layers(), label='Hidden layers', on_change=set_nn_num_layers)
        _log_reg_nn = get_nn_regularization()
        nn_regularization = mo.ui.slider(-6, 0, 0.5, _log_reg_nn, label='Regularization (log₁₀)', on_change=set_nn_regularization)
        nn_ensemble_size = mo.ui.slider(3, 10, 1, get_nn_ensemble_size(), label='Ensemble size', on_change=set_nn_ensemble_size)
    else:
        nn_activation = mo.Html(f"<div style='opacity: 0.4; pointer-events: none;'>{mo.ui.dropdown(['tanh'], value='tanh', label='Activation function')}</div>")
        nn_ensemble_method = mo.Html(f"<div style='opacity: 0.4; pointer-events: none;'>{mo.ui.dropdown(['seed', 'bootstrap'], value='seed', label='Ensemble method')}</div>")
        nn_hidden_units = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(5, 50, 5, get_nn_hidden_units(), label='Hidden units', disabled=True)}</div>")
        nn_num_layers = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(1, 3, 1, get_nn_num_layers(), label='Hidden layers', disabled=True)}</div>")
        nn_regularization = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(-6, 0, 0.5, -3, label='Regularization (log₁₀)', disabled=True)}</div>")
        nn_ensemble_size = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(3, 10, 1, 5, label='Ensemble size', disabled=True)}</div>")

    # Linear Methods tab: Two-column layout
    # First column: Shared settings, Bayesian fit, and Quantile regression
    if reg_enabled:
        shared_label = mo.md("*Shared settings:*")
    else:
        shared_label = mo.md("*Shared settings (enable a method below):*")

    linear_col1 = mo.vstack([
        shared_label,
        P_elem, aleatoric,
        mo.Html("<hr style='margin: 10px 0; border: 0; border-top: 1px solid #ddd;'>"),
        mo.left(bayesian),
        mo.Html("<hr style='margin: 5px 0; border: 0; border-top: 1px solid #ddd;'>"),
        mo.left(quantile),
        quantile_confidence,
        quantile_regularization
    ])

    # Second column: Conformal Prediction and POPS regression
    linear_col2 = mo.vstack([
        mo.left(conformal),
        calib_frac, zeta, cp_separator,
        mo.left(pops),
        percentile_clipping
    ])

    linear_methods_tab = mo.Html(f'''
    <div style="display: flex; gap: 20px;">
        <div style="width: 50%; min-width: 200px;">
            {linear_col1}
        </div>
        <div style="width: 50%; min-width: 200px;">
            {linear_col2}
        </div>
    </div>
    ''')

    # Kernel Methods tab: Two-column layout
    kernel_col1 = mo.vstack([
        mo.left(gp_regression),
        aleatoric_gp,
        gp_kernel_dropdown, gp_lengthscale, gp_support_radius,
        gp_opt_button_elem
    ])

    kernel_col2 = mo.vstack([
        gp_use_poly_mean_elem,
        gp_poly_mean_degree, gp_joint_inference, gp_mean_regularization
    ])

    kernel_methods_tab = mo.Html(f'''
    <div style="display: flex; gap: 20px;">
        <div style="width: 50%; min-width: 200px;">
            {kernel_col1}
        </div>
        <div style="width: 50%; min-width: 200px;">
            {kernel_col2}
        </div>
    </div>
    ''')

    # Non-linear Methods tab: Single column with checkbox at top
    nonlinear_methods_tab = mo.vstack([
        mo.left(neural_network),
        nn_activation,
        nn_ensemble_method,
        nn_hidden_units, nn_num_layers,
        nn_regularization, nn_ensemble_size
    ])

    # Create tabs for method-specific parameters
    method_params_tabs = mo.ui.tabs({
        "Linear Methods": linear_methods_tab,
        "Kernel Methods": kernel_methods_tab,
        "Non-linear Methods": nonlinear_methods_tab
    })

    # Two-column layout: Dataset parameters on left, Tabs on right
    dataset_column = mo.Html(f'''
    <div style="width: 30%; min-width: 200px;">
        {mo.vstack([data_label, function_dropdown, N_samples, filter_range, sigma, seed])}
    </div>
    ''')

    tabs_column = mo.Html(f'''
    <div style="width: 70%; min-width: 400px;">
        {method_params_tabs}
    </div>
    ''')

    controls = mo.hstack([
        dataset_column,
        tabs_column
    ], gap=0.2)

    mo.Html(f'''
    <div class="app-dashboard">
        {controls}
    </div>
    ''')
    return (
        N_samples,
        aleatoric,
        aleatoric_gp,
        function_dropdown,
        gp_lengthscale_slider,
        gp_support_radius_slider,
        seed,
        sigma,
    )


@app.cell(hide_code=True)
def _(bayes_log_ml, bayesian, conformal, gp_log_ml, gp_optimize_button, gp_regression, gp_sparsity, metrics_dict, mo, n, opt_message, qhat):
    # Display computed outputs below the dashboard (only values not in sliders)
    output_items = []

    if bayesian.value and bayes_log_ml != 0.0:
        output_items.append(f"Bayesian log ML: {bayes_log_ml:.1f}")

    if conformal.value:
        output_items.append(f"Calibration size (n): {n}")
        output_items.append(f"Quantile (q̂): {qhat:.2f}")

    if gp_regression.value:
        # Add optimized hyperparameters if available
        if gp_optimize_button.value and opt_message is not None and isinstance(opt_message, dict):
            if 'error' in opt_message:
                output_items.append(f"Opt error: {opt_message['error']}")
            else:
                # Show optimized log ML if available, otherwise current log ML
                log_ml_value = opt_message.get('log_ml', gp_log_ml)
                output_items.append(f"GP log ML: {log_ml_value:.1f}")

                if opt_message.get('lengthscale') is not None:
                    output_items.append(f"Opt lengthscale: {opt_message['lengthscale']:.3f}")
                if opt_message.get('support_radius') is not None:
                    output_items.append(f"Opt support: {opt_message['support_radius']:.3f}")
                if opt_message.get('noise') is not None:
                    output_items.append(f"Opt noise: {opt_message['noise']:.4f}")
                if opt_message.get('mean_reg') is not None:
                    output_items.append(f"Opt mean reg: {opt_message['mean_reg']:.4f}")
        else:
            # Show current log ML when not optimized
            output_items.append(f"GP log ML: {gp_log_ml:.1f}")

        output_items.append(f"Sparsity: {gp_sparsity:.1f}%")

    # Display outputs
    output_html = ""
    if output_items:
        outputs_text = " | ".join(output_items)
        output_html = f'''
        <div style="background-color: #e8f4f8; padding: 8px 15px; border-radius: 5px; margin: 0 auto 10px auto; max-width: 90%; text-align: center; font-size: 14px;">
            <b>Outputs:</b> {outputs_text}
        </div>
        '''

    # Display metrics table
    metrics_html = ""
    if metrics_dict:
        # Create table rows
        rows = []
        for method_name, method_metrics in metrics_dict.items():
            cov_pct = method_metrics['coverage']
            # Color code coverage: green if >85%, yellow if 70-85%, red if <70%
            if cov_pct >= 85:
                coverage_color = "#28a745"  # green
            elif cov_pct >= 70:
                coverage_color = "#ffc107"  # yellow
            else:
                coverage_color = "#dc3545"  # red

            rows.append(f'''
            <tr>
                <td style="text-align: left; padding: 5px 10px;"><b>{method_name}</b></td>
                <td style="text-align: center; padding: 5px 10px; color: {coverage_color};"><b>{cov_pct:.1f}%</b></td>
                <td style="text-align: center; padding: 5px 10px;">{method_metrics['mean_width']:.3f}</td>
                <td style="text-align: center; padding: 5px 10px;">{method_metrics['mse']:.4f}</td>
                <td style="text-align: center; padding: 5px 10px;">{method_metrics['fit_time']*1000:.1f} ms</td>
            </tr>
            ''')

        metrics_html = f'''
        <div style="background-color: #f8f9fa; padding: 10px 15px; border-radius: 5px; margin: 0 auto 10px auto; max-width: 90%;">
            <div style="text-align: center; font-size: 14px; margin-bottom: 8px;"><b>Performance Metrics</b></div>
            <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                <thead>
                    <tr style="border-bottom: 2px solid #dee2e6;">
                        <th style="text-align: left; padding: 5px 10px;">Method</th>
                        <th style="text-align: center; padding: 5px 10px;">Coverage</th>
                        <th style="text-align: center; padding: 5px 10px;">Interval Width</th>
                        <th style="text-align: center; padding: 5px 10px;">MSE</th>
                        <th style="text-align: center; padding: 5px 10px;">Fit Time</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        '''

    result = mo.Html(output_html + metrics_html)
    result
    return


@app.cell(hide_code=True)
def _(mo, set_last_enabled_method):
    bayesian = mo.ui.checkbox(False, label="<b>Bayesian fit</b>",
                               on_change=lambda v: set_last_enabled_method('linear') if v else None)
    conformal = mo.ui.checkbox(False, label="<b>Conformal prediction</b>",
                                on_change=lambda v: set_last_enabled_method('linear') if v else None)
    pops = mo.ui.checkbox(False, label="<b>POPS regression</b>",
                          on_change=lambda v: set_last_enabled_method('linear') if v else None)
    quantile = mo.ui.checkbox(False, label="<b>Quantile regression</b>",
                              on_change=lambda v: set_last_enabled_method('linear') if v else None)
    gp_regression = mo.ui.checkbox(False, label="<b>GP regression</b>",
                                    on_change=lambda v: set_last_enabled_method('gp') if v else None)
    neural_network = mo.ui.checkbox(False, label="<b>Neural network</b>",
                                     on_change=lambda v: set_last_enabled_method('nn') if v else None)
    gp_use_poly_mean = mo.ui.checkbox(False, label="Use polynomial mean function")
    return (
        bayesian,
        conformal,
        gp_regression,
        gp_use_poly_mean,
        neural_network,
        pops,
        quantile,
    )


@app.cell(hide_code=True)
def _(mo):
    # Button definitions - isolated cell with no state dependencies to prevent circular loops
    gp_optimize_button = mo.ui.run_button(label="Optimize hyperparameters")
    return (gp_optimize_button,)


@app.cell(hide_code=True)
def _(mo):
    # Use marimo state to preserve all slider values
    get_N_samples, set_N_samples = mo.state(500)
    get_sigma, set_sigma = mo.state(0.1)
    get_seed, set_seed = mo.state(0)
    get_filter_min, set_filter_min = mo.state(0.0)
    get_filter_max, set_filter_max = mo.state(5.0)
    get_function_type, set_function_type = mo.state('sin')
    get_P, set_P = mo.state(10)
    get_calib_frac, set_calib_frac = mo.state(0.2)
    get_zeta, set_zeta = mo.state(0.05)
    get_percentile_clipping, set_percentile_clipping = mo.state(0)

    # GP-specific state
    get_gp_kernel_type, set_gp_kernel_type = mo.state('rbf')
    get_gp_lengthscale, set_gp_lengthscale = mo.state(0.5)
    get_gp_support_radius, set_gp_support_radius = mo.state(1.5)
    # gp_use_poly_mean is now a simple checkbox without state (created in analysis checkboxes cell)
    get_gp_poly_mean_degree, set_gp_poly_mean_degree = mo.state(3)
    get_gp_joint_inference, set_gp_joint_inference = mo.state(False)
    get_gp_mean_regularization, set_gp_mean_regularization = mo.state(0.1)

    # Neural network-specific state
    get_nn_activation, set_nn_activation = mo.state('tanh')
    get_nn_ensemble_method, set_nn_ensemble_method = mo.state('seed')
    get_nn_hidden_units, set_nn_hidden_units = mo.state(20)
    get_nn_num_layers, set_nn_num_layers = mo.state(1)
    get_nn_regularization, set_nn_regularization = mo.state(-3)  # log10 scale
    get_nn_ensemble_size, set_nn_ensemble_size = mo.state(5)

    # Quantile regression-specific state
    get_quantile_confidence, set_quantile_confidence = mo.state(0.9)
    get_quantile_regularization, set_quantile_regularization = mo.state(0.01)

    # Button click count tracking to prevent infinite loops
    get_bayes_opt_count, set_bayes_opt_count = mo.state(0)
    get_gp_opt_count, set_gp_opt_count = mo.state(0)

    # Track last enabled method for inset display
    get_last_enabled_method, set_last_enabled_method = mo.state('linear')
    return (
        get_N_samples,
        get_P,
        get_calib_frac,
        get_filter_max,
        get_filter_min,
        get_function_type,
        get_gp_joint_inference,
        get_gp_kernel_type,
        get_gp_lengthscale,
        get_gp_mean_regularization,
        get_gp_poly_mean_degree,
        get_gp_support_radius,
        get_nn_ensemble_size,
        get_nn_hidden_units,
        get_nn_num_layers,
        get_nn_regularization,
        get_percentile_clipping,
        get_seed,
        get_sigma,
        get_zeta,
        set_N_samples,
        set_P,
        set_calib_frac,
        set_filter_max,
        set_filter_min,
        set_function_type,
        set_gp_joint_inference,
        set_gp_kernel_type,
        set_gp_lengthscale,
        set_gp_mean_regularization,
        set_gp_poly_mean_degree,
        set_gp_support_radius,
        set_nn_ensemble_size,
        set_nn_hidden_units,
        set_nn_num_layers,
        set_nn_regularization,
        set_percentile_clipping,
        set_seed,
        set_sigma,
        set_zeta,
    )


@app.cell(hide_code=True)
def _(
    N_samples,
    g,
    get_P,
    get_filter_max,
    get_filter_min,
    get_function_type,
    get_gp_kernel_type,
    get_gp_poly_mean_degree,
    gp_lengthscale_slider,
    gp_optimize_button,
    gp_support_radius_slider,
    gp_use_poly_mean,
    mo,
    np,
    optimize_gp_hyperparameters,
    seed,
    set_gp_lengthscale,
    set_gp_mean_regularization,
    set_gp_support_radius,
    sigma,
):
    # Perform GP optimization when button is clicked
    # Only run if button clicked AND GP sliders are available (GP regression enabled)
    opt_message = None

    if gp_optimize_button.value and gp_lengthscale_slider is not None:
        # Regenerate training data (use _ prefix to avoid variable redefinition)
        np.random.seed(seed.value)
        _x_train = np.append(np.random.uniform(-10, 10, size=N_samples.value), np.linspace(-10, 10, 2))
        _x_train = _x_train[(_x_train < get_filter_min()) | (_x_train > get_filter_max())]
        _x_train = np.sort(_x_train)
        _y_train = g(_x_train, noise_variance=sigma.value**2, function_type=get_function_type())

        # Set initial parameters - read from sliders, NOT state getters (avoids circular dependency)
        initial_params = {
            'lengthscale': gp_lengthscale_slider.value,
            'support_radius': gp_support_radius_slider.value,
            'degree': get_P(),
            'sigma': 0.1,
            'noise': sigma.value**2
        }

        # Get polynomial mean settings
        _use_poly_mean = gp_use_poly_mean.value
        # Use default values to avoid circular dependency (cell modifies set_gp_mean_regularization)
        _joint_inference = False  # Default: False
        _poly_degree = get_gp_poly_mean_degree()
        # Use default initial value for mean regularization (will be optimized)
        _mean_reg = 0.1

        # Optimize
        opt_lengthscale = None  # Initialize variables used in output formatting
        opt_support = None
        opt_mean_reg = None
        opt_noise = None

        try:
            optimized_params, log_ml = optimize_gp_hyperparameters(
                _x_train, _y_train, get_gp_kernel_type(), initial_params, max_iter=50,
                use_polynomial_mean=_use_poly_mean,
                poly_degree=_poly_degree,
                joint_inference=_joint_inference,
                mean_regularization_strength=_mean_reg
            )

            # Update sliders with optimized values
            if 'lengthscale' in optimized_params:
                # Clamp to slider bounds [0.1, 5.0]
                opt_lengthscale = np.clip(optimized_params['lengthscale'], 0.1, 5.0)
                set_gp_lengthscale(float(opt_lengthscale))

            if 'support_radius' in optimized_params:
                # Clamp to slider bounds [0.5, 5.0]
                opt_support = np.clip(optimized_params['support_radius'], 0.5, 5.0)
                set_gp_support_radius(float(opt_support))

            if 'mean_regularization' in optimized_params:
                # Clamp to valid range [10^-6, 10^1]
                opt_mean_reg = np.clip(optimized_params['mean_regularization'], 1e-6, 10.0)
                set_gp_mean_regularization(float(opt_mean_reg))

            if 'noise' in optimized_params:
                opt_noise = optimized_params['noise']

            # Store optimized parameters for display in output cell
            opt_message = {
                'lengthscale': float(opt_lengthscale) if opt_lengthscale is not None else None,
                'support_radius': float(opt_support) if opt_support is not None else None,
                'mean_reg': float(opt_mean_reg) if opt_mean_reg is not None else None,
                'noise': float(opt_noise) if opt_noise is not None else None,
                'log_ml': float(log_ml)
            }
        except Exception as e:
            opt_message = {'error': str(e)}

    return (opt_message,)


if __name__ == "__main__":
    app.run()
