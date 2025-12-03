"""
Gaussian Process kernels and prediction functions.

Pure numpy/scipy implementation for WebAssembly compatibility.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def rbf_kernel(X1, X2, lengthscale, signal_variance=1.0):
    """
    RBF/squared exponential kernel.

    Parameters
    ----------
    X1 : array (n1,)
        First set of points
    X2 : array (n2,)
        Second set of points
    lengthscale : float
        Kernel lengthscale
    signal_variance : float
        Signal variance (amplitude squared)

    Returns
    -------
    K : array (n1, n2)
        Kernel matrix
    """
    dists_sq = (X1[:, None] - X2[None, :]) ** 2
    K = signal_variance * np.exp(-0.5 * dists_sq / lengthscale**2)
    return K


def matern_kernel(X1, X2, lengthscale, signal_variance=1.0, nu=1.5):
    """
    Matérn kernel with nu=1.5 (Matérn-3/2).

    Parameters
    ----------
    X1 : array (n1,)
        First set of points
    X2 : array (n2,)
        Second set of points
    lengthscale : float
        Kernel lengthscale
    signal_variance : float
        Signal variance (amplitude squared)
    nu : float
        Smoothness parameter (only 1.5 implemented)

    Returns
    -------
    K : array (n1, n2)
        Kernel matrix
    """
    dists = np.abs(X1[:, None] - X2[None, :])
    r = dists / lengthscale
    sqrt3_r = np.sqrt(3.0) * r
    K = signal_variance * (1.0 + sqrt3_r) * np.exp(-sqrt3_r)
    return K


def bump_kernel(X1, X2, lengthscale, support_radius, signal_variance=1.0):
    """
    Wendland C2 compactly supported kernel.

    Parameters
    ----------
    X1 : array (n1,)
        First set of points
    X2 : array (n2,)
        Second set of points
    lengthscale : float
        Kernel lengthscale
    support_radius : float
        Compact support radius
    signal_variance : float
        Signal variance (amplitude squared)

    Returns
    -------
    K : array (n1, n2)
        Kernel matrix (sparse beyond support_radius)
    """
    dists = np.abs(X1[:, None] - X2[None, :])
    r = dists / (lengthscale * support_radius)
    r_clipped = np.clip(r, 0.0, 1.0)
    K = signal_variance * np.where(r < 1.0, (1.0 - r_clipped) ** 4 * (4.0 * r_clipped + 1.0), 0.0)
    return K


def polynomial_kernel(X1, X2, degree, sigma):
    """
    Polynomial kernel.

    Parameters
    ----------
    X1 : array (n1,)
        First set of points
    X2 : array (n2,)
        Second set of points
    degree : int
        Polynomial degree
    sigma : float
        Offset parameter

    Returns
    -------
    K : array (n1, n2)
        Kernel matrix
    """
    # Normalize inputs
    x1_norm = X1 * 0.1
    x2_norm = X2 * 0.1
    K = (sigma + x1_norm[:, None] * x2_norm[None, :]) ** degree
    return K


def custom_kernel(X1, X2, lengthscale, signal_variance, custom_code):
    """
    Execute user-defined kernel code.

    Parameters
    ----------
    X1, X2 : array (n1,), (n2,)
        Input points
    lengthscale : float
        Kernel lengthscale parameter
    signal_variance : float
        Kernel signal variance (amplitude squared)
    custom_code : str
        User code that defines K matrix

    Returns
    -------
    K : array (n1, n2)
        Kernel matrix, or None if execution failed
    error : str or None
        Error message if execution failed
    """
    import math
    import scipy

    safe_builtins = {
        'range': range, 'len': len, 'sum': sum, 'min': min, 'max': max,
        'abs': abs, 'round': round, 'int': int, 'float': float,
        'list': list, 'tuple': tuple, 'zip': zip, 'enumerate': enumerate,
        'True': True, 'False': False, 'None': None, 'print': print,
    }

    try:
        namespace = {
            'np': np, 'scipy': scipy, 'math': math,
            'X1': np.atleast_1d(X1), 'X2': np.atleast_1d(X2),
            'lengthscale': lengthscale, 'signal_variance': signal_variance
        }
        exec(custom_code, {"__builtins__": safe_builtins}, namespace)

        if 'K' not in namespace:
            return None, "Code must define 'K'"

        K = np.atleast_2d(np.asarray(namespace['K'], dtype=float))

        # Validate shape
        expected_shape = (len(np.atleast_1d(X1)), len(np.atleast_1d(X2)))
        if K.shape != expected_shape:
            return None, f"K shape {K.shape} doesn't match expected {expected_shape}"

        # Check for NaN/Inf
        if np.any(np.isnan(K)) or np.any(np.isinf(K)):
            return None, "Kernel matrix contains NaN or Inf values"

        return K, None

    except Exception as e:
        return None, str(e)


def compute_kernel_matrix(X1, X2, kernel_type, params):
    """
    Compute kernel matrix given type and parameters.

    Parameters
    ----------
    X1 : array (n1,)
        First set of points
    X2 : array (n2,)
        Second set of points
    kernel_type : str
        One of 'rbf', 'matern', 'bump', 'polynomial'
    params : dict
        Kernel parameters (lengthscale, support_radius, degree, sigma)

    Returns
    -------
    K : array (n1, n2)
        Kernel matrix
    """
    if kernel_type == "custom":
        K, error = custom_kernel(
            X1, X2,
            params.get("lengthscale", 1.0),
            params.get("signal_variance", 1.0),
            params.get("custom_code", "")
        )
        if error:
            raise ValueError(f"Custom kernel error: {error}")
        return K
    elif kernel_type == "bump":
        return bump_kernel(X1, X2, params["lengthscale"], params["support_radius"],
                          params.get("signal_variance", 1.0))
    elif kernel_type == "polynomial":
        return polynomial_kernel(X1, X2, params["degree"], params["sigma"])
    elif kernel_type == "rbf":
        return rbf_kernel(X1, X2, params["lengthscale"], params.get("signal_variance", 1.0))
    elif kernel_type == "matern":
        return matern_kernel(X1, X2, params["lengthscale"], params.get("signal_variance", 1.0))
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")


def gp_marginal_likelihood(X_train, y_train, K_train, noise):
    """
    Compute log marginal likelihood.

    log p(y|X) = -0.5 * y^T K^{-1} y - 0.5 * log|K| - n/2 * log(2π)

    Parameters
    ----------
    X_train : array (n,)
        Training inputs (unused, kept for API consistency)
    y_train : array (n,)
        Training targets
    K_train : array (n, n)
        Kernel matrix on training data
    noise : float
        Observation noise variance

    Returns
    -------
    float
        Log marginal likelihood
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


def gp_joint_marginal_likelihood(
    X_train, y_train, K_train, noise, Phi_train, mean_regularization_strength
):
    """
    Compute log marginal likelihood with joint Bayesian inference over basis mean coefficients.

    Model: y = Φβ + f + ε
    where β ~ N(0, λ⁻¹I), f ~ GP(0, K), ε ~ N(0, σ²I)

    Marginal: y ~ N(0, K_total) where K_total = ΦΛ⁻¹Φᵀ + K + σ²I

    Parameters
    ----------
    X_train : array (n_train,)
        Training inputs
    y_train : array (n_train,)
        Training targets
    K_train : array (n_train, n_train)
        GP kernel on training data
    noise : float
        Observation noise variance
    Phi_train : array (n_train, n_features)
        Pre-computed basis feature matrix
    mean_regularization_strength : float
        Regularization strength for basis coefficients

    Returns
    -------
    float
        Log marginal likelihood
    """
    n = len(y_train)

    if Phi_train is None:
        return gp_marginal_likelihood(X_train, y_train, K_train, noise)

    # Normalize basis features for numerical stability
    Phi_train_std = np.std(Phi_train, axis=0, keepdims=True)
    Phi_train_std[Phi_train_std < 1e-10] = 1.0
    Phi_train_normalized = Phi_train / Phi_train_std

    # Prior precision for basis coefficients
    lambda_prior = mean_regularization_strength
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
    GP prediction using Cholesky decomposition.

    Parameters
    ----------
    X_train : array (n_train,)
        Training inputs
    y_train : array (n_train,)
        Training targets
    X_test : array (n_test,)
        Test inputs
    K_train : array (n_train, n_train)
        Kernel on training data
    K_test_train : array (n_test, n_train)
        Cross kernel
    K_test : array (n_test, n_test)
        Kernel on test data
    noise : float
        Observation noise variance

    Returns
    -------
    y_mean : array (n_test,)
        Predictive mean
    y_std : array (n_test,)
        Predictive standard deviation
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


def gp_sample_posterior(X_train, y_train, X_test, K_train, K_test_train, K_test, noise, n_samples=10):
    """
    Draw samples from the GP posterior predictive distribution.

    Parameters
    ----------
    X_train : array (n_train,)
        Training inputs
    y_train : array (n_train,)
        Training targets
    X_test : array (n_test,)
        Test inputs
    K_train : array (n_train, n_train)
        Kernel on training data
    K_test_train : array (n_test, n_train)
        Cross kernel
    K_test : array (n_test, n_test)
        Kernel on test data
    noise : float
        Observation noise variance
    n_samples : int
        Number of posterior samples to draw

    Returns
    -------
    samples : array (n_test, n_samples)
        Posterior predictive samples
    """
    n_train = len(X_train)
    n_test = len(X_test)

    # Add noise to diagonal
    K_noisy = K_train + noise * np.eye(n_train)

    # Cholesky decomposition of training kernel
    try:
        L_train, lower = cho_factor(K_noisy, lower=True)
    except np.linalg.LinAlgError:
        K_noisy += 1e-3 * np.eye(n_train)
        L_train, lower = cho_factor(K_noisy, lower=True)

    # Solve for alpha = K^{-1} y
    alpha = cho_solve((L_train, lower), y_train)

    # Predictive mean
    y_mean = K_test_train @ alpha

    # Posterior covariance: K_** - K_*n @ K_nn^{-1} @ K_n*
    v = cho_solve((L_train, lower), K_test_train.T)
    K_post = K_test - K_test_train @ v

    # Add jitter for numerical stability
    jitter = 1e-6
    K_post += jitter * np.eye(n_test)

    # Ensure symmetry
    K_post = 0.5 * (K_post + K_post.T)

    # Cholesky decompose posterior covariance
    try:
        L_post = np.linalg.cholesky(K_post)
    except np.linalg.LinAlgError:
        # If Cholesky fails, add more jitter
        K_post += 1e-4 * np.eye(n_test)
        try:
            L_post = np.linalg.cholesky(K_post)
        except np.linalg.LinAlgError:
            # Fallback: return mean repeated n_samples times
            return np.tile(y_mean[:, None], (1, n_samples))

    # Sample: mean + L @ z where z ~ N(0, I)
    z = np.random.randn(n_test, n_samples)
    samples = y_mean[:, None] + L_post @ z

    return samples


def gp_loo_log_likelihood(y_train, K_train, noise):
    """
    Compute Leave-One-Out cross-validation log likelihood for GP (per point average).

    Uses efficient closed-form formula:
    - α = K^{-1} y
    - K_inv_ii = diagonal of K^{-1}
    - LOO mean: μ_i = y_i - α_i / K_inv_ii
    - LOO variance: σ²_i = 1 / K_inv_ii

    Parameters
    ----------
    y_train : array (n,)
        Training targets
    K_train : array (n, n)
        Kernel matrix on training data
    noise : float
        Observation noise variance

    Returns
    -------
    float
        Average LOO log likelihood per point
    """
    n = len(y_train)

    # Add noise to kernel
    K_noisy = K_train + noise * np.eye(n)

    try:
        # Compute K^{-1} and K^{-1}y
        L, lower = cho_factor(K_noisy, lower=True)
        alpha = cho_solve((L, lower), y_train)

        # Compute diagonal of K^{-1}
        K_inv = cho_solve((L, lower), np.eye(n))
        K_inv_diag = np.diag(K_inv)

        # Ensure positive diagonal (numerical stability)
        K_inv_diag = np.maximum(K_inv_diag, 1e-10)

        # LOO predictions
        loo_mean = y_train - alpha / K_inv_diag
        loo_var = 1.0 / K_inv_diag

        # LOO residuals
        loo_residuals = y_train - loo_mean

        # Gaussian log likelihood
        log_lik = -0.5 * (np.log(2 * np.pi * loo_var) + loo_residuals**2 / loo_var)

        return np.mean(log_lik)

    except np.linalg.LinAlgError:
        return -np.inf


def fit_gp_numpy(
    X_train,
    y_train,
    X_test,
    kernel_type="rbf",
    use_basis_mean=False,
    Phi_train=None,
    Phi_test=None,
    joint_inference=False,
    mean_regularization_strength=0.1,
    **kernel_params,
):
    """
    Fit GP using pure numpy/scipy.

    Parameters
    ----------
    X_train : array (n_train,)
        Training inputs
    y_train : array (n_train,)
        Training targets
    X_test : array (n_test,)
        Test inputs
    kernel_type : str
        One of 'rbf', 'matern', 'bump', 'polynomial'
    use_basis_mean : bool
        Whether to use basis mean function
    Phi_train : array (n_train, n_basis) or None
        Pre-computed basis features for training
    Phi_test : array (n_test, n_basis) or None
        Pre-computed basis features for testing
    joint_inference : bool
        If True do joint Bayesian inference over mean params and GP
    mean_regularization_strength : float
        Regularization for basis coefficients
    **kernel_params : dict
        Kernel hyperparameters (lengthscale, support_radius, degree, sigma, noise)

    Returns
    -------
    y_mean : array (n_test,)
        Full GP predictions (with basis mean added back)
    y_std : array (n_test,)
        Total uncertainty (GP + mean if joint inference)
    basis_mean : array (n_test,) or None
        Basis mean function (for plotting)
    y_std_gp : array (n_test,) or None
        GP uncertainty component only (for joint inference)
    y_std_mean : array (n_test,) or None
        Mean uncertainty component only (for joint inference)
    """
    X_train = np.atleast_1d(X_train).reshape(-1)
    y_train = np.atleast_1d(y_train).reshape(-1)
    X_test = np.atleast_1d(X_test).reshape(-1)

    # Basis mean function (if requested)
    if use_basis_mean and Phi_train is not None and Phi_test is not None:
        from sklearn.linear_model import BayesianRidge

        if joint_inference:
            # Joint Bayesian inference: marginalize over basis coefficients
            # Prior precision for basis coefficients
            lambda_prior = mean_regularization_strength
            Lambda_inv = (1.0 / lambda_prior) * np.eye(Phi_train.shape[1])

            # Normalize basis features to improve conditioning
            Phi_train_std = np.std(Phi_train, axis=0, keepdims=True)
            Phi_train_std[Phi_train_std < 1e-10] = 1.0
            Phi_train_normalized = Phi_train / Phi_train_std
            Phi_test_normalized = Phi_test / Phi_train_std

            # Update Lambda_inv to account for normalization
            Lambda_inv_normalized = Lambda_inv / (Phi_train_std.T @ Phi_train_std + 1e-10)

            # Store normalized features for joint inference
            Phi_train = Phi_train_normalized
            Phi_test = Phi_test_normalized
            Lambda_inv = Lambda_inv_normalized

            y_train_residual = None
            mean_test = None
        else:
            # Sequential inference: fit basis model first (point estimate)
            basis_model = BayesianRidge(fit_intercept=False)
            basis_model.fit(Phi_train, y_train)

            # Get basis mean predictions
            mean_train = basis_model.predict(Phi_train)
            mean_test = basis_model.predict(Phi_test)

            # Subtract mean from training data (fit GP on residuals)
            y_train_residual = y_train - mean_train
    else:
        y_train_residual = y_train
        mean_test = np.zeros_like(X_test)
        use_basis_mean = False
        joint_inference = False

    # Get noise level
    noise = kernel_params.get("noise", 0.1)

    # Compute kernel matrices
    if kernel_type == "bump":
        lengthscale = kernel_params.get("lengthscale", 1.0)
        support_radius = kernel_params.get("support_radius", 2.0)
        noise = max(noise, 1e-4)

        K_train = bump_kernel(X_train, X_train, lengthscale, support_radius)
        K_test_train = bump_kernel(X_test, X_train, lengthscale, support_radius)
        K_test = bump_kernel(X_test, X_test, lengthscale, support_radius)

    elif kernel_type == "polynomial":
        degree = min(kernel_params.get("degree", 10), 8)
        sigma = kernel_params.get("sigma", 0.1)
        noise = max(noise, 1e-3)

        K_train = polynomial_kernel(X_train, X_train, degree, sigma)
        K_test_train = polynomial_kernel(X_test, X_train, degree, sigma)
        K_test = polynomial_kernel(X_test, X_test, degree, sigma)

    elif kernel_type == "rbf":
        lengthscale = kernel_params.get("lengthscale", 1.0)
        noise = max(noise, 1e-6)

        K_train = rbf_kernel(X_train, X_train, lengthscale)
        K_test_train = rbf_kernel(X_test, X_train, lengthscale)
        K_test = rbf_kernel(X_test, X_test, lengthscale)

    elif kernel_type == "matern":
        lengthscale = kernel_params.get("lengthscale", 1.0)
        noise = max(noise, 1e-6)

        K_train = matern_kernel(X_train, X_train, lengthscale)
        K_test_train = matern_kernel(X_test, X_train, lengthscale)
        K_test = matern_kernel(X_test, X_test, lengthscale)

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    try:
        if joint_inference and use_basis_mean:
            # Joint Bayesian inference path
            K_beta = Phi_train @ Lambda_inv @ Phi_train.T
            K_total = K_beta + K_train + noise * np.eye(len(X_train))

            try:
                # Adaptive jitter for numerical stability
                jitter = 1e-8
                max_attempts = 5

                for attempt in range(max_attempts):
                    try:
                        L = np.linalg.cholesky(K_total + jitter * np.eye(len(X_train)))
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
                        break
                    except np.linalg.LinAlgError:
                        if attempt < max_attempts - 1:
                            jitter *= 10
                            continue
                        else:
                            raise

                # Prediction mean
                K_test_beta = Phi_test @ Lambda_inv @ Phi_train.T
                K_combined = K_test_beta + K_test_train
                y_mean = K_combined @ alpha

                # Prediction variance
                v = np.linalg.solve(L, K_combined.T)
                K_test_beta_diag = np.sum((Phi_test @ Lambda_inv) * Phi_test, axis=1)

                y_var_mean_prior = K_test_beta_diag
                y_var_gp_prior = np.diag(K_test)
                y_var_reduction = np.sum(v**2, axis=0)

                y_var_total = y_var_mean_prior + y_var_gp_prior - y_var_reduction
                y_var_total = np.maximum(y_var_total, 1e-10)
                y_std_total = np.sqrt(y_var_total)

                # Approximate split between mean and GP contributions
                fraction_mean = y_var_mean_prior / (
                    y_var_mean_prior + y_var_gp_prior + 1e-10
                )
                fraction_gp = y_var_gp_prior / (
                    y_var_mean_prior + y_var_gp_prior + 1e-10
                )

                y_std_mean_component = y_std_total * np.sqrt(fraction_mean)
                y_std_gp_component = y_std_total * np.sqrt(fraction_gp)

                # Posterior mean of polynomial coefficients for visualization
                Phi_beta_coef = Lambda_inv @ Phi_train.T @ alpha
                mean_test_posterior = Phi_test @ Phi_beta_coef

                return (
                    y_mean,
                    y_std_total,
                    mean_test_posterior,
                    y_std_gp_component,
                    y_std_mean_component,
                )

            except np.linalg.LinAlgError:
                print("Warning: Cholesky failed for joint inference")
                return np.zeros_like(X_test), np.ones_like(X_test), None, None, None
        else:
            # Sequential inference path: fit GP on residuals
            y_mean_residual, y_std = gp_predict(
                X_train, y_train_residual, X_test, K_train, K_test_train, K_test, noise
            )

            # Add polynomial mean back to predictions
            y_mean = y_mean_residual + mean_test

            # Check for NaNs
            if np.any(np.isnan(y_mean)) or np.any(np.isnan(y_std)):
                print(f"Warning: NaNs in GP prediction with {kernel_type} kernel")
                return np.zeros_like(X_test), np.ones_like(X_test), None, None, None

            return (
                y_mean,
                y_std,
                (mean_test if use_basis_mean and not joint_inference else None),
                None,
                None,
            )

    except Exception as e:
        print(f"Error in GP fitting with {kernel_type} kernel: {e}")
        return np.zeros_like(X_test), np.ones_like(X_test), None, None, None
