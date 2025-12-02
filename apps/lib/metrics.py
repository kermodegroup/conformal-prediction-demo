"""
Probabilistic metrics for uncertainty quantification evaluation.

Includes Log Likelihood and Continuous Ranked Probability Score (CRPS).
"""

import numpy as np
from scipy.stats import norm


def gaussian_log_likelihood_per_point(y_true, mu, sigma):
    """
    Compute average log likelihood per data point for Gaussian predictions.

    Parameters
    ----------
    y_true : array
        True values
    mu : array
        Predicted means
    sigma : array
        Predicted standard deviations

    Returns
    -------
    float
        Average log likelihood per point
    """
    sigma = np.maximum(sigma, 1e-10)
    log_lik = -0.5 * (np.log(2 * np.pi * sigma**2) + (y_true - mu) ** 2 / sigma**2)
    return np.mean(log_lik)


def crps_gaussian(y_true, mu, sigma):
    """
    Continuous Ranked Probability Score for Gaussian predictions (closed form).

    Lower CRPS is better (0 is perfect).

    Parameters
    ----------
    y_true : array
        True values
    mu : array
        Predicted means
    sigma : array
        Predicted standard deviations

    Returns
    -------
    float
        Mean CRPS
    """
    sigma = np.maximum(sigma, 1e-10)
    z = (y_true - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return np.mean(crps)


def crps_quantile_approx(y_true, y_lower, y_median, y_upper, alpha):
    """
    Approximate CRPS using pinball losses at 3 quantiles.

    For quantile regression which doesn't output a full distribution.

    Parameters
    ----------
    y_true : array
        True values
    y_lower : array
        Lower quantile predictions (alpha/2)
    y_median : array
        Median predictions (0.5)
    y_upper : array
        Upper quantile predictions (1 - alpha/2)
    alpha : float
        Significance level (e.g., 0.1 for 90% intervals)

    Returns
    -------
    float
        Approximate mean CRPS
    """
    # Pinball loss for lower quantile (alpha/2)
    tau_low = alpha / 2
    err_low = y_true - y_lower
    pinball_low = np.where(err_low >= 0, tau_low * err_low, (tau_low - 1) * err_low)

    # Pinball loss for median (0.5)
    err_med = y_true - y_median
    pinball_med = np.abs(err_med) * 0.5

    # Pinball loss for upper quantile (1 - alpha/2)
    tau_high = 1 - alpha / 2
    err_high = y_true - y_upper
    pinball_high = np.where(
        err_high >= 0, tau_high * err_high, (tau_high - 1) * err_high
    )

    # Average pinball losses as CRPS approximation
    return np.mean(pinball_low + pinball_med + pinball_high) / 3


def crps_ensemble(y_true, ensemble_predictions):
    """
    CRPS using empirical distribution from ensemble members.

    Parameters
    ----------
    y_true : array (n_samples,)
        True values
    ensemble_predictions : array (n_samples, n_members)
        Predictions from each ensemble member

    Returns
    -------
    float
        Mean CRPS
    """
    n_samples = len(y_true)
    crps_values = np.zeros(n_samples)

    for i in range(n_samples):
        y = y_true[i]
        preds = ensemble_predictions[i]
        # CRPS = E|X - y| - 0.5 * E|X - X'|
        # where X, X' are independent draws from the forecast distribution
        crps_values[i] = np.mean(np.abs(preds - y)) - 0.5 * np.mean(
            np.abs(preds[:, None] - preds[None, :])
        )

    return np.mean(crps_values)
