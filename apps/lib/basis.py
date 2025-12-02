"""
Basis feature functions for linear regression models.

Provides RBF, Fourier, Lennard-Jones, and custom basis transformations.
"""

import numpy as np


def make_rbf_features(X, n_basis, x_min, x_max, lengthscale=None):
    """
    Create RBF basis features with evenly spaced centers.

    Parameters
    ----------
    X : array (n_samples, 1)
        Input data
    n_basis : int
        Number of basis functions
    x_min : float
        Minimum x value for center placement
    x_max : float
        Maximum x value for center placement
    lengthscale : float, optional
        Width parameter. If None, uses half the spacing between centers.

    Returns
    -------
    features : array (n_samples, n_basis)
        RBF feature matrix
    centers : array (n_basis,)
        Center locations
    sigma : float
        Width parameter used
    """
    centers = np.linspace(x_min, x_max, n_basis)
    if lengthscale is None:
        spacing = (x_max - x_min) / (n_basis - 1) if n_basis > 1 else 1.0
        sigma = spacing * 0.5
    else:
        sigma = lengthscale
    x = X[:, 0]
    features = np.column_stack(
        [np.exp(-((x - c) ** 2) / (2 * sigma**2)) for c in centers]
    )
    return features, centers, sigma


def make_fourier_features(X, n_basis, x_min, x_max, lengthscale=None):
    """
    Create Fourier basis features (1, sin, cos pairs).

    Parameters
    ----------
    X : array (n_samples, 1)
        Input data
    n_basis : int
        Number of basis functions
    x_min : float
        Minimum x value (used for period calculation)
    x_max : float
        Maximum x value (used for period calculation)
    lengthscale : float, optional
        Half-period. If None, uses half the data range.

    Returns
    -------
    features : array (n_samples, n_basis)
        Fourier feature matrix
    L : float
        Half-period used
    """
    if lengthscale is None:
        L = (x_max - x_min) / 2
    else:
        L = lengthscale
    x = X[:, 0]
    features = [np.ones(len(x))]  # Constant term
    freq = 1
    while len(features) < n_basis:
        features.append(np.sin(freq * np.pi * x / L))
        if len(features) < n_basis:
            features.append(np.cos(freq * np.pi * x / L))
        freq += 1
    return np.column_stack(features[:n_basis]), L


def make_lj_features(X, n_basis, x_min, x_max):
    """
    Create Lennard-Jones inspired basis features.

    Features: constant, 1/r^12, 1/r^6, and inverse powers.

    Parameters
    ----------
    X : array (n_samples, 1)
        Input data
    n_basis : int
        Number of basis functions
    x_min : float
        Unused (kept for API consistency)
    x_max : float
        Unused (kept for API consistency)

    Returns
    -------
    features : array (n_samples, n_basis)
        LJ feature matrix
    offset : float
        Offset used in r = x/5 + offset mapping
    """
    offset = 1.5  # Hard-coded to match LJ ground truth
    x = X[:, 0]
    # Use same mapping as LJ ground truth: r = x/5 + offset
    r = x / 5.0 + offset
    r = np.maximum(r, 0.1)  # Safety clamp

    features = []
    # Order: constant, 1/r^12 (repulsive), 1/r^6 (attractive), then inverse powers
    if n_basis >= 1:
        features.append(np.ones(len(r)))  # Constant for vertical offset
    if n_basis >= 2:
        features.append(1.0 / r**12)  # Repulsive term
    if n_basis >= 3:
        features.append(1.0 / r**6)  # Attractive term
    # Additional inverse power terms: 1/r, 1/r^2, 1/r^3, ...
    power = 1
    while len(features) < n_basis:
        features.append(1.0 / r**power)
        power += 1

    return np.column_stack(features[:n_basis]), offset


def make_custom_features(X, n_basis, custom_code):
    """
    Execute user code to create custom basis features.

    Parameters
    ----------
    X : array (n_samples, 1)
        Input data
    n_basis : int
        Number of basis functions (available as P in code)
    custom_code : str
        Python code that defines 'features' variable

    Returns
    -------
    features : array (n_samples, n_basis) or None
        Custom feature matrix, or None on error
    error : str or None
        Error message if failed, None on success
    """
    import math
    import scipy

    # Safe subset of builtins needed for basic operations
    safe_builtins = {
        "range": range,
        "len": len,
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "list": list,
        "tuple": tuple,
        "zip": zip,
        "enumerate": enumerate,
        "True": True,
        "False": False,
        "None": None,
        "print": print,
    }
    try:
        namespace = {"np": np, "scipy": scipy, "math": math, "X": X, "P": n_basis}
        exec(custom_code, {"__builtins__": safe_builtins}, namespace)
        if "features" not in namespace:
            return None, "Code must define 'features'"
        features = np.atleast_2d(np.asarray(namespace["features"], dtype=float))
        # Ensure correct shape (n_samples, n_features)
        if features.shape[0] != X.shape[0]:
            features = features.T  # Try transpose
        # Check for NaN/Inf values
        if np.any(np.isnan(features)) or np.any(np.isinf(features)):
            return None, "Features contain NaN or Inf values"
        return features[:, :n_basis], None
    except Exception as e:
        return None, str(e)
