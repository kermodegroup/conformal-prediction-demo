"""
Data generation functions for regression demos.

Provides ground truth functions and dataset generation utilities.
"""

import numpy as np


def ground_truth(X, noise_variance, function_type="sin", custom_code=None):
    """
    Generate data from different ground truth functions.

    Parameters
    ----------
    X : array
        Input values
    noise_variance : float
        Variance of additive Gaussian noise
    function_type : str
        One of 'sin', 'witch', 'sinc', 'step', 'runge', 'lj', 'custom'
    custom_code : str, optional
        Python code defining y = f(X) for custom function type

    Returns
    -------
    y : array
        Function values with noise
    """
    if function_type == "sin":
        y = np.sin(X)

    elif function_type == "witch":
        # Witch of Agnesi: 1/(1+x²)
        y = 1.0 / (1.0 + X**2)

    elif function_type == "sinc":
        # sinc(x) = sin(x)/x, handle x=0 carefully
        y = np.where(X == 0, 1.0, np.sin(X) / X)

    elif function_type == "step":
        # Step function
        y = np.where(X < 0, -0.5, 0.5)

    elif function_type == "runge":
        # Runge function: 1/(1+25x²) - pathological for polynomial interpolation
        y = 1.0 / (1.0 + 25.0 * X**2)

    elif function_type == "lj":
        # Lennard-Jones potential: V(r) = 4ε[(σ/r)^12 - (σ/r)^6]
        # r = x/5 + 1.5 maps x=0 to r=1.5 (at sigma), x=5 to r=2.5, x=10 to r=3.5
        sigma = 1.5
        r = X / 5.0 + 1.5
        r = np.maximum(r, 0.3)  # Avoid singularity
        y = 4.0 * ((sigma / r) ** 12 - (sigma / r) ** 6)
        # Clip extreme values for display
        y = np.clip(y, -1.5, 1.5)

    elif function_type == "custom":
        y = _execute_custom_function(X, custom_code)

    else:
        y = np.sin(X)  # Default to sin

    return y + np.random.normal(scale=np.sqrt(noise_variance), size=X.shape)


def _execute_custom_function(X, custom_code):
    """
    Execute user-provided code to compute y = f(X).

    Parameters
    ----------
    X : array
        Input values
    custom_code : str
        Python code that must define 'y'

    Returns
    -------
    y : array
        Function values (NaN array if code fails)
    """
    import math
    import scipy

    # Safe subset of builtins
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
        namespace = {"np": np, "scipy": scipy, "math": math, "X": X}
        exec(custom_code, {"__builtins__": safe_builtins}, namespace)

        if "y" not in namespace:
            raise ValueError("Code must define 'y'")

        y = namespace["y"]
        y = np.atleast_1d(np.asarray(y, dtype=float))

        if y.shape != X.shape:
            y = np.broadcast_to(y, X.shape).copy()

        return y

    except Exception:
        # Return NaN on error - will suppress plotting
        return np.full_like(X, np.nan)


def generate_dataset(
    n_samples=500,
    noise_std=0.1,
    function_type="sin",
    custom_code=None,
    x_range=(-10, 10),
    filter_min=0.0,
    filter_max=5.0,
    filter_invert=False,
    random_state=None,
):
    """
    Generate training and test datasets.

    Parameters
    ----------
    n_samples : int
        Number of training samples
    noise_std : float
        Standard deviation of observation noise
    function_type : str
        Ground truth function type
    custom_code : str, optional
        Custom function code
    x_range : tuple
        (min, max) range for x values
    filter_min : float
        Minimum x for data filtering
    filter_max : float
        Maximum x for data filtering
    filter_invert : bool
        If True, keep only inside filter range; else exclude filter range
    random_state : int, optional
        Random seed

    Returns
    -------
    X_train : array (n_train, 1)
        Training inputs
    y_train : array (n_train,)
        Training targets
    X_test : array (n_test, 1)
        Test inputs (dense grid)
    y_test : array (n_test,)
        Ground truth at test points (no noise)
    """
    if random_state is not None:
        np.random.seed(random_state)

    x_min, x_max = x_range

    # Generate random training points plus boundary points
    x_train = np.append(
        np.random.uniform(x_min, x_max, size=n_samples),
        np.linspace(x_min, x_max, 2),
    )

    # Apply filter
    if filter_invert:
        # Keep inside range
        x_train = x_train[(x_train >= filter_min) & (x_train <= filter_max)]
    else:
        # Exclude inside range
        x_train = x_train[(x_train < filter_min) | (x_train > filter_max)]

    x_train = np.sort(x_train)

    # Generate noisy training data
    y_train = ground_truth(
        x_train,
        noise_variance=noise_std**2,
        function_type=function_type,
        custom_code=custom_code,
    )

    # Generate dense test grid (no noise)
    x_test = np.linspace(x_min, x_max, 1000)
    y_test = ground_truth(
        x_test, noise_variance=0, function_type=function_type, custom_code=custom_code
    )

    # Reshape to (n, 1) for sklearn compatibility
    X_train = x_train[:, None]
    X_test = x_test[:, None]

    return X_train, y_train, X_test, y_test
