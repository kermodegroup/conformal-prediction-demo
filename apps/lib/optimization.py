"""
Hyperparameter optimization for Gaussian Processes.

Provides marginal likelihood optimization for GP hyperparameters.
"""

import numpy as np
from scipy.optimize import minimize

from .kernels import (
    compute_kernel_matrix,
    gp_marginal_likelihood,
    gp_joint_marginal_likelihood,
)


def optimize_gp_hyperparameters(
    X_train,
    y_train,
    kernel_type,
    initial_params,
    max_iter=50,
    use_basis_mean=False,
    Phi_train=None,
    joint_inference=False,
    mean_regularization_strength=0.1,
):
    """
    Optimize GP hyperparameters by maximizing marginal likelihood.

    Parameters
    ----------
    X_train : array (n_train,)
        Training inputs
    y_train : array (n_train,)
        Training targets
    kernel_type : str
        Kernel type ('rbf', 'matern', 'bump', 'polynomial')
    initial_params : dict
        Initial hyperparameter values
    max_iter : int
        Maximum optimization iterations
    use_basis_mean : bool
        Whether to use basis mean function
    Phi_train : array (n_train, n_basis) or None
        Pre-computed basis features
    joint_inference : bool
        If True optimize mean regularization jointly
    mean_regularization_strength : float
        Initial mean regularization strength

    Returns
    -------
    optimized_params : dict
        Optimized hyperparameters
    log_marginal_likelihood : float
        Final log marginal likelihood
    """
    X_train = np.atleast_1d(X_train).reshape(-1)
    y_train = np.atleast_1d(y_train).reshape(-1)

    # Define bounds and parameterization based on kernel type
    if kernel_type == "rbf":
        pack_params, unpack_params, x0, bounds = _setup_rbf_params(
            initial_params,
            use_basis_mean,
            joint_inference,
            Phi_train,
            mean_regularization_strength,
        )

    elif kernel_type == "matern":
        pack_params, unpack_params, x0, bounds = _setup_matern_params(
            initial_params,
            use_basis_mean,
            joint_inference,
            Phi_train,
            mean_regularization_strength,
        )

    elif kernel_type == "bump":
        pack_params, unpack_params, x0, bounds = _setup_bump_params(
            initial_params,
            use_basis_mean,
            joint_inference,
            Phi_train,
            mean_regularization_strength,
        )

    elif kernel_type == "polynomial":
        pack_params, unpack_params, x0, bounds = _setup_polynomial_params(
            initial_params,
            use_basis_mean,
            joint_inference,
            Phi_train,
            mean_regularization_strength,
        )

    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Negative log marginal likelihood (to minimize)
    def neg_log_marginal_likelihood(x):
        params = unpack_params(x)
        noise = params.pop("noise")
        mean_reg = params.pop("mean_regularization", None)

        K_train = compute_kernel_matrix(X_train, X_train, kernel_type, params)

        if (
            mean_reg is not None
            and use_basis_mean
            and joint_inference
            and Phi_train is not None
        ):
            log_ml = gp_joint_marginal_likelihood(
                X_train, y_train, K_train, noise, Phi_train, mean_reg
            )
        else:
            log_ml = gp_marginal_likelihood(X_train, y_train, K_train, noise)

        return -log_ml

    # Optimize
    result = minimize(
        neg_log_marginal_likelihood,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": max_iter},
    )

    optimized_params = unpack_params(result.x)
    log_ml = -result.fun

    return optimized_params, log_ml


def _setup_rbf_params(
    initial_params,
    use_basis_mean,
    joint_inference,
    Phi_train,
    mean_regularization_strength,
):
    """Set up parameter packing for RBF kernel."""
    if use_basis_mean and joint_inference and Phi_train is not None:

        def pack_params(lengthscale, noise, mean_reg):
            return np.array([np.log(lengthscale), np.log(noise), np.log(mean_reg)])

        def unpack_params(x):
            return {
                "lengthscale": np.exp(x[0]),
                "noise": np.exp(x[1]),
                "mean_regularization": np.exp(x[2]),
            }

        x0 = pack_params(
            initial_params.get("lengthscale", 1.0),
            initial_params.get("noise", 0.1),
            mean_regularization_strength,
        )
        bounds = [(-2, 2), (-6, 0), (-6, 1)]
    else:

        def pack_params(lengthscale, noise):
            return np.array([np.log(lengthscale), np.log(noise)])

        def unpack_params(x):
            return {"lengthscale": np.exp(x[0]), "noise": np.exp(x[1])}

        x0 = pack_params(
            initial_params.get("lengthscale", 1.0),
            initial_params.get("noise", 0.1),
        )
        bounds = [(-2, 2), (-6, 0)]

    return pack_params, unpack_params, x0, bounds


def _setup_matern_params(
    initial_params,
    use_basis_mean,
    joint_inference,
    Phi_train,
    mean_regularization_strength,
):
    """Set up parameter packing for Matérn kernel (same as RBF)."""
    return _setup_rbf_params(
        initial_params,
        use_basis_mean,
        joint_inference,
        Phi_train,
        mean_regularization_strength,
    )


def _setup_bump_params(
    initial_params,
    use_basis_mean,
    joint_inference,
    Phi_train,
    mean_regularization_strength,
):
    """Set up parameter packing for bump kernel."""
    if use_basis_mean and joint_inference and Phi_train is not None:

        def pack_params(lengthscale, support_radius, noise, mean_reg):
            return np.array(
                [
                    np.log(lengthscale),
                    np.log(support_radius),
                    np.log(noise),
                    np.log(mean_reg),
                ]
            )

        def unpack_params(x):
            return {
                "lengthscale": np.exp(x[0]),
                "support_radius": np.exp(x[1]),
                "noise": np.exp(x[2]),
                "mean_regularization": np.exp(x[3]),
            }

        x0 = pack_params(
            initial_params.get("lengthscale", 1.0),
            initial_params.get("support_radius", 2.0),
            initial_params.get("noise", 0.1),
            mean_regularization_strength,
        )
        bounds = [(-2, 2), (-1, 2), (-6, 0), (-6, 1)]
    else:

        def pack_params(lengthscale, support_radius, noise):
            return np.array(
                [np.log(lengthscale), np.log(support_radius), np.log(noise)]
            )

        def unpack_params(x):
            return {
                "lengthscale": np.exp(x[0]),
                "support_radius": np.exp(x[1]),
                "noise": np.exp(x[2]),
            }

        x0 = pack_params(
            initial_params.get("lengthscale", 1.0),
            initial_params.get("support_radius", 2.0),
            initial_params.get("noise", 0.1),
        )
        bounds = [(-2, 2), (-1, 2), (-6, 0)]

    return pack_params, unpack_params, x0, bounds


def _setup_polynomial_params(
    initial_params,
    use_basis_mean,
    joint_inference,
    Phi_train,
    mean_regularization_strength,
):
    """Set up parameter packing for polynomial kernel (degree fixed)."""
    degree = initial_params.get("degree", 5)

    if use_basis_mean and joint_inference and Phi_train is not None:

        def pack_params(sigma, noise, mean_reg):
            return np.array([np.log(sigma), np.log(noise), np.log(mean_reg)])

        def unpack_params(x):
            return {
                "degree": degree,
                "sigma": np.exp(x[0]),
                "noise": np.exp(x[1]),
                "mean_regularization": np.exp(x[2]),
            }

        x0 = pack_params(
            initial_params.get("sigma", 0.1),
            initial_params.get("noise", 0.1),
            mean_regularization_strength,
        )
        bounds = [(-3, 1), (-6, 0), (-6, 1)]
    else:

        def pack_params(sigma, noise):
            return np.array([np.log(sigma), np.log(noise)])

        def unpack_params(x):
            return {"degree": degree, "sigma": np.exp(x[0]), "noise": np.exp(x[1])}

        x0 = pack_params(
            initial_params.get("sigma", 0.1),
            initial_params.get("noise", 0.1),
        )
        bounds = [(-3, 1), (-6, 0)]

    return pack_params, unpack_params, x0, bounds
