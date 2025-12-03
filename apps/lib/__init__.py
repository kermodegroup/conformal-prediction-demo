"""
Regression demo library - reusable components for Bayesian regression and UQ.
"""

from .metrics import (
    gaussian_log_likelihood_per_point,
    crps_gaussian,
    crps_quantile_approx,
    crps_ensemble,
)

from .basis import (
    make_rbf_features,
    make_fourier_features,
    make_lj_features,
    make_custom_features,
)

from .kernels import (
    rbf_kernel,
    matern_kernel,
    bump_kernel,
    polynomial_kernel,
    custom_kernel,
    compute_kernel_matrix,
    gp_predict,
    gp_sample_posterior,
    gp_marginal_likelihood,
    gp_joint_marginal_likelihood,
    gp_loo_log_likelihood,
    fit_gp_numpy,
)

from .models import (
    MyBayesianRidge,
    ConformalPrediction,
    NeuralNetworkRegression,
    QuantileRegressionUQ,
)

from .data import (
    ground_truth,
    generate_dataset,
)

from .optimization import (
    optimize_gp_hyperparameters,
)

__all__ = [
    # Metrics
    "gaussian_log_likelihood_per_point",
    "crps_gaussian",
    "crps_quantile_approx",
    "crps_ensemble",
    # Basis
    "make_rbf_features",
    "make_fourier_features",
    "make_lj_features",
    "make_custom_features",
    # Kernels
    "rbf_kernel",
    "matern_kernel",
    "bump_kernel",
    "polynomial_kernel",
    "custom_kernel",
    "compute_kernel_matrix",
    "gp_predict",
    "gp_sample_posterior",
    "gp_marginal_likelihood",
    "gp_joint_marginal_likelihood",
    "gp_loo_log_likelihood",
    "fit_gp_numpy",
    # Models
    "MyBayesianRidge",
    "ConformalPrediction",
    "NeuralNetworkRegression",
    "QuantileRegressionUQ",
    # Data
    "ground_truth",
    "generate_dataset",
    # Optimization
    "optimize_gp_hyperparameters",
]
