# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.5",
#     "scipy",
#     "seaborn==0.13.2",
#     "qrcode==8.2",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.Html('''
    <style>
        body, .marimo-container {
            margin: 0 !important;
            padding: 0 !important;
            height: 100vh;
            overflow: hidden;
        }

        .app-header {
            padding: 8px 16px;
            border-bottom: 1px solid #dee2e6;
            background-color: #fff;
        }

        .app-layout {
            display: flex;
            height: calc(100vh - 80px);
            align-items: flex-start;
            justify-content: center;
            gap: 2em;
            padding: 1em 0.5em;
        }

        .app-plot {
            flex: 1;
            min-width: 0;
            display: flex;
            justify-content: center;
            align-items: flex-start;
        }

        .app-plot img,
        .app-plot svg {
            max-width: 100%;
            height: auto;
        }

        .app-sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5em;
            padding: 1.5em;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            min-width: 280px;
        }

        .app-sidebar h4 {
            margin: 1em 0 0.5em 0;
            font-size: 0.9em;
            color: #495057;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 0.3em;
        }

        .app-sidebar h4:first-child {
            margin-top: 0;
        }
    </style>
    ''')
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.linalg as sla
    import scipy.optimize as opt

    import seaborn as sns
    sns.set_context('talk')
    return mo, np, plt, sla, opt, sns


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/gp-demo.html')
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # Convert to base64
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    qr_base64 = base64.b64encode(buffer.read()).decode()
    return (qr_base64,)


@app.cell(hide_code=True)
def _(mo, qr_base64):
    header = mo.Html(f'''
    <div class="app-header">
        <div style="display: flex; justify-content: space-between; align-items: center; margin: 0; padding: 0;">
            <div>
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>Gaussian Process Regression Demo</b>
                <br><span style="font-size: 16px;"><i>Live demos:</i>
                <a href="https://sciml.warwick.ac.uk/" target="_blank" style="color: #0066cc; text-decoration: none;">sciml.warwick.ac.uk</a>
                &nbsp;&nbsp;|&nbsp;&nbsp;
                <i>Code:</i>
                <a href="https://github.com/kermodegroup/demos" target="_blank" style="color: #0066cc; text-decoration: none;">github.com/kermodegroup/demos</a>
                </span></p>
            </div>
            <img src="data:image/png;base64,{qr_base64}" alt="QR Code" style="width: 100px; height: 100px; flex-shrink: 0;" />
        </div>
    </div>
    ''')
    return (header,)


@app.cell(hide_code=True)
def _(mo):
    # Target function selection
    target_dropdown = mo.ui.dropdown(
        options={
            'Sine': 'sin',
            'Step': 'step',
            'Runge': 'runge',
            'Witch of Agnesi': 'witch',
        },
        value='Sine',
        label='Target Function'
    )

    # Kernel type selection
    kernel_dropdown = mo.ui.dropdown(
        options={
            'Squared Exponential (RBF)': 'rbf',
            'Matern 3/2': 'matern32',
            'Matern 5/2': 'matern52',
        },
        value='Squared Exponential (RBF)',
        label='Kernel Type'
    )

    # Kernel hyperparameters
    variance_slider = mo.ui.slider(0.1, 5.0, 0.1, 1.0, label='Variance $v$')
    lengthscale_slider = mo.ui.slider(0.05, 1.0, 0.05, 0.3, label='Lengthscale $\\ell$')

    # Data parameters - start at 0 so posterior matches prior
    n_data_slider = mo.ui.slider(0, 50, 1, 0, label='$N$ data points')
    noise_slider = mo.ui.slider(0.01, 0.5, 0.01, 0.2, label='Noise $\\sigma_n$')
    seed_slider = mo.ui.slider(0, 100, 1, 42, label='Random seed')

    # Sampling controls
    n_samples_slider = mo.ui.slider(0, 30, 1, 5, label='$N$ samples')

    return (
        target_dropdown,
        kernel_dropdown,
        variance_slider,
        lengthscale_slider,
        n_data_slider,
        noise_slider,
        seed_slider,
        n_samples_slider,
    )


@app.cell(hide_code=True)
def _(mo, n_data_slider):
    # State for data point offset (can be negative to cancel slider)
    get_extra, set_extra = mo.state(0)

    def add_point(_):
        # Only add if total won't exceed 50
        total = n_data_slider.value + get_extra()
        if total < 50:
            set_extra(get_extra() + 1)

    def reset_to_zero(_):
        # Set offset to cancel out slider, resulting in 0 total
        set_extra(-n_data_slider.value)

    add_button = mo.ui.button(label="+1 Data Point", on_click=add_point)
    reset_button = mo.ui.button(label="Reset", on_click=reset_to_zero)

    return get_extra, add_button, reset_button


@app.cell(hide_code=True)
def _(np):
    def target_function(x, func_type):
        """Generate target function values."""
        if func_type == 'sin':
            return 0.5 + np.sin(2 * np.pi * x)
        elif func_type == 'step':
            return np.where(x < 0, -0.5, 0.5)
        elif func_type == 'runge':
            return 1.0 / (1.0 + 25.0 * x**2)
        elif func_type == 'witch':
            return 1.0 / (1.0 + x**2)
        return np.sin(x)

    def rbf_kernel(X1, X2, variance, lengthscale):
        """Squared exponential (RBF) kernel."""
        X1 = np.atleast_2d(X1).T if X1.ndim == 1 else X1
        X2 = np.atleast_2d(X2).T if X2.ndim == 1 else X2
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return variance * np.exp(-0.5 * sqdist / lengthscale**2)

    def matern32_kernel(X1, X2, variance, lengthscale):
        """Matern 3/2 kernel."""
        X1 = np.atleast_2d(X1).T if X1.ndim == 1 else X1
        X2 = np.atleast_2d(X2).T if X2.ndim == 1 else X2
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        r = np.sqrt(np.maximum(sqdist, 1e-12)) / lengthscale
        sqrt3 = np.sqrt(3.0)
        return variance * (1 + sqrt3 * r) * np.exp(-sqrt3 * r)

    def matern52_kernel(X1, X2, variance, lengthscale):
        """Matern 5/2 kernel."""
        X1 = np.atleast_2d(X1).T if X1.ndim == 1 else X1
        X2 = np.atleast_2d(X2).T if X2.ndim == 1 else X2
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        r = np.sqrt(np.maximum(sqdist, 1e-12)) / lengthscale
        sqrt5 = np.sqrt(5.0)
        return variance * (1 + sqrt5 * r + 5.0/3.0 * r**2) * np.exp(-sqrt5 * r)

    def get_kernel_func(kernel_type):
        """Return kernel function based on type."""
        if kernel_type == 'rbf':
            return rbf_kernel
        elif kernel_type == 'matern32':
            return matern32_kernel
        elif kernel_type == 'matern52':
            return matern52_kernel
        return rbf_kernel

    return target_function, rbf_kernel, matern32_kernel, matern52_kernel, get_kernel_func


@app.cell(hide_code=True)
def _(np, sla):
    def gp_posterior(X, y, X_star, kernel_func, variance, lengthscale, sigma_n):
        """Compute GP posterior mean and covariance using Cholesky decomposition."""
        N_data = len(X)

        # Compute kernel matrices
        K = kernel_func(X, X, variance, lengthscale)
        K_star = kernel_func(X_star, X_star, variance, lengthscale)
        k_star = kernel_func(X, X_star, variance, lengthscale)

        # Add noise to diagonal and compute Cholesky
        L = np.linalg.cholesky(K + sigma_n**2 * np.eye(N_data))

        # Solve for alpha: (K + sigma_n^2 I) alpha = y
        alpha = sla.solve_triangular(L.T, sla.solve_triangular(L, y, lower=True))

        # Posterior mean: k_star.T @ alpha
        f_mean = k_star.T @ alpha

        # Posterior covariance: K_star - k_star.T @ (K + sigma_n^2 I)^{-1} @ k_star
        v = sla.solve_triangular(L, k_star, lower=True)
        f_cov = K_star - v.T @ v

        return f_mean, f_cov

    def neg_log_marginal_likelihood(X, y, kernel_func, variance, lengthscale, sigma_n):
        """Compute negative log marginal likelihood for hyperparameter optimization."""
        N = len(X)
        if N == 0:
            return 0.0

        K = kernel_func(X, X, variance, lengthscale) + sigma_n**2 * np.eye(N)
        try:
            L = np.linalg.cholesky(K)
            alpha = sla.solve_triangular(L.T, sla.solve_triangular(L, y, lower=True))

            # Log marginal likelihood = -0.5 * y.T @ alpha - sum(log(diag(L))) - N/2 * log(2*pi)
            log_ml = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * N * np.log(2 * np.pi)
            return -log_ml  # Return negative for minimization
        except np.linalg.LinAlgError:
            return 1e10  # Return large value if Cholesky fails

    return gp_posterior, neg_log_marginal_likelihood


@app.cell(hide_code=True)
def _(
    mo, np, opt,
    target_function, get_kernel_func, neg_log_marginal_likelihood,
    target_dropdown, kernel_dropdown,
    variance_slider, lengthscale_slider,
    n_data_slider, noise_slider, seed_slider,
    get_extra,
):
    # State for optimized hyperparameters: None means use slider values
    # Stores (variance, lengthscale, noise) when optimized
    get_opt_params, set_opt_params = mo.state(None)
    get_opt_error, set_opt_error = mo.state(None)

    def run_optimization(_):
        """Optimize hyperparameters to maximize marginal likelihood."""
        try:
            set_opt_error(None)
            func_type = target_dropdown.value
            kernel_type = kernel_dropdown.value
            n_data = max(0, n_data_slider.value + get_extra())
            sigma_n = noise_slider.value
            seed = seed_slider.value

            if n_data < 2:
                set_opt_error("Need at least 2 data points")
                return

            kernel_func = get_kernel_func(kernel_type)

            # Generate training data (use slider noise for data generation)
            np.random.seed(seed)
            X_train = np.sort(np.random.uniform(-1, 1, n_data))
            y_train = target_function(X_train, func_type) + np.random.normal(0, sigma_n, n_data)

            # Objective function (optimize in log space for positivity)
            def objective(log_params):
                var, ell, noise = np.exp(log_params)
                return neg_log_marginal_likelihood(X_train, y_train, kernel_func, var, ell, noise)

            # Initial guess from slider values
            v0 = variance_slider.value
            l0 = lengthscale_slider.value
            n0 = noise_slider.value
            x0 = np.log([v0, l0, n0])

            # Optimize with bounds
            result = opt.minimize(
                objective, x0, method='L-BFGS-B',
                bounds=[(-2, 3), (-4, 1), (-4, 0)]  # log bounds
            )

            if result.success:
                opt_var, opt_ell, opt_noise = np.exp(result.x)
                set_opt_params((float(opt_var), float(opt_ell), float(opt_noise)))
            else:
                set_opt_error("Optimization failed")
        except Exception as e:
            set_opt_error(str(e))

    def reset_optimization(_):
        set_opt_params(None)
        set_opt_error(None)

    optimize_button = mo.ui.button(label="Optimize", on_click=run_optimization)
    reset_opt_button = mo.ui.button(label="Reset", on_click=reset_optimization)

    return get_opt_params, get_opt_error, optimize_button, reset_opt_button


@app.cell(hide_code=True)
def _(
    np, plt,
    target_function, get_kernel_func, gp_posterior,
    target_dropdown, kernel_dropdown, variance_slider, lengthscale_slider,
    n_data_slider, noise_slider, seed_slider, n_samples_slider,
    get_extra, get_opt_params,
):
    # Get parameter values
    func_type = target_dropdown.value
    kernel_type = kernel_dropdown.value
    n_data = max(0, n_data_slider.value + get_extra())
    seed = seed_slider.value
    n_samples = n_samples_slider.value

    # Use optimized params if available, otherwise use sliders
    opt_params = get_opt_params()
    if opt_params is not None:
        variance, lengthscale, sigma_n = opt_params
    else:
        variance = variance_slider.value
        lengthscale = lengthscale_slider.value
        sigma_n = noise_slider.value

    # Fixed y-axis limits
    y_min, y_max = -4, 4

    # Get kernel function
    kernel_func = get_kernel_func(kernel_type)

    # Generate training data
    np.random.seed(seed)
    if n_data > 0:
        X_train = np.sort(np.random.uniform(-1, 1, n_data))
        y_true_train = target_function(X_train, func_type)
        # Use slider noise for data generation, not optimized noise
        data_noise = noise_slider.value
        y_train = y_true_train + np.random.normal(0, data_noise, n_data)
    else:
        X_train = np.array([])
        y_train = np.array([])

    # Test points
    X_test = np.linspace(-1.2, 1.2, 200)
    y_true_test = target_function(X_test, func_type)

    # Create 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    axs = list(axs.flat)

    # Configure axes
    titles = ['Prior', 'Posterior', 'Posterior Samples', 'Uncertainty Breakdown']
    for ax, title in zip(axs, titles):
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=8)

    # Prior panel (top-left) - with prior samples
    ax = axs[0]
    prior_mean = np.zeros_like(X_test)
    prior_std = np.sqrt(variance) * np.ones_like(X_test)

    # Draw prior samples
    if n_samples > 0:
        K_prior = kernel_func(X_test, X_test, variance, lengthscale) + 1e-6 * np.eye(len(X_test))
        L_prior = np.linalg.cholesky(K_prior)
        np.random.seed(seed + 1000)  # Different seed for prior samples
        for _ in range(n_samples):
            z = np.random.randn(len(X_test))
            f_sample = prior_mean + L_prior @ z
            ax.plot(X_test, f_sample, 'C1-', alpha=0.4, lw=1)

    ax.plot(X_test, prior_mean, 'C0-', lw=2, label='Prior mean')
    ax.fill_between(X_test, prior_mean - 2*prior_std, prior_mean + 2*prior_std,
                   color='C0', alpha=0.2, label='$\\mu \\pm 2\\sigma$')
    ax.plot(X_test, y_true_test, 'k--', lw=1.5, label='True function')
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$f(x)$', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    # Posterior panel (top-right)
    ax = axs[1]

    if n_data > 0:
        f_mean, f_cov = gp_posterior(X_train, y_train, X_test, kernel_func, variance, lengthscale, sigma_n)
        f_std = np.sqrt(np.diag(f_cov))
        f_std_tot = np.sqrt(np.diag(f_cov) + sigma_n**2)
    else:
        f_mean = np.zeros_like(X_test)
        f_std = np.sqrt(variance) * np.ones_like(X_test)
        f_std_tot = np.sqrt(variance + sigma_n**2) * np.ones_like(X_test)

    ax.plot(X_test, f_mean, 'C0-', lw=2, label='Posterior mean')
    ax.fill_between(X_test, f_mean - 2*f_std, f_mean + 2*f_std,
                   color='C0', alpha=0.2, label='Epistemic')
    ax.fill_between(X_test, f_mean - 2*f_std_tot, f_mean - 2*f_std,
                   color='C1', alpha=0.2, label='Total')
    ax.fill_between(X_test, f_mean + 2*f_std, f_mean + 2*f_std_tot,
                   color='C1', alpha=0.2)
    ax.plot(X_test, y_true_test, 'k--', lw=1.5, label='True function')
    if n_data > 0:
        ax.scatter(X_train, y_train, c='red', s=50, zorder=5,
                  edgecolor='white', linewidth=1, label='Data')
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$f(x)$', fontsize=10)
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    # Posterior Samples panel (bottom-left)
    ax = axs[2]

    if n_data > 0:
        f_mean, f_cov = gp_posterior(X_train, y_train, X_test, kernel_func, variance, lengthscale, sigma_n)
        # Add jitter for numerical stability
        f_cov = f_cov + 1e-6 * np.eye(len(X_test))
    else:
        f_mean = np.zeros_like(X_test)
        f_cov = kernel_func(X_test, X_test, variance, lengthscale) + 1e-6 * np.eye(len(X_test))

    # Sample from posterior
    if n_samples > 0:
        L = np.linalg.cholesky(f_cov)
        np.random.seed(seed + 2000)  # Different seed for posterior samples
        for _ in range(n_samples):
            z = np.random.randn(len(X_test))
            f_sample = f_mean + L @ z
            ax.plot(X_test, f_sample, 'C1-', alpha=0.5, lw=1)

    ax.plot(X_test, f_mean, 'C0-', lw=2, label='Posterior mean')
    ax.plot(X_test, y_true_test, 'k--', lw=1.5, label='True function')
    if n_data > 0:
        ax.scatter(X_train, y_train, c='red', s=50, zorder=5,
                  edgecolor='white', linewidth=1, label='Data')
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$f(x)$', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.3)

    # Uncertainty Breakdown panel (bottom-right)
    ax = axs[3]

    if n_data > 0:
        f_mean, f_cov = gp_posterior(X_train, y_train, X_test, kernel_func, variance, lengthscale, sigma_n)
        epistemic = np.sqrt(np.diag(f_cov))
        aleatoric = sigma_n * np.ones_like(X_test)
        total = np.sqrt(np.diag(f_cov) + sigma_n**2)
    else:
        epistemic = np.sqrt(variance) * np.ones_like(X_test)
        aleatoric = sigma_n * np.ones_like(X_test)
        total = np.sqrt(variance + sigma_n**2) * np.ones_like(X_test)

    ax.fill_between(X_test, 0, epistemic, color='C0', alpha=0.5, label='Epistemic')
    ax.fill_between(X_test, epistemic, epistemic + aleatoric,
                   color='C1', alpha=0.5, label='Aleatoric')
    ax.plot(X_test, total, 'k-', lw=2, label='Total $\\sigma$')

    # Mark data locations
    if n_data > 0:
        for x_d in X_train:
            ax.axvline(x_d, color='red', alpha=0.3, lw=1)

    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('Uncertainty $\\sigma$', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0, 3.0)  # Fixed y-limit for uncertainty
    ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.0)
    gp_fig = fig
    return (gp_fig,)


@app.cell(hide_code=True)
def _(
    mo,
    target_dropdown, kernel_dropdown,
    variance_slider, lengthscale_slider,
    n_data_slider, noise_slider, seed_slider, n_samples_slider,
    add_button, reset_button, get_extra,
    optimize_button, reset_opt_button, get_opt_params, get_opt_error,
):
    # Function section
    func_section = mo.vstack([
        mo.Html("<h4>Function</h4>"),
        target_dropdown,
    ], gap="0.3em")

    # Kernel section with optimization
    _opt = get_opt_params()
    _err = get_opt_error()
    if _err is not None:
        _opt_info = mo.Html(f"<small style='color: #dc3545;'>Error: {_err}</small>")
    elif _opt is not None:
        _var, _ell, _noise = _opt
        _opt_info = mo.Html(f"<small style='color: #28a745;'>Optimized: v={_var:.3f}, ℓ={_ell:.3f}, σ={_noise:.3f}</small>")
    else:
        _opt_info = mo.Html("<small style='color: #6c757d;'>Using slider values</small>")

    kernel_section = mo.vstack([
        mo.Html("<h4>Kernel</h4>"),
        kernel_dropdown,
        variance_slider,
        lengthscale_slider,
        mo.hstack([optimize_button, reset_opt_button], gap="0.5em"),
        _opt_info,
    ], gap="0.3em")

    # Data section - show total count and buttons
    _total = max(0, n_data_slider.value + get_extra())
    data_section = mo.vstack([
        mo.Html("<h4>Data</h4>"),
        n_data_slider,
        mo.hstack([add_button, reset_button], gap="0.5em"),
        mo.Html(f"<small>Total: {_total} points</small>"),
        noise_slider,
        seed_slider,
    ], gap="0.3em")

    # Sampling section
    sampling_section = mo.vstack([
        mo.Html("<h4>Sampling</h4>"),
        n_samples_slider,
    ], gap="0.3em")

    sidebar = mo.vstack([func_section, kernel_section, data_section, sampling_section], gap="1em")

    sidebar_html = mo.Html(f'''
    <div class="app-sidebar">
        {sidebar}
    </div>
    ''')
    return (sidebar_html,)


@app.cell(hide_code=True)
def _(mo, header, gp_fig, sidebar_html):
    # Combined layout: header on top, plot on left, controls on right
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot">{mo.as_html(gp_fig)}</div>
        {sidebar_html}
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
