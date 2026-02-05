# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "pandas",
#     "numpy==2.2.5",
#     "scipy",
#     "pillow",
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
            flex-direction: column;
            justify-content: flex-start;
            align-items: stretch;
            z-index: 1;
            overflow: hidden;
        }

        .app-plot img,
        .app-plot svg {
            max-width: 100%;
            height: auto;
        }

        .app-sidebar-container {
            z-index: 10;
            position: relative;
            flex-shrink: 0;
            width: 300px;
        }

        .app-sidebar {
            display: flex;
            flex-direction: column;
            gap: 1.5em;
            padding: 1.5em;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            width: 100%;
        }

        @media (max-width: 768px) {
            .app-layout {
                flex-direction: column;
                height: auto;
                overflow-y: auto;
            }
            .app-plot {
                max-width: 100%;
                width: 100%;
            }
            .app-sidebar-container {
                width: 100%;
            }
            .app-sidebar {
                width: 100%;
            }
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
    import pandas as pd
    import altair as alt
    import scipy.linalg as sla
    import scipy.optimize as opt
    from scipy.stats import norm

    return alt, mo, np, norm, opt, pd, sla


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/wasm/gp-demo/')
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
    # State for custom code
    get_custom_code, set_custom_code = mo.state(
        "# Define y = f(x) where x is a numpy array\n"
        "# Available: np, math, x\n"
        "y = np.sin(2 * np.pi * x) + 0.5"
    )

    # Target function selection
    target_dropdown = mo.ui.dropdown(
        options={
            'Sine': 'sin',
            'Step': 'step',
            'Runge': 'runge',
            'Witch of Agnesi': 'witch',
            'Custom': 'custom',
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
    n_data_slider = mo.ui.slider(0, 50, 1, 0, label='$N$ random data points')
    noise_slider = mo.ui.slider(0.01, 0.5, 0.01, 0.2, label='Noise $\\sigma_n$')
    seed_slider = mo.ui.slider(0, 100, 1, 42, label='Random seed')

    # Click uncertainty slider - controls error bar size for clicked points
    click_uncertainty_slider = mo.ui.slider(0.05, 1.0, 0.05, 0.2, label='Data error $\\sigma_c$')

    # Sampling controls
    n_samples_slider = mo.ui.slider(0, 30, 1, 5, label='$N$ samples')

    # Custom function code editor and accordion
    custom_code_editor = mo.ui.code_editor(
        value=get_custom_code(),
        language="python",
        min_height=120,
        on_change=set_custom_code,
    )

    custom_function_accordion = mo.accordion({
        "Custom Function Code": mo.vstack([
            mo.md("Define `y = f(x)`. Available: `np`, `math`, `x` (numpy array)."),
            custom_code_editor,
        ])
    }, lazy=True)

    return (
        target_dropdown,
        kernel_dropdown,
        variance_slider,
        lengthscale_slider,
        n_data_slider,
        noise_slider,
        seed_slider,
        click_uncertainty_slider,
        n_samples_slider,
        get_custom_code,
        set_custom_code,
        custom_code_editor,
        custom_function_accordion,
    )


@app.cell(hide_code=True)
def _(mo):
    # State for clicked points: list of (x, y, error) tuples for heteroscedastic regression
    get_clicked_points, set_clicked_points = mo.state([])

    # State for optimized hyperparameters: None means use slider values
    get_opt_params, set_opt_params = mo.state(None)
    get_opt_error, set_opt_error = mo.state(None)

    return (
        get_clicked_points, set_clicked_points,
        get_opt_params, set_opt_params,
        get_opt_error, set_opt_error,
    )


@app.cell(hide_code=True)
def _(np):
    def target_function(x, func_type, custom_code=None):
        """Generate target function values."""
        if func_type == 'sin':
            return 0.5 + np.sin(2 * np.pi * x)
        elif func_type == 'step':
            return np.where(x < 0, -0.5, 0.5)
        elif func_type == 'runge':
            return 1.0 / (1.0 + 25.0 * x**2)
        elif func_type == 'witch':
            return 1.0 / (1.0 + x**2)
        elif func_type == 'custom' and custom_code:
            import math
            safe_builtins = {
                'range': range, 'len': len, 'sum': sum, 'min': min, 'max': max,
                'abs': abs, 'round': round, 'int': int, 'float': float,
                'True': True, 'False': False, 'None': None,
            }
            try:
                namespace = {'np': np, 'math': math, 'x': x}
                exec(custom_code, {"__builtins__": safe_builtins}, namespace)
                if 'y' not in namespace:
                    raise ValueError("Code must define 'y'")
                y = np.atleast_1d(np.asarray(namespace['y'], dtype=float))
                if y.shape != x.shape:
                    y = np.broadcast_to(y, x.shape).copy()
                return y
            except Exception:
                return np.full_like(x, np.nan, dtype=float)
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
    def gp_posterior_hetero(X, y, errors, X_star, kernel_func, variance, lengthscale):
        """Compute GP posterior with heteroscedastic (per-point) noise."""
        N_data = len(X)

        # Compute kernel matrices
        K = kernel_func(X, X, variance, lengthscale)
        K_star = kernel_func(X_star, X_star, variance, lengthscale)
        k_star = kernel_func(X, X_star, variance, lengthscale)

        # Heteroscedastic noise: diagonal matrix with per-point variances
        noise_var = np.diag(errors**2)

        # Add heteroscedastic noise and compute Cholesky
        L = np.linalg.cholesky(K + noise_var)

        # Solve for alpha
        alpha = sla.solve_triangular(L.T, sla.solve_triangular(L, y, lower=True))

        # Posterior mean
        f_mean = k_star.T @ alpha

        # Posterior covariance
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

    return gp_posterior_hetero, neg_log_marginal_likelihood


@app.cell(hide_code=True)
def _(np, pd):
    # Static click grid for point selection (100x100 = 10,000 points)
    _gx = np.linspace(-1.2, 1.2, 100)
    _gy = np.linspace(-4, 4, 100)
    click_grid_df = pd.DataFrame(
        [(x, y) for x in _gx for y in _gy],
        columns=['x', 'y']
    )
    return (click_grid_df,)


@app.cell(hide_code=True)
def _(
    mo, np, opt,
    target_function, get_kernel_func, neg_log_marginal_likelihood,
    target_dropdown, kernel_dropdown,
    variance_slider, lengthscale_slider,
    n_data_slider, noise_slider, seed_slider,
    get_clicked_points, set_clicked_points,
    get_opt_params, set_opt_params, get_opt_error, set_opt_error,
    get_custom_code,
):
    def run_optimization(_):
        """Optimize hyperparameters to maximize marginal likelihood."""
        try:
            set_opt_error(None)
            func_type = target_dropdown.value
            kernel_type = kernel_dropdown.value
            n_data = n_data_slider.value
            sigma_n = noise_slider.value
            seed = seed_slider.value
            clicked = get_clicked_points() or []
            custom_code = get_custom_code() if func_type == 'custom' else None

            # Combine slider-generated and clicked data
            if n_data > 0:
                np.random.seed(seed)
                X_slider = np.sort(np.random.uniform(-1, 1, n_data))
                y_slider = target_function(X_slider, func_type, custom_code) + np.random.normal(0, sigma_n, n_data)
            else:
                X_slider = np.array([])
                y_slider = np.array([])

            if clicked:
                clicked_arr = np.array(clicked)
                X_clicked = clicked_arr[:, 0]
                y_clicked = clicked_arr[:, 1]
            else:
                X_clicked = np.array([])
                y_clicked = np.array([])

            X_train = np.concatenate([X_slider, X_clicked]) if len(X_slider) > 0 or len(X_clicked) > 0 else np.array([])
            y_train = np.concatenate([y_slider, y_clicked]) if len(y_slider) > 0 or len(y_clicked) > 0 else np.array([])

            if len(X_train) < 2:
                set_opt_error("Need at least 2 data points")
                return

            kernel_func = get_kernel_func(kernel_type)

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

    def clear_clicked_points(_):
        set_clicked_points([])

    optimize_button = mo.ui.button(label="Optimize", on_click=run_optimization)
    reset_opt_button = mo.ui.button(label="Reset Opt", on_click=reset_optimization)
    clear_points_button = mo.ui.button(label="Clear Points", on_click=clear_clicked_points)

    return optimize_button, reset_opt_button, clear_points_button


@app.cell(hide_code=True)
def _(
    alt, np, pd, mo, norm,
    target_function, get_kernel_func, gp_posterior_hetero,
    target_dropdown, kernel_dropdown, variance_slider, lengthscale_slider,
    n_data_slider, noise_slider, seed_slider, n_samples_slider,
    click_uncertainty_slider,
    get_clicked_points, get_opt_params,
    click_grid_df,
    get_custom_code,
):
    # Get parameter values
    func_type = target_dropdown.value
    kernel_type = kernel_dropdown.value
    n_data = n_data_slider.value
    seed = seed_slider.value
    n_samples = n_samples_slider.value
    click_uncertainty = click_uncertainty_slider.value
    custom_code = get_custom_code() if func_type == 'custom' else None

    # Use optimized params if available, otherwise use sliders
    opt_params = get_opt_params()
    if opt_params is not None:
        variance, lengthscale, sigma_n = opt_params
    else:
        variance = variance_slider.value
        lengthscale = lengthscale_slider.value
        sigma_n = noise_slider.value

    # Fixed axis limits
    x_min, x_max = -1.2, 1.2
    y_min, y_max = -4, 4

    # Get kernel function
    kernel_func = get_kernel_func(kernel_type)

    # Generate slider-based training data (homoscedastic noise)
    np.random.seed(seed)
    if n_data > 0:
        X_slider = np.sort(np.random.uniform(-1, 1, n_data))
        y_true_slider = target_function(X_slider, func_type, custom_code)
        data_noise = noise_slider.value
        y_slider = y_true_slider + np.random.normal(0, data_noise, n_data)
        errors_slider = np.full(n_data, data_noise)  # Homoscedastic
    else:
        X_slider = np.array([])
        y_slider = np.array([])
        errors_slider = np.array([])

    # Get clicked points (heteroscedastic: x, y, error)
    clicked_points = get_clicked_points() or []
    if clicked_points:
        clicked_arr = np.array(clicked_points)
        X_clicked = clicked_arr[:, 0]
        y_clicked = clicked_arr[:, 1]
        errors_clicked = clicked_arr[:, 2]
    else:
        X_clicked = np.array([])
        y_clicked = np.array([])
        errors_clicked = np.array([])

    # Combine all training data
    all_X = []
    all_y = []
    all_errors = []

    if len(X_slider) > 0:
        all_X.append(X_slider)
        all_y.append(y_slider)
        all_errors.append(errors_slider)

    if len(X_clicked) > 0:
        all_X.append(X_clicked)
        all_y.append(y_clicked)
        all_errors.append(errors_clicked)

    if all_X:
        X_train = np.concatenate(all_X)
        y_train = np.concatenate(all_y)
        errors_train = np.concatenate(all_errors)
    else:
        X_train = np.array([])
        y_train = np.array([])
        errors_train = np.array([])

    total_points = len(X_train)

    # Test points for plotting
    X_test = np.linspace(x_min, x_max, 200)
    y_true_test = target_function(X_test, func_type, custom_code)

    # Compute GP posterior (heteroscedastic if we have any data)
    if total_points > 0:
        # Use heteroscedastic GP
        f_mean, f_cov = gp_posterior_hetero(X_train, y_train, errors_train, X_test, kernel_func, variance, lengthscale)
        # For total uncertainty, use mean of error variances as representative noise
        mean_noise = np.mean(errors_train)
    else:
        f_mean = np.zeros_like(X_test)
        f_cov = kernel_func(X_test, X_test, variance, lengthscale)
        mean_noise = sigma_n

    f_std = np.sqrt(np.diag(f_cov))
    f_std_tot = np.sqrt(np.diag(f_cov) + mean_noise**2)

    # --- Compute metrics ---
    # Test metrics: compare GP predictions to ground truth on test grid
    z_test = (y_true_test - f_mean) / f_std_tot
    rmse_test = np.sqrt(np.mean((f_mean - y_true_test)**2))
    mae_test = np.mean(np.abs(f_mean - y_true_test))
    # CRPS for Gaussian: σ * [z*(2*Φ(z) - 1) + 2*φ(z) - 1/√π]
    crps_test = np.mean(f_std_tot * (z_test * (2 * norm.cdf(z_test) - 1) + 2 * norm.pdf(z_test) - 1 / np.sqrt(np.pi)))
    # Log likelihood: log p(y | μ, σ)
    ll_test = np.mean(-0.5 * np.log(2 * np.pi) - np.log(f_std_tot) - 0.5 * z_test**2)

    # Training metrics: compare GP predictions at training locations to observed values
    if total_points > 0:
        # Compute GP prediction at training points (leave-one-out would be better but this is simpler)
        f_mean_train, f_cov_train = gp_posterior_hetero(X_train, y_train, errors_train, X_train, kernel_func, variance, lengthscale)
        f_std_train = np.sqrt(np.diag(f_cov_train) + np.mean(errors_train)**2)

        z_train = (y_train - f_mean_train) / f_std_train
        rmse_train = np.sqrt(np.mean((f_mean_train - y_train)**2))
        mae_train = np.mean(np.abs(f_mean_train - y_train))
        crps_train = np.mean(f_std_train * (z_train * (2 * norm.cdf(z_train) - 1) + 2 * norm.pdf(z_train) - 1 / np.sqrt(np.pi)))
        ll_train = np.mean(-0.5 * np.log(2 * np.pi) - np.log(f_std_train) - 0.5 * z_train**2)
    else:
        rmse_train = np.nan
        mae_train = np.nan
        crps_train = np.nan
        ll_train = np.nan

    # Generate samples from posterior
    samples_data = []
    if n_samples > 0:
        f_cov_jitter = f_cov + 1e-6 * np.eye(len(X_test))

        L = np.linalg.cholesky(f_cov_jitter)
        np.random.seed(seed + 2000)
        for i in range(n_samples):
            z = np.random.randn(len(X_test))
            f_sample = f_mean + L @ z
            for x_val, y_val in zip(X_test, f_sample):
                samples_data.append({'x': x_val, 'y': y_val, 'sample': f'Sample {i+1}'})

    # Build DataFrames
    gt_df = pd.DataFrame({'x': X_test, 'y': y_true_test})
    mean_df = pd.DataFrame({'x': X_test, 'y': f_mean})

    # Uncertainty bands
    band_df = pd.DataFrame({
        'x': X_test,
        'y_mean': f_mean,
        'y_lower_ep': f_mean - 2 * f_std,
        'y_upper_ep': f_mean + 2 * f_std,
        'y_lower_tot': f_mean - 2 * f_std_tot,
        'y_upper_tot': f_mean + 2 * f_std_tot,
    })

    # Slider-generated data points (blue) - with error bars
    if len(X_slider) > 0:
        slider_data_df = pd.DataFrame({
            'x': X_slider,
            'y': y_slider,
            'y_lower': y_slider - errors_slider,
            'y_upper': y_slider + errors_slider,
        })
    else:
        slider_data_df = pd.DataFrame(columns=['x', 'y', 'y_lower', 'y_upper'])

    # Clicked data points (red) - with custom error bars
    if len(X_clicked) > 0:
        clicked_data_df = pd.DataFrame({
            'x': X_clicked,
            'y': y_clicked,
            'y_lower': y_clicked - errors_clicked,
            'y_upper': y_clicked + errors_clicked,
        })
    else:
        clicked_data_df = pd.DataFrame(columns=['x', 'y', 'y_lower', 'y_upper'])

    # Samples
    samples_df = pd.DataFrame(samples_data) if samples_data else pd.DataFrame(columns=['x', 'y', 'sample'])

    # Define scales
    x_scale = alt.Scale(domain=[x_min, x_max])
    y_scale = alt.Scale(domain=[y_min, y_max])

    # Click selection for adding points
    click_select = alt.selection_point(on='click', nearest=True, fields=['x', 'y'], name='click_select')

    # Build chart layers
    # Total uncertainty band (outer, lighter)
    total_band = alt.Chart(band_df).mark_area(
        opacity=0.15, color='#ff7f0e'
    ).encode(
        x=alt.X('x:Q', scale=x_scale, title='x'),
        y=alt.Y('y_lower_tot:Q', scale=y_scale, title='f(x)'),
        y2='y_upper_tot:Q',
    )

    # Epistemic uncertainty band (inner, darker)
    epistemic_band = alt.Chart(band_df).mark_area(
        opacity=0.3, color='#1f77b4'
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y_lower_ep:Q', scale=y_scale),
        y2='y_upper_ep:Q',
    )

    # Ground truth line (dashed)
    gt_line = alt.Chart(gt_df).mark_line(
        color='black', strokeWidth=3, strokeDash=[5, 5], opacity=0.7
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale),
    )

    # Posterior mean line (solid)
    mean_line = alt.Chart(mean_df).mark_line(
        color='#1f77b4', strokeWidth=3
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale),
    )

    # Build layers list
    layers = [total_band, epistemic_band]

    # Samples (if any)
    if len(samples_df) > 0:
        samples_layer = alt.Chart(samples_df).mark_line(
            strokeWidth=1.5, opacity=0.4
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
            color=alt.Color('sample:N', legend=None)
        )
        layers.append(samples_layer)

    layers.append(mean_line)
    layers.append(gt_line)

    # Slider-generated data points (blue) with error bars
    if len(slider_data_df) > 0:
        slider_errorbars = alt.Chart(slider_data_df).mark_rule(
            color='#1f77b4', strokeWidth=2, opacity=0.6
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y_lower:Q', scale=y_scale),
            y2='y_upper:Q',
        )
        slider_points = alt.Chart(slider_data_df).mark_circle(
            color='#1f77b4', size=120, opacity=0.8
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
        )
        layers.append(slider_errorbars)
        layers.append(slider_points)

    # Clicked data points (red) with error bars
    if len(clicked_data_df) > 0:
        clicked_errorbars = alt.Chart(clicked_data_df).mark_rule(
            color='#d62728', strokeWidth=3, opacity=0.8
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y_lower:Q', scale=y_scale),
            y2='y_upper:Q',
        )
        clicked_points_layer = alt.Chart(clicked_data_df).mark_circle(
            color='#d62728', size=150, opacity=0.9
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
        )
        layers.append(clicked_errorbars)
        layers.append(clicked_points_layer)

    # Create manual 2-column legend
    clicked_legend_label = 'Clicked'

    # 2-column layout: [col, row] positions
    legend_data = pd.DataFrame({
        'label': ['Ground Truth', 'Posterior Mean', 'Epistemic ±2σ', 'Total ±2σ', 'Random Data', clicked_legend_label],
        'type': ['line', 'line', 'area', 'area', 'point', 'point'],
        'color': ['black', '#1f77b4', '#1f77b4', '#ff7f0e', '#1f77b4', '#d62728'],
        'dash': ['dashed', 'solid', 'solid', 'solid', 'solid', 'solid'],
        'col': [0, 1, 0, 1, 0, 1],  # Column 0 or 1
        'row': [0, 0, 1, 1, 2, 2],  # Row 0, 1, or 2
    })

    # Legend positioning
    legend_right = x_max - 0.03
    legend_top = y_max - 0.25
    row_spacing = 0.55
    col_width = 0.58
    col1_symbol_x = legend_right - 2 * col_width + 0.05
    col2_symbol_x = legend_right - col_width + 0.05

    legend_items_df = legend_data.copy()
    legend_items_df['x'] = legend_items_df['col'].apply(lambda c: col1_symbol_x if c == 0 else col2_symbol_x)
    legend_items_df['y'] = legend_items_df['row'].apply(lambda r: legend_top - r * row_spacing)
    legend_items_df['x_text'] = legend_items_df['x'] + 0.12

    n_rows = 3
    # Legend background
    legend_bg = alt.Chart(pd.DataFrame({
        'x': [col1_symbol_x - 0.1], 'x2': [legend_right],
        'y': [legend_top + 0.32], 'y2': [legend_top - (n_rows - 1) * row_spacing - 0.32]
    })).mark_rect(fill='white', stroke='#ccc', strokeWidth=1, cornerRadius=4, opacity=0.95).encode(
        x=alt.X('x:Q', scale=x_scale), x2='x2:Q',
        y=alt.Y('y:Q', scale=y_scale), y2='y2:Q'
    )
    layers.append(legend_bg)

    # Legend lines (for line items)
    line_legend_df = legend_items_df[legend_items_df['type'] == 'line'].copy()
    for _, row in line_legend_df.iterrows():
        dash_pattern = [5, 5] if row['dash'] == 'dashed' else []
        line_seg = alt.Chart(pd.DataFrame({
            'x': [row['x'] - 0.06, row['x'] + 0.06],
            'y': [row['y'], row['y']]
        })).mark_line(
            color=row['color'], strokeWidth=2.5, strokeDash=dash_pattern,
            opacity=0.8 if row['dash'] == 'dashed' else 1.0
        ).encode(x=alt.X('x:Q', scale=x_scale), y=alt.Y('y:Q', scale=y_scale))
        layers.append(line_seg)

    # Legend squares (for area items)
    area_legend_df = legend_items_df[legend_items_df['type'] == 'area'].copy()
    for _, row in area_legend_df.iterrows():
        opacity = 0.5 if row['label'] == 'Epistemic ±2σ' else 0.3
        area_sym = alt.Chart(pd.DataFrame({
            'x': [row['x'] - 0.045], 'x2': [row['x'] + 0.045],
            'y': [row['y'] - 0.16], 'y2': [row['y'] + 0.16]
        })).mark_rect(fill=row['color'], opacity=opacity, stroke=row['color'], strokeWidth=1).encode(
            x=alt.X('x:Q', scale=x_scale), x2='x2:Q',
            y=alt.Y('y:Q', scale=y_scale), y2='y2:Q'
        )
        layers.append(area_sym)

    # Legend circles with error bars (for point items)
    point_legend_df = legend_items_df[legend_items_df['type'] == 'point'].copy()
    # Add error bar sizes: Random Data uses noise slider, Clicked uses click_uncertainty
    data_noise = noise_slider.value
    point_legend_df = point_legend_df.copy()
    point_legend_df['error'] = point_legend_df['label'].apply(
        lambda lbl: data_noise if lbl == 'Random Data' else click_uncertainty
    )
    # Scale error bars for legend display (map data units to reasonable legend size)
    error_scale = 0.6  # Scale factor for visibility in legend
    point_legend_df['y_lower'] = point_legend_df['y'] - point_legend_df['error'] * error_scale
    point_legend_df['y_upper'] = point_legend_df['y'] + point_legend_df['error'] * error_scale

    # Error bars
    for _, row in point_legend_df.iterrows():
        errbar = alt.Chart(pd.DataFrame({
            'x': [row['x']],
            'y_lower': [row['y_lower']],
            'y_upper': [row['y_upper']]
        })).mark_rule(color=row['color'], strokeWidth=2, opacity=0.7).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y_lower:Q', scale=y_scale),
            y2='y_upper:Q'
        )
        layers.append(errbar)

    # Points on top of error bars
    point_sym = alt.Chart(point_legend_df).mark_circle(size=50, opacity=0.9).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale),
        color=alt.Color('color:N', scale=None, legend=None)
    )
    layers.append(point_sym)

    # Legend text labels
    legend_text = alt.Chart(legend_items_df).mark_text(
        align='left', fontSize=9, font='sans-serif'
    ).encode(
        x=alt.X('x_text:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale),
        text='label:N'
    )
    layers.append(legend_text)

    # Invisible click layer (must be on top for interaction)
    click_layer = alt.Chart(click_grid_df).mark_point(
        opacity=0, size=100
    ).encode(
        x=alt.X('x:Q', scale=x_scale),
        y=alt.Y('y:Q', scale=y_scale)
    ).add_params(click_select)
    layers.append(click_layer)

    # Combine layers
    chart = alt.layer(*layers).properties(
        width='container', height=450,
        title='Gaussian Process Posterior'
    ).configure_axis(
        grid=True, gridOpacity=0.3,
        labelFontSize=14, titleFontSize=16
    ).configure_title(
        fontSize=18
    ).configure_legend(
        labelFontSize=12,
        symbolSize=100
    )

    interactive_chart = mo.ui.altair_chart(chart)

    # Count points for display
    n_slider_points = len(X_slider)
    n_clicked_points = len(X_clicked)

    # Bundle metrics for table
    metrics = {
        'rmse_train': rmse_train, 'rmse_test': rmse_test,
        'mae_train': mae_train, 'mae_test': mae_test,
        'crps_train': crps_train, 'crps_test': crps_test,
        'll_train': ll_train, 'll_test': ll_test,
    }

    return interactive_chart, total_points, n_slider_points, n_clicked_points, click_uncertainty, metrics


@app.cell(hide_code=True)
def _(interactive_chart):
    # Pass-through display cell
    chart_display = interactive_chart
    return (chart_display,)


@app.cell(hide_code=True)
def _(mo, np, metrics):
    # Format metric value
    def fmt(v):
        return '—' if np.isnan(v) else f'{v:.4f}'

    # Metrics table
    metrics_table = mo.Html(f'''
    <table style="width: 100%; border-collapse: collapse; font-size: 13px; max-width: 400px; margin: 0 auto;">
        <thead>
            <tr style="border-bottom: 2px solid #dee2e6;">
                <th style="text-align: left; padding: 6px 8px;">Metric</th>
                <th style="text-align: right; padding: 6px 8px;">Train</th>
                <th style="text-align: right; padding: 6px 8px;">Test</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px 8px;">RMSE</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(metrics['rmse_train'])}</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(metrics['rmse_test'])}</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px 8px;">MAE</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(metrics['mae_train'])}</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(metrics['mae_test'])}</td>
            </tr>
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 6px 8px;">CRPS</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(metrics['crps_train'])}</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(metrics['crps_test'])}</td>
            </tr>
            <tr>
                <td style="padding: 6px 8px;">Log Lik.</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(metrics['ll_train'])}</td>
                <td style="text-align: right; padding: 6px 8px; font-family: monospace;">{fmt(metrics['ll_test'])}</td>
            </tr>
        </tbody>
    </table>
    ''')
    return (metrics_table,)


@app.cell(hide_code=True)
def _(interactive_chart, click_grid_df, get_clicked_points, set_clicked_points, click_uncertainty, pd):
    # Click handler - reads selection, updates state with current uncertainty
    _current = get_clicked_points() or []
    _filtered = interactive_chart.apply_selection(click_grid_df)

    # Check if valid DataFrame
    if _filtered is not None and isinstance(_filtered, pd.DataFrame) and len(_filtered) > 0 and len(_filtered) < len(click_grid_df):
        # User clicked on a point
        _new_x = float(_filtered['x'].iloc[0])
        _new_y = float(_filtered['y'].iloc[0])
        _new_point = (_new_x, _new_y, click_uncertainty)
        # Only add if x position not already present (avoid duplicates at same x)
        _existing_x = [p[0] for p in _current]
        if _new_x not in _existing_x:
            set_clicked_points(_current + [_new_point])

    # Return nothing to avoid circular deps
    return ()


@app.cell(hide_code=True)
def _(
    mo,
    target_dropdown, kernel_dropdown,
    variance_slider, lengthscale_slider,
    n_data_slider, noise_slider, seed_slider, n_samples_slider,
    click_uncertainty_slider,
    optimize_button, reset_opt_button, clear_points_button,
    get_clicked_points, get_opt_params, get_opt_error,
    n_slider_points, n_clicked_points,
    custom_function_accordion,
):
    # Function section
    func_section = mo.vstack([
        mo.Html("<h4>Function</h4>"),
        target_dropdown,
        custom_function_accordion,
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

    # Data section
    _n_clicked = len(get_clicked_points() or [])

    data_section = mo.vstack([
        mo.Html("<h4>Data</h4>"),
        n_data_slider,
        noise_slider,
        seed_slider,
        mo.Html("<h4>Click to Add</h4>"),
        click_uncertainty_slider,
        mo.hstack([clear_points_button], gap="0.5em"),
        mo.Html(f"<small>Random: {n_slider_points} | Clicked: {_n_clicked}</small>"),
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
        <p style="font-size: 0.85em; color: #666; margin-top: 1em;">
            <b>Tip:</b> Click on plot to add points (red). Adjust uncertainty slider to change error bar size for new points. Brush indicator in top-right shows current size.
        </p>
    </div>
    ''')
    return (sidebar_html,)


@app.cell(hide_code=True)
def _(mo, header, chart_display, sidebar_html, metrics_table):
    # Combined layout: header on top, plot on left, controls on right
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot">
            {mo.as_html(chart_display)}
            <div style="margin-top: 1em;">
                {mo.as_html(metrics_table)}
            </div>
        </div>
        <div class="app-sidebar-container">
            {sidebar_html}
        </div>
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
