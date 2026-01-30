# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.5",
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

    import seaborn as sns
    sns.set_context('talk')
    return mo, np, plt, sns


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/kernel-demo.html')
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
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>Kernel Explorer</b>
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
    # Kernel type selection
    kernel_dropdown = mo.ui.dropdown(
        options={
            'Squared Exponential (RBF)': 'rbf',
            'Matern 3/2': 'matern32',
            'Matern 5/2': 'matern52',
            'Rational Quadratic': 'rq',
        },
        value='Squared Exponential (RBF)',
        label='Kernel Type'
    )

    # Kernel hyperparameters
    variance_slider = mo.ui.slider(0.1, 5.0, 0.1, 1.0, label='Variance $v$')
    lengthscale_slider = mo.ui.slider(0.1, 2.0, 0.05, 0.5, label='Lengthscale $\\ell$')
    alpha_slider = mo.ui.slider(0.1, 5.0, 0.1, 1.0, label='Alpha $\\alpha$')

    # 2D kernel parameters
    ell1_slider = mo.ui.slider(0.1, 2.0, 0.05, 0.5, label='$\\ell_1$')
    ell2_slider = mo.ui.slider(0.1, 2.0, 0.05, 0.5, label='$\\ell_2$')

    # Sampling controls
    n_samples_slider = mo.ui.slider(1, 50, 1, 10, label='$N$ samples')
    seed_slider = mo.ui.slider(0, 100, 1, 42, label='Random seed')

    # View selector
    view_radio = mo.ui.radio(
        options={"1D Kernel": "1d", "2D Kernel": "2d"},
        value="1D Kernel",
        label="View"
    )

    return (
        kernel_dropdown,
        variance_slider,
        lengthscale_slider,
        alpha_slider,
        ell1_slider,
        ell2_slider,
        n_samples_slider,
        seed_slider,
        view_radio,
    )


@app.cell(hide_code=True)
def _(np):
    def rbf_kernel(X1, X2, variance, lengthscale):
        """Squared exponential (RBF) kernel."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        if X1.shape[0] == 1 or (X1.ndim == 2 and X1.shape[1] == 1 and len(X1.shape) == 2):
            X1 = X1.T if X1.shape[0] == 1 else X1
        if X2.shape[0] == 1 or (X2.ndim == 2 and X2.shape[1] == 1 and len(X2.shape) == 2):
            X2 = X2.T if X2.shape[0] == 1 else X2
        # Handle 1D case
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return variance * np.exp(-0.5 * sqdist / lengthscale**2)

    def matern32_kernel(X1, X2, variance, lengthscale):
        """Matern 3/2 kernel."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        r = np.sqrt(np.maximum(sqdist, 1e-12)) / lengthscale
        sqrt3 = np.sqrt(3.0)
        return variance * (1 + sqrt3 * r) * np.exp(-sqrt3 * r)

    def matern52_kernel(X1, X2, variance, lengthscale):
        """Matern 5/2 kernel."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        r = np.sqrt(np.maximum(sqdist, 1e-12)) / lengthscale
        sqrt5 = np.sqrt(5.0)
        return variance * (1 + sqrt5 * r + 5.0/3.0 * r**2) * np.exp(-sqrt5 * r)

    def rq_kernel(X1, X2, variance, lengthscale, alpha):
        """Rational Quadratic kernel."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        if X1.ndim == 1:
            X1 = X1.reshape(-1, 1)
        if X2.ndim == 1:
            X2 = X2.reshape(-1, 1)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return variance * (1 + sqdist / (2 * alpha * lengthscale**2)) ** (-alpha)

    def rbf_kernel_2d(X1, X2, variance, ell1, ell2):
        """2D RBF kernel with anisotropic lengthscales."""
        # Scale inputs by lengthscales
        X1_scaled = X1 / np.array([ell1, ell2])
        X2_scaled = X2 / np.array([ell1, ell2])
        sqdist = np.sum(X1_scaled**2, 1).reshape(-1, 1) + np.sum(X2_scaled**2, 1) - 2 * X1_scaled @ X2_scaled.T
        return variance * np.exp(-0.5 * sqdist)

    def matern32_kernel_2d(X1, X2, variance, ell1, ell2):
        """2D Matern 3/2 kernel with anisotropic lengthscales."""
        X1_scaled = X1 / np.array([ell1, ell2])
        X2_scaled = X2 / np.array([ell1, ell2])
        sqdist = np.sum(X1_scaled**2, 1).reshape(-1, 1) + np.sum(X2_scaled**2, 1) - 2 * X1_scaled @ X2_scaled.T
        r = np.sqrt(np.maximum(sqdist, 1e-12))
        sqrt3 = np.sqrt(3.0)
        return variance * (1 + sqrt3 * r) * np.exp(-sqrt3 * r)

    def matern52_kernel_2d(X1, X2, variance, ell1, ell2):
        """2D Matern 5/2 kernel with anisotropic lengthscales."""
        X1_scaled = X1 / np.array([ell1, ell2])
        X2_scaled = X2 / np.array([ell1, ell2])
        sqdist = np.sum(X1_scaled**2, 1).reshape(-1, 1) + np.sum(X2_scaled**2, 1) - 2 * X1_scaled @ X2_scaled.T
        r = np.sqrt(np.maximum(sqdist, 1e-12))
        sqrt5 = np.sqrt(5.0)
        return variance * (1 + sqrt5 * r + 5.0/3.0 * r**2) * np.exp(-sqrt5 * r)

    def rq_kernel_2d(X1, X2, variance, ell1, ell2, alpha):
        """2D Rational Quadratic kernel with anisotropic lengthscales."""
        X1_scaled = X1 / np.array([ell1, ell2])
        X2_scaled = X2 / np.array([ell1, ell2])
        sqdist = np.sum(X1_scaled**2, 1).reshape(-1, 1) + np.sum(X2_scaled**2, 1) - 2 * X1_scaled @ X2_scaled.T
        return variance * (1 + sqdist / (2 * alpha)) ** (-alpha)

    def get_kernel_func(kernel_type):
        """Return 1D kernel function based on type."""
        if kernel_type == 'rbf':
            return rbf_kernel
        elif kernel_type == 'matern32':
            return matern32_kernel
        elif kernel_type == 'matern52':
            return matern52_kernel
        elif kernel_type == 'rq':
            return rq_kernel
        return rbf_kernel

    def get_kernel_2d_func(kernel_type):
        """Return 2D kernel function based on type."""
        if kernel_type == 'rbf':
            return rbf_kernel_2d
        elif kernel_type == 'matern32':
            return matern32_kernel_2d
        elif kernel_type == 'matern52':
            return matern52_kernel_2d
        elif kernel_type == 'rq':
            return rq_kernel_2d
        return rbf_kernel_2d

    return (
        rbf_kernel, matern32_kernel, matern52_kernel, rq_kernel,
        rbf_kernel_2d, matern32_kernel_2d, matern52_kernel_2d, rq_kernel_2d,
        get_kernel_func, get_kernel_2d_func
    )


@app.cell(hide_code=True)
def _(
    np, plt,
    get_kernel_func, get_kernel_2d_func,
    kernel_dropdown, variance_slider, lengthscale_slider, alpha_slider,
    ell1_slider, ell2_slider, n_samples_slider, seed_slider,
    view_radio
):
    from matplotlib.gridspec import GridSpec

    # Get parameter values
    kernel_type = kernel_dropdown.value
    variance = variance_slider.value
    lengthscale = lengthscale_slider.value
    alpha = alpha_slider.value
    ell1 = ell1_slider.value
    ell2 = ell2_slider.value
    n_samples = n_samples_slider.value
    seed = seed_slider.value
    current_view = view_radio.value

    np.random.seed(seed)

    if current_view == '2d':
        # 2D Kernel view
        fig = plt.figure(figsize=(7, 7))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

        N = 25  # Grid resolution for 2D
        x_2d = np.linspace(-2, 2, N)
        X1, X2 = np.meshgrid(x_2d, x_2d)
        X_2d = np.column_stack([X1.ravel(), X2.ravel()])

        # Get 2D kernel function
        kernel_2d_func = get_kernel_2d_func(kernel_type)

        # Top-left: 2D kernel k(x, 0)
        ax0 = fig.add_subplot(gs[0, 0])
        origin = np.array([[0.0, 0.0]])
        if kernel_type == 'rq':
            K_2d = kernel_2d_func(X_2d, origin, variance, ell1, ell2, alpha).reshape(N, N)
        else:
            K_2d = kernel_2d_func(X_2d, origin, variance, ell1, ell2).reshape(N, N)
        contour = ax0.contourf(x_2d, x_2d, K_2d, levels=20, cmap='viridis')
        ax0.set_xlabel('$x_1$', fontsize=10)
        ax0.set_ylabel('$x_2$', fontsize=10)
        ax0.set_title('2D Kernel $k(\\mathbf{x}, \\mathbf{0})$', fontsize=11)
        ax0.set_aspect('equal')
        ax0.tick_params(labelsize=8)
        plt.colorbar(contour, ax=ax0, shrink=0.8)

        # Top-right: Covariance matrix
        ax1 = fig.add_subplot(gs[0, 1])
        if kernel_type == 'rq':
            K_cov_2d = kernel_2d_func(X_2d, X_2d, variance, ell1, ell2, alpha)
        else:
            K_cov_2d = kernel_2d_func(X_2d, X_2d, variance, ell1, ell2)
        im = ax1.imshow(K_cov_2d, cmap='viridis', origin='upper')
        ax1.set_xlabel('Grid point $i$', fontsize=10)
        ax1.set_ylabel('Grid point $j$', fontsize=10)
        ax1.set_title(f'Covariance Matrix ({N*N}x{N*N})', fontsize=11)
        ax1.tick_params(labelsize=8)
        plt.colorbar(im, ax=ax1, shrink=0.8)

        # Bottom: GP sample
        ax2 = fig.add_subplot(gs[1, :])
        K_sample = K_cov_2d + 1e-6 * np.eye(N*N)
        L = np.linalg.cholesky(K_sample)
        z = np.random.randn(N*N)
        sample_2d = (L @ z).reshape(N, N)

        contour_sample = ax2.contourf(x_2d, x_2d, sample_2d, levels=20, cmap='RdBu_r')
        ax2.set_xlabel('$x_1$', fontsize=10)
        ax2.set_ylabel('$x_2$', fontsize=10)
        ax2.set_title('GP Sample $f(\\mathbf{x}) \\sim \\mathcal{GP}$', fontsize=11)
        ax2.set_aspect('equal')
        ax2.tick_params(labelsize=8)
        plt.colorbar(contour_sample, ax=ax2, shrink=0.5)

    else:
        # 1D Kernel view
        fig = plt.figure(figsize=(7, 7))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

        kernel_func = get_kernel_func(kernel_type)

        # 1D Kernel (top-left)
        ax0 = fig.add_subplot(gs[0, 0])
        x = np.linspace(-4, 4, 200)
        if kernel_type == 'rq':
            k_vals = kernel_func(x, np.array([0.0]), variance, lengthscale, alpha)[:, 0]
        else:
            k_vals = kernel_func(x, np.array([0.0]), variance, lengthscale)[:, 0]
        ax0.plot(x, k_vals, 'C0-', lw=2)
        ax0.axhline(0, color='gray', lw=0.5, ls='--')
        ax0.axvline(0, color='gray', lw=0.5, ls='--')
        ax0.set_xlabel('$x$', fontsize=10)
        ax0.set_ylabel('$k(x, 0)$', fontsize=10)
        ax0.set_title('1D Kernel', fontsize=11)
        ax0.set_ylim(0, 2)
        ax0.tick_params(labelsize=8)
        ax0.grid(True, alpha=0.3)

        # Covariance Matrix (top-right)
        ax1 = fig.add_subplot(gs[0, 1])
        N_cov = 50
        X_cov = np.linspace(-2, 2, N_cov)
        if kernel_type == 'rq':
            K_cov = kernel_func(X_cov, X_cov, variance, lengthscale, alpha)
        else:
            K_cov = kernel_func(X_cov, X_cov, variance, lengthscale)
        im = ax1.imshow(K_cov, cmap='viridis', origin='upper',
                       extent=[-2, 2, 2, -2])
        ax1.set_xlabel('$x$', fontsize=10)
        ax1.set_ylabel("$x'$", fontsize=10)
        ax1.set_title('Covariance Matrix', fontsize=11)
        ax1.tick_params(labelsize=8)
        plt.colorbar(im, ax=ax1, shrink=0.8)

        # GP Samples (bottom, spanning both columns)
        ax2 = fig.add_subplot(gs[1, :])
        N_sample = 100
        X_sample = np.linspace(-2, 2, N_sample)
        if kernel_type == 'rq':
            K_sample = kernel_func(X_sample, X_sample, variance, lengthscale, alpha)
        else:
            K_sample = kernel_func(X_sample, X_sample, variance, lengthscale)
        K_sample = K_sample + 1e-6 * np.eye(N_sample)
        L = np.linalg.cholesky(K_sample)

        # Sample from the GP
        for _ in range(n_samples):
            z = np.random.randn(N_sample)
            sample = L @ z
            ax2.plot(X_sample, sample, 'C1-', alpha=0.5, lw=1)

        # Plot mean and uncertainty
        mean = np.zeros(N_sample)
        std = np.sqrt(variance) * np.ones(N_sample)
        ax2.plot(X_sample, mean, 'C0-', lw=2, label='Mean')
        ax2.fill_between(X_sample, mean - 2*std, mean + 2*std,
                       color='C0', alpha=0.2, label='$\\mu \\pm 2\\sigma$')
        ax2.set_xlabel('$x$', fontsize=10)
        ax2.set_ylabel('$f(x)$', fontsize=10)
        ax2.set_title('GP Prior Samples', fontsize=11)
        ax2.legend(fontsize=8)
        ax2.tick_params(labelsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout(pad=1.0)
    kernel_fig = fig
    return (kernel_fig,)


@app.cell(hide_code=True)
def _(
    mo,
    kernel_dropdown, variance_slider, lengthscale_slider, alpha_slider,
    ell1_slider, ell2_slider, n_samples_slider, seed_slider,
    view_radio
):
    # Get current kernel type and view
    _current_kernel = kernel_dropdown.value
    _current_view = view_radio.value

    # View selector section
    view_section = mo.vstack([
        view_radio,
    ], gap="0.3em")

    # Kernel section
    kernel_widgets = [kernel_dropdown, variance_slider, lengthscale_slider]
    if _current_kernel == 'rq':
        kernel_widgets.append(alpha_slider)

    kernel_section = mo.vstack([
        mo.Html("<h4>Kernel</h4>"),
        *kernel_widgets,
    ], gap="0.3em")

    # Build sidebar based on current view
    if _current_view == '2d':
        # 2D view: show 2D lengthscales and seed
        kernel_2d_section = mo.vstack([
            mo.Html("<h4>2D Lengthscales</h4>"),
            ell1_slider,
            ell2_slider,
        ], gap="0.3em")
        sampling_section = mo.vstack([
            mo.Html("<h4>Sampling</h4>"),
            seed_slider,
        ], gap="0.3em")
        sidebar = mo.vstack([view_section, kernel_section, kernel_2d_section, sampling_section], gap="1em")
    else:
        # 1D view: show sampling controls
        sampling_section = mo.vstack([
            mo.Html("<h4>Sampling</h4>"),
            n_samples_slider,
            seed_slider,
        ], gap="0.3em")
        sidebar = mo.vstack([view_section, kernel_section, sampling_section], gap="1em")

    sidebar_html = mo.Html(f'''
    <div class="app-sidebar">
        {sidebar}
    </div>
    ''')
    return (sidebar_html,)


@app.cell(hide_code=True)
def _(mo, header, kernel_fig, sidebar_html):
    # Combined layout: header on top, plot on left, controls on right
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot">{mo.as_html(kernel_fig)}</div>
        {sidebar_html}
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
