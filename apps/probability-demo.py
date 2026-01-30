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
    import scipy.stats as st

    import seaborn as sns
    sns.set_context('talk')
    return mo, np, plt, sns, st


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/probability-demo.html')
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
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>Probability Distributions Demo</b>
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
    # Distribution selection
    dist_dropdown = mo.ui.dropdown(
        options={
            'Normal (Gaussian)': 'normal',
            'Uniform': 'uniform',
            'Exponential': 'exponential',
            'Beta': 'beta',
            'Gamma': 'gamma',
        },
        value='Normal (Gaussian)',
        label='Distribution'
    )

    # Normal distribution parameters
    mu_slider = mo.ui.slider(-3.0, 3.0, 0.1, 0.0, label='Mean $\\mu$')
    sigma_slider = mo.ui.slider(0.1, 3.0, 0.1, 1.0, label='Std dev $\\sigma$')

    # Uniform distribution parameters
    a_slider = mo.ui.slider(-3.0, 2.0, 0.1, -1.0, label='Lower bound $a$')
    b_slider = mo.ui.slider(-2.0, 3.0, 0.1, 1.0, label='Upper bound $b$')

    # Exponential distribution parameter
    lambda_slider = mo.ui.slider(0.1, 3.0, 0.1, 1.0, label='Rate $\\lambda$')

    # Beta distribution parameters
    alpha_slider = mo.ui.slider(0.1, 5.0, 0.1, 2.0, label='$\\alpha$')
    beta_slider = mo.ui.slider(0.1, 5.0, 0.1, 2.0, label='$\\beta$')

    # Gamma distribution parameters
    k_slider = mo.ui.slider(0.1, 5.0, 0.1, 2.0, label='Shape $k$')
    theta_slider = mo.ui.slider(0.1, 3.0, 0.1, 1.0, label='Scale $\\theta$')

    # Panel visibility checkboxes
    show_pdf = mo.ui.checkbox(True, label='<b>PDF</b>')
    show_cdf = mo.ui.checkbox(True, label='<b>CDF</b>')
    show_samples = mo.ui.checkbox(False, label='<b>Samples</b>')

    # Number of samples
    n_samples_slider = mo.ui.slider(10, 1000, 10, 100, label='$N$ samples')

    return (
        dist_dropdown,
        mu_slider,
        sigma_slider,
        a_slider,
        b_slider,
        lambda_slider,
        alpha_slider,
        beta_slider,
        k_slider,
        theta_slider,
        show_pdf,
        show_cdf,
        show_samples,
        n_samples_slider,
    )


@app.cell(hide_code=True)
def _(st):
    def get_distribution(dist_type, params):
        """Return scipy.stats distribution object based on type and parameters."""
        if dist_type == 'normal':
            return st.norm(loc=params['mu'], scale=params['sigma'])
        elif dist_type == 'uniform':
            a, b = params['a'], params['b']
            return st.uniform(loc=a, scale=b - a)
        elif dist_type == 'exponential':
            return st.expon(scale=1.0 / params['lambda'])
        elif dist_type == 'beta':
            return st.beta(params['alpha'], params['beta'])
        elif dist_type == 'gamma':
            return st.gamma(params['k'], scale=params['theta'])
        return st.norm(0, 1)

    def get_x_range(dist_type):
        """Return fixed x range for each distribution type."""
        # Use fixed ranges so effect of parameter changes is clearly visible
        if dist_type == 'normal':
            return -10, 10
        elif dist_type == 'uniform':
            return -4, 4
        elif dist_type == 'exponential':
            return 0, 8
        elif dist_type == 'beta':
            return -0.05, 1.05
        elif dist_type == 'gamma':
            return 0, 15
        return -5, 5

    def get_y_max(dist_type):
        """Return fixed y-axis max for PDF panel."""
        if dist_type == 'normal':
            return 1.5
        elif dist_type == 'uniform':
            return 1.5
        elif dist_type == 'exponential':
            return 3.5
        elif dist_type == 'beta':
            return 4.0
        elif dist_type == 'gamma':
            return 1.0
        return 1.5

    def get_dist_label(dist_type, params):
        """Return LaTeX label for the distribution."""
        if dist_type == 'normal':
            return rf"$\mathcal{{N}}({params['mu']:.1f}, {params['sigma']:.1f}^2)$"
        elif dist_type == 'uniform':
            return rf"$\mathrm{{Uniform}}({params['a']:.1f}, {params['b']:.1f})$"
        elif dist_type == 'exponential':
            return rf"$\mathrm{{Exponential}}({params['lambda']:.1f})$"
        elif dist_type == 'beta':
            return rf"$\mathrm{{Beta}}({params['alpha']:.1f}, {params['beta']:.1f})$"
        elif dist_type == 'gamma':
            return rf"$\mathrm{{Gamma}}({params['k']:.1f}, {params['theta']:.1f})$"
        return ""
    return get_distribution, get_x_range, get_y_max, get_dist_label


@app.cell(hide_code=True)
def _(
    np, plt,
    get_distribution, get_x_range, get_y_max, get_dist_label,
    dist_dropdown, mu_slider, sigma_slider,
    a_slider, b_slider, lambda_slider,
    alpha_slider, beta_slider, k_slider, theta_slider,
    show_pdf, show_cdf, show_samples,
    n_samples_slider
):
    # Get current distribution type
    dist_type = dist_dropdown.value

    # Build parameters dict based on distribution type
    params = {}
    if dist_type == 'normal':
        params = {'mu': mu_slider.value, 'sigma': sigma_slider.value}
    elif dist_type == 'uniform':
        # Ensure a < b
        a_val = min(a_slider.value, b_slider.value - 0.1)
        b_val = max(b_slider.value, a_slider.value + 0.1)
        params = {'a': a_val, 'b': b_val}
    elif dist_type == 'exponential':
        params = {'lambda': lambda_slider.value}
    elif dist_type == 'beta':
        params = {'alpha': alpha_slider.value, 'beta': beta_slider.value}
    elif dist_type == 'gamma':
        params = {'k': k_slider.value, 'theta': theta_slider.value}

    # Get distribution and x range
    dist = get_distribution(dist_type, params)
    x_min, x_max = get_x_range(dist_type)
    x = np.linspace(x_min, x_max, 500)
    dist_label = get_dist_label(dist_type, params)

    # Count active panels
    active_panels = sum([show_pdf.value, show_cdf.value, show_samples.value, True])  # Stats always shown
    if active_panels == 0:
        active_panels = 1  # Default to at least one panel

    # Create 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    axs = list(axs.flat)

    # Configure all axes with titles
    titles = ['PDF', 'CDF', 'Samples', 'Statistics']
    for ax, title in zip(axs, titles):
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=8)

    # PDF panel (top-left)
    if show_pdf.value:
        ax = axs[0]
        pdf_vals = dist.pdf(x)
        ax.plot(x, pdf_vals, 'C0-', lw=2, label=dist_label)
        ax.fill_between(x, 0, pdf_vals, alpha=0.3)
        ax.set_xlabel('$x$', fontsize=10)
        ax.set_ylabel('$f_X(x)$', fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, get_y_max(dist_type))
        ax.grid(True, alpha=0.3)

    # CDF panel (top-right)
    if show_cdf.value:
        ax = axs[1]
        cdf_vals = dist.cdf(x)
        ax.plot(x, cdf_vals, 'C1-', lw=2, label=dist_label)
        ax.set_xlabel('$x$', fontsize=10)
        ax.set_ylabel('$F_X(x)$', fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)

    # Samples panel (bottom-left)
    if show_samples.value:
        ax = axs[2]
        n_samples = n_samples_slider.value
        samples = dist.rvs(size=n_samples)
        ax.hist(samples, bins=30, density=True, alpha=0.7, color='C2', edgecolor='white')
        # Overlay PDF for comparison
        ax.plot(x, dist.pdf(x), 'k--', lw=1.5, label='PDF')
        ax.set_xlabel('$x$', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, get_y_max(dist_type))
        ax.grid(True, alpha=0.3)

    # Statistics panel (bottom-right) - always shown
    ax = axs[3]
    ax.axis('off')

    # Compute statistics
    mean = dist.mean()
    var = dist.var()
    std = dist.std()
    median = dist.median()

    stats_text = (
        f"Mean: $\\mu = {mean:.3f}$\n\n"
        f"Variance: $\\sigma^2 = {var:.3f}$\n\n"
        f"Std Dev: $\\sigma = {std:.3f}$\n\n"
        f"Median: ${median:.3f}$"
    )

    ax.text(0.1, 0.8, stats_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout(pad=1.0)
    prob_fig = fig
    return (prob_fig,)


@app.cell(hide_code=True)
def _(
    mo,
    dist_dropdown,
    mu_slider, sigma_slider,
    a_slider, b_slider,
    lambda_slider,
    alpha_slider, beta_slider,
    k_slider, theta_slider,
    show_pdf, show_cdf, show_samples,
    n_samples_slider
):
    # Get current distribution type to show relevant sliders
    current_dist = dist_dropdown.value

    # Distribution selection section
    dist_section = mo.vstack([
        mo.Html("<h4>Distribution</h4>"),
        dist_dropdown,
    ], gap="0.3em")

    # Parameters section - show relevant sliders based on distribution
    param_widgets = []
    if current_dist == 'normal':
        param_widgets = [mu_slider, sigma_slider]
    elif current_dist == 'uniform':
        param_widgets = [a_slider, b_slider]
    elif current_dist == 'exponential':
        param_widgets = [lambda_slider]
    elif current_dist == 'beta':
        param_widgets = [alpha_slider, beta_slider]
    elif current_dist == 'gamma':
        param_widgets = [k_slider, theta_slider]

    param_section = mo.vstack([
        mo.Html("<h4>Parameters</h4>"),
        *param_widgets,
    ], gap="0.3em")

    # Panels section
    panel_section = mo.vstack([
        mo.Html("<h4>Panels</h4>"),
        show_pdf,
        show_cdf,
        show_samples,
    ], gap="0.2em")

    # Samples section (only show when samples panel is active)
    samples_section = mo.vstack([
        mo.Html("<h4>Sampling</h4>"),
        n_samples_slider,
    ], gap="0.3em") if show_samples.value else mo.Html("")

    sidebar = mo.vstack([dist_section, param_section, panel_section, samples_section], gap="1em")

    sidebar_html = mo.Html(f'''
    <div class="app-sidebar">
        {sidebar}
    </div>
    ''')
    return (sidebar_html,)


@app.cell(hide_code=True)
def _(mo, header, prob_fig, sidebar_html):
    # Combined layout: header on top, plot on left, controls on right
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot">{mo.as_html(prob_fig)}</div>
        {sidebar_html}
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
