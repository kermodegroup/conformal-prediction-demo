# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "pandas",
#     "numpy==2.2.5",
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
            justify-content: flex-start;
            gap: 1em;
            padding: 1em;
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

    return alt, mo, np, pd


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://kermodegroup.github.io/demos/kernel-demo.html')
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
            'Matérn 3/2': 'matern32',
            'Matérn 5/2': 'matern52',
            'Rational Quadratic': 'rq',
        },
        value='Squared Exponential (RBF)',
        label='Kernel Type'
    )

    # Kernel hyperparameters (debounce=True reduces flickering)
    variance_slider = mo.ui.slider(0.1, 5.0, 0.1, 1.0, label='Variance $v$', debounce=True)
    lengthscale_slider = mo.ui.slider(0.1, 2.0, 0.05, 0.5, label='Lengthscale $\\ell$', debounce=True)
    alpha_slider = mo.ui.slider(0.1, 5.0, 0.1, 1.0, label='Alpha $\\alpha$', debounce=True)

    # Sampling controls
    n_samples_slider = mo.ui.slider(1, 10, 1, 5, label='$N$ samples', debounce=True)
    seed_slider = mo.ui.slider(0, 100, 1, 42, label='Random seed', debounce=True)

    return (
        kernel_dropdown,
        variance_slider,
        lengthscale_slider,
        alpha_slider,
        n_samples_slider,
        seed_slider,
    )


@app.cell(hide_code=True)
def _(np):
    def rbf_kernel(X1, X2, variance, lengthscale):
        """Squared exponential (RBF) kernel."""
        X1 = np.atleast_1d(X1).reshape(-1, 1)
        X2 = np.atleast_1d(X2).reshape(-1, 1)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return variance * np.exp(-0.5 * sqdist / lengthscale**2)

    def matern32_kernel(X1, X2, variance, lengthscale):
        """Matérn 3/2 kernel."""
        X1 = np.atleast_1d(X1).reshape(-1, 1)
        X2 = np.atleast_1d(X2).reshape(-1, 1)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        r = np.sqrt(np.maximum(sqdist, 1e-12)) / lengthscale
        sqrt3 = np.sqrt(3.0)
        return variance * (1 + sqrt3 * r) * np.exp(-sqrt3 * r)

    def matern52_kernel(X1, X2, variance, lengthscale):
        """Matérn 5/2 kernel."""
        X1 = np.atleast_1d(X1).reshape(-1, 1)
        X2 = np.atleast_1d(X2).reshape(-1, 1)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        r = np.sqrt(np.maximum(sqdist, 1e-12)) / lengthscale
        sqrt5 = np.sqrt(5.0)
        return variance * (1 + sqrt5 * r + 5.0/3.0 * r**2) * np.exp(-sqrt5 * r)

    def rq_kernel(X1, X2, variance, lengthscale, alpha):
        """Rational Quadratic kernel."""
        X1 = np.atleast_1d(X1).reshape(-1, 1)
        X2 = np.atleast_1d(X2).reshape(-1, 1)
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return variance * (1 + sqdist / (2 * alpha * lengthscale**2)) ** (-alpha)

    def get_kernel_func(kernel_type):
        """Return kernel function based on type."""
        if kernel_type == 'rbf':
            return rbf_kernel
        elif kernel_type == 'matern32':
            return matern32_kernel
        elif kernel_type == 'matern52':
            return matern52_kernel
        elif kernel_type == 'rq':
            return rq_kernel
        return rbf_kernel

    def get_kernel_label(kernel_type):
        """Return display label for kernel type."""
        labels = {
            'rbf': 'RBF',
            'matern32': 'Matérn 3/2',
            'matern52': 'Matérn 5/2',
            'rq': 'Rational Quadratic',
        }
        return labels.get(kernel_type, 'RBF')

    return (
        rbf_kernel, matern32_kernel, matern52_kernel, rq_kernel,
        get_kernel_func, get_kernel_label
    )


@app.cell(hide_code=True)
def _(mo):
    # State for clicked x position on kernel plot
    get_clicked_x, set_clicked_x = mo.state(None)
    return get_clicked_x, set_clicked_x


@app.cell(hide_code=True)
def _(
    alt, np, pd, mo,
    get_kernel_func, get_kernel_label,
    kernel_dropdown, variance_slider, lengthscale_slider, alpha_slider,
    n_samples_slider, seed_slider,
    get_clicked_x,
):
    # Get parameter values
    kernel_type = kernel_dropdown.value
    variance = variance_slider.value
    lengthscale = lengthscale_slider.value
    alpha = alpha_slider.value
    n_samples = n_samples_slider.value
    seed = seed_slider.value

    np.random.seed(seed)

    kernel_func = get_kernel_func(kernel_type)
    kernel_label = get_kernel_label(kernel_type)

    # Kernel plot parameters
    x_kernel = np.linspace(-4, 4, 200)
    if kernel_type == 'rq':
        k_vals = kernel_func(x_kernel, np.array([0.0]), variance, lengthscale, alpha)[:, 0]
    else:
        k_vals = kernel_func(x_kernel, np.array([0.0]), variance, lengthscale)[:, 0]

    kernel_df = pd.DataFrame({
        'x': x_kernel,
        'k': k_vals,
    })

    # Covariance matrix parameters
    N_cov = 50
    X_cov = np.linspace(-2, 2, N_cov)
    if kernel_type == 'rq':
        K_cov = kernel_func(X_cov, X_cov, variance, lengthscale, alpha)
    else:
        K_cov = kernel_func(X_cov, X_cov, variance, lengthscale)

    # Flatten covariance matrix for Altair heatmap
    cov_data = []
    for i in range(N_cov):
        for j in range(N_cov):
            cov_data.append({
                'x': X_cov[j],
                'y': X_cov[i],
                'k': K_cov[i, j],
                'row_idx': i,
                'col_idx': j,
            })
    cov_df = pd.DataFrame(cov_data)

    # GP samples
    N_sample = 100
    X_sample = np.linspace(-2, 2, N_sample)
    if kernel_type == 'rq':
        K_sample = kernel_func(X_sample, X_sample, variance, lengthscale, alpha)
    else:
        K_sample = kernel_func(X_sample, X_sample, variance, lengthscale)
    K_sample = K_sample + 1e-6 * np.eye(N_sample)
    L = np.linalg.cholesky(K_sample)

    samples_data = []
    for s in range(n_samples):
        z = np.random.randn(N_sample)
        sample = L @ z
        for xi, yi in zip(X_sample, sample):
            samples_data.append({'x': xi, 'y': yi, 'sample': s})
    samples_df = pd.DataFrame(samples_data)

    # Mean and uncertainty band data
    mean_df = pd.DataFrame({
        'x': X_sample,
        'mean': np.zeros(N_sample),
        'upper': 2 * np.sqrt(variance),
        'lower': -2 * np.sqrt(variance),
    })

    # Click grid for kernel plot selection
    click_grid_df = pd.DataFrame({'x': np.linspace(-4, 4, 200)})

    # Get clicked x position
    clicked_x = get_clicked_x()

    # Compute highlighted row index if clicked
    highlighted_row_idx = None
    if clicked_x is not None:
        # Map clicked_x to covariance matrix row
        # X_cov ranges from -2 to 2, so we need to clamp clicked_x
        clamped_x = np.clip(clicked_x, -2, 2)
        highlighted_row_idx = int(np.round((clamped_x - (-2)) / (2 - (-2)) * (N_cov - 1)))

    # === Build Altair Charts ===

    # Click selection
    click_select = alt.selection_point(on='click', nearest=True, fields=['x'], name='click_select')

    # --- Kernel Plot k(x, 0) ---
    kernel_x_scale = alt.Scale(domain=[-4, 4])
    kernel_y_scale = alt.Scale(domain=[0, variance * 1.1])

    kernel_line = alt.Chart(kernel_df).mark_line(
        color='steelblue', strokeWidth=2
    ).encode(
        x=alt.X('x:Q', scale=kernel_x_scale, title='x'),
        y=alt.Y('k:Q', scale=kernel_y_scale, title='k(x, 0)'),
        tooltip=[
            alt.Tooltip('x:Q', format='.2f', title='x'),
            alt.Tooltip('k:Q', format='.4f', title='k(x, 0)'),
        ]
    )

    kernel_area = alt.Chart(kernel_df).mark_area(
        opacity=0.3, color='steelblue'
    ).encode(
        x=alt.X('x:Q', scale=kernel_x_scale),
        y=alt.Y('k:Q', scale=kernel_y_scale),
    )

    # Zero reference lines
    zero_h_df = pd.DataFrame({'y': [0]})
    zero_v_df = pd.DataFrame({'x': [0]})
    zero_h = alt.Chart(zero_h_df).mark_rule(strokeDash=[4, 4], color='gray', strokeWidth=2).encode(
        y=alt.Y('y:Q', scale=kernel_y_scale)
    )
    zero_v = alt.Chart(zero_v_df).mark_rule(strokeDash=[4, 4], color='gray', strokeWidth=2).encode(
        x=alt.X('x:Q', scale=kernel_x_scale)
    )

    kernel_layers = [kernel_area, kernel_line, zero_h, zero_v]

    # Vertical rule and point marker at clicked position
    if clicked_x is not None:
        rule_df = pd.DataFrame({'x': [clicked_x]})
        clicked_rule = alt.Chart(rule_df).mark_rule(
            strokeDash=[4, 4], color='red', strokeWidth=2
        ).encode(
            x=alt.X('x:Q', scale=kernel_x_scale)
        )
        kernel_layers.append(clicked_rule)

        # Point marker
        if kernel_type == 'rq':
            clicked_k = kernel_func(np.array([clicked_x]), np.array([0.0]), variance, lengthscale, alpha)[0, 0]
        else:
            clicked_k = kernel_func(np.array([clicked_x]), np.array([0.0]), variance, lengthscale)[0, 0]
        point_df = pd.DataFrame({'x': [clicked_x], 'k': [clicked_k]})
        clicked_point = alt.Chart(point_df).mark_circle(
            color='red', size=100
        ).encode(
            x=alt.X('x:Q', scale=kernel_x_scale),
            y=alt.Y('k:Q', scale=kernel_y_scale)
        )
        kernel_layers.append(clicked_point)

    # Invisible click layer
    click_layer = alt.Chart(click_grid_df).mark_point(
        opacity=0, size=1
    ).encode(
        x=alt.X('x:Q', scale=kernel_x_scale)
    ).add_params(click_select)
    kernel_layers.append(click_layer)

    kernel_chart = alt.layer(*kernel_layers).properties(
        width='container', height=210,
        title=f'{kernel_label} Kernel k(x, 0)'
    ).configure_axis(
        grid=True, gridOpacity=0.3,
        labelFontSize=14, titleFontSize=16
    ).configure_title(
        fontSize=18
    )

    # --- Covariance Matrix Heatmap ---
    cov_heatmap = alt.Chart(cov_df).mark_rect().encode(
        x=alt.X('col_idx:O', title='x', axis=alt.Axis(
            values=list(range(0, N_cov, 10)),
            labelExpr=f"datum.value == 0 ? '-2' : datum.value == {N_cov//2} ? '0' : datum.value == {N_cov-1} ? '2' : ''"
        )),
        y=alt.Y('row_idx:O', title="x'", axis=alt.Axis(
            values=list(range(0, N_cov, 10)),
            labelExpr=f"datum.value == 0 ? '-2' : datum.value == {N_cov//2} ? '0' : datum.value == {N_cov-1} ? '2' : ''"
        )),
        color=alt.Color('k:Q', scale=alt.Scale(scheme='viridis'), title="k(x, x')"),
        tooltip=[
            alt.Tooltip('x:Q', format='.2f', title='x'),
            alt.Tooltip('y:Q', format='.2f', title="x'"),
            alt.Tooltip('k:Q', format='.4f', title="k(x, x')"),
        ]
    )

    cov_layers = [cov_heatmap]

    # Highlight row at clicked position
    if highlighted_row_idx is not None:
        highlight_df = cov_df[cov_df['row_idx'] == highlighted_row_idx]
        highlight = alt.Chart(highlight_df).mark_rect(
            stroke='red', strokeWidth=2, fill='transparent'
        ).encode(
            x=alt.X('col_idx:O'),
            y=alt.Y('row_idx:O'),
        )
        cov_layers.append(highlight)

    cov_chart = alt.layer(*cov_layers).properties(
        width='container', height=210,
        title="Covariance Matrix K(x, x')"
    ).configure_axis(
        labelFontSize=14, titleFontSize=16
    ).configure_title(
        fontSize=18
    ).configure_legend(
        titleFontSize=14, labelFontSize=12
    )

    # --- GP Samples Chart ---
    gp_x_scale = alt.Scale(domain=[-2, 2])
    gp_y_scale = alt.Scale(domain=[-4, 4])

    # Sample lines
    sample_lines = alt.Chart(samples_df).mark_line(
        opacity=0.6, strokeWidth=2
    ).encode(
        x=alt.X('x:Q', scale=gp_x_scale, title='x'),
        y=alt.Y('y:Q', scale=gp_y_scale, title='f(x)'),
        color=alt.Color('sample:N', legend=None, scale=alt.Scale(scheme='category10')),
    )

    # Mean line
    mean_line = alt.Chart(mean_df).mark_line(
        color='steelblue', strokeWidth=2
    ).encode(
        x=alt.X('x:Q', scale=gp_x_scale),
        y=alt.Y('mean:Q', scale=gp_y_scale),
    )

    # Uncertainty band
    uncertainty_band = alt.Chart(mean_df).mark_area(
        opacity=0.2, color='steelblue'
    ).encode(
        x=alt.X('x:Q', scale=gp_x_scale),
        y=alt.Y('lower:Q', scale=gp_y_scale),
        y2='upper:Q',
    )

    gp_chart = alt.layer(uncertainty_band, sample_lines, mean_line).properties(
        width='container', height=250,
        title='GP Prior Samples f(x) ~ GP(0, k)'
    ).configure_axis(
        grid=True, gridOpacity=0.3,
        labelFontSize=14, titleFontSize=16
    ).configure_title(
        fontSize=18
    )

    # === Combine Charts ===
    kernel_interactive = mo.ui.altair_chart(kernel_chart)

    # Store data for click handler
    current_clicked_x = clicked_x
    current_variance = variance
    current_lengthscale = lengthscale

    return (
        kernel_interactive, cov_chart, gp_chart,
        click_grid_df, current_clicked_x, current_variance, current_lengthscale,
        kernel_type, highlighted_row_idx
    )


@app.cell(hide_code=True)
def _(kernel_interactive, click_grid_df, set_clicked_x):
    # Click handler - reads selection, updates state
    _filtered = kernel_interactive.apply_selection(click_grid_df)

    if len(_filtered) > 0 and len(_filtered) < len(click_grid_df):
        _new_x = float(_filtered['x'].iloc[0])
        set_clicked_x(_new_x)

    return ()


@app.cell(hide_code=True)
def _(
    mo,
    kernel_dropdown, variance_slider, lengthscale_slider, alpha_slider,
    n_samples_slider, seed_slider,
    kernel_type,
):
    # Kernel section
    kernel_widgets = [kernel_dropdown, variance_slider, lengthscale_slider]
    if kernel_type == 'rq':
        kernel_widgets.append(alpha_slider)

    kernel_section = mo.vstack([
        mo.Html("<h4>Kernel</h4>"),
        *kernel_widgets,
    ], gap="0.3em")

    # Sampling section
    sampling_section = mo.vstack([
        mo.Html("<h4>Sampling</h4>"),
        n_samples_slider,
        seed_slider,
    ], gap="0.3em")

    sidebar_content = mo.vstack([kernel_section, sampling_section], gap="1em")

    return (sidebar_content,)


@app.cell(hide_code=True)
def _(
    mo, header,
    kernel_interactive, cov_chart, gp_chart,
    sidebar_content,
    current_clicked_x, current_variance, current_lengthscale,
    kernel_type, get_kernel_func, get_kernel_label,
    alpha_slider, highlighted_row_idx, np
):
    # Build statistics table
    kernel_label_display = get_kernel_label(kernel_type)

    stats_rows = f'''
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>Kernel</b></td>
            <td style="padding: 4px 12px; text-align: right;">{kernel_label_display}</td>
        </tr>
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>Variance (v)</b></td>
            <td style="padding: 4px 12px; text-align: right;">{current_variance:.2f}</td>
        </tr>
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>Lengthscale (ℓ)</b></td>
            <td style="padding: 4px 12px; text-align: right;">{current_lengthscale:.2f}</td>
        </tr>
    '''

    if kernel_type == 'rq':
        stats_rows += f'''
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>Alpha (α)</b></td>
            <td style="padding: 4px 12px; text-align: right;">{alpha_slider.value:.2f}</td>
        </tr>
        '''

    # Add clicked point info if available
    if current_clicked_x is not None:
        _kernel_func = get_kernel_func(kernel_type)
        if kernel_type == 'rq':
            clicked_k_val = _kernel_func(np.array([current_clicked_x]), np.array([0.0]), current_variance, current_lengthscale, alpha_slider.value)[0, 0]
        else:
            clicked_k_val = _kernel_func(np.array([current_clicked_x]), np.array([0.0]), current_variance, current_lengthscale)[0, 0]

        stats_rows += f'''
        <tr style="border-top: 1px solid #dee2e6;">
            <td colspan="2" style="padding: 8px 0 4px 0; color: #666;"><i>At x₀ = {current_clicked_x:.3f}</i></td>
        </tr>
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>k(x₀, 0)</b></td>
            <td style="padding: 4px 12px; text-align: right; color: red;">{clicked_k_val:.4f}</td>
        </tr>
        '''
        if highlighted_row_idx is not None:
            stats_rows += f'''
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>Cov row</b></td>
            <td style="padding: 4px 12px; text-align: right; color: red;">{highlighted_row_idx}</td>
        </tr>
            '''
    stats_table = f'''
        <table style="border-collapse: collapse; font-size: 13px; width: 100%;">
            {stats_rows}
        </table>
    '''

    # Build chart layout
    # Kernel plot and covariance matrix side by side, GP samples below
    cov_html = mo.as_html(cov_chart)
    gp_html = mo.as_html(gp_chart)

    charts_layout = mo.vstack([
        mo.hstack([kernel_interactive, cov_html], gap="1em", align="start", widths="equal"),
        gp_html,
    ], gap="1em")

    # Combined layout
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot">
            {mo.as_html(charts_layout)}
        </div>
        <div class="app-sidebar-container">
            <div class="app-sidebar">
                {sidebar_content}
            </div>
            <div style="margin-top: 1em; padding: 0.75em; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
                <div style="font-size: 0.9em; font-weight: bold; color: #495057; margin-bottom: 0.5em;">Statistics</div>
                {stats_table}
            </div>
            <div style="margin-top: 1em; padding: 0.5em 0.75em; color: #666; font-size: 12px;">
                <b>Tip:</b> Click the kernel plot to select x₀ and highlight the corresponding row in the covariance matrix.
            </div>
        </div>
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
