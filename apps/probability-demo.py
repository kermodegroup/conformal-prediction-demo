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
    import scipy.stats as st

    return alt, mo, np, pd, st


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/wasm/probability-demo/')
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

    # Normal distribution parameters (debounce=True reduces flickering)
    mu_slider = mo.ui.slider(-3.0, 3.0, 0.1, 0.0, label='Mean $\\mu$', debounce=True)
    sigma_slider = mo.ui.slider(0.1, 3.0, 0.1, 1.0, label='Std dev $\\sigma$', debounce=True)

    # Uniform distribution parameters
    a_slider = mo.ui.slider(-3.0, 2.0, 0.1, -1.0, label='Lower bound $a$', debounce=True)
    b_slider = mo.ui.slider(-2.0, 3.0, 0.1, 1.0, label='Upper bound $b$', debounce=True)

    # Exponential distribution parameter
    lambda_slider = mo.ui.slider(0.1, 3.0, 0.1, 1.0, label='Rate $\\lambda$', debounce=True)

    # Beta distribution parameters
    alpha_slider = mo.ui.slider(0.1, 5.0, 0.1, 2.0, label='$\\alpha$', debounce=True)
    beta_slider = mo.ui.slider(0.1, 5.0, 0.1, 2.0, label='$\\beta$', debounce=True)

    # Gamma distribution parameters
    k_slider = mo.ui.slider(0.1, 5.0, 0.1, 2.0, label='Shape $k$', debounce=True)
    theta_slider = mo.ui.slider(0.1, 3.0, 0.1, 1.0, label='Scale $\\theta$', debounce=True)

    # Panel visibility checkboxes
    show_pdf = mo.ui.checkbox(True, label='<span style="color: steelblue;"><b>PDF</b></span>')
    show_cdf = mo.ui.checkbox(True, label='<span style="color: orange;"><b>CDF</b></span>')
    show_samples = mo.ui.checkbox(False, label='<span style="color: green;"><b>Samples</b></span>')

    # Number of samples
    n_samples_slider = mo.ui.slider(10, 1000, 10, 100, label='$N$ samples', debounce=True)

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
        """Return fixed y-axis max for PDF."""
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
        """Return label for the distribution."""
        if dist_type == 'normal':
            return f"N({params['mu']:.1f}, {params['sigma']:.1f}²)"
        elif dist_type == 'uniform':
            return f"Uniform({params['a']:.1f}, {params['b']:.1f})"
        elif dist_type == 'exponential':
            return f"Exponential({params['lambda']:.1f})"
        elif dist_type == 'beta':
            return f"Beta({params['alpha']:.1f}, {params['beta']:.1f})"
        elif dist_type == 'gamma':
            return f"Gamma({params['k']:.1f}, {params['theta']:.1f})"
        return ""

    return get_distribution, get_x_range, get_y_max, get_dist_label




@app.cell(hide_code=True)
def _(mo):
    # State for clicked x position
    get_clicked_x, set_clicked_x = mo.state(None)
    return get_clicked_x, set_clicked_x


@app.cell(hide_code=True)
def _(
    alt, np, pd, mo,
    get_distribution, get_x_range, get_dist_label,
    dist_dropdown, mu_slider, sigma_slider,
    a_slider, b_slider, lambda_slider,
    alpha_slider, beta_slider, k_slider, theta_slider,
    show_pdf, show_cdf, show_samples,
    n_samples_slider,
    get_clicked_x,
):
    # Get current distribution type
    dist_type = dist_dropdown.value

    # Build parameters dict based on distribution type
    params = {}
    if dist_type == 'normal':
        params = {'mu': mu_slider.value, 'sigma': sigma_slider.value}
    elif dist_type == 'uniform':
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
    dist_label = get_dist_label(dist_type, params)

    # Generate x values and compute PDF/CDF
    x = np.linspace(x_min, x_max, 500)
    pdf_vals = dist.pdf(x)
    cdf_vals = dist.cdf(x)

    # Create line DataFrame for PDF and CDF
    line_df = pd.DataFrame({
        'x': x,
        'pdf': pdf_vals,
        'cdf': cdf_vals,
    })

    # Create click grid for point selection (within current x range)
    click_grid_df = pd.DataFrame({'x': np.linspace(x_min, x_max, 200)})

    # Generate samples and histogram data if enabled
    hist_df = pd.DataFrame()
    sample_mean = None
    sample_std = None
    if show_samples.value:
        n_samples = n_samples_slider.value
        samples = dist.rvs(size=n_samples)
        # Calculate sample statistics (on all samples, not just valid ones)
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)  # Use ddof=1 for sample std dev
        # Filter samples to valid range for histogram
        samples_valid = samples[(samples >= x_min) & (samples <= x_max)]
        if len(samples_valid) > 0:
            # Use bins='auto' (Freedman-Diaconis / Sturges hybrid, same as matplotlib default)
            # Specify range to keep bins within x-axis bounds
            # density=True so histogram integrates to 1 (comparable with PDF)
            hist_counts, bin_edges = np.histogram(
                samples_valid, bins='auto', range=(x_min, x_max), density=True
            )
            hist_df = pd.DataFrame({
                'x1': bin_edges[:-1],
                'x2': bin_edges[1:],
                'y1': np.zeros(len(hist_counts)),
                'y2': hist_counts,
            })

    # Get clicked x position
    clicked_x = get_clicked_x()

    # Define scales
    x_scale = alt.Scale(domain=[x_min, x_max])
    left_scale = alt.Scale(domain=[0, 1])
    cdf_scale = alt.Scale(domain=[0, 1])

    # Click selection
    click_select = alt.selection_point(on='click', nearest=True, fields=['x'], name='click_select')

    # Build chart layers
    layers = []

    # PDF area and line (left y-axis)
    if show_pdf.value:
        pdf_area = alt.Chart(line_df).mark_area(
            opacity=0.3, color='steelblue', clip=True
        ).encode(
            x=alt.X('x:Q', scale=x_scale, title='x'),
            y=alt.Y('pdf:Q', scale=left_scale, title='Density / Frequency')
        )
        pdf_line = alt.Chart(line_df).mark_line(
            color='steelblue', strokeWidth=2, clip=True
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('pdf:Q', scale=left_scale)
        )
        layers.extend([pdf_area, pdf_line])

    # Histogram bars (left y-axis, density normalized to match PDF)
    if show_samples.value and len(hist_df) > 0:
        hist_bars = alt.Chart(hist_df).mark_rect(
            opacity=0.5, color='green', clip=True
        ).encode(
            x=alt.X('x1:Q', scale=x_scale),
            x2=alt.X2('x2:Q'),
            y=alt.Y('y1:Q', scale=left_scale),
            y2=alt.Y2('y2:Q')
        )
        layers.append(hist_bars)

    # CDF line (right y-axis, independent scale)
    cdf_chart = None
    if show_cdf.value:
        cdf_chart = alt.Chart(line_df).mark_line(
            color='orange', strokeWidth=2
        ).encode(
            x=alt.X('x:Q', scale=x_scale, title='x'),
            y=alt.Y('cdf:Q', scale=cdf_scale, title='CDF F(x)')
        )

    # Vertical rule at clicked position
    if clicked_x is not None and x_min <= clicked_x <= x_max:
        rule_df = pd.DataFrame({'x': [clicked_x]})
        rule = alt.Chart(rule_df).mark_rule(
            strokeDash=[4, 4], color='gray', strokeWidth=2
        ).encode(
            x=alt.X('x:Q', scale=x_scale)
        )
        layers.append(rule)

        # Point marker on PDF
        if show_pdf.value:
            clicked_pdf = dist.pdf(clicked_x)
            point_df = pd.DataFrame({'x': [clicked_x], 'pdf': [clicked_pdf]})
            pdf_point = alt.Chart(point_df).mark_circle(
                color='steelblue', size=100, clip=True
            ).encode(
                x=alt.X('x:Q', scale=x_scale),
                y=alt.Y('pdf:Q', scale=left_scale)
            )
            layers.append(pdf_point)

    # Invisible click layer
    click_layer = alt.Chart(click_grid_df).mark_point(
        opacity=0, size=1
    ).encode(
        x=alt.X('x:Q', scale=x_scale)
    ).add_params(click_select)

    # Combine PDF/histogram layers
    if layers:
        pdf_combined = alt.layer(*layers)
    else:
        # Empty chart if nothing to show
        pdf_combined = alt.Chart(line_df).mark_point(opacity=0).encode(
            x=alt.X('x:Q', scale=x_scale, title='x'),
            y=alt.Y('pdf:Q', scale=left_scale, title='Density / Frequency')
        )

    # Build final chart with dual y-axis
    if cdf_chart is not None:
        # Add CDF point marker if clicked
        if clicked_x is not None and x_min <= clicked_x <= x_max:
            clicked_cdf = dist.cdf(clicked_x)
            cdf_point_df = pd.DataFrame({'x': [clicked_x], 'cdf': [clicked_cdf]})
            cdf_point = alt.Chart(cdf_point_df).mark_circle(
                color='orange', size=100
            ).encode(
                x=alt.X('x:Q', scale=x_scale),
                y=alt.Y('cdf:Q', scale=cdf_scale)
            )
            cdf_chart = alt.layer(cdf_chart, cdf_point)

        chart = alt.layer(
            pdf_combined, cdf_chart, click_layer
        ).resolve_scale(
            y='independent'
        ).properties(
            width='container', height=400,
            title=f'{dist_label} - Click to inspect'
        )
    else:
        chart = alt.layer(
            pdf_combined, click_layer
        ).properties(
            width='container', height=400,
            title=f'{dist_label} - Click to inspect'
        )

    chart = chart.configure_axis(
        grid=True, gridOpacity=0.3,
        labelFontSize=14, titleFontSize=16
    ).configure_title(
        fontSize=18
    )

    interactive_chart = mo.ui.altair_chart(chart)

    # Store distribution and sample stats for stats table
    current_dist = dist
    current_clicked_x = clicked_x
    current_sample_mean = sample_mean
    current_sample_std = sample_std

    return interactive_chart, current_dist, current_clicked_x, click_grid_df, current_sample_mean, current_sample_std


@app.cell(hide_code=True)
def _(interactive_chart):
    chart_display = interactive_chart
    return (chart_display,)


@app.cell(hide_code=True)
def _(interactive_chart, click_grid_df, set_clicked_x):
    # Click handler - reads selection, updates state
    _filtered = interactive_chart.apply_selection(click_grid_df)

    if len(_filtered) > 0 and len(_filtered) < len(click_grid_df):
        _new_x = float(_filtered['x'].iloc[0])
        set_clicked_x(_new_x)

    return ()


@app.cell(hide_code=True)
def _(current_dist, current_clicked_x):
    # Compute statistics for table
    dist_mean = current_dist.mean()
    dist_var = current_dist.var()
    dist_std = current_dist.std()
    dist_median = current_dist.median()

    # Clicked point values
    if current_clicked_x is not None:
        clicked_pdf_val = current_dist.pdf(current_clicked_x)
        clicked_cdf_val = current_dist.cdf(current_clicked_x)
    else:
        clicked_pdf_val = None
        clicked_cdf_val = None

    return dist_mean, dist_var, dist_std, dist_median, clicked_pdf_val, clicked_cdf_val


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
    current_dist_type = dist_dropdown.value

    # Distribution selection section
    dist_section = mo.vstack([
        mo.Html("<h4>Distribution</h4>"),
        dist_dropdown,
    ], gap="0.3em")

    # Parameters section - show relevant sliders based on distribution
    param_widgets = []
    if current_dist_type == 'normal':
        param_widgets = [mu_slider, sigma_slider]
    elif current_dist_type == 'uniform':
        param_widgets = [a_slider, b_slider]
    elif current_dist_type == 'exponential':
        param_widgets = [lambda_slider]
    elif current_dist_type == 'beta':
        param_widgets = [alpha_slider, beta_slider]
    elif current_dist_type == 'gamma':
        param_widgets = [k_slider, theta_slider]

    param_section = mo.vstack([
        mo.Html("<h4>Parameters</h4>"),
        *param_widgets,
    ], gap="0.3em")

    # Display section
    display_section = mo.vstack([
        mo.Html("<h4>Display</h4>"),
        show_pdf,
        show_cdf,
        show_samples,
    ], gap="0.2em")

    # Samples section (only show when samples enabled)
    samples_section = mo.vstack([
        mo.Html("<h4>Sampling</h4>"),
        n_samples_slider,
    ], gap="0.3em") if show_samples.value else mo.Html("")

    sidebar_content = mo.vstack([dist_section, param_section, display_section, samples_section], gap="1em")

    return (sidebar_content,)


@app.cell(hide_code=True)
def _(
    mo, header, chart_display, sidebar_content,
    dist_mean, dist_var, dist_std, dist_median,
    clicked_pdf_val, clicked_cdf_val, current_clicked_x,
    current_sample_mean, current_sample_std
):
    # Build statistics table
    stats_rows = f'''
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>Mean (μ)</b></td>
            <td style="padding: 4px 12px; text-align: right;">{dist_mean:.4f}</td>
        </tr>
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>Variance (σ²)</b></td>
            <td style="padding: 4px 12px; text-align: right;">{dist_var:.4f}</td>
        </tr>
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>Std Dev (σ)</b></td>
            <td style="padding: 4px 12px; text-align: right;">{dist_std:.4f}</td>
        </tr>
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>Median</b></td>
            <td style="padding: 4px 12px; text-align: right;">{dist_median:.4f}</td>
        </tr>
    '''

    # Add sample statistics if samples are enabled
    if current_sample_mean is not None:
        stats_rows += f'''
        <tr style="border-top: 1px solid #dee2e6;">
            <td style="padding: 8px 12px 4px 0;"><b>Sample Mean</b></td>
            <td style="padding: 8px 12px 4px 12px; text-align: right; color: green;">{current_sample_mean:.4f}</td>
        </tr>
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>Sample Std</b></td>
            <td style="padding: 4px 12px; text-align: right; color: green;">{current_sample_std:.4f}</td>
        </tr>
        '''

    # Add clicked point info if available
    if current_clicked_x is not None and clicked_pdf_val is not None:
        stats_rows += f'''
        <tr style="border-top: 1px solid #dee2e6;">
            <td colspan="2" style="padding: 8px 0 4px 0; color: #666;"><i>At x = {current_clicked_x:.3f}</i></td>
        </tr>
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>f(x)</b></td>
            <td style="padding: 4px 12px; text-align: right; color: steelblue;">{clicked_pdf_val:.4f}</td>
        </tr>
        <tr>
            <td style="padding: 4px 12px 4px 0;"><b>P(X < x)</b></td>
            <td style="padding: 4px 12px; text-align: right; color: orange;">{clicked_cdf_val:.4f}</td>
        </tr>
        '''
    else:
        stats_rows += '''
        <tr style="border-top: 1px solid #dee2e6;">
            <td colspan="2" style="padding: 8px 0 4px 0; color: #999; font-style: italic;">Click chart to inspect a point</td>
        </tr>
        '''

    stats_table = f'''
        <table style="border-collapse: collapse; font-size: 13px; width: 100%;">
            {stats_rows}
        </table>
    '''

    # Combined layout
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot">
            {mo.as_html(chart_display)}
        </div>
        <div class="app-sidebar-container">
            <div class="app-sidebar">
                {sidebar_content}
            </div>
            <div style="margin-top: 1em; padding: 0.75em; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
                <div style="font-size: 0.9em; font-weight: bold; color: #495057; margin-bottom: 0.5em;">Statistics</div>
                {stats_table}
            </div>
        </div>
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
