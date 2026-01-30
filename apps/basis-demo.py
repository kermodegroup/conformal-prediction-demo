# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.5",
#     "scikit-learn==1.6.1",
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
    qr.add_data('https://sciml.warwick.ac.uk/basis-demo.html')
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
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>Basis Function Explorer</b>
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
    # Basis type selection (multiselect)
    basis_multiselect = mo.ui.multiselect(
        options={
            'Polynomial': 'polynomial',
            'RBF': 'rbf',
            'Fourier': 'fourier',
        },
        value=['Polynomial'],
        label='Basis Types'
    )

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

    # Polynomial parameters
    degree_slider = mo.ui.slider(1, 25, 1, 5, label='Degree $m$')

    # RBF parameters
    num_rbf_slider = mo.ui.slider(2, 20, 1, 10, label='Number of RBFs')
    ell_slider = mo.ui.slider(0.05, 1.0, 0.05, 0.2, label='Lengthscale $\\ell$')

    # Fourier parameters
    num_fourier_slider = mo.ui.slider(1, 20, 1, 5, label='Number of terms')

    # Data parameters
    n_points_slider = mo.ui.slider(10, 200, 10, 50, label='$N$ data points')
    noise_slider = mo.ui.slider(0.0, 0.5, 0.05, 0.1, label='Noise $\\sigma$')
    seed_slider = mo.ui.slider(0, 100, 1, 42, label='Random seed')

    return (
        basis_multiselect,
        target_dropdown,
        degree_slider,
        num_rbf_slider,
        ell_slider,
        num_fourier_slider,
        n_points_slider,
        noise_slider,
        seed_slider,
    )


@app.cell(hide_code=True)
def _(mo, basis_multiselect):
    # Dynamic dropdown populated from currently selected bases
    # This must be in a separate cell to react to multiselect changes
    _selected = basis_multiselect.value
    _basis_names = {'polynomial': 'Polynomial', 'rbf': 'RBF', 'fourier': 'Fourier'}

    if _selected:
        _options = {_basis_names[b]: b for b in _selected}
        # Default to first selected
        _default = list(_options.keys())[0]
    else:
        _options = {'(none)': 'none'}
        _default = '(none)'

    detail_basis_dropdown = mo.ui.dropdown(
        options=_options,
        value=_default,
        label='Detail View'
    )
    return (detail_basis_dropdown,)


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

    def polynomial_basis(x, degree):
        """Compute polynomial basis functions."""
        # x can be 1D array, return shape (len(x), degree+1)
        return np.column_stack([x**i for i in range(degree + 1)])

    def rbf_basis(x, centers, ell):
        """Compute RBF basis functions."""
        # x: 1D array of length n
        # centers: 1D array of length m
        # return: (n, m) array
        x = x.reshape(-1, 1)
        centers = centers.reshape(1, -1)
        return np.exp(-0.5 * (x - centers)**2 / ell**2)

    def fourier_basis(x, num_terms, L=1.0):
        """Compute Fourier basis functions (sin and cos)."""
        # Return constant term plus sin/cos pairs
        basis = [np.ones_like(x)]
        for j in range(1, num_terms + 1):
            basis.append(np.cos(2 * j * np.pi * x / L))
            basis.append(np.sin(2 * j * np.pi * x / L))
        return np.column_stack(basis)

    def fit_least_squares(Phi, y):
        """Fit using least squares."""
        w, residuals, rank, s = np.linalg.lstsq(Phi, y, rcond=None)
        return w

    return target_function, polynomial_basis, rbf_basis, fourier_basis, fit_least_squares


@app.cell(hide_code=True)
def _(
    np, plt,
    target_function, polynomial_basis, rbf_basis, fourier_basis, fit_least_squares,
    basis_multiselect, target_dropdown, detail_basis_dropdown,
    degree_slider, num_rbf_slider, ell_slider, num_fourier_slider,
    n_points_slider, noise_slider, seed_slider,
):
    # Get parameter values
    basis_types = basis_multiselect.value  # Now a list
    func_type = target_dropdown.value
    n_points = n_points_slider.value
    noise = noise_slider.value
    seed = seed_slider.value

    # Generate training data
    np.random.seed(seed)
    X_train = np.random.uniform(-1, 1, n_points)
    y_true_train = target_function(X_train, func_type)
    y_train = y_true_train + np.random.normal(0, noise, n_points)

    # Test points for plotting
    X_test = np.linspace(-1, 1, 200)
    y_true_test = target_function(X_test, func_type)

    # Colors and labels for each basis type
    basis_colors = {'polynomial': 'C1', 'rbf': 'C2', 'fourier': 'C4'}
    basis_names = {'polynomial': 'Polynomial', 'rbf': 'RBF', 'fourier': 'Fourier'}

    # Store results for each basis type
    results = {}
    condition_numbers = {}

    for basis_type in basis_types:
        if basis_type == 'polynomial':
            degree = degree_slider.value
            Phi_train = polynomial_basis(X_train, degree)
            Phi_test = polynomial_basis(X_test, degree)
        elif basis_type == 'rbf':
            num_rbf = num_rbf_slider.value
            ell = ell_slider.value
            centers = np.linspace(-1, 1, num_rbf)
            Phi_train = rbf_basis(X_train, centers, ell)
            Phi_test = rbf_basis(X_test, centers, ell)
        elif basis_type == 'fourier':
            num_terms = num_fourier_slider.value
            Phi_train = fourier_basis(X_train, num_terms)
            Phi_test = fourier_basis(X_test, num_terms)
        else:
            continue

        # Compute condition number
        cond = np.linalg.cond(Phi_train)
        condition_numbers[basis_type] = cond

        # Fit the model
        w = fit_least_squares(Phi_train, y_train)
        y_pred = Phi_test @ w

        results[basis_type] = {
            'Phi_train': Phi_train,
            'Phi_test': Phi_test,
            'w': w,
            'y_pred': y_pred,
        }

    # Create 2x2 grid
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    axs = list(axs.flat)

    # Configure axes
    titles = ['Data & Fit', 'Basis Functions', 'Weighted Components', 'Weights']
    for ax, title in zip(axs, titles):
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=8)

    # Data & Fit panel (top-left) - always shown
    ax = axs[0]
    ax.scatter(X_train, y_train, c='C0', s=30, alpha=0.7, label='Data', edgecolor='white', linewidth=0.5)
    ax.plot(X_test, y_true_test, 'k--', lw=1.5, label='True')
    for basis_type, res in results.items():
        ax.plot(X_test, res['y_pred'], color=basis_colors[basis_type], lw=2,
               label=basis_names[basis_type])
    ax.set_xlabel('$x$', fontsize=10)
    ax.set_ylabel('$y$', fontsize=10)
    ax.legend(fontsize=7, loc='best')
    ax.set_xlim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    # Get the selected basis for detail panels (from dropdown)
    detail_basis = detail_basis_dropdown.value
    # Fall back to first available if dropdown selection not in results
    if detail_basis not in results and results:
        detail_basis = list(results.keys())[0]

    # Basis Functions panel (top-right) - show selected basis
    if detail_basis in results:
        ax = axs[1]
        Phi_test = results[detail_basis]['Phi_test']
        num_basis = Phi_test.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, num_basis))
        n_show = min(num_basis, 10)
        for i in range(n_show):
            ax.plot(X_test, Phi_test[:, i], color=colors[i], lw=1.5, alpha=0.8,
                   label=rf'$j={i}$')
        ax.set_xlabel('$x$', fontsize=10)
        ax.set_ylabel(rf'$\phi_j(x)$ ({basis_names[detail_basis]})', fontsize=10)
        ax.legend(fontsize=6, ncol=2, loc='upper right')
        ax.set_xlim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

    # Weighted Components panel (bottom-left) - show selected basis
    if detail_basis in results:
        ax = axs[2]
        Phi_test = results[detail_basis]['Phi_test']
        w = results[detail_basis]['w']
        y_pred = results[detail_basis]['y_pred']
        num_basis = Phi_test.shape[1]
        colors = plt.cm.viridis(np.linspace(0, 1, num_basis))
        n_show = min(num_basis, 10)
        for i in range(n_show):
            ax.plot(X_test, Phi_test[:, i] * w[i], color=colors[i], lw=1.5, alpha=0.8,
                   label=rf'$j={i}$')
        ax.plot(X_test, y_pred, 'k-', lw=2, label='Sum')
        ax.set_xlabel('$x$', fontsize=10)
        ax.set_ylabel(rf'$w_j \phi_j(x)$ ({basis_names[detail_basis]})', fontsize=10)
        ax.legend(fontsize=6, ncol=2, loc='upper right')
        ax.set_xlim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

    # Weights panel (bottom-right) - show selected basis
    if detail_basis in results:
        ax = axs[3]
        w = results[detail_basis]['w']
        num_basis = len(w)
        indices = np.arange(num_basis)
        colors = ['C0' if wi >= 0 else 'C3' for wi in w]
        ax.bar(indices, w, color=colors, alpha=0.7, edgecolor='white')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_xlabel('Basis index $j$', fontsize=10)
        ax.set_ylabel(rf'$w_j$ ({basis_names[detail_basis]})', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(pad=1.0)
    basis_fig = fig
    return basis_fig, condition_numbers


@app.cell(hide_code=True)
def _(
    mo,
    basis_multiselect, target_dropdown, detail_basis_dropdown,
    degree_slider, num_rbf_slider, ell_slider, num_fourier_slider,
    n_points_slider, noise_slider, seed_slider,
):
    # Get current basis types to show relevant sliders
    current_bases = basis_multiselect.value

    # Function section
    func_section = mo.vstack([
        mo.Html("<h4>Function</h4>"),
        target_dropdown,
    ], gap="0.3em")

    # Basis section - show sliders for all selected basis types
    basis_widgets = [basis_multiselect]
    if 'polynomial' in current_bases:
        basis_widgets.append(degree_slider)
    if 'rbf' in current_bases:
        basis_widgets.extend([num_rbf_slider, ell_slider])
    if 'fourier' in current_bases:
        basis_widgets.append(num_fourier_slider)

    # Add detail dropdown only if multiple bases selected
    if len(current_bases) > 1:
        basis_widgets.append(detail_basis_dropdown)

    basis_section = mo.vstack([
        mo.Html("<h4>Basis</h4>"),
        *basis_widgets,
    ], gap="0.3em")

    # Data section
    data_section = mo.vstack([
        mo.Html("<h4>Data</h4>"),
        n_points_slider,
        noise_slider,
        seed_slider,
    ], gap="0.3em")

    sidebar = mo.vstack([func_section, basis_section, data_section], gap="1em")

    sidebar_html = mo.Html(f'''
    <div class="app-sidebar">
        {sidebar}
    </div>
    ''')
    return (sidebar_html,)


@app.cell(hide_code=True)
def _(mo, header, basis_fig, condition_numbers, sidebar_html):
    # Build condition number display
    if condition_numbers:
        _cond_items = []
        for _bt, _cond in condition_numbers.items():
            _name = {'polynomial': 'Polynomial', 'rbf': 'RBF', 'fourier': 'Fourier'}[_bt]
            # Color code: green if good (<1e4), yellow if moderate, red if bad (>1e8)
            if _cond < 1e4:
                _color = "#28a745"  # green
            elif _cond < 1e8:
                _color = "#ffc107"  # yellow
            else:
                _color = "#dc3545"  # red
            _cond_items.append(f'<span style="color:{_color}; margin-right: 1.5em;"><b>{_name}:</b> {_cond:.2e}</span>')
        cond_html = ''.join(_cond_items)
    else:
        cond_html = '<span style="color:#6c757d;">No basis selected</span>'

    # Combined layout: header on top, plot and sidebar, condition numbers at bottom
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div>
            <div class="app-plot">{mo.as_html(basis_fig)}</div>
            <div style="margin-top: 0.5em; padding: 0.5em; background: #f8f9fa; border-radius: 4px; font-size: 13px;">
                <b>Condition number:</b> {cond_html}
            </div>
        </div>
        {sidebar_html}
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
