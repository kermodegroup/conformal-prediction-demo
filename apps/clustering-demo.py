# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "altair",
#     "pandas",
#     "numpy==2.2.5",
#     "scikit-learn==1.6.1",
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
            justify-content: center;
            align-items: flex-start;
            z-index: 1;
        }

        .app-plot img,
        .app-plot svg {
            max-width: 100%;
            height: auto;
        }

        .app-sidebar {
            z-index: 10;
            position: relative;
            display: flex;
            flex-direction: column;
            gap: 1.5em;
            padding: 1.5em;
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            width: 30%;
            min-width: 280px;
            max-width: 400px;
            flex-shrink: 0;
        }

        @media (max-width: 768px) {
            .app-layout {
                flex-direction: column;
                height: auto;
                overflow-y: auto;
            }
            .app-sidebar {
                width: 100%;
                max-width: none;
                min-width: auto;
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
    from scipy.optimize import linear_sum_assignment
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    return alt, mo, np, pd, linear_sum_assignment, make_blobs, KMeans, GaussianMixture


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://kermodegroup.github.io/demos/clustering-demo.html')
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
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>K-Means and GMM Clustering Demo</b>
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
    # Data generation controls
    n_clusters_slider = mo.ui.slider(2, 6, 1, 3, label='Number of clusters $K$')
    sigma_slider = mo.ui.slider(0.1, 2.0, 0.1, 1.0, label='Cluster spread $\\sigma$')
    seed_slider = mo.ui.slider(0, 10, 1, 2, label='Random seed')

    # Covariance transform controls
    Sxx_slider = mo.ui.slider(0.5, 2.0, 0.1, 1.0, label='$\\Sigma_{xx}$')
    Sxy_slider = mo.ui.slider(-0.5, 0.5, 0.1, 0.0, label='$\\Sigma_{xy}$')
    Syy_slider = mo.ui.slider(0.5, 2.0, 0.1, 1.0, label='$\\Sigma_{yy}$')

    # View dropdown (replaces checkboxes)
    view_dropdown = mo.ui.dropdown(
        options={
            'Ground Truth': 'ground_truth',
            'Unlabelled': 'unlabelled',
            'K-Means': 'kmeans',
            'GMM': 'gmm',
        },
        value='Ground Truth',
        label='View'
    )

    # Checkbox to overlay ground truth on clustering views
    show_ground_truth = mo.ui.checkbox(False, label='Overlay ground truth')

    return (
        n_clusters_slider,
        sigma_slider,
        seed_slider,
        Sxx_slider,
        Sxy_slider,
        Syy_slider,
        view_dropdown,
        show_ground_truth,
    )


@app.cell(hide_code=True)
def _(np, make_blobs):
    def generate_cluster_data(n_samples, n_clusters, sigma, Sxx, Sxy, Syy, random_state):
        """Generate clustered data with covariance transform."""
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=2,
            centers=n_clusters,
            cluster_std=sigma,
            shuffle=True,
            random_state=random_state
        )
        # Apply covariance transform
        T = np.array([[Sxx, -Sxy],
                      [-Sxy, Syy]])
        X = X @ T
        return X, y
    return (generate_cluster_data,)


@app.cell(hide_code=True)
def _(np):
    def compute_ellipse_points(mean, covar, n_std=2.0, n_points=100):
        """Compute points along ellipse perimeter for Altair rendering."""
        # Eigendecomposition for ellipse parameters
        eigenvalues, eigenvectors = np.linalg.eigh(covar)
        # Rotation angle from first eigenvector
        angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])

        # Parametric ellipse
        t = np.linspace(0, 2 * np.pi, n_points)
        a = n_std * np.sqrt(eigenvalues[0])
        b = n_std * np.sqrt(eigenvalues[1])

        # Unrotated ellipse
        x = a * np.cos(t)
        y = b * np.sin(t)

        # Rotate and translate
        x_rot = x * np.cos(angle) - y * np.sin(angle) + mean[0]
        y_rot = x * np.sin(angle) + y * np.cos(angle) + mean[1]

        return x_rot, y_rot
    return (compute_ellipse_points,)


@app.cell(hide_code=True)
def _(
    alt, np, pd, linear_sum_assignment,
    KMeans, GaussianMixture,
    generate_cluster_data, compute_ellipse_points,
    n_clusters_slider, sigma_slider, seed_slider,
    Sxx_slider, Sxy_slider, Syy_slider,
    view_dropdown, show_ground_truth
):
    # Get parameter values
    K = n_clusters_slider.value
    sigma = sigma_slider.value
    random_state = seed_slider.value
    Sxx = Sxx_slider.value
    Sxy = Sxy_slider.value
    Syy = Syy_slider.value
    view = view_dropdown.value

    # Generate data
    X, y = generate_cluster_data(
        n_samples=150,
        n_clusters=K,
        sigma=sigma,
        Sxx=Sxx,
        Sxy=Sxy,
        Syy=Syy,
        random_state=random_state
    )

    # Define colors for up to 6 clusters
    cluster_colors = ['#e41a1c', '#4daf4a', '#377eb8', '#ff7f00', '#00bfc4', '#c040c0']
    # Emoji squares matching cluster colors: red, green, blue, orange, cyan, purple
    cluster_emojis = ['🟥', '🟩', '🟦', '🟧', '🔵', '🟪']
    color_scale = alt.Scale(domain=list(range(K)), range=cluster_colors[:K])

    # Fixed axis limits
    x_min, x_max = -12, 12
    y_min, y_max = -12, 12
    x_scale = alt.Scale(domain=[x_min, x_max])
    y_scale = alt.Scale(domain=[y_min, y_max])

    # Fit models
    km = KMeans(n_clusters=K, init='random', n_init=10, max_iter=300, tol=1e-4, random_state=4)
    y_km_raw = km.fit_predict(X)

    gm = GaussianMixture(n_components=K, random_state=0)
    gm.fit(X)
    y_gmm_raw = gm.predict(X)
    proba_raw = gm.predict_proba(X)

    # Match cluster labels to ground truth using Hungarian algorithm
    def match_labels_to_ground_truth(y_true, y_pred, n_clusters):
        """Find optimal label mapping to maximize agreement with ground truth."""
        # Build contingency matrix: rows=true labels, cols=predicted labels
        contingency = np.zeros((n_clusters, n_clusters), dtype=int)
        for t, p in zip(y_true, y_pred):
            contingency[t, p] += 1
        # Use Hungarian algorithm to find optimal assignment (maximize overlap)
        # linear_sum_assignment minimizes, so negate the contingency matrix
        row_ind, col_ind = linear_sum_assignment(-contingency)
        # mapping: predicted label -> true label
        mapping = {col: row for row, col in zip(row_ind, col_ind)}
        return mapping

    # Remap K-Means labels
    km_mapping = match_labels_to_ground_truth(y, y_km_raw, K)
    y_km = np.array([km_mapping[label] for label in y_km_raw])

    # Remap GMM labels and probabilities
    gmm_mapping = match_labels_to_ground_truth(y, y_gmm_raw, K)
    y_gmm = np.array([gmm_mapping[label] for label in y_gmm_raw])
    # Reorder probability columns to match new label order
    proba = np.zeros_like(proba_raw)
    for old_label, new_label in gmm_mapping.items():
        proba[:, new_label] = proba_raw[:, old_label]

    # Convert hex colors to RGB for interpolation
    def hex_to_rgb(hex_color):
        h = hex_color.lstrip('#')
        return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])

    def rgb_to_hex(rgb):
        return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    cluster_rgb = np.array([hex_to_rgb(c) for c in cluster_colors[:K]])

    # Compute interpolated colors for GMM based on probabilities
    # Weighted average of RGB values
    gmm_rgb = proba @ cluster_rgb  # Shape: (n_samples, 3)
    gmm_rgb = np.clip(gmm_rgb, 0, 255)
    gmm_hex = [rgb_to_hex(rgb) for rgb in gmm_rgb]

    # Build main DataFrame with all columns
    df_dict = {
        'x': X[:, 0],
        'y': X[:, 1],
        'true_label': y,
        'kmeans_label': y_km,
        'gmm_label': y_gmm,
        'gmm_color': gmm_hex,
    }
    # Add probability columns for GMM tooltips
    for i in range(K):
        df_dict[f'p_c{i}'] = proba[:, i]

    # Build formatted probability string with colored emoji squares
    prob_strings = []
    for row_idx in range(len(X)):
        parts = [f"{cluster_emojis[i]} {proba[row_idx, i]:.2f}" for i in range(K)]
        prob_strings.append("  ".join(parts))
    df_dict['prob_tooltip'] = prob_strings

    df = pd.DataFrame(df_dict)

    # K-Means centers DataFrame
    centers_km_df = pd.DataFrame({
        'x': km.cluster_centers_[:, 0],
        'y': km.cluster_centers_[:, 1],
    })

    # GMM centers DataFrame
    centers_gmm_df = pd.DataFrame({
        'x': gm.means_[:, 0],
        'y': gm.means_[:, 1],
    })

    # View titles
    view_titles = {
        'ground_truth': 'Ground Truth',
        'unlabelled': 'Unlabelled Data',
        'kmeans': 'K-Means Clustering',
        'gmm': 'Gaussian Mixture Model',
    }

    # Create legend layers for consistent display across all views
    show_gt_overlay = show_ground_truth.value and view in ('kmeans', 'gmm')
    legend_df = pd.DataFrame({'cluster': list(range(K))})
    legend_layers = []

    if view == 'ground_truth':
        # Just cluster colors
        legend_layers.append(alt.Chart(legend_df).mark_circle(opacity=0, size=120).encode(
            color=alt.Color('cluster:N', scale=color_scale, legend=alt.Legend(title='Cluster'))
        ))
    elif view == 'unlabelled':
        # Placeholder legend to maintain consistent sizing
        legend_layers.append(alt.Chart(legend_df).mark_circle(opacity=0, size=120).encode(
            color=alt.Color('cluster:N', scale=color_scale, legend=alt.Legend(title='Cluster'))
        ))
    elif view == 'kmeans':
        # Cluster colors (filled circles)
        legend_layers.append(alt.Chart(legend_df).mark_circle(opacity=0, size=120).encode(
            color=alt.Color('cluster:N', scale=color_scale, legend=alt.Legend(title='Cluster'))
        ))
        # Cluster centers (cross)
        legend_layers.append(alt.Chart(pd.DataFrame({'label': ['Centers']})).mark_point(
            opacity=0, shape='cross', size=200, strokeWidth=3
        ).encode(
            color=alt.value('black'),
            shape=alt.Shape('label:N', legend=alt.Legend(title='Markers', symbolStrokeColor='black'))
        ))
        # Ground truth overlay (unfilled circles)
        if show_gt_overlay:
            legend_layers.append(alt.Chart(legend_df).mark_circle(
                opacity=0, size=290, filled=False, strokeWidth=2
            ).encode(
                stroke=alt.Color('cluster:N', scale=color_scale, legend=alt.Legend(title='Cluster', symbolFillColor='transparent'))
            ))
    elif view == 'gmm':
        # Cluster colors (filled circles)
        legend_layers.append(alt.Chart(legend_df).mark_circle(opacity=0, size=120).encode(
            color=alt.Color('cluster:N', scale=color_scale, legend=alt.Legend(title='Cluster'))
        ))
        # Covariance ellipses (lines) and centers (cross)
        markers_df = pd.DataFrame({'label': ['Centers', 'Covariance']})
        legend_layers.append(alt.Chart(markers_df).mark_point(opacity=0, size=200).encode(
            shape=alt.Shape('label:N', scale=alt.Scale(
                domain=['Centers', 'Covariance'],
                range=['cross', 'stroke']
            ), legend=alt.Legend(title='Markers', symbolStrokeColor='black', symbolStrokeWidth=2))
        ))
        # Ground truth overlay (unfilled circles)
        if show_gt_overlay:
            legend_layers.append(alt.Chart(legend_df).mark_circle(
                opacity=0, size=290, filled=False, strokeWidth=2
            ).encode(
                stroke=alt.Color('cluster:N', scale=color_scale, legend=alt.Legend(title='Cluster', symbolFillColor='transparent'))
            ))

    # Build chart based on view
    layers = legend_layers.copy()  # Always include legend layer(s) first

    if view == 'ground_truth':
        points = alt.Chart(df).mark_circle(size=120).encode(
            x=alt.X('x:Q', scale=x_scale, title='x'),
            y=alt.Y('y:Q', scale=y_scale, title='y'),
            color=alt.Color('true_label:N', scale=color_scale, legend=None),
            tooltip=[alt.Tooltip('true_label:N', title='Cluster')]
        )
        layers.append(points)

    elif view == 'unlabelled':
        points = alt.Chart(df).mark_circle(
            size=120, color='white', stroke='black', strokeWidth=1
        ).encode(
            x=alt.X('x:Q', scale=x_scale, title='x'),
            y=alt.Y('y:Q', scale=y_scale, title='y'),
        )
        layers.append(points)

    elif view == 'kmeans':
        points = alt.Chart(df).mark_circle(size=120).encode(
            x=alt.X('x:Q', scale=x_scale, title='x'),
            y=alt.Y('y:Q', scale=y_scale, title='y'),
            color=alt.Color('kmeans_label:N', scale=color_scale, legend=None),
            tooltip=[
                alt.Tooltip('kmeans_label:N', title='Predicted'),
                alt.Tooltip('true_label:N', title='True'),
            ]
        )
        centers = alt.Chart(centers_km_df).mark_point(
            shape='cross', size=200, strokeWidth=3, color='black', filled=False
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
        )
        layers.extend([points, centers])

    elif view == 'gmm':
        # Build ellipses for each component (using remapped cluster indices)
        ellipse_data = []
        for i in range(K):
            remapped_cluster = gmm_mapping[i]  # Map raw GMM cluster to ground truth cluster
            x_ell, y_ell = compute_ellipse_points(gm.means_[i], gm.covariances_[i], n_std=2.0)
            for j, (xe, ye) in enumerate(zip(x_ell, y_ell)):
                ellipse_data.append({
                    'x': xe,
                    'y': ye,
                    'cluster': remapped_cluster,
                    'order': j,
                })
        ellipse_df = pd.DataFrame(ellipse_data)

        # Draw ellipses as colored outline lines
        for i in range(K):
            cluster_ellipse_df = ellipse_df[ellipse_df['cluster'] == i]
            ellipse_line = alt.Chart(cluster_ellipse_df).mark_line(
                strokeWidth=2.5,
                opacity=0.7,
            ).encode(
                x=alt.X('x:Q', scale=x_scale),
                y=alt.Y('y:Q', scale=y_scale),
                color=alt.value(cluster_colors[i]),
                order='order:Q',
            )
            layers.append(ellipse_line)

        # Use interpolated colors based on cluster probabilities
        points = alt.Chart(df).mark_circle(size=120, stroke='black', strokeWidth=0.5).encode(
            x=alt.X('x:Q', scale=x_scale, title='x'),
            y=alt.Y('y:Q', scale=y_scale, title='y'),
            color=alt.Color('gmm_color:N', scale=None, legend=None),
            tooltip=[
                alt.Tooltip('prob_tooltip:N', title='Probabilities'),
                alt.Tooltip('true_label:N', title='True'),
            ]
        )
        centers = alt.Chart(centers_gmm_df).mark_point(
            shape='cross', size=200, strokeWidth=3, color='black', filled=False
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
        )
        layers.extend([points, centers])

    # Add ground truth overlay for K-Means and GMM views
    if show_ground_truth.value and view in ('kmeans', 'gmm'):
        gt_overlay = alt.Chart(df).mark_circle(
            size=290, filled=False, strokeWidth=2
        ).encode(
            x=alt.X('x:Q', scale=x_scale),
            y=alt.Y('y:Q', scale=y_scale),
            stroke=alt.Color('true_label:N', scale=color_scale, legend=None),
        )
        layers.append(gt_overlay)

    # Combine layers
    chart = alt.layer(*layers).properties(
        width=550, height=550,
        title=view_titles.get(view, 'Clustering')
    ).configure_axis(
        grid=True, gridOpacity=0.3,
        labelFontSize=14, titleFontSize=16
    ).configure_title(
        fontSize=18
    )

    clustering_chart = chart
    # Export data for confusion matrix
    clustering_data = {
        'true_labels': y,
        'kmeans_labels': y_km,
        'gmm_labels': y_gmm,
        'K': K,
        'view': view,
    }
    return clustering_chart, clustering_data


@app.cell(hide_code=True)
def _(mo, np, clustering_data):
    # Build confusion matrix for K-Means or GMM views
    _view = clustering_data['view']
    _K = clustering_data['K']
    _true_labels = clustering_data['true_labels']

    if _view in ('kmeans', 'gmm'):
        if _view == 'kmeans':
            _pred_labels = clustering_data['kmeans_labels']
            _method_name = 'K-Means'
        else:
            _pred_labels = clustering_data['gmm_labels']
            _method_name = 'GMM'

        # Compute confusion matrix
        _conf_matrix = np.zeros((_K, _K), dtype=int)
        for _t, _p in zip(_true_labels, _pred_labels):
            _conf_matrix[_t, _p] += 1

        # Compute accuracy
        _accuracy = np.trace(_conf_matrix) / np.sum(_conf_matrix) * 100

        # Build HTML table
        _colors = ['#e41a1c', '#4daf4a', '#377eb8', '#ff7f00', '#00bfc4', '#c040c0']

        _header_row = '<th style="padding: 4px 8px; border: 1px solid #ccc; background: #f0f0f0;"></th>'
        for _j in range(_K):
            _header_row += f'<th style="padding: 4px 8px; border: 1px solid #ccc; background: {_colors[_j]}20; color: {_colors[_j]}; font-weight: bold;">P{_j}</th>'

        _rows_html = ""
        for _i in range(_K):
            _row = f'<td style="padding: 4px 8px; border: 1px solid #ccc; background: {_colors[_i]}20; color: {_colors[_i]}; font-weight: bold;">T{_i}</td>'
            for _j in range(_K):
                _val = _conf_matrix[_i, _j]
                # Highlight diagonal (correct predictions)
                if _i == _j:
                    _bg = '#d4edda'  # Light green
                else:
                    _bg = '#f8d7da' if _val > 0 else '#fff'  # Light red if errors
                _row += f'<td style="padding: 4px 8px; border: 1px solid #ccc; background: {_bg}; text-align: center;">{_val}</td>'
            _rows_html += f'<tr>{_row}</tr>'

        _table_html = f'''
        <table style="border-collapse: collapse; font-size: 12px; margin-top: 0.5em;">
            <tr>{_header_row}</tr>
            {_rows_html}
        </table>
        '''

        confusion_matrix_html = mo.Html(f'''
        <h4 style="margin: 0 0 0.3em 0; font-size: 0.9em; color: #495057; border-bottom: 1px solid #dee2e6; padding-bottom: 0.3em;">
            {_method_name} Confusion Matrix
        </h4>
        <div style="font-size: 11px; color: #666; margin-bottom: 0.3em;">T=True, P=Predicted</div>
        {_table_html}
        <div style="font-size: 12px; margin-top: 0.5em;"><b>Accuracy: {_accuracy:.1f}%</b></div>
        ''')
    else:
        # No confusion matrix for ground truth or unlabelled views
        confusion_matrix_html = mo.Html("")

    return (confusion_matrix_html,)


@app.cell(hide_code=True)
def _(
    mo,
    n_clusters_slider, sigma_slider, seed_slider,
    Sxx_slider, Sxy_slider, Syy_slider,
    view_dropdown, show_ground_truth
):
    # Vertical sidebar with grouped controls
    data_section = mo.vstack([
        mo.Html("<h4>Data</h4>"),
        n_clusters_slider,
        sigma_slider,
        seed_slider,
    ], gap="0.3em")

    cov_section = mo.vstack([
        mo.Html("<h4>Covariance</h4>"),
        Sxx_slider,
        Sxy_slider,
        Syy_slider,
    ], gap="0.3em")

    view_section = mo.vstack([
        mo.Html("<h4>View</h4>"),
        view_dropdown,
        show_ground_truth,
    ], gap="0.3em")

    sidebar = mo.vstack([data_section, cov_section, view_section], gap="1em")
    return (sidebar,)


@app.cell(hide_code=True)
def _(mo, header, clustering_chart, sidebar, confusion_matrix_html):
    # Combined layout: header on top, plot on left, controls on right
    sidebar_html = mo.Html(f'''
    <div class="app-sidebar">
        {sidebar}
        {confusion_matrix_html}
    </div>
    ''')
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot">{mo.as_html(clustering_chart)}</div>
        {sidebar_html}
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
