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

__generated_with = "0.18.1"
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
            flex-shrink: 0;
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
            margin: 0 0 0.5em 0;
            font-size: 0.9em;
            color: #495057;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 0.3em;
        }
    </style>
    ''')
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    import seaborn as sns
    sns.set_context('talk')
    return KMeans, GaussianMixture, make_blobs, mo, mpatches, np, plt, sns


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    # Generate QR code
    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/clustering-demo.html')
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
                <br><span style="font-size: 16px;"><i>Live demo:</i>
                <a href="https://sciml.warwick.ac.uk/clustering-demo.html" target="_blank" style="color: #0066cc; text-decoration: none;">sciml.warwick.ac.uk</a>
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

    # Panel visibility checkboxes (Ground Truth always visible)
    show_unlabelled = mo.ui.checkbox(False, label='<b>Unlabelled</b>')
    show_kmeans = mo.ui.checkbox(False, label='<b>K-Means</b>')
    show_gmm = mo.ui.checkbox(False, label='<b>GMM</b>')

    return (
        n_clusters_slider,
        sigma_slider,
        seed_slider,
        Sxx_slider,
        Sxy_slider,
        Syy_slider,
        show_unlabelled,
        show_kmeans,
        show_gmm,
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
def _(np, mpatches):
    def draw_gmm_ellipse(ax, mean, covar, color, n_std=2.0):
        """Draw confidence ellipse for a Gaussian component."""
        # Eigendecomposition for ellipse parameters
        eigenvalues, eigenvectors = np.linalg.eigh(covar)
        # Width and height are 2*n_std*sqrt(eigenvalue)
        width = 2 * n_std * np.sqrt(eigenvalues[0])
        height = 2 * n_std * np.sqrt(eigenvalues[1])
        # Rotation angle from first eigenvector
        angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

        # Create and add ellipse patch
        ellipse = mpatches.Ellipse(
            mean, width, height,
            angle=angle,
            color=color,
            alpha=0.15,
            linewidth=2,
            fill=True
        )
        ax.add_patch(ellipse)
        return ellipse
    return (draw_gmm_ellipse,)


@app.cell(hide_code=True)
def _(
    np, plt, KMeans, GaussianMixture,
    generate_cluster_data, draw_gmm_ellipse,
    n_clusters_slider, sigma_slider, seed_slider,
    Sxx_slider, Sxy_slider, Syy_slider,
    show_unlabelled, show_kmeans, show_gmm
):
    # Get parameter values
    K = n_clusters_slider.value
    sigma = sigma_slider.value
    random_state = seed_slider.value
    Sxx = Sxx_slider.value
    Sxy = Sxy_slider.value
    Syy = Syy_slider.value

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
    colors = np.array([
        [1, 0, 0],      # Red
        [0, 0.7, 0],    # Green
        [0, 0, 1],      # Blue
        [1, 0.8, 0],    # Yellow
        [0, 0.8, 0.8],  # Cyan
        [0.8, 0, 0.8],  # Magenta
    ])

    # Always use 2x2 grid for consistent layout
    fig, axs = plt.subplots(2, 2, figsize=(7, 7))
    axs = list(axs.flat)

    # Fixed axis limits for consistent plot sizes
    xlim = (-12, 12)
    ylim = (-12, 12)

    # Configure all axes with same limits and titles for consistent sizing
    titles = ['Ground Truth', 'Unlabelled', 'K-Means', 'GMM']
    for ax, title in zip(axs, titles):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(labelsize=8)
        ax.set_title(title, fontsize=11)

    # Ground Truth panel (top-left) - always visible
    ax = axs[0]
    for i in range(K):
        mask = y == i
        ax.scatter(X[mask, 0], X[mask, 1], s=60, color=colors[i],
                  marker='o', edgecolor='black', linewidth=0.5)

    # Unlabelled panel (top-right)
    if show_unlabelled.value:
        ax = axs[1]
        ax.scatter(X[:, 0], X[:, 1], c='white', marker='o',
                  edgecolor='black', s=60, linewidth=0.5)

    # K-Means panel (bottom-left)
    if show_kmeans.value:
        ax = axs[2]
        km = KMeans(n_clusters=K, init='random', n_init=10,
                   max_iter=300, tol=1e-4, random_state=4)
        y_km = km.fit_predict(X)

        for i in range(K):
            mask = y_km == i
            ax.scatter(X[mask, 0], X[mask, 1], s=60, color=colors[i],
                      marker='o', edgecolor='black', linewidth=0.5)
        ax.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
                  s=200, marker='x', c='black', linewidths=2)

    # GMM panel (bottom-right)
    if show_gmm.value:
        ax = axs[3]
        gm = GaussianMixture(n_components=K, random_state=0)
        gm.fit(X)
        proba = gm.predict_proba(X)

        # Soft assignment colors: blend RGB based on probabilities
        proba_rgb = np.zeros((len(X), 3))
        for i in range(K):
            proba_rgb += proba[:, i:i+1] * colors[i:i+1]
        proba_rgb = np.clip(proba_rgb, 0, 1)

        ax.scatter(X[:, 0], X[:, 1], s=60, c=proba_rgb,
                  marker='o', edgecolor='black', linewidth=0.5)

        # Draw means
        ax.scatter(gm.means_[:, 0], gm.means_[:, 1],
                  s=200, marker='x', c='black', linewidths=2)

        # Draw confidence ellipses
        for i, (mean, covar) in enumerate(zip(gm.means_, gm.covariances_)):
            draw_gmm_ellipse(ax, mean, covar, colors[i], n_std=2.0)

    plt.tight_layout(pad=1.0)
    clustering_fig = fig
    return (clustering_fig,)


@app.cell(hide_code=True)
def _(
    mo,
    n_clusters_slider, sigma_slider, seed_slider,
    Sxx_slider, Sxy_slider, Syy_slider,
    show_unlabelled, show_kmeans, show_gmm
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

    panel_section = mo.vstack([
        mo.Html("<h4>Panels</h4>"),
        show_unlabelled,
        show_kmeans,
        show_gmm,
    ], gap="0.2em")

    sidebar = mo.vstack([data_section, cov_section, panel_section], gap="1em")

    sidebar_html = mo.Html(f'''
    <div class="app-sidebar">
        {sidebar}
    </div>
    ''')
    return (sidebar_html,)


@app.cell(hide_code=True)
def _(mo, header, clustering_fig, sidebar_html):
    # Combined layout: header on top, plot on left, controls on right
    mo.Html(f'''
    {header}
    <div class="app-layout">
        <div class="app-plot">{mo.as_html(clustering_fig)}</div>
        {sidebar_html}
    </div>
    ''')
    return


if __name__ == "__main__":
    app.run()
