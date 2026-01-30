# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.5",
#     "seaborn==0.13.2",
#     "qrcode==8.2",
#     "jax",
#     "jaxlib",
#     "equinox",
#     "optax",
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
            gap: clamp(0.3em, 1vh, 1em);
            padding: clamp(0.5em, 1.5vh, 1.5em);
            background-color: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #dee2e6;
            min-width: 364px;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
        }

        .app-sidebar h4 {
            margin: clamp(0.3em, 1vh, 1em) 0 clamp(0.2em, 0.5vh, 0.5em) 0;
            font-size: 0.9em;
            color: #495057;
            border-bottom: 1px solid #dee2e6;
            padding-bottom: 0.2em;
        }

        .app-sidebar h4:first-child {
            margin-top: 0;
        }

        .app-sidebar .marimo-ui-element {
            margin-bottom: clamp(0.1em, 0.3vh, 0.3em);
        }
    </style>
    ''')
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')

    import seaborn as sns
    sns.set_context('talk')
    return eqx, jax, jnp, mo, np, optax, plt, sns


@app.cell(hide_code=True)
def _():
    import qrcode
    import io
    import base64

    qr = qrcode.QRCode(version=1, box_size=10, border=4)
    qr.add_data('https://sciml.warwick.ac.uk/mlp-demo/')
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

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
                <p style='font-size: 24px; margin: 0; padding: 0; line-height: 1.3;'><b>MLP Regression Demo</b>
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
    # Data controls
    n_points_slider = mo.ui.slider(10, 100, 5, 30, label='Training points $N$')
    noise_slider = mo.ui.slider(0.0, 0.5, 0.05, 0.1, label='Noise level $\\sigma$')
    function_dropdown = mo.ui.dropdown(
        options={'Sine': 'sin', 'Step': 'step', 'Runge': 'runge', 'Witch of Agnesi': 'witch'},
        value='Sine',
        label='Target function'
    )
    seed_slider = mo.ui.slider(0, 10, 1, 0, label='Random seed')

    # Network controls
    width_slider = mo.ui.slider(8, 128, 8, 32, label='Width (units per layer)')
    depth_slider = mo.ui.slider(1, 5, 1, 2, label='Depth (hidden layers)')
    activation_dropdown = mo.ui.dropdown(
        options={'ReLU': 'relu', 'Tanh': 'tanh', 'Sigmoid': 'sigmoid'},
        value='ReLU',
        label='Activation function'
    )

    # Training controls
    lr_slider = mo.ui.slider(-4, -1, 0.5, -2, label='Learning rate (log₁₀)')
    optimizer_dropdown = mo.ui.dropdown(
        options={'Adam': 'adam', 'SGD': 'sgd'},
        value='Adam',
        label='Optimizer'
    )
    epochs_slider = mo.ui.slider(100, 5000, 100, 1000, label='Number of epochs')

    # Action buttons
    train_button = mo.ui.run_button(label='Train')
    reset_button = mo.ui.run_button(label='Reset Weights')
    store_button = mo.ui.run_button(label='Store Fit')
    clear_stored_button = mo.ui.run_button(label='Clear Stored')

    return (
        n_points_slider,
        noise_slider,
        function_dropdown,
        seed_slider,
        width_slider,
        depth_slider,
        activation_dropdown,
        lr_slider,
        optimizer_dropdown,
        epochs_slider,
        train_button,
        reset_button,
        store_button,
        clear_stored_button,
    )


@app.cell(hide_code=True)
def _(np):
    def generate_data(n_points, noise_std, function_type, seed):
        """Generate training data from target function with noise."""
        rng = np.random.default_rng(seed)
        X = rng.uniform(-2, 2, n_points)
        X = np.sort(X)

        if function_type == 'sin':
            y_true = np.sin(np.pi * X)
        elif function_type == 'step':
            y_true = np.where(X < 0, -0.5, 0.5)
        elif function_type == 'runge':
            y_true = 1.0 / (1.0 + 25.0 * X**2)
        elif function_type == 'witch':
            y_true = 1.0 / (1.0 + X**2)
        else:
            y_true = np.sin(np.pi * X)

        y = y_true + rng.normal(0, noise_std, n_points)
        return X, y, y_true

    def get_ground_truth(X, function_type):
        """Get ground truth values for plotting."""
        if function_type == 'sin':
            return np.sin(np.pi * X)
        elif function_type == 'step':
            return np.where(X < 0, -0.5, 0.5)
        elif function_type == 'runge':
            return 1.0 / (1.0 + 25.0 * X**2)
        elif function_type == 'witch':
            return 1.0 / (1.0 + X**2)
        else:
            return np.sin(np.pi * X)

    return generate_data, get_ground_truth


@app.cell(hide_code=True)
def _(eqx, jax, jnp):
    def get_activation(name):
        """Get JAX activation function by name."""
        if name == 'relu':
            return jax.nn.relu
        elif name == 'tanh':
            return jnp.tanh
        elif name == 'sigmoid':
            return jax.nn.sigmoid
        else:
            return jax.nn.relu

    class MLP(eqx.Module):
        """Multi-layer perceptron for 1D regression."""
        layers: list

        def __init__(self, key, width, depth, activation):
            keys = jax.random.split(key, depth + 1)
            layers = []

            # Input layer: scalar -> width
            layers.append(eqx.nn.Linear('scalar', width, key=keys[0]))
            layers.append(activation)

            # Hidden layers
            for i in range(depth - 1):
                layers.append(eqx.nn.Linear(width, width, key=keys[i + 1]))
                layers.append(activation)

            # Output layer: width -> scalar
            layers.append(eqx.nn.Linear(width, 'scalar', key=keys[-1]))

            self.layers = layers

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    return get_activation, MLP


@app.cell(hide_code=True)
def _(eqx, jax, jnp):
    @eqx.filter_jit
    def train_step(model, opt_state, optimizer, x, y):
        """Single training step with gradient update."""
        def loss_fn(model):
            pred = jax.vmap(model)(x)
            return jnp.mean((pred - y) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    def train_model(model, optimizer, x, y, n_epochs):
        """Train model for n_epochs and return loss history."""
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        losses = []

        for _ in range(n_epochs):
            model, opt_state, loss = train_step(model, opt_state, optimizer, x, y)
            losses.append(float(loss))

        return model, losses

    return train_step, train_model


@app.cell(hide_code=True)
def _(mo):
    # State for model and training results
    get_model, set_model = mo.state(None)
    get_losses, set_losses = mo.state([])
    get_trained, set_trained = mo.state(False)
    # State for tracking if fit is stale (parameters changed since last train)
    get_train_params, set_train_params = mo.state(None)  # dict of params used during training
    get_last_pred, set_last_pred = mo.state(None)  # last prediction array
    # State for stored fits (list of dicts with 'pred', 'losses', 'label')
    get_stored_fits, set_stored_fits = mo.state([])
    return (get_model, set_model, get_losses, set_losses, get_trained, set_trained,
            get_train_params, set_train_params, get_last_pred, set_last_pred,
            get_stored_fits, set_stored_fits)


@app.cell(hide_code=True)
def _(
    jax, jnp, np, optax,
    n_points_slider, noise_slider, function_dropdown, seed_slider,
    width_slider, depth_slider, activation_dropdown,
    lr_slider, optimizer_dropdown, epochs_slider,
    train_button, reset_button, store_button, clear_stored_button,
    generate_data, get_activation, MLP, train_model,
    get_model, set_model, get_losses, set_losses, get_trained, set_trained,
    get_train_params, set_train_params, get_last_pred, set_last_pred,
    get_stored_fits, set_stored_fits,
):
    # Generate training data
    X_train, y_train, _ = generate_data(
        n_points_slider.value,
        noise_slider.value,
        function_dropdown.value,
        seed_slider.value
    )
    X_jax = jnp.array(X_train)
    y_jax = jnp.array(y_train)

    # Get network parameters
    width = width_slider.value
    depth = depth_slider.value
    activation_name = activation_dropdown.value
    activation = get_activation(activation_name)
    learning_rate = 10 ** lr_slider.value
    n_epochs = epochs_slider.value

    # Current parameters dict for staleness tracking
    current_params = {
        'n_points': n_points_slider.value,
        'noise': noise_slider.value,
        'function': function_dropdown.value,
        'seed': seed_slider.value,
        'width': width,
        'depth': depth,
        'activation': activation_name,
        'lr': lr_slider.value,
        'optimizer': optimizer_dropdown.value,
        'epochs': n_epochs,
    }

    # Create optimizer
    if optimizer_dropdown.value == 'adam':
        optimizer = optax.adam(learning_rate)
    else:
        optimizer = optax.sgd(learning_rate)

    # Initialize or reset model
    key = jax.random.PRNGKey(seed_slider.value + 42)

    # Handle reset button
    if reset_button.value:
        _model = MLP(key, width, depth, activation)
        set_model(_model)
        set_losses([])
        set_trained(False)
        set_train_params(None)
        set_last_pred(None)

    # Handle train button
    if train_button.value:
        # Check if parameters changed (fit is stale) - if so, reinitialize model
        _train_params = get_train_params()
        _is_stale = _train_params is not None and _train_params != current_params

        if _is_stale:
            # Parameters changed - start fresh with new model
            _current_model = MLP(key, width, depth, activation)
            _prev_losses = []
        else:
            # Continue training existing model
            _current_model = get_model()
            if _current_model is None:
                _current_model = MLP(key, width, depth, activation)
            _prev_losses = get_losses() or []

        # Train model
        _trained_model, _new_losses = train_model(_current_model, optimizer, X_jax, y_jax, n_epochs)
        set_model(_trained_model)

        # Accumulate losses (or start fresh if stale)
        set_losses(_prev_losses + _new_losses)
        set_trained(True)

        # Store training parameters and compute prediction for staleness tracking
        set_train_params(current_params.copy())
        _X_plot = jnp.linspace(-2, 2, 200)
        _y_pred = np.array(jax.vmap(_trained_model)(_X_plot))
        set_last_pred(_y_pred)

    # Handle store button - save current fit for comparison
    if store_button.value:
        _last_pred = get_last_pred()
        _losses = get_losses()
        if _last_pred is not None and _losses:
            _stored = get_stored_fits()
            _label = f"Fit {len(_stored) + 1}: W={width}, D={depth}, {activation_name}"
            _new_fit = {
                'pred': _last_pred.copy(),
                'losses': list(_losses),
                'label': _label,
            }
            set_stored_fits(_stored + [_new_fit])

    # Handle clear stored button
    if clear_stored_button.value:
        set_stored_fits([])

    # Initialize model if needed
    if get_model() is None:
        set_model(MLP(key, width, depth, activation))

    return X_train, y_train, X_jax, y_jax, width, depth, activation, learning_rate, n_epochs, optimizer, key, current_params


@app.cell(hide_code=True)
def _(
    jax, jnp, np, plt,
    X_train, y_train, function_dropdown, current_params,
    get_ground_truth, get_model, get_losses, get_trained,
    get_train_params, get_last_pred, get_stored_fits,
):
    # Create figure with two vertically stacked subplots (2:1 aspect ratio each)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.8))

    # Top plot: Data + Fit
    X_plot = np.linspace(-2, 2, 200)
    y_gt = get_ground_truth(X_plot, function_dropdown.value)

    # Plot ground truth
    ax1.plot(X_plot, y_gt, 'k-', lw=2, label='Ground truth', alpha=0.7)

    # Plot training data
    ax1.scatter(X_train, y_train, c='C0', s=40, alpha=0.7, label='Training data', zorder=5)

    # Plot stored fits first (so current fit appears on top)
    _stored_fits = get_stored_fits()
    _stored_colors = ['C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for _i, _fit in enumerate(_stored_fits):
        _color = _stored_colors[_i % len(_stored_colors)]
        ax1.plot(X_plot, _fit['pred'], color=_color, lw=1.5, alpha=0.8,
                 label=_fit['label'], zorder=4)
        # Plot stored loss curves with same color and label
        _stored_epochs = np.arange(1, len(_fit['losses']) + 1)
        ax2.semilogy(_stored_epochs, _fit['losses'], color=_color, lw=1.5, alpha=0.8,
                     label=_fit['label'])

    # Check if fit is stale (parameters changed since last training)
    _train_params = get_train_params()
    _last_pred = get_last_pred()
    _is_stale = _train_params is not None and _train_params != current_params

    # Plot MLP prediction if trained
    _model = get_model()
    if _model is not None and get_trained():
        if _is_stale and _last_pred is not None:
            # Show stale prediction in light grey
            ax1.plot(X_plot, _last_pred, color='lightgrey', lw=2, label='Current (stale)', zorder=3)
        else:
            # Show current prediction in color
            X_jax_plot = jnp.array(X_plot)
            _y_pred = jax.vmap(_model)(X_jax_plot)
            ax1.plot(X_plot, np.array(_y_pred), 'C1-', lw=2, label='Current fit')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_xlim(-2.1, 2.1)
    ax1.set_ylim(-1.5, 1.5)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title('Data and Model Fit')
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Loss curve
    _losses = get_losses()
    if _losses or _stored_fits:
        if _losses:
            _epochs = np.arange(1, len(_losses) + 1)
            if _is_stale:
                # Show stale loss curve in light grey
                ax2.semilogy(_epochs, _losses, color='lightgrey', lw=1.5, label='Current (stale)')
            else:
                ax2.semilogy(_epochs, _losses, 'C1-', lw=1.5, label='Current fit')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3, which='both')
        # Set x-axis limit to max epochs across all fits
        _max_epochs = max([len(_losses)] if _losses else [0] +
                         [len(f['losses']) for f in _stored_fits])
        if _max_epochs > 0:
            ax2.set_xlim(1, _max_epochs)
        if _stored_fits or _losses:
            ax2.legend(loc='upper right', fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'Click "Train" to start',
                 ha='center', va='center', transform=ax2.transAxes,
                 fontsize=14, color='gray')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.set_title('Training Loss')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_output = fig
    return (plot_output,)


@app.cell(hide_code=True)
def _(
    mo,
    n_points_slider, noise_slider, function_dropdown, seed_slider,
    width_slider, depth_slider, activation_dropdown,
    lr_slider, optimizer_dropdown, epochs_slider,
    train_button, reset_button, store_button, clear_stored_button,
):
    sidebar = mo.Html(f'''
    <div class="app-sidebar">
        <h4>Data</h4>
        {n_points_slider}
        {noise_slider}
        {function_dropdown}
        {seed_slider}

        <h4>Network Architecture</h4>
        {width_slider}
        {depth_slider}
        {activation_dropdown}

        <h4>Training</h4>
        {lr_slider}
        {optimizer_dropdown}
        {epochs_slider}

        <div style="display: flex; gap: 0.5em; margin-top: 1em; flex-wrap: wrap;">
            {train_button}
            {reset_button}
            {store_button}
            {clear_stored_button}
        </div>
    </div>
    ''')
    return (sidebar,)


@app.cell(hide_code=True)
def _(mo, header, plot_output, sidebar):
    mo.vstack([
        header,
        mo.Html(f'''
        <div class="app-layout">
            <div class="app-plot">{mo.as_html(plot_output)}</div>
            {sidebar}
        </div>
        ''')
    ])
    return


if __name__ == "__main__":
    app.run()
