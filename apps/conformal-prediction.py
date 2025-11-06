# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "watchdog",
#     "matplotlib==3.10.1",
#     "numpy==2.2.5",
#     "popsregression==0.3.4",
#     "scikit-learn==1.6.1",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.17.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import PolynomialFeatures
    from POPSRegression import POPSRegression

    class RadialBasisFunctions:
        """
        A set of linear basis functions.

        Arguments:
        X   -  The centers of the radial basis functions.
        ell -  The assumed lengthscale.
        """

        def __init__(self, X, ell):
            self.X = X
            self.ell = ell
            self.num_basis = X.shape[0]

        def __call__(self, x):
            return np.exp(-0.5 * (x - self.X) ** 2 / self.ell**2).flatten()

    def design_matrix(X, phi):
        """
        Arguments:

        X   -  The observed inputs
        phi -  The basis functions
        """
        num_observations = X.shape[0]
        num_basis = phi.num_basis
        Phi = np.zeros((num_observations, num_basis))
        for i in range(num_observations):
            Phi[i, :] = phi(X[i, :])
        return Phi    

    # Customize default plotting style
    import seaborn as sns
    sns.set_context('talk')
    return (
        BayesianRidge,
        POPSRegression,
        PolynomialFeatures,
        mo,
        np,
        plt,
        train_test_split,
    )


@app.cell
def _(np):
    def noise(size, variance):
        return np.random.normal(scale=np.sqrt(variance), size=size)

    def g(X, noise_variance):
        return np.sin(X) + noise(X.shape, noise_variance)
    return (g,)


@app.cell
def _(BayesianRidge, np):
    class MyBayesianRidge(BayesianRidge):
        def predict(self, X, return_std=False, aleatoric=False):
            y_pred = super().predict(X)
            if not return_std:
                return y_pred
            y_var = np.sum((X @ self.sigma_) * X, axis=1)
            if aleatoric:
                y_var += 1.0 / self.alpha_
            return y_pred, np.sqrt(y_var)

    class MyPOPSRegression(MyBayesianRidge):
        def fit(self, X, y, prior=None, clipping=0.05, n_samples=100):
            super().fit(X, y)       
            num_observations, num_basis = X.shape
            if prior is None:
                prior = np.eye(num_basis)
            H = prior.T @ prior + X.T @ X
            dθ = np.zeros((num_observations, num_basis))
            for i in range(num_observations):
                V = np.linalg.solve(H, X[i, :])
                leverage = X[i, :].T @ V
                E        = X[i, :].T @ self.coef_
                dy       = y[i] - E
                dθ[i, :] = (dy / leverage) * V
            self._dθ = dθ

            U, S, Vh = np.linalg.svd(self._dθ, full_matrices=False)
            projected = self._dθ @ Vh.T
            num_basis = projected.shape[1]
            lower  = [np.quantile(projected[:, i], clipping) for i in range(num_basis) ]
            upper  = [np.quantile(projected[:, i], 1.0 - clipping) for i in range(num_basis) ] 
            bounds = np.c_[[lower, upper]].T

            δθ = np.zeros((n_samples, num_basis))
            for j in range(n_samples):
                u = np.random.uniform(num_basis)
                δθ[j, :] = (Vh @ (bounds[:, 0] + bounds[:, 1] * u)) + self.coef_
            self._misspecification_sigma = δθ.T @ δθ / n_samples

        def predict(self, X, return_std=False, aleatoric=False):
            y_pred = super().predict(X)
            if return_std:
                y_std = ((X @ self._misspecification_sigma) * X).sum(axis=1)
                if aleatoric:
                    y_std = np.sqrt(y_std**2 + 1.0 / self.alpha_)
                return (y_pred, y_std)
            else:
                return y_pred        

    class ConformalPrediction(MyBayesianRidge):
        def get_scores(self, X, y, aleatoric=False):
            y_pred, y_std = self.predict(X, return_std=True, rescale=False, aleatoric=aleatoric)
            residuals = (y_pred - y) / y_std
            scores = np.abs(residuals)
            return scores

        def calibrate(self, X_calib, y_calib, zeta=0.05, aleatoric=False):
            scores = self.get_scores(X_calib, y_calib, aleatoric=aleatoric)
            n = float(len(y_calib))
            self.qhat = np.quantile(scores, np.ceil((n+1)*(1.0-zeta))/n, method='higher')
            return self.qhat

        def predict(self, X, return_std=False, aleatoric=False, rescale=True):
            res = super().predict(X, return_std=return_std, aleatoric=aleatoric)
            if not return_std:
                return res
            y_pred, y_std = res
            if rescale:
                y_std = y_std * self.qhat
            return y_pred, y_std
    return ConformalPrediction, MyBayesianRidge


@app.cell(hide_code=True)
def _(
    ConformalPrediction,
    MyBayesianRidge,
    N_samples,
    POPSRegression,
    PolynomialFeatures,
    aleatoric,
    bayesian,
    conformal,
    g,
    get_N_samples,
    get_P,
    get_calib_frac,
    get_leverage_percentile,
    get_percentile_clipping,
    get_seed,
    get_sigma,
    get_zeta,
    np,
    plt,
    pops,
    seed,
    sigma,
    train_test_split,
):
    def get_data(N_samples=500, sigma=0.1):
        x_train = np.append(np.random.uniform(-10, 10, size=N_samples), np.linspace(-10, 10, 2))
        x_train = x_train[(x_train < 0) | (x_train > 5.0)]
        x_train = np.sort(x_train)
        y_train = g(x_train, noise_variance=sigma**2)
        X_train = x_train[:, None]

        x_test = np.linspace(-10, 10, 1000)
        y_test = g(x_test, 0)
        X_test = x_test[:, None]

        return X_train, y_train, X_test, y_test

    fig, ax = plt.subplots(figsize=(12, 6))
    np.random.seed(seed.value)
    X_data, y_data, X_test, y_test = get_data(N_samples.value, sigma=sigma.value)

    X_train, X_calib, y_train, y_calib = train_test_split(X_data, y_data, test_size=get_calib_frac(), random_state=get_seed())
    n = len(y_calib)

    poly = PolynomialFeatures(degree=get_P()-1, include_bias=True)
    Phi_train = poly.fit_transform(X_train)
    Phi_test = poly.transform(X_test)
    Phi_calib = poly.transform(X_calib)

    b = MyBayesianRidge(fit_intercept=False) 
    p = POPSRegression(fit_intercept=False, percentile_clipping=get_percentile_clipping(), leverage_percentile=get_leverage_percentile())
    c = ConformalPrediction(fit_intercept=False)

    ax.plot(X_test[:, 0], y_test, 'k-', label='Truth')
    ax.plot(X_train[:, 0], y_train, 'b.', label='Train')
    ax.plot(X_calib[:, 0], y_calib, 'c.', label='Calibration')
    ax.axvline(0.0, ls='--', color='k')
    ax.axvline(5.0, ls='--', color='k')

    for model, color, label in zip((b, c, p), ('C2', 'C1', 'C0'), 
                                   ('Bayesian uncertainty', 'Conformal prediction', 'POPS regression')):

        if label == 'Bayesian uncertainty' and not bayesian.value:
            continue

        if label == 'Conformal prediction' and not conformal.value:
            continue

        if label == 'POPS regression' and not pops.value:
            continue

        model.fit(Phi_train, y_train)
        kwargs = {
            'return_std': True,
            'aleatoric': aleatoric.value,
        }        
        if label == 'Conformal prediction':
            qhat = model.calibrate(Phi_calib, y_calib, zeta=get_zeta(), aleatoric=aleatoric.value)
            kwargs['rescale'] = True

        if label == 'POPS regression':
            y_pred, y_std = model.predict(Phi_test, return_std=True)
            if aleatoric.value:
                y_std = np.sqrt(y_std**2 + 1.0 / model.alpha_)
        else:
            y_pred, y_std = model.predict(Phi_test, **kwargs)
        ax.plot(X_test[:, 0], y_pred, color=color, lw=3)
        ax.fill_between(X_test[:, 0], y_pred - y_std, y_pred + y_std, alpha=0.5, color=color, label=label)

    caption = fr'$N=${get_N_samples()} data, $\sigma$={get_sigma():.2f} noise'
    if bayesian.value or conformal.value:
        caption += fr', $P=${get_P()} params'
    if conformal.value:
        caption += fr', $n=${n} calib,  $\zeta$={get_zeta():.2f},  $\hat{{q}}=${qhat:.1f}'
    ax.set_title(caption)
    # print(caption)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(loc='lower left')
    fig
    return


@app.cell
def _(
    aleatoric,
    bayesian,
    conformal,
    get_N_samples,
    get_P,
    get_calib_frac,
    get_leverage_percentile,
    get_percentile_clipping,
    get_seed,
    get_sigma,
    get_zeta,
    mo,
    pops,
    set_N_samples,
    set_P,
    set_calib_frac,
    set_leverage_percentile,
    set_percentile_clipping,
    set_seed,
    set_sigma,
    set_zeta,
):
    data_label = mo.md("**Dataset parameters**")
    N_samples = mo.ui.slider(50, 1000, 50, get_N_samples(), label='Samples $N$', on_change=set_N_samples)
    sigma = mo.ui.slider(0.001, 0.3, 0.005, get_sigma(), label=r'$\sigma$ noise', on_change=set_sigma)
    seed = mo.ui.slider(0, 10, get_seed(), label="Random seed", on_change=set_seed)

    # Regression parameters with conditional styling
    reg_enabled = bayesian.value or conformal.value or pops.value

    if reg_enabled:
        reg_label = mo.md("**Regression parameters**")
        P_elem = mo.ui.slider(5, 15, 1, get_P(), label="Fit parameters $P$", on_change=set_P)
    else:
        reg_label = mo.Html("<p style='color: #d0d0d0; font-weight: bold;'>Regression parameters</p>")
        P_elem = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(5, 15, 1, get_P(), label='Fit parameters $P$', disabled=True, on_change=set_P)}</div>")

    # Conformal prediction section with conditional styling
    if conformal.value:
        cp_label = mo.md("**Conformal prediction parameters**")
        calib_frac = mo.ui.slider(0.05, 0.5, 0.05, get_calib_frac(), label="Calibration fraction", on_change=set_calib_frac)
        zeta = mo.ui.slider(0.05, 0.3, 0.05, get_zeta(), label=r"Coverage $\zeta$", on_change=set_zeta)
    else:
        cp_label = mo.Html("<p style='color: #d0d0d0; font-weight: bold;'>Conformal prediction parameters</p>")
        calib_frac = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.05, 0.5, 0.05, get_calib_frac(), label='Calibration fraction', disabled=True, on_change=set_calib_frac)}</div>")
        zeta = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0.05, 0.3, 0.05, get_zeta(), label=r'Coverage $\zeta$', disabled=True, on_change=set_zeta)}</div>")

    # POPS regression section with conditional styling
    if pops.value:
        pops_label = mo.md("**POPS regression parameters**")
        percentile_clipping = mo.ui.slider(0, 10, 1, get_percentile_clipping(), label="Percentile clipping", on_change=set_percentile_clipping)
        leverage_percentile = mo.ui.slider(0, 99, 5, get_leverage_percentile(), label="Leverage percentile", on_change=set_leverage_percentile)
    else:
        pops_label = mo.Html("<p style='color: #d0d0d0; font-weight: bold;'>POPS regression parameters</p>")
        percentile_clipping = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0, 10, 1, get_percentile_clipping(), label='Percentile clipping', disabled=True, on_change=set_percentile_clipping)}</div>")
        leverage_percentile = mo.Html(f"<div style='opacity: 0.4;'>{mo.ui.slider(0, 99, 5, get_leverage_percentile(), label='Leverage percentile', disabled=True, on_change=set_leverage_percentile)}</div>")

    mo.hstack([
        mo.vstack([mo.left(bayesian), mo.left(conformal), mo.left(pops), mo.left(aleatoric)]),
        mo.vstack([data_label, N_samples, sigma, seed, reg_label, P_elem]),
        mo.vstack([cp_label, calib_frac, zeta]),
        mo.vstack([pops_label, percentile_clipping, leverage_percentile])
    ])
    return N_samples, seed, sigma


@app.cell(hide_code=True)
def _(mo):
    bayesian = mo.ui.checkbox(False, label="Bayesian fit")
    conformal = mo.ui.checkbox(False, label="Conformal prediction")
    pops = mo.ui.checkbox(False, label="POPS regression")
    aleatoric = mo.ui.checkbox(False, label="Aleatoric uncertainty")
    return aleatoric, bayesian, conformal, pops


@app.cell
def _(mo):
    # Use marimo state to preserve all slider values
    get_N_samples, set_N_samples = mo.state(500)
    get_sigma, set_sigma = mo.state(0.1)
    get_seed, set_seed = mo.state(0)
    get_P, set_P = mo.state(10)
    get_calib_frac, set_calib_frac = mo.state(0.2)
    get_zeta, set_zeta = mo.state(0.05)
    get_percentile_clipping, set_percentile_clipping = mo.state(0)
    get_leverage_percentile, set_leverage_percentile = mo.state(50)
    return (
        get_N_samples,
        get_P,
        get_calib_frac,
        get_leverage_percentile,
        get_percentile_clipping,
        get_seed,
        get_sigma,
        get_zeta,
        set_N_samples,
        set_P,
        set_calib_frac,
        set_leverage_percentile,
        set_percentile_clipping,
        set_seed,
        set_sigma,
        set_zeta,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
