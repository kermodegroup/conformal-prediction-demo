# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.1",
#     "numpy==2.2.5",
#     "popsregression==0.3.4",
#     "scikit-learn==1.6.1",
#     "seaborn==0.13.2",
# ]
# ///

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import BayesianRidge
    from sklearn.preprocessing import PolynomialFeatures

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

    from POPSRegression import POPSRegression

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
def _(BayesianRidge, POPSRegression, g, np):
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

    class MyBayesianRidge(BayesianRidge):
        def predict(self, X, return_std=False, aleatoric=False):
            y_pred = super().predict(X)
            if not return_std:
                return y_pred
            y_var = np.sum((X @ self.sigma_) * X, axis=1)
            if aleatoric:
                y_var += 1.0 / self.alpha_
            return y_pred, np.sqrt(y_var)


    class MyPOPSRegression(POPSRegression):
        def predict(self, X, return_std=False, aleatoric=False):
            res = super().predict(X, return_epistemic_std=return_std)
            if return_std:
                y_pred, y_std = res
                if aleatoric:
                    y_std = np.sqrt(y_std**2 + 1.0 / self.alpha_)
                res = (y_pred, y_std)
            return res

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

    return ConformalPrediction, MyBayesianRidge, MyPOPSRegression, get_data


@app.cell(hide_code=True)
def _(
    ConformalPrediction,
    MyBayesianRidge,
    MyPOPSRegression,
    N_samples,
    P,
    PolynomialFeatures,
    aleatoric,
    bayesian,
    calib_frac,
    conformal,
    get_data,
    np,
    plt,
    pops,
    seed,
    sigma,
    train_test_split,
    zeta,
):
    fig, ax = plt.subplots(figsize=(12, 6))
    np.random.seed(seed.value)
    X_data, y_data, X_test, y_test = get_data(N_samples.value, sigma=sigma.value)

    X_train, X_calib, y_train, y_calib = train_test_split(X_data, y_data, test_size=calib_frac.value, random_state=seed.value)
    n = len(y_calib)

    poly = PolynomialFeatures(degree=P.value-1, include_bias=True)
    Phi_train = poly.fit_transform(X_train)
    Phi_test = poly.transform(X_test)
    Phi_calib = poly.transform(X_calib)

    b = MyBayesianRidge(fit_intercept=False) 
    p = MyPOPSRegression(resampling_method='sobol', fit_intercept=False)
    c = ConformalPrediction(fit_intercept=False)

    ax.plot(X_test[:, 0], y_test, 'k-', label='Truth')
    ax.plot(X_train[:, 0], y_train, 'b.', label='Train');
    ax.plot(X_calib[:, 0], y_calib, 'c.', label='Calibration');
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
            qhat = model.calibrate(Phi_calib, y_calib, zeta=zeta.value, aleatoric=aleatoric.value)
            kwargs['rescale'] = True

        y_pred, y_std = model.predict(Phi_test, **kwargs)
        if label == 'Bayesian uncertainty':
            ax.plot(X_test[:, 0], y_pred, color=color, label='Mean Prediction', lw=3)
        ax.fill_between(X_test[:, 0], y_pred - y_std, y_pred + y_std, alpha=0.5, color=color, label=label)

    caption = fr'$N=${N_samples.value} data, $\sigma$={sigma.value:.2f} noise'
    if bayesian.value or conformal.value:
        caption += fr', $P=${P.value} params'
    if conformal.value:
        caption += fr', $n=${n} calib,  $\zeta$={zeta.value:.2f},  $\hat{{q}}=${qhat:.1f}'
    ax.set_title(caption)
    # print(caption)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.legend(loc='lower left')
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    bayesian = mo.ui.checkbox(False, label="Bayesian fit")
    conformal = mo.ui.checkbox(False, label="Conformal prediction")
    pops = mo.ui.checkbox(False, label="POPS regression")
    aleatoric = mo.ui.checkbox(False, label="Aleatoric uncertainty")
    N_samples = mo.ui.slider(50, 1000, 50, 500, label='Data samples $N$')
    sigma = mo.ui.slider(0.001, 0.3, 0.005, 0.1, label=r'$\sigma$ noise')
    calib_frac = mo.ui.slider(0.05, 0.5, 0.05, 0.2, label="Calibration fraction")
    P = mo.ui.slider(5, 15, 1, 10, label="Parameters $P$")
    zeta = mo.ui.slider(0.05, 0.3, 0.05, label=r"Coverage $\zeta$")
    seed = mo.ui.slider(0, 10, label="Random seed")
    mo.hstack([
        mo.vstack([mo.left(bayesian), mo.left(conformal), mo.left(aleatoric)]),
        mo.vstack([N_samples, sigma, calib_frac]),
        mo.vstack([P, zeta, seed])
    ])
    return (
        N_samples,
        P,
        aleatoric,
        bayesian,
        calib_frac,
        conformal,
        pops,
        seed,
        sigma,
        zeta,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
