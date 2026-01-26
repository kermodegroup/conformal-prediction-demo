# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "jax",
#     "jaxlib",
#     "matplotlib",
# ]
# ///

import marimo

__generated_with = "0.19.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # JAX Test Notebook

        This notebook tests live server deployment with JAX (incompatible with WASM).
        """
    )
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    return jax, jnp


@app.cell
def _(jax):
    # Show JAX configuration
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    return


@app.cell
def _(jnp):
    # Simple JAX computation
    x = jnp.linspace(0, 2 * jnp.pi, 100)
    y = jnp.sin(x)
    return x, y


@app.cell
def _(x, y):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.title("sin(x) computed with JAX")
    plt.xlabel("x")
    plt.ylabel("sin(x)")
    plt.grid(True)
    plt.gca()
    return (plt,)


@app.cell
def _(jax, jnp):
    # Demonstrate JIT compilation
    @jax.jit
    def compute(a, b):
        return jnp.dot(a, b)

    a = jnp.ones((100, 100))
    b = jnp.ones((100, 100))
    result = compute(a, b)
    print(f"Matrix multiplication result shape: {result.shape}")
    print(f"Sum of result: {result.sum()}")
    return a, b, compute, result


if __name__ == "__main__":
    app.run()
