# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains interactive research demonstrations built with [marimo](https://marimo.io), a reactive Python notebook framework. Demos are deployed to two locations:
- **WASM notebooks**: https://kermodegroup.github.io/demos (public, no auth)
- **Live notebooks**: https://sciml.warwick.ac.uk (SSO protected)

## Repository Structure

```
demos/
├── apps/                    # WASM-compatible notebooks (static HTML export)
│   └── lib/                 # Shared library modules
├── notebooks/               # Live server notebooks (require native deps)
├── scripts/
│   ├── build.py             # WASM HTML export script
│   ├── categorize_notebooks.py  # WASM vs live detection
│   ├── generate_index.py    # GitHub Pages index generator
│   ├── deploy.sh            # Server-side deployment (systemd + nginx)
│   └── deploy-warwick.sh    # Local deployment script (live notebooks only)
├── server/
│   └── app.py               # FastAPI server for live notebooks
└── .github/workflows/
    └── pages.yml            # GitHub Actions for WASM deployment
```

## Deployment Architecture

### Split Deployment

Notebooks are automatically categorized and deployed to different hosts:

1. **WASM (GitHub Pages)** - Notebooks in `apps/` with pure Python dependencies
   - Deployed via GitHub Actions to kermodegroup.github.io/demos
   - Exported to HTML+WASM via `marimo export html-wasm`
   - No authentication required
   - Runs entirely in browser via Pyodide

2. **Live (sciml.warwick.ac.uk)** - Notebooks in `notebooks/` with native dependencies
   - Deployed manually via `deploy-warwick.sh`
   - Each notebook runs as a separate marimo process
   - Proxied through nginx with WebSocket support
   - Protected by University of Warwick SSO

### WASM Incompatible Packages

The `scripts/categorize_notebooks.py` detects these packages and routes to live:
- ML frameworks: `jax`, `torch`, `tensorflow`
- File system: `watchdog`, `psutil`
- Database drivers: `psycopg2`, `mysqlclient`
- Native extensions: `opencv-python`, `cryptography`, `grpcio`, `pyarrow`, etc.

## Commands

### Local Development

```bash
# Run a notebook locally
marimo run apps/regression-demo.py
marimo edit notebooks/jax-test.py

# Test WASM compatibility detection
python scripts/categorize_notebooks.py --offline

# Build WASM notebooks locally
python scripts/build.py --sync-lib --output-dir _wasm_site
```

### Deployment

**WASM notebooks** deploy automatically via GitHub Actions when changes are pushed to `apps/`.

**Live notebooks** require manual deployment (2FA authentication):

```bash
# Deploy live notebooks to sciml.warwick.ac.uk
./scripts/deploy-warwick.sh
```

The deploy script:
1. Categorizes notebooks into WASM vs live
2. Syncs dependencies on server
3. Deploys live notebooks to `~/marimo-server/notebooks/`
4. Restarts marimo service on server

## Server Infrastructure

**Live server:** sciml.warwick.ac.uk

```
/home/ubuntu/marimo-server/
├── app.py           # FastAPI entry point
├── deploy.sh        # Server-side deployment script
├── notebooks/       # Live notebook .py files
└── .venv/           # Python environment with marimo + deps
```

**Services:**
- nginx: SSL termination, SSO auth, reverse proxy
- systemd: marimo server process
- Let's Encrypt: SSL certificates

## GitHub Pages Configuration

After initial setup, configure GitHub Pages in repo settings:
1. Go to Settings → Pages
2. Change Source from "Deploy from a branch" to "GitHub Actions"

## Marimo Notebook Conventions

- Include dependencies in PEP 723 script metadata at top of file
- Avoid `watchdog` and other dev-only deps that break WASM
- Use `mo.ui.*` for interactive elements
- Last expression in cell is auto-displayed
- For matplotlib: use `plt.gca()` not `plt.show()`

## Library Modules (`apps/lib/`)

Shared code for regression demos:

| Module | Description |
|--------|-------------|
| `models.py` | `MyBayesianRidge`, `ConformalPrediction`, `NeuralNetworkRegression`, `QuantileRegressionUQ` |
| `kernels.py` | GP kernels (RBF, Matérn, bump, polynomial) |
| `basis.py` | Basis feature functions (RBF, Fourier, LJ) |
| `metrics.py` | Probabilistic metrics (log likelihood, CRPS) |
| `data.py` | Ground truth functions and data generation |
| `optimization.py` | GP hyperparameter optimization |

**Sync workflow:** When modifying lib/, run `--check-sync` to verify inline copies in notebooks match.
